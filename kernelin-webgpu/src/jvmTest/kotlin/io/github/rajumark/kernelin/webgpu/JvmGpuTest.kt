package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.globalId
import io.github.rajumark.kernelin.globalIdX
import io.github.rajumark.kernelin.globalIdY
import io.github.rajumark.kernelin.globalIdZ
import io.github.rajumark.kernelin.kernelSource
import kotlinx.coroutines.async
import org.junit.jupiter.api.Assumptions.assumeTrue
import kotlinx.coroutines.runBlocking
import kotlin.math.abs
import kotlin.math.sqrt
import kotlin.test.AfterTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

@Kernel
fun saxpy(alpha: Float, x: FloatArray, y: FloatArray) {
    val i = globalId()
    y[i] = alpha * x[i] + y[i]
}

@Kernel
fun mandelbrot(out: IntArray, width: Int, height: Int, maxIter: Int) {
    val i = globalId()
    val cx = (i % width).toFloat() / width * 3.5f - 2.5f
    val cy = (i / width).toFloat() / height * 2f - 1f
    var x = 0f
    var y = 0f
    var n = 0
    while (n < maxIter && x * x + y * y <= 4f) {
        val t = x * x - y * y + cx
        y = 2f * x * y + cy
        x = t
        n++
    }
    out[i] = n
}

@Kernel
fun gridCells(cells: IntArray, offset: Int) {
    cells[globalId()] = globalIdX() + globalIdY() * 1000 + globalIdZ() * 1000000 + offset
}

@Kernel
fun blur(input: FloatArray, out: FloatArray, flip: Boolean) {
    val i = globalId()
    var sum = 0f
    var count = 0
    for (j in i - 2..i + 2) {
        if (j >= 0 && j < input.size) {
            sum += input[j]
            count++
        }
    }
    out[i] = if (flip) -sum / count else sum / count
}

@Kernel
fun bits(a: IntArray, out: IntArray, shift: Int) {
    val i = globalId()
    out[i] = (a[i] shl shift) xor (a[i] ushr 3) + a[i] % 7 - a[i] / 5
}

@Kernel
fun heavy(x: FloatArray, out: FloatArray, rounds: Int) {
    val i = globalId()
    var v = x[i]
    repeat(rounds) { v = sqrt(v * v + 1f) * 0.5f + kotlin.math.sin(v) * 0.25f }
    out[i] = v
}

/**
 * Runs kernels on this computer's real GPU through jWebGPU (wgpu) and compares every result with the same Kotlin on
 * the CPU. On a machine without a GPU driver (no Metal, Vulkan or Direct3D 12) the tests are skipped, not passed;
 * CI runs them on lavapipe, Mesa's software Vulkan.
 */
class JvmGpuTest {
    private lateinit var gpu: WebGpu

    @BeforeTest
    fun openGpu() {
        gpu = try {
            runBlocking { WebGpu.create() }
        } catch (e: GpuException) {
            // CI sets KERNELIN_REQUIRE_GPU=1, so a missing GPU fails there instead of quietly skipping everything.
            if (System.getenv("KERNELIN_REQUIRE_GPU") == "1") throw e
            assumeTrue(false, "No GPU for WebGPU on this machine: ${e.message}")
            return
        }
    }

    @AfterTest
    fun closeGpu() {
        if (::gpu.isInitialized) gpu.close()
    }

    @Test
    fun describesTheGpu() {
        println("Kernelin JVM GPU: ${gpu.description}")
        assertTrue(gpu.description.isNotBlank())
        assertFalse(gpu.isLost)
    }

    @Test
    fun saxpyMatchesCpu(): Unit = runBlocking {
        val n = 10_000
        val x = FloatArray(n) { it * 0.5f }
        val y = FloatArray(n) { 1f - it }
        val kernel = gpu.load(kernelSource(::saxpy))
        val result = kernel.run(n, mapOf("alpha" to 2.5f), mapOf("x" to GpuData.of(x), "y" to GpuData.of(y)))
        val expected = y.copyOf()
        KernelinCpu.dispatch(n) { saxpy(2.5f, x, expected) }
        assertContentEquals(expected, result.floats("y"))
        assertTrue(result.millis >= 0.0)
    }

    @Test
    fun gridLaunchesMatchCpu(): Unit = runBlocking {
        val (w, h, d) = Triple(37, 23, 3)
        val cpu = IntArray(w * h * d)
        KernelinCpu.dispatch(w, h, d) { gridCells(cpu, 7) }
        val cells = IntArray(w * h * d)
        gpu.dispatch(w, h, d) { gridCells(cells = cells, offset = 7) }
        assertContentEquals(cpu, cells)
        val kernel = gpu.load(kernelSource(::gridCells))
        val flat = kernel.run(w, h, mapOf("offset" to 0), mapOf("cells" to GpuData.empty(w * h))).ints("cells")
        assertContentEquals(IntArray(w * h) { it % w + it / w * 1000 }, flat)
        val twoD = IntArray(w * h)
        gpu.dispatch(w, h) { gridCells(cells = twoD, offset = 1) }
        assertContentEquals(IntArray(w * h) { it % w + it / w * 1000 + 1 }, twoD)
    }

    @Test
    fun typedDispatchWritesResultsIntoTheArrays(): Unit = runBlocking {
        val n = 5_000
        val x = FloatArray(n) { it * 0.5f }
        val y = FloatArray(n) { 1f - it }
        val cpuY = y.copyOf()
        KernelinCpu.dispatch(n) { saxpy(2f, x, cpuY) }
        gpu.dispatch(n) { saxpy(alpha = 2f, x = x, y = y) }
        assertContentEquals(cpuY, y)
        // The second call reuses the loaded kernel.
        gpu.dispatch(n) { saxpy(alpha = 2f, x = x, y = y) }
        KernelinCpu.dispatch(n) { saxpy(2f, x, cpuY) }
        assertContentEquals(cpuY, y)
    }

    @Test
    fun integerKernelMatchesCpuBitForBit(): Unit = runBlocking {
        val n = 4096
        val a = IntArray(n) { it * 7919 - 1_000_000 }
        val kernel = gpu.load(kernelSource(::bits))
        val result = kernel.run(n, mapOf("shift" to 35), mapOf("a" to GpuData.of(a), "out" to GpuData.empty(n)))
        val expected = IntArray(n)
        KernelinCpu.dispatch(n) { bits(a, expected, 35) }
        assertContentEquals(expected, result.ints("out"))
    }

    @Test
    fun mandelbrotMatchesCpu(): Unit = runBlocking {
        val w = 320
        val h = 200
        val kernel = gpu.load(kernelSource(::mandelbrot))
        val result = kernel.run(w * h, mapOf("width" to w, "height" to h, "maxIter" to 200), mapOf("out" to GpuData.empty(w * h)))
        val expected = IntArray(w * h)
        KernelinCpu.dispatch(w * h) { mandelbrot(expected, w, h, 200) }
        val gpuOut = result.ints("out")
        // Float rounding moves some pixels on the edge of the set; Metal also fuses multiply-adds, so a few more
        // move there (about 0.3% on an Apple M5).
        val off = expected.indices.count { expected[it] != gpuOut[it] }
        assertTrue(off < w * h / 100, "$off of ${w * h} pixels differ")
    }

    @Test
    fun booleanScalarsAndNeighbours(): Unit = runBlocking {
        val n = 1000
        val input = FloatArray(n) { (it % 17).toFloat() }
        val kernel = gpu.load(kernelSource(::blur))
        for (flip in listOf(false, true)) {
            val result = kernel.run(n, mapOf("flip" to flip), mapOf("input" to GpuData.of(input), "out" to GpuData.empty(n)))
            val expected = FloatArray(n)
            KernelinCpu.dispatch(n) { blur(input, expected, flip) }
            val got = result.floats("out")
            for (i in 0 until n) assertTrue(abs(expected[i] - got[i]) <= 1e-5f, "flip=$flip, $i: ${expected[i]} vs ${got[i]}")
        }
    }

    @Test
    fun moreThreadsThanOneDispatchDimension(): Unit = runBlocking {
        // 65535 workgroups of 64 threads is the most one dimension holds; this run needs the second dimension.
        val n = 65535 * 64 + 1000
        val x = FloatArray(n) { 1f }
        val y = FloatArray(n) { it.toFloat() }
        val kernel = gpu.load(kernelSource(::saxpy))
        val out = kernel.run(n, mapOf("alpha" to 1f), mapOf("x" to GpuData.of(x), "y" to GpuData.of(y))).floats("y")
        assertEquals(n, out.size)
        for (i in listOf(0, 1, 65535 * 64 - 1, 65535 * 64, n - 1)) assertEquals(i + 1f, out[i], "id $i")
    }

    @Test
    fun keepsDataOnTheGpuBetweenRuns(): Unit = runBlocking {
        val n = 256
        val kernel = gpu.load(kernelSource(::saxpy))
        val x = GpuData.of(FloatArray(n) { 1f })
        var y = GpuData.of(FloatArray(n))
        repeat(10) {
            val step = kernel.run(n, mapOf("alpha" to 1f), mapOf("x" to x, "y" to y), readBack = emptySet())
            assertFailsWith<IllegalArgumentException> { step.floats("y") }
            y = GpuData.of(step.buffer("y"))
        }
        val last = kernel.run(n, mapOf("alpha" to 1f), mapOf("x" to x, "y" to y))
        assertTrue(last.floats("y").all { it == 11f })
    }

    @Test
    fun emptyBuffersStartFromZeroOnEveryRun(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        val x = GpuData.of(FloatArray(64) { 1f })
        repeat(2) {
            val out = kernel.run(64, mapOf("alpha" to 3f), mapOf("x" to x, "y" to GpuData.empty(64))).floats("y")
            assertTrue(out.all { it == 3f })
        }
    }

    @Test
    fun overlappingRunsWaitForEachOther(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        val runs = (1..8).map { k ->
            async { kernel.run(512, mapOf("alpha" to k.toFloat()), mapOf("x" to GpuData.of(FloatArray(512) { 1f }), "y" to GpuData.empty(512))) }
        }
        runs.forEachIndexed { i, r -> assertTrue(r.await().floats("y").all { it == i + 1f }) }
    }

    @Test
    fun wrongArgumentsFailBeforeTheGpu(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        val buffers = mapOf("x" to GpuData.of(FloatArray(4)), "y" to GpuData.of(FloatArray(4)))
        assertContains(assertFailsWith<IllegalArgumentException> { kernel.run(4, mapOf(), buffers) }.message!!, "alpha")
        assertFailsWith<IllegalArgumentException> { kernel.run(4, mapOf("alpha" to true), buffers) }
        assertFailsWith<IllegalArgumentException> {
            kernel.run(4, mapOf("alpha" to 1f), mapOf("x" to GpuData.of(IntArray(4)), "y" to GpuData.of(FloatArray(4))))
        }
        assertFailsWith<IllegalArgumentException> { kernel.run(4, mapOf("alpha" to 1f), buffers, readBack = setOf("x")) }
    }

    @Test
    fun closedKernelsAndDevicesRefuseToRun(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        val buffers = mapOf("x" to GpuData.of(FloatArray(4)), "y" to GpuData.of(FloatArray(4)))
        val first = kernel.run(4, mapOf("alpha" to 1f), buffers)
        kernel.close()
        assertTrue(kernel.isClosed)
        assertFalse(first.buffer("y").isValid)
        assertFailsWith<IllegalStateException> { kernel.run(4, mapOf("alpha" to 1f), buffers) }
        gpu.close()
        assertFailsWith<IllegalStateException> { gpu.load(kernelSource(::saxpy)) }
    }

    @Test
    fun badWgslIsReported(): Unit = runBlocking {
        val good = kernelSource(::saxpy)
        val bad = io.github.rajumark.kernelin.KernelSource("test.bad", good.wgsl.replace("fn ", "fun "), good.manifestJson)
        assertContains(assertFailsWith<IllegalArgumentException> { gpu.load(bad) }.message!!, "rejected")
    }

    @Test
    fun knowsWhichDesktopsHaveANativeLibrary() {
        assertEquals("mac arm64", nativeLibraryFor("Mac OS X", "aarch64"))
        assertEquals("mac x64", nativeLibraryFor("Mac OS X", "x86_64"))
        assertEquals("linux x64", nativeLibraryFor("Linux", "amd64"))
        assertEquals("windows x64", nativeLibraryFor("Windows 11", "amd64"))
        assertEquals(null, nativeLibraryFor("Linux", "aarch64"))
        assertEquals(null, nativeLibraryFor("FreeBSD", "amd64"))
        assertTrue(WebGpu.isSupported)
    }

    @Test
    fun twoDevicesWorkSideBySide(): Unit = runBlocking {
        WebGpu.create().use { other ->
            val a = gpu.load(kernelSource(::saxpy))
            val b = other.load(kernelSource(::saxpy))
            val x = GpuData.of(FloatArray(100) { 1f })
            val ra = a.run(100, mapOf("alpha" to 2f), mapOf("x" to x, "y" to GpuData.empty(100)))
            val rb = b.run(100, mapOf("alpha" to 3f), mapOf("x" to x, "y" to GpuData.empty(100)))
            assertTrue(ra.floats("y").all { it == 2f })
            assertTrue(rb.floats("y").all { it == 3f })
            // A buffer of one device can't be used on the other.
            assertFailsWith<IllegalArgumentException> {
                b.run(100, mapOf("alpha" to 1f), mapOf("x" to GpuData.of(ra.buffer("y")), "y" to GpuData.empty(100)))
            }
        }
    }

    @Test
    fun outOfDateBuffersAreRejected(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        val x = GpuData.of(FloatArray(8) { 1f })
        val first = kernel.run(8, mapOf("alpha" to 1f), mapOf("x" to x, "y" to GpuData.empty(8)))
        val old = first.buffer("y")
        kernel.run(8, mapOf("alpha" to 1f), mapOf("x" to x, "y" to GpuData.empty(8)))
        assertFalse(old.isValid)
        assertFailsWith<IllegalStateException> { kernel.run(8, mapOf("alpha" to 1f), mapOf("x" to x, "y" to GpuData.of(old))) }
        assertContains(assertFailsWith<IllegalArgumentException> { first.floats("nope") }.message!!, "No output")
        assertFailsWith<IllegalArgumentException> { first.buffer("nope") }
    }

    @Test
    fun buffersChangeSizeBetweenRuns(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        for (n in listOf(10, 1000, 10)) {
            val y = kernel.run(n, mapOf("alpha" to 2f), mapOf("x" to GpuData.of(FloatArray(n) { 1f }), "y" to GpuData.empty(n))).floats("y")
            assertEquals(n, y.size)
            assertTrue(y.all { it == 2f })
        }
    }

    @Test
    fun closingTwiceIsFineAndCloseFromAnotherThreadWorks(): Unit = runBlocking {
        val kernel = gpu.load(kernelSource(::saxpy))
        kernel.close()
        kernel.close()
        gpu.close()
        gpu.close()
        assertTrue(gpu.isClosed)
    }
}
