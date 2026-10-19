package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.globalId
import io.github.rajumark.kernelin.kernelSource
import kotlinx.coroutines.async
import org.junit.Assume.assumeTrue
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
 * Runs kernels on the device's real GPU through AndroidX WebGPU and compares every result with the same Kotlin on
 * the CPU. Run on a phone or emulator: gradle :kernelin-webgpu:connectedAndroidDeviceTest
 */
class AndroidGpuTest {
    private lateinit var gpu: WebGpu

    /** Emulators only offer Dawn's "Null" adapter, which runs nothing: these tests are skipped there, not passed. */
    @BeforeTest
    fun openGpu() {
        gpu = try {
            runBlocking { WebGpu.create() }
        } catch (e: GpuException) {
            assumeTrue("No real GPU on this device: ${e.message}", false)
            return
        }
    }

    @AfterTest
    fun closeGpu() {
        if (::gpu.isInitialized) gpu.close()
    }

    @Test
    fun describesTheGpu() {
        println("Kernelin Android GPU: ${gpu.description}")
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
        // Float rounding can move a few pixels on the edge of the set by an iteration or two.
        val off = expected.indices.count { expected[it] != gpuOut[it] }
        assertTrue(off < w * h / 500, "$off of ${w * h} pixels differ")
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
}
