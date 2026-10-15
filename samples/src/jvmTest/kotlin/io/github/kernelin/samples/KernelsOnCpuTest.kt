package io.github.kernelin.samples

import io.github.kernelin.KernelinCpu
import io.github.kernelin.KernelinShaders
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals

/** The same @Kernel functions that run on the GPU are ordinary Kotlin, so they can be unit tested on the CPU. */
class KernelsOnCpuTest {
    @Test
    fun vectorAdd() {
        val a = floatArrayOf(1f, 2f, 3f)
        val b = floatArrayOf(10f, 20f, 30f)
        val out = FloatArray(3)
        KernelinCpu.dispatch(3) { vectorAdd(a, b, out) }
        assertEquals(listOf(11f, 22f, 33f), out.toList())
    }

    @Test
    fun saxpy() {
        val x = floatArrayOf(1f, 2f)
        val y = floatArrayOf(1f, 1f)
        KernelinCpu.dispatch(2) { saxpy(3f, x, y) }
        assertEquals(listOf(4f, 7f), y.toList())
    }

    @Test
    fun `mandelbrot - origin is inside the set, far corner escapes at once`() {
        val w = 7
        val h = 4
        val out = IntArray(w * h)
        KernelinCpu.dispatch(w * h) { mandelbrot(out, w, h, 100) }
        // Pixel (5, 2) maps to (0.0, 0.0), which never escapes.
        assertEquals(100, out[2 * w + 5])
        // Pixel (0, 0) maps to (-2.5, -1.0), which escapes on the first iteration.
        assertEquals(1, out[0])
    }

    @Test
    fun `blur averages the neighbourhood and handles the edges`() {
        val input = FloatArray(20) { 1f }
        val out = FloatArray(20)
        KernelinCpu.dispatch(20) { blur(input, out) }
        assertEquals(List(20) { 1f }, out.toList())
    }

    @Test
    fun `shaders ship inside the jar`() {
        assertContains(KernelinShaders.wgsl("io.github.kernelin.samples.mandelbrot"), "@compute @workgroup_size(64)")
        assertContains(KernelinShaders.manifest("io.github.kernelin.samples.mandelbrot"), "\"maxIter\"")
    }
}
