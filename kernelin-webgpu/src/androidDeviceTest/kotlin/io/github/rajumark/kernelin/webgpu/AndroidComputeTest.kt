package io.github.rajumark.kernelin.webgpu

import android.os.Build
import io.github.rajumark.kernelin.KernelinCpu
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * [Compute] on a real device: the GPU on a phone, the CPU on an emulator (which has no real GPU for WebGPU). Unlike
 * [AndroidGpuTest] this never skips: either way the results must be right.
 */
class AndroidComputeTest {
    private val isEmulator = Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish")

    @Test
    fun runsOnTheGpuOrFallsBackToTheCpuWithTheSameResults(): Unit = runBlocking {
        val n = 50_000
        val x = FloatArray(n) { it * 0.5f }
        val y = FloatArray(n) { 1f - it }
        val expected = y.copyOf()
        KernelinCpu.dispatch(n) { saxpy(2f, x, expected) }

        Compute.create().use { compute ->
            println("Kernelin Compute: ${compute.description}")
            if (isEmulator) assertFalse(compute.isGpu, "Emulators have no real GPU for WebGPU")
            compute.dispatch(n) { saxpy(alpha = 2f, x = x, y = y) }
            assertContentEquals(expected, y)
        }
    }

    @Test
    fun useGpuFalseRunsOnAllCpuCores(): Unit = runBlocking {
        val compute = Compute.create(useGpu = false)
        assertFalse(compute.isGpu)
        val out = IntArray(1000)
        compute.dispatch(1000) { bits(a = IntArray(1000) { it }, out = out, shift = 1) }
        assertTrue(out.isNotEmpty())
        val expected = IntArray(1000)
        KernelinCpu.dispatch(1000) { bits(IntArray(1000) { it }, expected, 1) }
        assertContentEquals(expected, out)
        assertEquals("the CPU was chosen (useGpu = false)", compute.cpuReason)
    }
}
