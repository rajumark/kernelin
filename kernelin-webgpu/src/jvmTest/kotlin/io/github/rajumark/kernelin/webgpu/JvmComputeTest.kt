package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelinCpu
import kotlinx.coroutines.runBlocking
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * [Compute] in a JVM app: the GPU when the computer has one, else all CPU cores. Unlike [JvmGpuTest] this never
 * skips: either way the results must be right.
 */
class JvmComputeTest {
    @Test
    fun runsOnTheGpuOrFallsBackToTheCpuWithTheSameResults(): Unit = runBlocking {
        val n = 50_000
        val x = FloatArray(n) { it * 0.5f }
        val y = FloatArray(n) { 1f - it }
        val expected = y.copyOf()
        KernelinCpu.dispatch(n) { saxpy(2f, x, expected) }

        Compute.create().use { compute ->
            println("Kernelin Compute: ${compute.description}")
            if (System.getenv("KERNELIN_REQUIRE_GPU") == "1") assertTrue(compute.isGpu, compute.description)
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
        val expected = IntArray(1000)
        KernelinCpu.dispatch(1000) { bits(IntArray(1000) { it }, expected, 1) }
        assertContentEquals(expected, out)
        assertEquals("the CPU was chosen (useGpu = false)", compute.cpuReason)
    }
}
