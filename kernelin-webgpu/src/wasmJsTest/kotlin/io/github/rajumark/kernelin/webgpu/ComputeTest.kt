package io.github.rajumark.kernelin.webgpu

import kotlinx.coroutines.test.runTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertNull
import kotlin.test.assertTrue

/**
 * [Compute]: the GPU when there is one, the CPU when not. The fake GPU never runs shaders, so a run that went to the
 * GPU fills `out` with 42, and a run on the CPU computes the real values ([scaleInto] with factor 7).
 */
class ComputeTest {
    private val a = floatArrayOf(1f, 2f, 3f)
    private val onCpu = intArrayOf(7, 14, 21)
    private val onGpu = intArrayOf(42, 42, 42)

    @BeforeTest
    fun setUp() {
        installFakeGpu()
        fakeOnDispatchFill(": out", 42)
    }

    private suspend fun Compute.run(): IntArray = IntArray(3).also { out -> dispatch(3) { scaleInto(factor = 7, on = true, a = a, out = out) } }

    @Test
    fun usesTheGpuWhenThereIsOne() = runTest {
        val compute = Compute.create()
        assertTrue(compute.isGpu)
        assertNull(compute.cpuReason)
        assertTrue(compute.description.startsWith("GPU: "), compute.description)
        assertContentEquals(onGpu, compute.run())
    }

    @Test
    fun fallsBackToTheCpuWithoutWebGpu() = runTest {
        removeWebGpu()
        val compute = Compute.create()
        assertFalse(compute.isGpu)
        assertContains(compute.cpuReason!!, "WebGPU is not available")
        assertContains(compute.description, "CPU (no GPU")
        assertContentEquals(onCpu, compute.run())
    }

    @Test
    fun useGpuFalseAlwaysRunsOnTheCpu() = runTest {
        val compute = Compute.create(useGpu = false)
        assertFalse(compute.isGpu)
        assertContentEquals(onCpu, compute.run())
    }

    @Test
    fun aLostDeviceSwitchesToTheCpuForThatRunAndLaterOnes() = runTest {
        val compute = Compute.create()
        assertContentEquals(onGpu, compute.run())
        fakeLoseDevice("GPU process crashed")
        tick()
        assertContentEquals(onCpu, compute.run(), "The run that hit the lost device still gives the right answer")
        assertFalse(compute.isGpu)
        assertContains(compute.cpuReason!!, "GPU device lost")
        assertContains(compute.cpuReason!!, "GPU process crashed")
        assertContentEquals(onCpu, compute.run())
    }

    @Test
    fun closeFreesTheGpuAndLaterRunsUseTheCpu() = runTest {
        val compute = Compute.create()
        compute.close()
        compute.close()
        assertFalse(compute.isGpu)
        assertEquals("closed", compute.cpuReason)
        assertContentEquals(onCpu, compute.run())
    }
}

private fun removeWebGpu(): Unit = js("{ globalThis.navigator = {}; }")
