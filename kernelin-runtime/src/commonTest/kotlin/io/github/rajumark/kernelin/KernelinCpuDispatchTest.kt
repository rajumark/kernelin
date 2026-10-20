package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** Runs on every target: JVM, JS, Wasm (when enabled) and Kotlin/Native. */
class KernelinCpuDispatchTest {
    @Test
    fun dispatchVisitsEveryIdOnceInOrder() {
        val seen = mutableListOf<Int>()
        KernelinCpu.dispatch(5) { seen += globalId() }
        assertContentEquals(listOf(0, 1, 2, 3, 4), seen)
    }

    @Test
    fun globalIdOutsideADispatchFailsClearly() {
        assertFailsWith<IllegalStateException> { globalId() }
        KernelinCpu.dispatch(1) { }
        assertFailsWith<IllegalStateException> { globalId() }
    }

    @Test
    fun emptyDispatchDoesNothing() {
        KernelinCpu.dispatch(0) { error("must not run") }
    }

    @Test
    fun idIsResetWhenTheKernelThrows() {
        assertFailsWith<IllegalArgumentException> { KernelinCpu.dispatch(3) { require(globalId() < 1) } }
        assertFailsWith<IllegalStateException> { globalId() }
    }

    @Test
    fun kernelSourceWithoutThePluginExplainsTheFix() {
        // This module is built without the compiler plugin, so the call is not replaced.
        val e = assertFailsWith<IllegalStateException> { kernelSource(::dispatchVisitsEveryIdOnceInOrder) }
        kotlin.test.assertTrue("Kernelin compiler plugin" in e.message.orEmpty())
    }

    @Test
    fun dispatch2DVisitsRowByRow() {
        val seen = mutableListOf<String>()
        KernelinCpu.dispatch(3, 2) { seen += "${globalIdX()},${globalIdY()},${globalIdZ()}=${globalId()}" }
        assertContentEquals(listOf("0,0,0=0", "1,0,0=1", "2,0,0=2", "0,1,0=3", "1,1,0=4", "2,1,0=5"), seen)
    }

    @Test
    fun dispatch3DVisitsLayerByLayer() {
        val seen = mutableListOf<Int>()
        KernelinCpu.dispatch(2, 3, 4) {
            val x = globalIdX()
            val y = globalIdY()
            val z = globalIdZ()
            assertEquals(globalId(), x + y * 2 + z * 6)
            seen += x * 100 + y * 10 + z
        }
        assertEquals(24, seen.size)
        assertEquals(listOf(0, 100, 10, 110, 20, 120, 1), seen.take(7))
        assertEquals(123, seen.last())
    }

    @Test
    fun gridIdsInA1DDispatch() {
        val seen = mutableListOf<Int>()
        KernelinCpu.dispatch(4) { seen += globalIdX() * 100 + globalIdY() * 10 + globalIdZ() }
        assertContentEquals(listOf(0, 100, 200, 300), seen)
    }

    @Test
    fun gridIdsOutsideADispatchFailWithTheirName() {
        val e = assertFailsWith<IllegalStateException> { globalIdY() }
        kotlin.test.assertTrue("globalIdY()" in e.message.orEmpty(), e.message)
        assertFailsWith<IllegalStateException> { globalIdX() }
        assertFailsWith<IllegalStateException> { globalIdZ() }
    }

    @Test
    fun badGridSizesAreRejected() {
        KernelinCpu.dispatch(0, 5) { error("must not run") }
        KernelinCpu.dispatch(5, 5, 0) { error("must not run") }
        assertFailsWith<IllegalArgumentException> { KernelinCpu.dispatch(-1, 5) { } }
        assertFailsWith<IllegalArgumentException> { KernelinCpu.dispatch(1, 1, -1) { } }
        val e = assertFailsWith<IllegalArgumentException> { KernelinCpu.dispatch(65536, 65536) { } }
        kotlin.test.assertTrue("4294967296" in e.message.orEmpty(), e.message)
    }
}
