package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertContentEquals
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
}
