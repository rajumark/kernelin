package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

class KernelinCpuTest {
    @Test
    fun `dispatch visits every id once in order`() {
        val seen = mutableListOf<Int>()
        KernelinCpu.dispatch(5) { seen += globalId() }
        assertContentEquals(listOf(0, 1, 2, 3, 4), seen)
    }

    @Test
    fun `parallel dispatch covers every id exactly once`() {
        val hits = IntArray(100_003)
        KernelinCpu.dispatchParallel(hits.size) { hits[globalId()]++ }
        assertContentEquals(IntArray(hits.size) { 1 }, hits)
    }

    @Test
    fun `globalId outside a dispatch fails clearly`() {
        assertFailsWith<IllegalStateException> { globalId() }
        KernelinCpu.dispatch(1) { }
        assertFailsWith<IllegalStateException> { globalId() }
    }

    @Test
    fun `empty dispatch does nothing`() {
        KernelinCpu.dispatch(0) { error("must not run") }
        KernelinCpu.dispatchParallel(0) { error("must not run") }
    }
}
