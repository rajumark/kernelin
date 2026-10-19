package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertContentEquals

/** JVM-only: [KernelinCpu.dispatchParallel]. The rest is in commonTest. */
class KernelinCpuTest {
    @Test
    fun `parallel dispatch covers every id exactly once`() {
        val hits = IntArray(100_003)
        KernelinCpu.dispatchParallel(hits.size) { hits[globalId()]++ }
        assertContentEquals(IntArray(hits.size) { 1 }, hits)
    }

    @Test
    fun `empty parallel dispatch does nothing`() {
        KernelinCpu.dispatchParallel(0) { error("must not run") }
    }
}
