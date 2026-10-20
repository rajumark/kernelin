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

    @Test
    fun `parallel 2D and 3D dispatch give every cell its own x, y and z`() {
        val (w, h, d) = Triple(301, 97, 3)
        val cells = IntArray(w * h * d)
        KernelinCpu.dispatchParallel(w, h, d) { cells[globalIdX() + globalIdY() * w + globalIdZ() * w * h] += globalId() + 1 }
        assertContentEquals(IntArray(cells.size) { it + 1 }, cells)
        val flat = IntArray(w * h)
        KernelinCpu.dispatchParallel(w, h) { flat[globalId()] = globalIdY() * 1000 + globalIdX() + globalIdZ() }
        assertContentEquals(IntArray(w * h) { (it / w) * 1000 + it % w }, flat)
    }
}
