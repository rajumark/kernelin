package hello

import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.kernelSource

fun main() {
    val out = IntArray(8)
    KernelinCpu.dispatchParallel(out.size) { squares(out) }
    println("JVM (all cores): ${out.toList()}")
    println("WGSL lines: ${kernelSource(::squares).wgsl.lines().size}")
}
