package hello

import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.kernelSource

/** On Kotlin/Native: run the kernel on the CPU. kernelSource() works because the compiler plugin embedded the WGSL. */
fun main() {
    val out = IntArray(8)
    KernelinCpu.dispatch(out.size) { squares(out) }
    println("Kotlin/Native: ${out.toList()}")
    println("WGSL lines: ${kernelSource(::squares).wgsl.lines().size}")
}
