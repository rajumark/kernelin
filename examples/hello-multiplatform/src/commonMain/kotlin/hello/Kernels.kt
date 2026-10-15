package hello

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId

/** Shared by every target: runs on the JVM, in Wasm, and (as WGSL) on the GPU. */
@Kernel
fun squares(out: IntArray) {
    val i = globalId()
    out[i] = i * i
}
