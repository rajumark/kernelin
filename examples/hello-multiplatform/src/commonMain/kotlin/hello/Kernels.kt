package hello

import io.github.kernelin.Kernel
import io.github.kernelin.globalId

/** Shared by every target: runs on the JVM, in Wasm, and (as WGSL) on the GPU. */
@Kernel
fun squares(out: IntArray) {
    val i = globalId()
    out[i] = i * i
}
