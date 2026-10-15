package io.github.kernelin

import kotlin.reflect.KFunction

/** The GPU code Kernelin generated for one `@Kernel` function. */
class KernelSource(
    /** Fully qualified Kotlin name of the kernel, e.g. `io.github.kernelin.samples.saxpy`. */
    val name: String,
    /** WGSL compute shader with entry point `main`. */
    val wgsl: String,
    /** JSON description of buffers and parameters (see [KernelManifest]). */
    val manifestJson: String,
) {
    val manifest: KernelManifest by lazy { KernelManifest.parse(manifestJson) }

    override fun toString() = "KernelSource($name)"
}

/**
 * Returns the generated GPU code for [kernel], e.g. `kernelSource(::saxpy)`.
 *
 * The Kernelin compiler plugin replaces this call at compile time with the shader itself,
 * so it works on every platform (JVM, Wasm, ...) without files or reflection.
 */
fun kernelSource(@Suppress("UNUSED_PARAMETER") kernel: KFunction<*>): KernelSource =
    throw IllegalStateException("kernelSource() was not replaced: apply the Kernelin compiler plugin to this module")
