package io.github.rajumark.kernelin

import kotlin.reflect.KFunction

/** The GPU code Kernelin generated for one `@Kernel` function. */
public class KernelSource(
    /** Fully qualified Kotlin name of the kernel, e.g. `io.github.rajumark.kernelin.samples.saxpy`. */
    public val name: String,
    wgsl: String,
    manifestJson: String,
) {
    // The compiler embeds the shader where kernelSource() is called. An incremental build may skip recompiling
    // that caller when only the kernel's body changed, so platforms that also ship the shader next to the
    // compiled kernel (the JVM) prefer that copy: it's always rebuilt together with the kernel.
    private val fresh: Pair<String, String>? = freshShader(name)

    /** WGSL compute shader with entry point `main`. */
    public val wgsl: String = fresh?.first ?: wgsl

    /** JSON description of buffers and parameters (see [KernelManifest]). */
    public val manifestJson: String = fresh?.second ?: manifestJson

    /** The parsed [manifestJson]. */
    public val manifest: KernelManifest by lazy { KernelManifest.parse(manifestJson) }

    override fun toString(): String = "KernelSource($name)"
}

/**
 * Returns the generated GPU code for [kernel], e.g. `kernelSource(::saxpy)`.
 *
 * The Kernelin compiler plugin replaces this call at compile time with the shader itself,
 * so it works on every platform (JVM, Wasm, ...) without files or reflection.
 */
public fun kernelSource(@Suppress("UNUSED_PARAMETER") kernel: KFunction<*>): KernelSource =
    throw IllegalStateException("kernelSource() was not replaced: apply the Kernelin compiler plugin to this module")

/** The shader written next to the compiled kernel, when the platform has one (JVM resources). */
internal expect fun freshShader(name: String): Pair<String, String>?
