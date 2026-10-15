package io.github.rajumark.kernelin

/**
 * Reads the WGSL that the compiler plugin generated for a kernel.
 *
 * The plugin writes `kernelin/<fq.name>.wgsl` and `kernelin/<fq.name>.json` into the
 * compilation's resources, so they ship inside the jar next to the compiled code.
 */
object KernelinShaders {
    fun wgsl(kernelFqName: String): String = read("kernelin/$kernelFqName.wgsl")

    /** JSON description of the kernel's buffers and parameters, used by GPU hosts. */
    fun manifest(kernelFqName: String): String = read("kernelin/$kernelFqName.json")

    private fun read(path: String): String {
        val stream = KernelinShaders::class.java.classLoader.getResourceAsStream(path)
            ?: error("No Kernelin shader found at '$path'. Is the Kernelin compiler plugin applied?")
        return stream.bufferedReader().use { it.readText() }
    }
}
