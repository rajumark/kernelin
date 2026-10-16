package io.github.rajumark.kernelin

/** Reads `kernelin/<name>.wgsl` and `.json`, which the compiler plugin writes next to the kernel's classes. */
internal actual fun freshShader(name: String): Pair<String, String>? {
    val loaders = listOfNotNull(Thread.currentThread().contextClassLoader, KernelSource::class.java.classLoader)
    for (loader in loaders) {
        val wgsl = loader.getResourceAsStream("kernelin/$name.wgsl")?.use { it.readBytes().decodeToString() } ?: continue
        val manifest = loader.getResourceAsStream("kernelin/$name.json")?.use { it.readBytes().decodeToString() } ?: continue
        return wgsl to manifest
    }
    return null
}
