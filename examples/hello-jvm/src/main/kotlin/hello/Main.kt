package hello

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.globalId
import io.github.rajumark.kernelin.kernelSource

/** Makes every pixel a bit brighter, keeping values between 0 and 1. */
@Kernel
fun brighten(pixels: FloatArray, amount: Float) {
    val i = globalId()
    pixels[i] = (pixels[i] + amount).coerceIn(0f, 1f)
}

fun main() {
    // 1. It's plain Kotlin, so it runs on the CPU (and in unit tests) as is.
    val pixels = floatArrayOf(0.1f, 0.5f, 0.95f)
    KernelinCpu.dispatch(pixels.size) { brighten(pixels, 0.1f) }
    println("CPU result: ${pixels.toList()}")

    // 2. The Kernelin compiler plugin also turned it into a GPU compute shader.
    val source = kernelSource(::brighten)
    println("Generated WGSL for ${source.name} (params: ${source.manifest.params.map { it.name }}):")
    println(source.wgsl)
}
