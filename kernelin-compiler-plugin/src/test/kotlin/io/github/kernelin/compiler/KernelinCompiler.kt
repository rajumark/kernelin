package io.github.kernelin.compiler

import org.jetbrains.kotlin.cli.common.ExitCode
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.nio.file.Files

/** Result of compiling Kotlin sources with the real compiler and the packaged Kernelin plugin. */
class Compilation(val exitCode: ExitCode, val output: String, private val outDir: File) {
    val ok get() = exitCode == ExitCode.OK

    fun wgsl(kernel: String): String = outDir.resolve("kernelin/$kernel.wgsl").readText()

    fun manifest(kernel: String): String = outDir.resolve("kernelin/$kernel.json").readText()

    fun wgslFiles(): List<File> = outDir.resolve("kernelin").listFiles { f -> f.extension == "wgsl" }?.sortedBy { it.name }.orEmpty()

    val errors: List<String> get() = output.lines().filter { "error:" in it }
}

object KernelinCompiler {
    private val pluginJar = System.getProperty("kernelin.pluginJar")
        ?: error("kernelin.pluginJar is not set; run the tests through Gradle")

    fun compile(source: String, fileName: String = "Kernels.kt"): Compilation {
        val dir = Files.createTempDirectory("kernelin-test").toFile()
        val src = dir.resolve("src/$fileName").apply { parentFile.mkdirs(); writeText(source) }
        val out = dir.resolve("out").apply { mkdirs() }
        val buffer = ByteArrayOutputStream()
        val exit = K2JVMCompiler().exec(
            PrintStream(buffer),
            src.path,
            "-d", out.path,
            "-classpath", System.getProperty("java.class.path"),
            "-no-stdlib", "-no-reflect",
            "-jvm-target", "21",
            "-module-name", "kernelin_test",
            "-Xplugin=$pluginJar",
        )
        return Compilation(exit, buffer.toString(), out)
    }
}
