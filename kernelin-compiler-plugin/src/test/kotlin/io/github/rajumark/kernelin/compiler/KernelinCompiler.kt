package io.github.rajumark.kernelin.compiler

import org.jetbrains.kotlin.cli.common.ExitCode
import org.jetbrains.kotlin.cli.jvm.K2JVMCompiler
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.PrintStream
import java.nio.file.Files

/** Result of compiling Kotlin sources with the real compiler and the packaged Kernelin plugin. */
class Compilation(val exitCode: ExitCode, val output: String, val outDir: File) {
    val ok get() = exitCode == ExitCode.OK

    fun wgsl(kernel: String): String = outDir.resolve("kernelin/$kernel.wgsl").readText()

    fun manifest(kernel: String): String = outDir.resolve("kernelin/$kernel.json").readText()

    fun wgslFiles(): List<File> = outDir.resolve("kernelin").listFiles { f -> f.extension == "wgsl" }?.sortedBy { it.name }.orEmpty()

    val errors: List<String> get() = output.lines().filter { "error:" in it }

    /** Loads a compiled class, with the test classpath (runtime, stdlib) as parent. */
    fun loadClass(name: String): Class<*> =
        java.net.URLClassLoader(arrayOf(outDir.toURI().toURL()), javaClass.classLoader).loadClass(name)
}

object KernelinCompiler {
    private val pluginJar = System.getProperty("kernelin.pluginJar")
        ?: error("kernelin.pluginJar is not set; run the tests through Gradle")

    /** Compiles [source] alone; with [out] from an earlier compilation, like an incremental build does. */
    fun compile(source: String, fileName: String = "Kernels.kt", out: File? = null): Compilation =
        compileFiles(mapOf(fileName to source), out)

    /** Compiles several source files (file name to source) into one module; [out] as for [compile]. */
    fun compileFiles(files: Map<String, String>, out: File? = null): Compilation {
        val dir = Files.createTempDirectory("kernelin-test").toFile()
        val srcs = files.map { (name, text) -> dir.resolve("src/$name").apply { parentFile.mkdirs(); writeText(text) }.path }
        val classpath = listOfNotNull(out?.path, System.getProperty("java.class.path")).joinToString(File.pathSeparator)
        val out = out ?: dir.resolve("out").apply { mkdirs() }
        val buffer = ByteArrayOutputStream()
        val exit = K2JVMCompiler().exec(
            PrintStream(buffer),
            *srcs.toTypedArray(),
            "-d", out.path,
            "-classpath", classpath,
            "-no-stdlib", "-no-reflect",
            "-jvm-target", "21",
            "-module-name", "kernelin_test",
            "-Xplugin=$pluginJar",
        )
        return Compilation(exit, buffer.toString(), out)
    }
}
