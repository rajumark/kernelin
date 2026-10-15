package io.github.rajumark.kernelin.compiler

import java.io.File
import java.util.concurrent.TimeUnit
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.test.fail

class WgslTranslationTest {

    private fun compileOk(source: String): Compilation {
        val c = KernelinCompiler.compile("import io.github.rajumark.kernelin.*\nimport kotlin.math.*\n\n$source")
        if (!c.ok) fail("Compilation failed:\n${c.output}")
        c.wgslFiles().forEach(Naga::validate)
        return c
    }

    @Test
    fun `vector add becomes a compute shader with storage buffers`() {
        val c = compileOk(
            """
            @Kernel
            fun vectorAdd(a: FloatArray, b: FloatArray, out: FloatArray) {
                val i = globalId()
                out[i] = a[i] + b[i]
            }
            """,
        )
        val wgsl = c.wgsl("vectorAdd")
        assertContains(wgsl, "@group(0) @binding(1) var<storage, read> a: array<f32>;")
        assertContains(wgsl, "@group(0) @binding(2) var<storage, read> b: array<f32>;")
        assertContains(wgsl, "@group(0) @binding(3) var<storage, read_write> out: array<f32>;")
        assertContains(wgsl, "@compute @workgroup_size(64)")
        assertContains(wgsl, "out[i] = (a[i] + b[i]);")
        assertContains(c.manifest("vectorAdd"), "\"access\": \"read_write\"")
    }

    @Test
    fun `scalar parameters go into a uniform struct`() {
        val c = compileOk(
            """
            @Kernel
            fun scale(factor: Float, times: Int, data: FloatArray) {
                val i = globalId()
                data[i] = data[i] * factor * times
            }
            """,
        )
        val wgsl = c.wgsl("scale")
        assertContains(wgsl, "struct KernelinParams {\n  kernelin_size: u32,\n  factor: f32,\n  times: i32,\n}")
        assertContains(wgsl, "kernelin_params.factor")
        assertContains(wgsl, "f32(kernelin_params.times)")
    }

    @Test
    fun `control flow, loops and math are translated`() {
        val c = compileOk(
            """
            fun square(x: Float): Float = x * x
            fun Float.half(): Float = this / 2f

            @Kernel
            fun everything(input: FloatArray, ints: IntArray, out: FloatArray, n: Int) {
                val i = globalId()
                var acc = 0f
                for (k in 0 until n) acc += input[k]
                for (k in 0..3) acc += k
                for (k in 10 downTo 0 step 2) acc -= k
                for (k in ints.indices) { if (ints[k] < 0) continue; if (ints[k] > 100) break; acc += ints[k] }
                var w = 0
                while (w < 5) { w++ }
                do { w-- } while (w > 2)
                val grade = when {
                    acc > 10f -> 3
                    acc > 5f -> 2
                    else -> 1
                }
                val pick = if (i % 2 == 0 && grade != 1 || w == 2) sqrt(acc) else abs(acc)
                val sub = when (grade) { 1 -> 0.5f; 2 -> 0.25f; else -> 0.125f }
                out[i] = square(pick).half() + sin(acc) * cos(acc) + max(1f, min(2f, acc)) +
                    acc.coerceIn(0f, 1f) + sub + exp(0.1f) + ln(2f) + acc.pow(2f) + floor(acc) + atan2(acc, 1f) +
                    (ints[i] shl 1).toFloat() + (ints[i] and 7) - (-acc) + i.toFloat() / n
                out[i] += 1f
                out[i]++
            }
            """,
        )
        val wgsl = c.wgsl("everything")
        assertContains(wgsl, "fn square(x: f32) -> f32")
        assertContains(wgsl, "fn half(receiver: f32) -> f32")
        assertContains(wgsl, "for (var k: i32 = 0i; k < kernelin_end; k = k + 1i)")
        assertContains(wgsl, "for (var k_2: i32 = 10i; k_2 >= kernelin_end_2; k_2 = k_2 - 2i)")
        assertContains(wgsl, "i32(arrayLength(&ints))")
        assertContains(wgsl, "continuing {")
        assertContains(wgsl, "select(")
        assertContains(wgsl, "clamp(")
        assertContains(wgsl, "continue;")
        assertContains(wgsl, "break;")
    }

    @Test
    fun `wgsl keywords and kernelin names are renamed`() {
        val c = compileOk(
            """
            @Kernel
            fun names(loop: FloatArray, kernelin_id: Int) {
                val let = globalId()
                val fn = 1f
                loop[let] = fn + kernelin_id
            }
            """,
        )
        val wgsl = c.wgsl("names")
        assertContains(wgsl, "var<storage, read_write> loop_k: array<f32>;")
        assertContains(wgsl, "kernelin_id_k: i32")
        assertContains(wgsl, "let let_k: i32 = i32(kernelin_id);")
    }

    @Test
    fun `non-kernel code is untouched and still runs on the jvm`() {
        val c = compileOk("fun plain(x: Int) = x + 1")
        assertTrue(c.wgslFiles().isEmpty())
    }

    @Test
    fun `kernelSource is replaced by the generated shader at compile time`() {
        val c = compileOk(
            """
            @Kernel
            fun fill(out: FloatArray, value: Float) {
                out[globalId()] = value
            }

            fun source(): KernelSource = kernelSource(::fill)
            """,
        )
        val source = c.loadClass("KernelsKt").getMethod("source").invoke(null) as io.github.rajumark.kernelin.KernelSource
        assertEquals("fill", source.name)
        assertEquals(c.wgsl("fill"), source.wgsl)
        assertContains(source.wgsl, "out[i32(kernelin_id)] = kernelin_params.value;")
        assertEquals(listOf("out", "value"), source.manifest.params.map { it.name })
        assertEquals(16, source.manifest.uniformBytes)
    }

    @Test
    fun `kernelSource rejects functions that are not kernels`() {
        val c = compileError("fun plain(x: Int) = x\nfun source() = kernelSource(::plain)")
        assertContains(c.errors.joinToString("\n"), "'plain' isn't annotated with @Kernel")
    }

    // ------------------------------------------------------------------ errors

    private fun compileError(source: String): Compilation {
        val c = KernelinCompiler.compile("import io.github.rajumark.kernelin.*\nimport kotlin.math.*\n\n$source")
        assertTrue(!c.ok, "expected a compile error, got:\n${c.output}")
        return c
    }

    @Test
    fun `strings are rejected with a clear message and location`() {
        val c = compileError(
            """
            @Kernel
            fun bad(out: FloatArray) {
                val s = "hello ${'$'}{globalId()}"
                out[0] = 1f
            }
            """,
        )
        val error = c.errors.single { "strings aren't supported" in it || "variable 's'" in it }
        assertContains(error, "Kernels.kt:")
        assertContains(error, "(in @Kernel bad)")
    }

    @Test
    fun `double, lambdas, recursion and bad types are rejected`() {
        val c = compileError(
            """
            fun fact(n: Int): Int = if (n <= 1) 1 else n * fact(n - 1)

            @Kernel
            fun bad(out: FloatArray, list: List<Float>) {
                val d = 1.0
                repeat(3) { out[it] = 0f }
                out[0] = fact(3).toFloat()
            }

            @Kernel
            fun returnsValue(out: FloatArray): Int = 1
            """,
        )
        val text = c.errors.joinToString("\n")
        assertContains(text, "kernels accept FloatArray, IntArray, Int and Float")
        assertContains(text, "must return Unit")
    }

    @Test
    fun `body errors are all reported together`() {
        val c = compileError(
            """
            fun fact(n: Int): Int = if (n <= 1) 1 else n * fact(n - 1)

            @Kernel
            fun bad(out: FloatArray) {
                val d = 1.0
                repeat(3) { out[it] = 0f }
                out[0] = fact(3).toFloat()
                outer@ for (a in 0 until 3) { for (b in 0 until 3) { if (b == 1) break@outer } }
            }
            """,
        )
        val text = c.errors.joinToString("\n")
        assertContains(text, "Double isn't supported")
        assertContains(text, "lambdas aren't supported")
        assertContains(text, "recursion isn't allowed")
        assertContains(text, "outer loop label")
        assertTrue(c.errors.size >= 4, text)
    }
}

/** Validates generated WGSL with naga (from the wgpu project) when it's installed. */
object Naga {
    private val binary: File? = listOf(
        System.getenv("NAGA"),
        System.getProperty("user.home") + "/.cargo/bin/naga",
        "/opt/homebrew/bin/naga",
    ).firstNotNullOfOrNull { p -> p?.let(::File)?.takeIf { it.canExecute() } }

    val available get() = binary != null

    fun validate(file: File) {
        val naga = binary ?: return
        val process = ProcessBuilder(naga.path, file.path).redirectErrorStream(true).start()
        val output = process.inputStream.bufferedReader().readText()
        process.waitFor(30, TimeUnit.SECONDS)
        assertEquals(0, process.exitValue(), "naga rejected ${file.name}:\n$output\n--- shader ---\n${file.readText()}")
    }
}
