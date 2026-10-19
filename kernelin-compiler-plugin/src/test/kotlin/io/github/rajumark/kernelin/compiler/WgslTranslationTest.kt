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
        assertContains(wgsl, "kernelin_coerce_in(acc, 0.0f, 1.0f)")
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
    fun `kernelSource uses the saved shader when only the calling file is recompiled`() {
        // An incremental build recompiles Main.kt alone: fill comes from the earlier output, without a body.
        val kernels = compileOk(
            """
            @Kernel
            fun fill(out: FloatArray, value: Float) {
                out[globalId()] = value
            }
            """,
        )
        val caller = "import io.github.rajumark.kernelin.*\n\nfun source(): KernelSource = kernelSource(::fill)"
        val withoutShaders = kernels.outDir.resolveSibling("without-shaders")
        kernels.outDir.copyRecursively(withoutShaders)
        withoutShaders.resolve("kernelin").deleteRecursively()
        val c = KernelinCompiler.compile(caller, "Main.kt", out = kernels.outDir)
        if (!c.ok) fail("Compilation failed:\n${c.output}")
        val source = c.loadClass("MainKt").getMethod("source").invoke(null) as io.github.rajumark.kernelin.KernelSource
        assertEquals(kernels.wgsl("fill"), source.wgsl)
        assertEquals(kernels.manifest("fill"), source.manifestJson)

        // Without the saved shader there is nothing to embed: a clear error, not a wrong shader.
        val missing = KernelinCompiler.compile(caller, "Main.kt", out = withoutShaders)
        assertContains(missing.errors.joinToString("\n"), "'fill' must be a @Kernel from this module")
    }

    @Test
    fun `kernelSource rejects functions that are not kernels`() {
        val c = compileError("fun plain(x: Int) = x\nfun source() = kernelSource(::plain)")
        assertContains(c.errors.joinToString("\n"), "'plain' isn't annotated with @Kernel")
    }

    @Test
    fun `labeled jumps become flags and repeat becomes a for loop`() {
        val c = compileOk(
            """
            @Kernel
            fun k(out: IntArray) {
                var s = 0
                outer@ for (a in 0 until 3) { for (b in 0 until 3) { if (b == a) continue@outer; s += b } }
                repeat(4) { if (it == 2) return@repeat; s += it }
                out[globalId()] = s
            }
            """,
        )
        val wgsl = c.wgsl("k")
        assertContains(wgsl, "var kernelin_continue: bool = false;")
        assertContains(wgsl, "if (kernelin_continue) { kernelin_continue = false; continue; }")
        assertContains(wgsl, "for (var it: i32 = 0i; it < kernelin_end_2; it = it + 1i)")
    }

    @Test
    fun `constant expressions are computed with Kotlin rules`() {
        val c = compileOk(
            """
            @Kernel
            fun k(out: IntArray, f: FloatArray) {
                val i = globalId()
                out[i] = (-75 shl 28) + (Int.MAX_VALUE + 1) + (7 shl 33) + (if (3 > 2) 10 else 20)
                f[i] = Float.MAX_VALUE * 2f + (0f / 0f) + sqrt(16f) + (1 shl 2).toFloat()
            }
            """,
        )
        val wgsl = c.wgsl("k")
        // -75 shl 28 wraps to 1342177280; MAX_VALUE + 1 wraps to MIN_VALUE; 7 shl 33 == 7 shl 1 == 14.
        assertContains(wgsl, "out[i] = 1342177294i;".replace("1342177294", (1342177280 + Int.MIN_VALUE + 14 + 10).toString()))
        // Infinity and NaN can't be WGSL constants, so they read a runtime zero.
        assertContains(wgsl, "var<private> kernelin_zero: u32;")
        assertContains(wgsl, "bitcast<f32>(")
    }

    @Test
    fun `pow never calls WGSL pow with a negative or zero base`() {
        // WGSL (and Vulkan) leave pow(x, y) undefined for x < 0 and for x == 0, y <= 0; Kotlin defines both.
        val c = compileOk(
            """
            @Kernel
            fun k(a: FloatArray, b: FloatArray, out: FloatArray) {
                val i = globalId()
                out[i] = a[i].pow(b[i]) + a[i].pow(2)
            }
            """,
        )
        val wgsl = c.wgsl("k")
        assertContains(wgsl, "out[i] = (kernelin_pow(a[i], b[i]) + kernelin_pow(a[i], f32(2i)));")
        assertContains(wgsl, "fn kernelin_pow(x: f32, y: f32) -> f32 {")
        assertContains(wgsl, "pow(abs(x), y)")
    }

    @Test
    fun `constant integer division by zero is a compile error`() {
        val c = compileError("@Kernel\nfun k(out: IntArray) { out[globalId()] = 5 / 0 }")
        assertContains(c.errors.joinToString("\n"), "integer division by zero")
    }

    @Test
    fun `for loops that could pass Int MAX_VALUE count their steps instead`() {
        val c = compileOk(
            """
            @Kernel
            fun k(out: IntArray, n: Int, s: Int) {
                val i = globalId()
                var x = 0
                for (j in 0 until n) x += j
                for (j in 0..n step s) x += j
                for (j in n downTo -n step 2) x += j
                for (j in 10 downTo 0 step 2) x += j
                for (j in 0..Int.MAX_VALUE) { if (j > i) break; x += j }
                out[i] = x
            }
            """,
        )
        val wgsl = c.wgsl("k")
        // `0 until n` can't overflow: j < n <= Int.MAX_VALUE, so j + 1 fits.
        assertContains(wgsl, "for (var j: i32 = 0i; j < kernelin_end; j = j + 1i)")
        // The others count the steps left (unsigned, so even Int.MIN_VALUE..Int.MAX_VALUE fits) and skip the loop
        // when the step is 0 or less (Kotlin throws there).
        assertContains(wgsl, "if (kernelin_step > 0i && kernelin_first <= kernelin_end_1) {")
        assertContains(wgsl, "var kernelin_left: u32 = (u32(kernelin_end_1) - u32(kernelin_first)) / u32(kernelin_step);")
        assertContains(wgsl, "break if kernelin_left == 4294967295u;")
        assertContains(wgsl, "j_1 = bitcast<i32>(bitcast<u32>(j_1) + bitcast<u32>(kernelin_step));")
        assertContains(wgsl, "(u32(kernelin_first_1) - u32(kernelin_end_2)) / 2u")
        assertContains(wgsl, "(u32(kernelin_end_4) - u32(kernelin_first_2)) / 1u")
        // Constant ends that leave room for one more step keep the plain loop.
        assertContains(wgsl, "for (var j_3: i32 = 10i; j_3 >= kernelin_end_3; j_3 = j_3 - 2i)")
    }

    @Test
    fun `a constant step of zero or less and a second step are compile errors`() {
        val text = compileError(
            """
            @Kernel
            fun k(out: IntArray) {
                for (j in 0..10 step 0) out[j] = 1
                for (j in 10 downTo 0 step -2) out[j] = 1
                for (j in 0..10 step 2 step 3) out[j] = 1
            }
            """,
        ).errors.joinToString("\n")
        assertContains(text, "step must be positive, but it is 0 (Kotlin throws IllegalArgumentException)")
        assertContains(text, "step must be positive, but it is -2")
        assertContains(text, "only one step per range is supported")
    }

    @Test
    fun `Float min, max, coerce, sign and compareTo follow Kotlin for NaN and -0f`() {
        val c = compileOk(
            """
            @Kernel
            fun k(a: FloatArray, out: FloatArray, ints: IntArray) {
                val i = globalId()
                out[i] = min(a[i], 1f) + max(a[i], 1f) + a[i].coerceIn(0f, 1f) + sign(a[i]) + min(-0f, 0f)
                ints[i] = a[i].compareTo(1f) + (if (-0f < 0f) 10 else 20) + (-0f).compareTo(0f) * 100
            }
            """,
        )
        val wgsl = c.wgsl("k")
        assertContains(wgsl, "kernelin_min(a[i], 1.0f)")
        assertContains(wgsl, "kernelin_max(a[i], 1.0f)")
        assertContains(wgsl, "kernelin_coerce_in(a[i], 0.0f, 1.0f)")
        assertContains(wgsl, "kernelin_sign(a[i])")
        assertContains(wgsl, "kernelin_compare(a[i], 1.0f)")
        // Constants too: min(-0f, 0f) is -0f, `-0f < 0f` is false, but (-0f).compareTo(0f) is -1.
        assertContains(wgsl, "+ -0.0f);")
        assertContains(wgsl, "ints[i] = ((kernelin_compare(a[i], 1.0f) + 20i) + -100i);")
    }

    @Test
    fun `the manifest is valid JSON for any kernel name`() {
        // Backticks allow quotes and spaces in a Kotlin name; the manifest used to put them in unescaped.
        val c = compileOk(
            """
            @Kernel
            fun `say "hi" twice`(`the out`: FloatArray, n: Int) {
                `the out`[globalId()] = n.toFloat()
            }
            """,
        )
        val json = c.manifest("say \"hi\" twice")
        assertContains(json, "\"kernel\": \"say \\\"hi\\\" twice\"")
        val manifest = io.github.rajumark.kernelin.KernelManifest.parse(json)
        assertEquals("say \"hi\" twice", manifest.kernel)
        assertEquals(listOf("the_out", "n"), manifest.params.map { it.name })
    }

    @Test
    fun `new features report clear errors when misused`() {
        val c = compileError(
            """
            @Kernel
            fun k(out: FloatArray, n: Int) {
                val i = globalId()
                if (i in 0..10 step 2) out[i] = 1f
                out[i] = out[i].toDouble().toFloat()
            }
            """,
        )
        val text = c.errors.joinToString("\n")
        assertContains(text, "'in' supports ranges written in place")
        assertContains(text, "only Double constants converted right away")
    }

    @Test
    fun `helpers named like WGSL built-ins are renamed`() {
        // Found by a real project: a random-number helper called uniform() shadowed var<uniform> and Chrome
        // rejected the shader ("cannot use function 'uniform' as address space").
        val c = KernelinCompiler.compile(
            """
            import io.github.rajumark.kernelin.*

            fun uniform(x: Int): Float = (x and 255).toFloat()
            fun storage(x: Float): Float = x * 2f
            fun read_write(x: Float): Float = x + 1f
            fun fract(x: Float): Float = x - x.toInt().toFloat()
            fun global_invocation_id(): Int = 3

            @Kernel
            fun k(out: FloatArray) {
                val i = globalId()
                out[i] = read_write(storage(uniform(i))) + fract(1.5f) + global_invocation_id()
            }
            """.trimIndent(),
        )
        assertTrue(c.ok, c.output)
        val wgsl = c.wgsl("k")
        for (name in listOf("uniform", "storage", "read_write", "fract", "global_invocation_id")) {
            assertTrue("fn ${name}_k(" in wgsl, "helper '$name' should be renamed:\n$wgsl")
            assertTrue("fn $name(" !in wgsl, wgsl)
        }
        c.wgslFiles().forEach(Naga::validate)
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
        val error = c.errors.first { "strings aren't supported" in it || "variable 's'" in it }
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
        assertContains(text, "kernels accept FloatArray, IntArray, Int, Float and Boolean")
        assertContains(text, "must return Unit")
    }

    @Test
    fun `body errors are all reported together`() {
        // Frontend checks (also shown in the IDE) report every mistake at once; the build stops there.
        val c = compileError(
            """
            @Kernel
            fun bad(out: FloatArray) {
                val d = 1.0
                run { out[0] = 2f }
                val r = if (out[0] > 1f) "yes" else "no"
            }
            """,
        )
        val text = c.errors.joinToString("\n")
        assertContains(text, "Double isn't supported")
        assertContains(text, "lambdas aren't supported")
        assertContains(text, "strings aren't supported")
        assertTrue(c.errors.size >= 3, text)
    }

    @Test
    fun `recursion is rejected`() {
        val c = compileError(
            """
            fun fact(n: Int): Int = if (n <= 1) 1 else n * fact(n - 1)

            @Kernel
            fun bad(out: FloatArray) {
                out[0] = fact(3).toFloat()
            }
            """,
        )
        assertContains(c.errors.joinToString("\n"), "recursion isn't allowed")
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
