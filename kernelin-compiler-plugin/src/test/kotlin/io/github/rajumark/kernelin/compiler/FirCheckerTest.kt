package io.github.rajumark.kernelin.compiler

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * The frontend (FIR) checks in KernelinFirCheckers.kt: the errors IntelliJ IDEA and Android Studio show while
 * you type. They must catch common mistakes, point at the right code, and never reject a valid kernel.
 */
class FirCheckerTest {
    private fun compile(source: String) =
        KernelinCompiler.compile("import io.github.rajumark.kernelin.*\nimport kotlin.math.*\n\n${source.trimIndent()}\n")

    private fun errorsOf(source: String): String {
        val c = compile(source)
        assertFalse(c.ok, "expected a compile error, got:\n${c.output}")
        return c.errors.joinToString("\n")
    }

    @Test
    fun `kernel signature mistakes`() {
        val text = errorsOf(
            """
            @Kernel fun returnsValue(out: FloatArray): Int = 1
            @Kernel fun <T> generic(out: FloatArray) {}
            @Kernel suspend fun suspending(out: FloatArray) {}
            @Kernel fun FloatArray.receiver() {}
            @Kernel fun badParams(a: DoubleArray, b: Long, c: Float?, d: String) {}
            class Holder { @Kernel fun member(out: FloatArray) {} }
            """,
        )
        assertContains(text, "must return Unit")
        assertContains(text, "can't have type parameters")
        assertContains(text, "can't be suspend")
        assertContains(text, "top-level function without receivers (in @Kernel receiver)")
        assertContains(text, "top-level function without receivers (in @Kernel member)")
        assertContains(text, "parameter 'a' has type DoubleArray")
        assertContains(text, "parameter 'b' has type Long; kernels accept FloatArray, IntArray, Int, Float and Boolean (Long isn't supported on the GPU; use Int)")
        assertContains(text, "parameter 'c' has type Float?")
        assertContains(text, "parameter 'd' has type String")
    }

    @Test
    fun `expressions the GPU can't run`() {
        val text = errorsOf(
            """
            @Kernel
            fun k(out: FloatArray, n: Int) {
                val i = globalId()
                out[i] = (out[i] * 2.0).toFloat()
                out[i] = (i + 1L).toFloat()
                if (n < 0) throw IllegalStateException()
                try { out[i] = 1f } catch (e: Exception) { }
                val any: Any = n
                if (any is Int) out[i] = 2f
                out[i] = (any as Int).toFloat()
                out[i] = n.toString().length.toFloat()
            }
            """,
        )
        assertContains(text, "Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f) (in @Kernel k)")
        assertContains(text, "Long isn't supported on the GPU; use Int")
        assertContains(text, "throwing exceptions isn't supported")
        assertContains(text, "try/catch isn't supported")
        assertContains(text, "type checks and casts (is/as) aren't supported")
        assertContains(text, "variable 'any' has type Any")
    }

    @Test
    fun `strings and nullable values`() {
        val text = errorsOf(
            """
            @Kernel
            fun k(out: FloatArray, n: Int) {
                val i = globalId()
                val label = "item ${'$'}i"
                val maybe: Int? = if (n > 0) n else null
                out[i] = (maybe ?: 0).toFloat()
                out[i] = maybe!!.toFloat()
            }
            """,
        )
        assertContains(text, "variable 'label' has type String; only Int, Float and Boolean are supported (strings aren't supported on the GPU)")
        assertContains(text, "strings aren't supported on the GPU (in @Kernel k)")
        assertContains(text, "variable 'maybe' has type Int?")
        assertContains(text, "nullable values aren't supported on the GPU")
    }

    @Test
    fun `calls the GPU can't run`() {
        val text = errorsOf(
            """
            @Kernel fun other(out: FloatArray) { out[globalId()] = 1f }

            fun applyTwice(x: Float, f: (Float) -> Float): Float = f(f(x))

            @Kernel
            fun k(out: FloatArray) {
                val i = globalId()
                println(i)
                other(out)
                val tmp = FloatArray(4)
                out.forEach { out[i] += it }
                i.let { out[it] = 1f }
                out[i] = applyTwice(out[i]) { it * 2f }
                out[i] = listOf(1f, 2f).size.toFloat()
            }
            """,
        )
        assertContains(text, "'kotlin.io.println' can't run on the GPU")
        assertContains(text, "a kernel can't call another @Kernel function")
        assertContains(text, "creating objects or arrays isn't supported in kernels")
        assertContains(text, "lambdas aren't supported in kernels yet ('forEach' takes one)")
        assertContains(text, "lambdas aren't supported in kernels yet ('let' takes one)")
        assertContains(text, "lambdas aren't supported in kernels yet ('applyTwice' takes one)")
        assertContains(text, "'kotlin.collections.listOf' can't run on the GPU")
    }

    @Test
    fun `errors point at the code to fix`() {
        val c = compile(
            """
            @Kernel
            fun k(out: FloatArray) {
                out[0] = (out[0] * 2.0).toFloat()
            }
            """,
        )
        // Line 6 of the file (2 import lines + blank line come first), column of the 2.0 literal.
        val error = c.errors.single()
        assertContains(error, "Kernels.kt:6:24:")
    }

    @Test
    fun `frontend errors stop the build before the translator runs`() {
        val text = errorsOf(
            """
            fun fact(n: Int): Int = if (n <= 1) 1 else n * fact(n - 1)

            @Kernel
            fun k(out: FloatArray) {
                val d = 1.0
                out[0] = fact(3).toFloat()
            }
            """,
        )
        assertContains(text, "variable 'd' has type Double")
        assertFalse("recursion" in text, text)
    }

    @Test
    fun `valid kernels and CPU-only code are never flagged`() {
        val c = compile(
            """
            fun cpuOnly(values: List<Double>): String = values.sum().toString() + "!"

            fun clamp01(x: Float): Float = x.coerceIn(0f, 1f)

            @Kernel
            fun k(input: FloatArray, flags: IntArray, out: FloatArray, n: Int, on: Boolean) {
                val i = globalId()
                val src = input
                @Suppress("UNUSED_VARIABLE") val unused = 0
                var acc = 0f
                repeat(3) { r -> acc += r.toFloat() }
                for (j in input.indices step 2) acc += src[j]
                for (j in 10 downTo 0) if (j in 2..<5 || j !in 0 until 8) acc += 1f
                if (flags[i] in 800 downTo 400) acc += 4f
                acc += when (flags[i]) { in -10..10 -> 1f; else -> 0f }
                acc += PI.toFloat() + E.toFloat() + sqrt(2f) + 2.0.toFloat()
                if (acc.isNaN() || !on) acc = 0f
                out[i] = clamp01(acc / n)
            }
            """,
        )
        assertTrue(c.ok, c.output)
    }
}
