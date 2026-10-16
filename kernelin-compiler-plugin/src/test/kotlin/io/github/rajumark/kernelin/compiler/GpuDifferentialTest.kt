package io.github.rajumark.kernelin.compiler

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Every test runs one kernel twice, as plain Kotlin on the CPU and as Kernelin's WGSL on a real GPU,
 * and requires the same results. This checks what the generated code *does*, not just that it's valid.
 */
class GpuDifferentialTest {
    private fun ints(n: Int, seed: Int, from: Int = -1000, until: Int = 1000) =
        Random(seed).let { r -> IntArray(n) { r.nextInt(from, until) } }

    private fun nonZeroInts(n: Int, seed: Int) = ints(n, seed).map { if (it == 0) 7 else it }.toIntArray()

    private fun floats(n: Int, seed: Int, scale: Float = 10f) =
        Random(seed).let { r -> FloatArray(n) { (r.nextFloat() * 2f - 1f) * scale } }

    private val n = 4096

    @Test
    fun `integer arithmetic, bit operations and overflow`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: IntArray, b: IntArray, out: IntArray) {
            val i = globalId()
            val x = a[i]
            val y = b[i]
            out[i] = x * y + x / y - x % y + (x shl 3) - (x shr 2) + (x ushr 28) +
                (x and y) - (x or 5) + (x xor y) + x.inv() - (-x) + x * 1_000_003 + Int.MAX_VALUE
        }
        """,
        "k", n, mapOf("a" to ints(n, 1), "b" to nonZeroInts(n, 2), "out" to IntArray(n)),
    ).let { }

    @Test
    fun `float arithmetic, mixed Int and Float, conversions`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: FloatArray, b: FloatArray, out: FloatArray, ints: IntArray) {
            val i = globalId()
            val x = a[i]
            val y = b[i]
            out[i] = x * y + x / (y * y + 1f) - x % 3f + i * 0.5f + i.toFloat() / 7 + (x - y).toInt()
            ints[i] = (x * 1e9f).toInt() + (-x).toInt() + (x * 3f).toInt() % 5 +
                ((x - x) / (y - y)).toInt() + (x * 1e30f).toInt() / 3
        }
        """,
        "k", n, mapOf("a" to floats(n, 3), "b" to floats(n, 4), "out" to FloatArray(n), "ints" to IntArray(n)),
        relTol = 1e-5, absTol = 1e-5,
    ).let { }

    @Test
    fun `NaN and infinity checks`() {
        // Special values come in through the data. (Making them on the GPU with x / 0f isn't portable:
        // WGSL leaves float division by zero undefined, and some GPUs return an ordinary number.)
        val special = floatArrayOf(Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, -0f, 1e38f, -1e38f)
        val a = floats(999, 12).also { v -> for (i in v.indices) if (i % 2 == 0) v[i] = special[(i / 2) % special.size] }
        GpuDifferential.assertSame(
            """
            @Kernel
            fun k(a: FloatArray, out: IntArray) {
                val i = globalId()
                val v = a[i]
                out[i] = (if (v.isNaN()) 1 else 0) + (if (v.isInfinite()) 2 else 0) + (if (v.isFinite()) 4 else 0) + v.toInt() * 8
            }
            """,
            "k", 999, mapOf("a" to a, "out" to IntArray(999)),
        )
    }

    @Test
    fun `comparisons, logic and if or when as values`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: IntArray, out: IntArray) {
            val i = globalId()
            val x = a[i]
            val big = x > 500 || x < -500
            val even = x % 2 == 0
            val grade = when {
                x > 800 -> 4
                x > 300 -> 3
                x > -300 -> 2
                else -> 1
            }
            val bucket = when (grade) { 4 -> 40; 3 -> 30; else -> 10 }
            val flags = (if (big) 1 else 0) + (if (even && !big) 2 else 0) + (if (big xor even) 4 else 0) +
                (if (x != 0 && x >= -10 && x <= 10) 8 else 0)
            out[i] = grade * 1000 + bucket * 10 + flags
        }
        """,
        "k", n, mapOf("a" to ints(n, 5), "out" to IntArray(n)),
    ).let { }

    @Test
    fun `every loop form with break and continue`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: IntArray, out: IntArray, n: Int) {
            val i = globalId()
            var s = 0
            for (j in 0 until n) s += a[(i + j) % n]
            for (j in 0..3) s += j
            for (j in 10 downTo 0 step 3) s -= j
            for (j in 0..<5) { if (j == 1) continue; if (j == 4) break; s += j * 100 }
            for (j in a.indices) { if (j > 3) break; s += a[j] }
            var w = 0
            while (w < 7) { w++; if (w == 2) continue; s += w }
            do { w-- } while (w > i % 5)
            out[i] = s + w
        }
        """,
        "k", 512, mapOf("a" to ints(512, 6), "out" to IntArray(512), "n" to 37),
    ).let { }

    @Test
    fun `labeled break and continue across two and three loops`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(out: IntArray) {
            val i = globalId()
            var s = 0
            outer@ for (a in 0 until 6) {
                for (b in 0 until 6) {
                    if (b == a) continue@outer
                    if (a + b == i % 9) break@outer
                    s += a * 10 + b
                }
            }
            top@ for (a in 0 until 4) {
                for (b in 0 until 4) {
                    var c = 0
                    while (c < 4) {
                        c++
                        if ((a + b + c + i) % 7 == 0) continue@top
                        if (a * b * c == 18) break@top
                        s += c
                    }
                }
            }
            out[i] = s
        }
        """,
        "k", 256, mapOf("out" to IntArray(256)),
    ).let { }

    @Test
    fun `repeat with return at repeat and break out of an outer loop`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(out: IntArray) {
            val i = globalId()
            var s = 0
            repeat(10) {
                if (it % 3 == i % 3) return@repeat
                s += it
            }
            for (round in 0 until 5) {
                repeat(5) { r ->
                    if (r + round == i % 7) break
                    s += r * round
                }
            }
            out[i] = s
        }
        """,
        "k", 256, mapOf("out" to IntArray(256)),
    ).let { }

    @Test
    fun `in ranges for Int and Float`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: IntArray, f: FloatArray, out: IntArray) {
            val i = globalId()
            val x = a[i]
            var r = 0
            if (x in -100..100) r += 1
            if (x !in 0 until 500) r += 2
            if (x in 800 downTo 400) r += 4
            if (f[i] in -0.5f..0.5f) r += 8
            if (x + 1 in 0..<10) r += 16
            r += when (x) {
                in -1000..-500 -> 100
                in 500..1000 -> 200
                else -> 300
            }
            out[i] = r
        }
        """,
        "k", n, mapOf("a" to ints(n, 7), "f" to floats(n, 8, 1f), "out" to IntArray(n)),
    ).let { }

    @Test
    fun `kotlin math functions`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(a: FloatArray, out: FloatArray) {
            val i = globalId()
            val x = a[i]
            val p = abs(x) + 0.5f
            out[i] = sqrt(p) + sin(x) + cos(x) + tan(x * 0.1f) + exp(x * 0.1f) + ln(p) + log2(p) + log10(p) +
                p.pow(1.5f) + atan2(x, 2f) + atan(x) + tanh(x * 0.3f) + hypot(x, 3f) + floor(x) + ceil(x) +
                round(x) + truncate(x) + min(x, 1f) + max(x, -1f) + sign(x) + x.coerceIn(-2f, 2f) +
                PI.toFloat() * 0.1f + E.toFloat() * 0.1f
        }
        """,
        "k", n, mapOf("a" to floats(n, 9), "out" to FloatArray(n)),
        // WGSL allows a few ULPs (and ~2^-11 absolute for sin/cos), so math gets a looser tolerance.
        absTol = 5e-3, relTol = 1e-3,
    ).let { }

    @Test
    fun `helpers - scalar, extension, Unit, and array parameters`() = GpuDifferential.assertSame(
        """
        fun square(x: Float): Float = x * x
        fun Float.half(): Float = this / 2f
        fun sumFirst(a: FloatArray, n: Int): Float { var s = 0f; for (k in 0 until n) s += a[k]; return s }
        fun bump(a: FloatArray, i: Int, by: Float) { a[i] += by }

        @Kernel
        fun k(a: FloatArray, b: FloatArray, out: FloatArray) {
            val i = globalId()
            out[i] = square(a[i]).half() + sumFirst(a, 4) - sumFirst(b, 3)
            bump(out, i, 1f)
            bump(out, i, b[i])
        }
        """,
        "k", n, mapOf("a" to floats(n, 10), "b" to floats(n, 11), "out" to FloatArray(n)),
        relTol = 1e-5, absTol = 1e-4,
    ).let { }

    @Test
    fun `Boolean parameters`() {
        val source = """
            @Kernel
            fun k(out: IntArray, double: Boolean, negate: Boolean) {
                val i = globalId()
                var v = i
                if (double) v *= 2
                if (negate && !double) v = -v
                out[i] = v
            }
        """
        for (d in listOf(true, false)) for (neg in listOf(true, false)) {
            val r = GpuDifferential.assertSame(source, "k", 64, mapOf("out" to IntArray(64), "double" to d, "negate" to neg))
            assertEquals(if (d) 20 else if (neg) -10 else 10, (r.gpu.getValue("out") as IntArray)[10])
        }
    }

    @Test
    fun `more than 4 million threads use a 2D dispatch`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun k(out: IntArray) {
            val i = globalId()
            out[i] = i * 3 + 1
        }
        """,
        "k", 5_000_003, mapOf("out" to IntArray(5_000_003)),
    ).let { }

    @Test
    fun `mandelbrot from the samples`() = GpuDifferential.assertSame(
        """
        @Kernel
        fun mandelbrot(out: IntArray, width: Int, height: Int, maxIter: Int) {
            val id = globalId()
            val x0 = (id % width).toFloat() / width * 3.5f - 2.5f
            val y0 = (id / width).toFloat() / height * 2.0f - 1.0f
            var x = 0f
            var y = 0f
            var iter = 0
            while (x * x + y * y <= 4f && iter < maxIter) {
                val xt = x * x - y * y + x0
                y = 2f * x * y + y0
                x = xt
                iter++
            }
            out[id] = iter
        }
        """,
        "mandelbrot", 256 * 160, mapOf("out" to IntArray(256 * 160), "width" to 256, "height" to 160, "maxIter" to 200),
        // GPUs may fuse multiply-adds, so a few pixels on the set's edge take a different number of steps.
        mismatchFraction = 0.01,
    ).let { }
}
