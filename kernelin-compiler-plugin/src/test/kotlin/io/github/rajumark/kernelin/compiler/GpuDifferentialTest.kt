package io.github.rajumark.kernelin.compiler

import kotlin.math.pow
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

    // A helper that writes an array and returns a value. Calls to it must run only where Kotlin runs them.
    private val countingHelper = """
        fun count(hits: IntArray, i: Int, by: Int): Int { hits[i] += by; return by }
    """

    @Test
    fun `if and when values run a writing helper only in the taken branch`() = GpuDifferential.assertSame(
        countingHelper + """
        @Kernel
        fun k(a: IntArray, out: IntArray, hits: IntArray) {
            val i = globalId()
            val x = a[i]
            val v = if (x > 0) count(hits, i, 1) else count(hits, i, 10)
            val w = when {
                x > 500 -> count(hits, i, 100)
                x > -500 -> count(hits, i, 1000) + 1
                else -> 7
            }
            val u = if (x % 2 == 0) { val t = count(hits, i, 10000); t * 2 } else 3
            out[i] = v + w + u
        }
        """,
        "k", n, mapOf("a" to ints(n, 13), "out" to IntArray(n), "hits" to IntArray(n)),
    ).let { }

    @Test
    fun `and and or skip a writing helper on the right like Kotlin`() = GpuDifferential.assertSame(
        countingHelper + """
        @Kernel
        fun k(a: IntArray, out: IntArray, hits: IntArray) {
            val i = globalId()
            val x = a[i]
            var r = 0
            // count(...) in 0..5 needs a temporary, which used to be computed before the && even when x <= 0.
            if (x > 0 && count(hits, i, 1) in 0..5) r += 1
            if (x > 300 || count(hits, i, 10) !in 0..5) r += 2
            val both = x < -300 && count(hits, i, 100) in 0..1000
            if (both) r += 4
            out[i] = r
        }
        """,
        "k", n, mapOf("a" to ints(n, 14), "out" to IntArray(n), "hits" to IntArray(n)),
    ).let { }

    @Test
    fun `pow with a negative base matches Kotlin`() {
        // Bases in -10..10 (including negatives and zero) with integer and fractional exponents.
        val r = Random(15)
        val base = FloatArray(n) { if (it % 17 == 0) 0f else (r.nextFloat() * 2f - 1f) * 10f }
        val exps = floatArrayOf(0f, 1f, 2f, 3f, -1f, -2f, 0.5f, 1.5f, -0.5f)
        val exp = FloatArray(n) { exps[it % exps.size] }
        GpuDifferential.assertSame(
            """
            @Kernel
            fun k(a: FloatArray, b: FloatArray, out: FloatArray, outInt: FloatArray) {
                val i = globalId()
                out[i] = a[i].pow(b[i])
                outInt[i] = a[i].pow(3) + a[i].pow(2)
            }
            """,
            "k", n, mapOf("a" to base, "b" to exp, "out" to FloatArray(n), "outInt" to FloatArray(n)),
            absTol = 1e-4, relTol = 1e-4,
        )
    }

    // Ranges from data, with ends near Int.MIN_VALUE / MAX_VALUE and big steps. A loop that went past its end
    // used to wrap around and never stop on the GPU; the `n > 40` break keeps such a bug finite in this test.
    private val loopKernel = """
        @Kernel
        fun k(a: IntArray, b: IntArray, s: IntArray, up: IntArray, until: IntArray, down: IntArray, fixed: IntArray) {
            val i = globalId()
            var n = 0
            var sum = 0
            for (j in a[i]..b[i] step s[i]) { n++; sum += j; if (n > 40) break }
            up[i] = n * 1000003 + sum
            n = 0; sum = 0
            for (j in a[i] until b[i] step s[i]) { n++; sum += j; if (n > 40) break }
            until[i] = n * 1000003 + sum
            n = 0; sum = 0
            for (j in b[i] downTo a[i] step s[i]) { n++; sum += j; if (n > 40) break }
            down[i] = n * 1000003 + sum
            n = 0
            for (j in Int.MAX_VALUE - 5..Int.MAX_VALUE) { n++; if (n > 40) break }
            for (j in Int.MIN_VALUE + 5 downTo Int.MIN_VALUE) { n++; if (n > 40) break }
            for (j in 0..Int.MAX_VALUE) { n++; if (n > 50) break }
            for (j in Int.MAX_VALUE - 9 until Int.MAX_VALUE step 4) { n += 100 }
            // Constant ranges of 2^31 and 2^32 values: these crashed lavapipe's shader compiler (SIGFPE).
            for (j in Int.MIN_VALUE..0) { n += 1000; if (n > 60000) break }
            for (j in Int.MIN_VALUE..Int.MAX_VALUE) { n += 1000; if (n > 120000) break }
            for (j in Int.MAX_VALUE downTo Int.MIN_VALUE) { n += 1000; if (n > 180000) break }
            fixed[i] = n
        }
    """

    @Test
    fun `for loops stop at the end of the range like Kotlin, even near Int MAX_VALUE`() {
        val big = 1 shl 30
        val edge = listOf(
            Triple(0, Int.MAX_VALUE, big), Triple(Int.MAX_VALUE - 10, Int.MAX_VALUE, 1),
            Triple(Int.MAX_VALUE - 10, Int.MAX_VALUE, 3), Triple(Int.MIN_VALUE, Int.MIN_VALUE + 10, 3),
            Triple(Int.MIN_VALUE, Int.MAX_VALUE, big + 7), Triple(-big, big, big), Triple(5, 5, 1),
            Triple(6, 5, 1), Triple(Int.MIN_VALUE, Int.MIN_VALUE, 1), Triple(Int.MAX_VALUE, Int.MAX_VALUE, 2),
            Triple(-3, Int.MIN_VALUE, 1), Triple(Int.MAX_VALUE - 1, Int.MAX_VALUE, Int.MAX_VALUE),
        )
        val size = 1024
        val r = Random(16)
        val rows = List(size) { i -> edge.getOrElse(i) { Triple(r.nextInt(-50, 50), r.nextInt(-50, 50), r.nextInt(1, 7)) } }
        GpuDifferential.assertSame(
            loopKernel, "k", size,
            mapOf(
                "a" to IntArray(size) { rows[it].first }, "b" to IntArray(size) { rows[it].second },
                "s" to IntArray(size) { rows[it].third },
                "up" to IntArray(size), "until" to IntArray(size), "down" to IntArray(size), "fixed" to IntArray(size),
            ),
        )
    }

    @Test
    fun `a step of zero or less runs no iterations on the GPU`() {
        // Kotlin throws IllegalArgumentException here, so there is no CPU result: the GPU must just skip the loop.
        val size = 4
        val gpu = GpuDifferential.runGpu(
            GpuDifferential.compile(loopKernel), "k", size,
            mapOf(
                "a" to IntArray(size) { 0 }, "b" to IntArray(size) { 10 }, "s" to intArrayOf(0, -1, Int.MIN_VALUE, -5),
                "up" to IntArray(size), "until" to IntArray(size), "down" to IntArray(size), "fixed" to IntArray(size),
            ),
        )
        for (name in listOf("up", "until", "down")) {
            assertEquals(listOf(0, 0, 0, 0), (gpu.getValue(name) as IntArray).toList(), name)
        }
    }

    @Test
    fun `Float min, max, coerce, compareTo and sign match Kotlin for NaN and -0f`() {
        // Every pair of special values, then random values with some NaN and -0f mixed in. WGSL leaves min/max
        // with NaN and the sign of a zero result unspecified; Kotlin defines both. Compared bit for bit.
        val special = floatArrayOf(Float.NaN, -0f, 0f, 1f, -1f, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 2.5f, -0.5f)
        val pairs = special.size * special.size
        val size = pairs + 2000
        val r = Random(17)
        fun pick(i: Int, f: (Int) -> Int) = if (i < pairs) special[f(i)] else when (r.nextInt(8)) {
            0 -> Float.NaN
            1 -> -0f
            2 -> 0f
            else -> (r.nextFloat() * 2f - 1f) * 4f
        }
        val a = FloatArray(size) { pick(it) { i -> i / special.size } }
        val b = FloatArray(size) { pick(it) { i -> i % special.size } }
        GpuDifferential.assertSame(
            """
            @Kernel
            fun k(a: FloatArray, b: FloatArray, mn: FloatArray, mx: FloatArray, co: FloatArray, lohi: FloatArray, sg: FloatArray, cmp: IntArray) {
                val i = globalId()
                val x = a[i]
                val y = b[i]
                mn[i] = min(x, y)
                mx[i] = max(x, y)
                // Kotlin throws when the minimum is above the maximum, so keep the bounds in order.
                co[i] = if (y <= 1f || y.isNaN()) x.coerceIn(y, 1f) else x.coerceIn(1f, y)
                lohi[2 * i] = x.coerceAtLeast(y)
                lohi[2 * i + 1] = x.coerceAtMost(y)
                sg[i] = sign(x)
                cmp[i] = x.compareTo(y) + 3 * x.compareTo(i % 3) + 9 * y.compareTo(-0f)
            }
            """,
            "k", size,
            mapOf(
                "a" to a, "b" to b, "mn" to FloatArray(size), "mx" to FloatArray(size), "co" to FloatArray(size),
                "lohi" to FloatArray(2 * size), "sg" to FloatArray(size), "cmp" to IntArray(size),
            ),
            exactFloats = true,
        )
    }

    @Test
    fun `hypot and log10 stay accurate for very big, very small and near-1 values`() {
        // length(vec2(x, y)) squares its inputs, so hypot(1e30f, 1e30f) overflowed and hypot(1e-30f, 0f) gave 0;
        // log(x) * 0.434f lost most digits near x = 1. Kotlin computes both in Double and rounds once.
        val r = Random(18)
        val special = floatArrayOf(
            0f, -0f, 1f, 10f, 100f, 1000f, 1e-30f, 1e30f, 3e38f, -1f, Float.NaN, Float.POSITIVE_INFINITY,
            1.0000001f, 0.99999994f, 1.0001f, 0.9999f, 1.5f, 0.70710677f, 2f, 0.5f,
        )
        val size = 4096
        val a = FloatArray(size) { i ->
            special.getOrElse(i) {
                when (r.nextInt(3)) {
                    0 -> 1f + (r.nextFloat() - 0.5f) * 1e-3f
                    1 -> 10f.pow(r.nextFloat() * 60f - 30f)
                    else -> r.nextFloat() * 100f
                }
            }
        }
        val b = FloatArray(size) { i -> if (i < special.size) special[(i * 7) % special.size] else 10f.pow(r.nextFloat() * 60f - 30f) }
        GpuDifferential.assertSame(
            """
            @Kernel
            fun k(a: FloatArray, b: FloatArray, h: FloatArray, l: FloatArray) {
                val i = globalId()
                h[i] = hypot(a[i], b[i])
                l[i] = log10(a[i])
            }
            """,
            "k", size, mapOf("a" to a, "b" to b, "h" to FloatArray(size), "l" to FloatArray(size)),
            // A few ULPs: WGSL's division and sqrt aren't correctly rounded.
            absTol = 0.0, relTol = 1e-6,
        )
    }

    @Test
    fun `random numbers have the same bits on the CPU and the GPU`() {
        val size = 100_000
        GpuDifferential.assertSame(
            """
            fun dice(seed: Int, i: Int): Int = randomIntBelow(6, seed, i, 4) + 1

            @Kernel
            fun k(seed: Int, ints: IntArray, below: IntArray, floats: FloatArray) {
                val i = globalId()
                // Large seeds and the indexes near Int.MAX_VALUE exercise the u32 wrap-around.
                ints[i] = randomInt(seed, i) xor randomInt(seed, Int.MAX_VALUE - i, -i)
                below[i] = randomIntBelow(1 + i * 7919, seed, i, 2) + randomIntBelow(Int.MAX_VALUE, -seed, i, 3) % 1000 + dice(seed, i)
                floats[i] = randomFloat(seed, i, 1)
            }
            """,
            "k", size,
            mapOf("seed" to -1_234_567_891, "ints" to IntArray(size), "below" to IntArray(size), "floats" to FloatArray(size)),
            exactFloats = true,
        )
    }

    @Test
    fun `random normal numbers agree on the CPU and the GPU`() {
        val size = 50_000
        val r = GpuDifferential.assertSame(
            """
            @Kernel
            fun k(seed: Int, out: FloatArray) {
                val i = globalId()
                out[i] = randomNormal(seed, i, 9)
            }
            """,
            "k", size, mapOf("seed" to 31, "out" to FloatArray(size)),
            // sqrt, ln and cos round a little differently on the GPU.
            absTol = 1e-5, relTol = 1e-4,
        )
        val gpu = r.gpu.getValue("out") as FloatArray
        val mean = gpu.sumOf { it.toDouble() } / size
        val variance = gpu.sumOf { (it - mean) * (it - mean) } / size
        assertEquals(0.0, mean, 0.03)
        assertEquals(1.0, variance, 0.04)
    }

    @Test
    fun `2D thread ids give each cell its column and row`() {
        val (w, h) = 37 to 23
        val r = GpuDifferential.assertSame(
            GpuDifferential.compile(
                """
                fun row(): Int = globalIdY()

                @Kernel
                fun k(cells: IntArray, ids: IntArray, scale: Float) {
                    val x = globalIdX()
                    cells[globalId()] = (x * scale).toInt() + row() * 1000 + globalIdZ() * 1000000
                    ids[globalId()] = x + row() * $w
                }
                """,
            ),
            "k", GpuDifferential.Grid(w, h), mapOf("cells" to IntArray(w * h), "ids" to IntArray(w * h), "scale" to 2f),
        )
        val cells = r.cpu.getValue("cells") as IntArray
        assertEquals(36 * 2 + 22 * 1000, cells[w * h - 1])
        assertEquals((0 until w * h).toList(), (r.cpu.getValue("ids") as IntArray).toList(), "globalId() is x + y * width")
    }

    @Test
    fun `3D thread ids, and the same kernel in a 1D launch`() {
        val c = GpuDifferential.compile(
            """
            @Kernel
            fun k(out: IntArray, offset: Int) {
                out[globalId()] = globalIdX() + globalIdY() * 100 + globalIdZ() * 10000 + offset
            }
            """,
        )
        val (w, h, d) = Triple(9, 5, 4)
        val r = GpuDifferential.assertSame(c, "k", GpuDifferential.Grid(w, h, d), mapOf("out" to IntArray(w * h * d), "offset" to 7))
        assertEquals(8 + 4 * 100 + 3 * 10000 + 7, (r.cpu.getValue("out") as IntArray).last())
        // In a 1D launch, globalIdX() is globalId() and y = z = 0.
        val line = GpuDifferential.assertSame(c, "k", 100, mapOf("out" to IntArray(100), "offset" to 0))
        assertEquals((0 until 100).toList(), (line.cpu.getValue("out") as IntArray).toList())
    }

    @Test
    fun `2D box blur reads its neighbors like an image kernel`() {
        val (w, h) = 61 to 45
        val c = GpuDifferential.compile(
            """
            @Kernel
            fun blur(src: FloatArray, dst: FloatArray, width: Int, height: Int) {
                val x = globalIdX()
                val y = globalIdY()
                var sum = 0f
                var count = 0
                for (dy in -1..1) for (dx in -1..1) {
                    val nx = x + dx
                    val ny = y + dy
                    if (nx in 0 until width && ny in 0 until height) {
                        sum += src[ny * width + nx]
                        count++
                    }
                }
                dst[y * width + x] = sum / count
            }
            """,
        )
        val args = mapOf("src" to floats(w * h, 21), "dst" to FloatArray(w * h), "width" to w, "height" to h)
        GpuDifferential.assertSame(c, "blur", GpuDifferential.Grid(w, h), args, relTol = 1e-5, absTol = 1e-5)
    }

    @Test
    fun `a 2D launch bigger than 65535 workgroups still covers every cell once`() {
        // 4100 x 1100 threads need 70470 workgroups of 64: more than one dimension holds, so the host spills
        // into a second dimension and the shader skips the extra threads.
        val (w, h) = 4100 to 1100
        val r = GpuDifferential.assertSame(
            GpuDifferential.compile(
                """
                @Kernel
                fun k(out: IntArray) {
                    out[globalId()] += globalIdX() * 2 + globalIdY() * 3 + 1
                }
                """,
            ),
            "k", GpuDifferential.Grid(w, h), mapOf("out" to IntArray(w * h)),
        )
        assertEquals((w - 1) * 2 + (h - 1) * 3 + 1, (r.gpu.getValue("out") as IntArray).last())
    }
}
