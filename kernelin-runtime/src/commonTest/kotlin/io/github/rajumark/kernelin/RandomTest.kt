package io.github.rajumark.kernelin

import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/** Kernelin's random numbers: fixed bits on every platform, and sane statistics. */
class RandomTest {
    private val n = 100_000

    /** The PCG hash written with UInt, as the WGSL does it, to check the Int version wraps the same way. */
    private fun pcgReference(v: UInt): UInt {
        val state = v * 747796405u + 2891336453u
        val word = ((state shr ((state shr 28) + 4u).toInt()) xor state) * 277803737u
        return (word shr 22) xor word
    }

    private fun bitsReference(seed: Int, index: Int, stream: Int): UInt =
        pcgReference(index.toUInt() + pcgReference(stream.toUInt() + pcgReference(seed.toUInt())))

    @Test
    fun randomIntMatchesTheUnsignedReference() {
        val values = listOf(0, 1, -1, 7, 123_456_789, Int.MAX_VALUE, Int.MIN_VALUE)
        for (seed in values) for (index in values) for (stream in values) {
            assertEquals(bitsReference(seed, index, stream).toInt(), randomInt(seed, index, stream), "($seed, $index, $stream)")
        }
    }

    @Test
    fun valuesAreFixedOnEveryPlatform() {
        // Pinned: changing the generator would change every user's results.
        assertContentEquals(GOLDEN_INTS, IntArray(GOLDEN_INTS.size) { randomInt(42, it, 3) })
        assertEquals(GOLDEN_INTS[0].ushr(8) / 16_777_216f, randomFloat(42, 0, 3))
        assertEquals(randomInt(5, 9), randomInt(5, 9, 0))
    }

    @Test
    fun seedIndexAndStreamAllChangeTheValue() {
        val a = randomInt(1, 2, 3)
        assertTrue(a != randomInt(0, 2, 3) && a != randomInt(1, 0, 3) && a != randomInt(1, 2, 0))
        // For one seed and stream, different indexes never give the same bits (the hash is one-to-one).
        val seen = HashSet<Int>()
        for (i in 0 until n) assertTrue(seen.add(randomInt(7, i, 0)), "index $i repeats a value")
    }

    @Test
    fun randomFloatIsUniformInZeroToOne() {
        for (stream in 0 until 3) {
            val bins = IntArray(100)
            var sum = 0.0
            var sumSq = 0.0
            for (i in 0 until n) {
                val u = randomFloat(2024, i, stream)
                assertTrue(u >= 0f && u < 1f, "out of range: $u")
                bins[(u * bins.size).toInt()]++
                sum += u
                sumSq += u.toDouble() * u
            }
            val mean = sum / n
            val variance = sumSq / n - mean * mean
            assertTrue(abs(mean - 0.5) < 0.005, "mean $mean")
            assertTrue(abs(variance - 1.0 / 12) < 0.002, "variance $variance")
            assertChiSquareOk(bins, n)
        }
    }

    @Test
    fun randomIntBelowIsUniformAndInRange() {
        for (bound in listOf(1, 2, 10, 37, Int.MAX_VALUE)) {
            for (i in 0 until 1000) {
                val v = randomIntBelow(bound, 9, i, 1)
                assertTrue(v in 0 until bound, "$v not below $bound")
            }
        }
        val bins = IntArray(10)
        for (i in 0 until n) bins[randomIntBelow(10, 3, i)]++
        assertChiSquareOk(bins, n)
        assertFailsWith<IllegalArgumentException> { randomIntBelow(0, 1, 2) }
    }

    @Test
    fun randomBitsAreBalanced() {
        // Every one of the 32 bits is set about half the time, also the low bits.
        val ones = IntArray(32)
        for (i in 0 until n) {
            val v = randomInt(11, i, 5)
            for (b in 0 until 32) if ((v ushr b) and 1 == 1) ones[b]++
        }
        for (b in 0 until 32) assertTrue(abs(ones[b] - n / 2) < 1000, "bit $b set ${ones[b]} times")
    }

    @Test
    fun randomNormalHasMeanZeroAndVarianceOne() {
        var sum = 0.0
        var sumSq = 0.0
        var sum4 = 0.0
        for (i in 0 until n) {
            val z = randomNormal(77, i, 2).toDouble()
            assertTrue(z.isFinite() && abs(z) < 6.0, "z = $z")
            sum += z
            sumSq += z * z
            sum4 += z * z * z * z
        }
        val mean = sum / n
        val variance = sumSq / n - mean * mean
        assertTrue(abs(mean) < 0.02, "mean $mean")
        assertTrue(abs(variance - 1.0) < 0.03, "variance $variance")
        // The fourth moment of a normal distribution is 3 (a uniform one would give 1.8).
        assertTrue(abs(sum4 / n - 3.0) < 0.15, "fourth moment ${sum4 / n}")
    }

    /** Pearson's chi-square against equal bins, with a bound about 5 standard deviations above its mean. */
    private fun assertChiSquareOk(bins: IntArray, total: Int) {
        val expected = total.toDouble() / bins.size
        val chi2 = bins.sumOf { (it - expected) * (it - expected) / expected }
        val dof = bins.size - 1
        val limit = dof + 5 * kotlin.math.sqrt(2.0 * dof)
        assertTrue(chi2 < limit, "chi-square $chi2 >= $limit for ${bins.toList()}")
    }

    private companion object {
        val GOLDEN_INTS = intArrayOf(-819049653, -173502500, 1048660287, -1445677254, -1410506178, 170171857, 364824900, -523447125)
    }
}
