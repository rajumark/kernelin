package io.github.rajumark.kernelin

import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sqrt

/*
 * Random numbers for kernels. They are counter based and keep no state: the same (seed, index, stream) always
 * gives the same number, on the CPU and on the GPU, however the work is split into threads. Use the thread's
 * globalId() as the index and count the draws of one thread with the stream (0, 1, 2, ...).
 *
 * The bits come from three rounds of the PCG hash (Jarzynski and Olano, "Hash Functions for GPU Rendering", 2020).
 * For one seed and stream, every index gets a different 32-bit value. Not for cryptography.
 */

/** The PCG hash: a fast mix of 32 bits where every input gives a different output. */
private fun pcgHash(v: Int): Int {
    val state = v * 747796405 + -1403630843 // 2891336453 as a u32
    val word = ((state ushr ((state ushr 28) + 4)) xor state) * 277803737
    return (word ushr 22) xor word
}

/**
 * A random Int (all 32 bits random) for draw [stream] of thread [index], from [seed].
 * Gives the same bits on the CPU and on the GPU.
 */
public fun randomInt(seed: Int, index: Int, stream: Int = 0): Int =
    pcgHash(index + pcgHash(stream + pcgHash(seed)))

/**
 * A random Float in `[0, 1)` (never 1) for draw [stream] of thread [index], from [seed]. It is a multiple of
 * 2^-24, the top 24 bits of [randomInt], so the CPU and the GPU give exactly the same Float.
 */
public fun randomFloat(seed: Int, index: Int, stream: Int = 0): Float =
    (randomInt(seed, index, stream) ushr 8).toFloat() / TWO_POW_24

/**
 * A random Int in `0 until bound` for draw [stream] of thread [index], from [seed]. Gives the same Int on the
 * CPU and on the GPU. Uses the top bits of a 32 x 32-bit product, so the bias is at most `bound / 2^32`.
 *
 * [bound] must be positive: the CPU throws [IllegalArgumentException], the GPU (which can't throw) returns 0.
 */
public fun randomIntBelow(bound: Int, seed: Int, index: Int, stream: Int = 0): Int {
    require(bound > 0) { "bound must be positive, was $bound" }
    return (((randomInt(seed, index, stream).toLong() and 0xffffffffL) * bound) ushr 32).toInt()
}

/**
 * A random Float from the standard normal distribution (mean 0, standard deviation 1) for draw [stream] of thread
 * [index], from [seed], by the Box-Muller method. It uses ln, sqrt and cos, which the GPU computes a little
 * differently, so CPU and GPU agree to about 1e-5 but not bit for bit. Values are within about ±5.8.
 */
public fun randomNormal(seed: Int, index: Int, stream: Int = 0): Float {
    val bits = randomInt(seed, index, stream)
    val u1 = 1f - (bits ushr 8).toFloat() / TWO_POW_24 // (0, 1], so ln() is safe
    val u2 = (pcgHash(bits) ushr 8).toFloat() / TWO_POW_24
    // An angle in [-pi, pi), where the GPU's cos is most accurate.
    return sqrt(-2f * ln(u1)) * cos(u2 * 6.2831855f - 3.1415927f)
}

// Dividing by 2^24 is exact everywhere. Multiplying by 5.9604645E-8f isn't on Kotlin/JS, where a Float is a
// JS double and that literal isn't exactly 2^-24 as a double.
private const val TWO_POW_24 = 16_777_216f
