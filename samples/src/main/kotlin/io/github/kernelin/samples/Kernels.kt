package io.github.kernelin.samples

import io.github.kernelin.Kernel
import io.github.kernelin.globalId
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

/** out = a + b, element by element. Memory-bound: tiny math per element. */
@Kernel
fun vectorAdd(a: FloatArray, b: FloatArray, out: FloatArray) {
    val i = globalId()
    out[i] = a[i] + b[i]
}

/** y = alpha * x + y (the classic BLAS "saxpy"). */
@Kernel
fun saxpy(alpha: Float, x: FloatArray, y: FloatArray) {
    val i = globalId()
    y[i] += alpha * x[i]
}

/** Iteration count of the Mandelbrot set for one pixel. Compute-bound. */
@Kernel
fun mandelbrot(out: IntArray, width: Int, height: Int, maxIter: Int) {
    val id = globalId()
    val px = id % width
    val py = id / width
    val x0 = px.toFloat() / width * 3.5f - 2.5f
    val y0 = py.toFloat() / height * 2.0f - 1.0f
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

/** Heavy per-element math with a loop and a helper function. */
@Kernel
fun waves(input: FloatArray, out: FloatArray, rounds: Int) {
    val i = globalId()
    var v = input[i]
    for (r in 0 until rounds) {
        v = smooth(sin(v) * cos(v * 0.5f) + r * 0.001f)
    }
    out[i] = v
}

fun smooth(x: Float): Float = x / (1f + sqrt(x * x))

/** 1D box blur with radius 4: each output is the average of 9 neighbours. */
@Kernel
fun blur(input: FloatArray, out: FloatArray) {
    val i = globalId()
    val n = input.size
    var sum = 0f
    var count = 0
    for (d in -4..4) {
        val j = i + d
        if (j >= 0 && j < n) {
            sum += input[j]
            count++
        }
    }
    out[i] = sum / count
}
