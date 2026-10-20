package io.github.rajumark.kernelin.samples.android

import io.github.rajumark.kernelin.KernelSource
import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.kernelSource
import io.github.rajumark.kernelin.samples.blur
import io.github.rajumark.kernelin.samples.mandelbrot
import io.github.rajumark.kernelin.samples.saxpy
import io.github.rajumark.kernelin.samples.vectorAdd
import io.github.rajumark.kernelin.samples.waves
import io.github.rajumark.kernelin.webgpu.GpuData
import io.github.rajumark.kernelin.webgpu.GpuKernel
import io.github.rajumark.kernelin.webgpu.GpuResult
import io.github.rajumark.kernelin.webgpu.WebGpu
import kotlin.math.abs
import kotlin.random.Random

/**
 * One benchmark: the same kernel on the CPU (1 core and all cores, through [KernelinCpu]) and on the GPU (native
 * WebGPU). Timing rules match the browser runner (samples/web/runner.html), so the numbers can be compared:
 * medians of 5 runs after 2 warm-ups; `gpuKernelMs` is one dispatch with the data already on the GPU (20 chained
 * runs); `gpuEndToEndMs` is upload + run + read back into Kotlin arrays.
 */
class BenchCase(
    val name: String,
    val source: KernelSource,
    val size: Int,
    val scalars: Map<String, Any>,
    inputs: Map<String, Any>,
    outputs: Map<String, Any>,
    val cpu: (buffers: Map<String, Any>, parallel: Boolean) -> Unit,
) {
    /** Inputs by parameter name (FloatArray or IntArray). */
    val inputs: MutableMap<String, Any> = LinkedHashMap(inputs)

    /**
     * Every buffer the kernel writes, as a zeroed prototype. A buffer that is also an input uses the input instead,
     * so its unused prototype doesn't take up heap.
     */
    val outputs: MutableMap<String, Any> = LinkedHashMap(outputs.mapValues { (name, v) -> inputs[name] ?: v })

    /**
     * Lets go of the arrays once the case is measured. A finished coroutine can still hold a reference to the case,
     * and the next case's 64 MB arrays only fit in a phone's app heap once these are gone.
     */
    fun release() {
        inputs.clear()
        outputs.clear()
    }
}

class BenchResult(
    val name: String,
    val size: Int,
    val gpuKernelMs: Double?,
    val gpuEndToEndMs: Double?,
    val cpuSingleMs: Double,
    val cpuParallelMs: Double,
    val matchPct: Double?,
    val maxAbsErr: Double?,
) {
    fun json(): String = "{\"name\":\"$name\",\"size\":$size,\"gpuKernelMs\":${num(gpuKernelMs)},\"gpuEndToEndMs\":${num(gpuEndToEndMs)}," +
        "\"cpuSingleMs\":${num(cpuSingleMs)},\"cpuParallelMs\":${num(cpuParallelMs)},\"matchPct\":${num(matchPct)},\"maxAbsErr\":${num(maxAbsErr)}}"
}

private fun num(v: Double?): String = if (v == null) "null" else "%.3f".format(java.util.Locale.ROOT, v).trimEnd('0').trimEnd('.')

/** The five benchmark cases of `gradle :samples:bench`; [scale] divides the sizes for quick runs (1 = full size). */
fun benchCases(scale: Int): List<() -> BenchCase> {
    val big = (16 shl 20) / scale
    val mid = (4 shl 20) / scale
    val side = 2048 / isqrt(scale)
    return listOf(
        {
            BenchCase(
                "vectorAdd", kernelSource(::vectorAdd), big, emptyMap(),
                inputs = mapOf("a" to randomFloats(big, 1), "b" to randomFloats(big, 2)), outputs = mapOf("out" to FloatArray(big)),
            ) { b, p ->
                val a = b["a"] as FloatArray; val bb = b["b"] as FloatArray; val out = b["out"] as FloatArray
                dispatch(big, p) { vectorAdd(a, bb, out) }
            }
        },
        {
            BenchCase(
                "saxpy", kernelSource(::saxpy), big, mapOf("alpha" to 2.5f),
                inputs = mapOf("x" to randomFloats(big, 3), "y" to randomFloats(big, 4)), outputs = mapOf("y" to FloatArray(big)),
            ) { b, p ->
                val x = b["x"] as FloatArray; val y = b["y"] as FloatArray
                dispatch(big, p) { saxpy(2.5f, x, y) }
            }
        },
        {
            BenchCase(
                "blur", kernelSource(::blur), big, emptyMap(),
                inputs = mapOf("input" to randomFloats(big, 5)), outputs = mapOf("out" to FloatArray(big)),
            ) { b, p ->
                val input = b["input"] as FloatArray; val out = b["out"] as FloatArray
                dispatch(big, p) { blur(input, out) }
            }
        },
        {
            BenchCase(
                "waves", kernelSource(::waves), mid, mapOf("rounds" to 64),
                inputs = mapOf("input" to randomFloats(mid, 6)), outputs = mapOf("out" to FloatArray(mid)),
            ) { b, p ->
                val input = b["input"] as FloatArray; val out = b["out"] as FloatArray
                dispatch(mid, p) { waves(input, out, 64) }
            }
        },
        {
            BenchCase(
                "mandelbrot", kernelSource(::mandelbrot), side * side, mapOf("width" to side, "height" to side, "maxIter" to 256),
                inputs = emptyMap(), outputs = mapOf("out" to IntArray(side * side)),
            ) { b, p ->
                val out = b["out"] as IntArray
                dispatch(side * side, p) { mandelbrot(out, side, side, 256) }
            }
        },
    )
}

/** Runs one case. [gpu] is null when the device has no usable GPU; then only the CPU is timed. */
suspend fun BenchCase.measure(gpu: WebGpu?): BenchResult {
    val cpuSingle = median { val b = fresh(); time { cpu(b, false) } }
    val cpuParallel = median { val b = fresh(); time { cpu(b, true) } }
    if (gpu == null) return BenchResult(name, size, null, null, cpuSingle, cpuParallel, null, null)

    // Only the outputs, copied: fresh() reuses the zeroed output arrays.
    var expected = fresh().also { cpu(it, true) }.filterKeys { it in outputs }.mapValues { (_, v) -> copy(v) }
    val kernel = gpu.load(source)
    try {
        val last = LastResult()
        val endToEnd = medianSuspend { endToEnd(kernel, last) }
        // Checked now, so the result's memory is free again for the kernel-only runs.
        val (match, maxErr) = compare(expected, last.result!!)
        last.result = null
        expected = emptyMap()
        val kernelOnly = medianSuspend { (chained(kernel, 20) - chained(kernel, 0)) / 20 }
        return BenchResult(name, size, kernelOnly, endToEnd, cpuSingle, cpuParallel, match, maxErr)
    } finally {
        kernel.close()
    }
}

/** Holds the result of the latest end-to-end run, for checking it against the CPU. */
private class LastResult(var result: GpuResult? = null)

/**
 * Milliseconds for one run: upload, run, and read every output into a Kotlin array. Its own function, not part of
 * the inline median loop, so no coroutine slot keeps the previous result alive: the 16M-element cases need its
 * 64 MB back before the next run fits in a phone's app heap.
 */
private suspend fun BenchCase.endToEnd(kernel: GpuKernel, last: LastResult): Double {
    last.result = null
    val data = gpuData()
    val start = System.nanoTime()
    val r = kernel.run(size, scalars, data)
    outputs.keys.forEach { read(r, it) }
    val ms = (System.nanoTime() - start) / 1e6
    last.result = r
    return ms
}

/**
 * Milliseconds for [runs] dispatches that keep every buffer on the GPU, plus one that reads an output back (so the
 * GPU work is finished). Calling it with 20 and 0 and subtracting leaves the time of 20 dispatches.
 */
private suspend fun BenchCase.chained(kernel: GpuKernel, runs: Int): Double {
    var r = kernel.run(size, scalars, gpuData(), readBack = emptySet())
    fun onGpu(r: GpuResult) = kernel.manifest.buffers.associate { it.name to GpuData.of(r.buffer(it.name)) }
    val start = System.nanoTime()
    repeat(runs) { r = kernel.run(size, scalars, onGpu(r), readBack = emptySet()) }
    kernel.run(size, scalars, onGpu(r), readBack = setOf(outputs.keys.first()))
    return (System.nanoTime() - start) / 1e6
}

/** Same rule as the browser runner: Ints must be equal, Floats within 1e-4 relative. */
private fun BenchCase.compare(expected: Map<String, Any>, result: GpuResult): Pair<Double, Double> {
    var total = 0L
    var bad = 0L
    var maxErr = 0.0
    for (name in outputs.keys) {
        when (val want = expected.getValue(name)) {
            is IntArray -> {
                val got = result.ints(name)
                total += want.size
                for (i in want.indices) if (got[i] != want[i]) bad++
            }
            is FloatArray -> {
                val got = result.floats(name)
                total += want.size
                for (i in want.indices) {
                    val err = abs(got[i].toDouble() - want[i].toDouble())
                    if (err > maxErr) maxErr = err
                    if (!(err <= 1e-4 * (1 + abs(want[i].toDouble())))) bad++
                }
            }
        }
    }
    return 100.0 * (total - bad) / total to maxErr
}

/**
 * The data for one run, so every run starts from the same values: a copy of each input the kernel also writes,
 * the zeroed output arrays (reused, not copied), and the read-only inputs as they are. Copying only what the kernel
 * writes keeps the 16M-element cases inside a phone's app heap.
 */
private fun BenchCase.fresh(): Map<String, Any> =
    inputs.filterKeys { it !in outputs } + outputs.mapValues { (name, v) -> inputs[name]?.let(::copy) ?: v.also(::zero) }

/**
 * The data for one GPU run. [GpuData.of] copies, so the original inputs are used as they are; outputs that aren't
 * also inputs start as [GpuData.empty] (zeroed on the GPU, no upload).
 */
private fun BenchCase.gpuData(): Map<String, GpuData> = (outputs + inputs).mapValues { (name, v) ->
    when {
        name !in inputs -> GpuData.empty(if (v is FloatArray) v.size else (v as IntArray).size)
        v is FloatArray -> GpuData.of(v)
        v is IntArray -> GpuData.of(v)
        else -> error("unsupported buffer $v")
    }
}

private fun BenchCase.read(r: GpuResult, name: String): Any =
    if (outputs.getValue(name) is IntArray) r.ints(name) else r.floats(name)

private fun zero(v: Any) = when (v) {
    is FloatArray -> v.fill(0f)
    is IntArray -> v.fill(0)
    else -> error("unsupported buffer $v")
}

private fun copy(v: Any): Any = when (v) {
    is FloatArray -> v.copyOf()
    is IntArray -> v.copyOf()
    else -> error("unsupported buffer $v")
}

private inline fun time(body: () -> Unit): Double {
    val start = System.nanoTime()
    body()
    return (System.nanoTime() - start) / 1e6
}

private inline fun median(sample: () -> Double): Double {
    repeat(2) { sample() }
    return List(5) { sample() }.sorted()[2]
}

private suspend inline fun medianSuspend(sample: () -> Double): Double {
    repeat(2) { sample() }
    return List(5) { sample() }.sorted()[2]
}

private fun dispatch(size: Int, parallel: Boolean, kernel: () -> Unit) =
    if (parallel) KernelinCpu.dispatchParallel(size, kernel) else KernelinCpu.dispatch(size, kernel)

private fun randomFloats(n: Int, seed: Int) = Random(seed).let { r -> FloatArray(n) { r.nextFloat() * 2f - 1f } }

private fun isqrt(n: Int): Int = kotlin.math.sqrt(n.toDouble()).toInt().coerceAtLeast(1)
