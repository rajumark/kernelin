package io.github.rajumark.kernelin.samples

import io.github.rajumark.kernelin.KernelSource
import io.github.rajumark.kernelin.KernelinShaders
import io.github.rajumark.kernelin.webgpu.GpuData
import io.github.rajumark.kernelin.webgpu.GpuKernel
import io.github.rajumark.kernelin.webgpu.GpuResult
import io.github.rajumark.kernelin.webgpu.WebGpu
import kotlin.math.abs
import kotlinx.coroutines.runBlocking

/**
 * Times every case on the CPU and on this computer's GPU (the JVM GPU host, no browser), checks the GPU results
 * against the CPU, and prints one JSON line per case, in the format of the files in `benchmarks/`.
 *
 * gpuKernelMs: data stays on the GPU, average of 20 back-to-back runs. gpuEndToEndMs: upload, run and read back.
 */
internal fun benchmarkOnGpu(cases: List<Case>): Unit = runBlocking {
    WebGpu.create().use { gpu ->
        println("GPU: ${gpu.description}")
        val lines = cases.map { case ->
            case.measure()
            val fq = "io.github.rajumark.kernelin.samples.${case.kernel}"
            val kernel = gpu.load(KernelSource(fq, KernelinShaders.wgsl(fq), KernelinShaders.manifest(fq)))
            try {
                val scalars: Map<String, Any> = case.scalars
                var last: GpuResult? = null
                val endToEnd = median {
                    val data = gpuData(case.freshBuffers())
                    val start = System.nanoTime()
                    last = kernel.run(case.size, scalars, data).also { r -> case.outputs.keys.forEach { r.bytes(it) } }
                    (System.nanoTime() - start) / 1e6
                }
                val kernelOnly = median { (chained(kernel, case, 20) - chained(kernel, case, 0)) / 20 }
                val (match, maxErr) = compare(case.expected, last!!)
                println("  %-12s GPU kernel: %8.2f ms   GPU end to end: %8.2f ms   match %.4f%%".format(case.name, kernelOnly, endToEnd, match))
                "{\"name\":\"${case.name}\",\"size\":${case.size},\"gpuKernelMs\":${"%.3f".format(kernelOnly)}," +
                    "\"gpuEndToEndMs\":${"%.3f".format(endToEnd)},\"cpuSingleMs\":${"%.3f".format(case.singleMs)}," +
                    "\"cpuParallelMs\":${"%.3f".format(case.parallelMs)},\"matchPct\":${"%.4f".format(match)},\"maxAbsErr\":$maxErr}"
            } finally {
                kernel.close()
            }
        }
        println(lines.joinToString(",\n", prefix = "GPU_RESULTS [\n", postfix = "\n]"))
    }
}

/** Milliseconds for [runs] runs that keep every buffer on the GPU, plus one that reads an output back. */
private suspend fun chained(kernel: GpuKernel, case: Case, runs: Int): Double {
    val scalars: Map<String, Any> = case.scalars
    var r = kernel.run(case.size, scalars, gpuData(case.freshBuffers()), readBack = emptySet())
    fun onGpu(r: GpuResult) = kernel.manifest.buffers.associate { it.name to GpuData.of(r.buffer(it.name)) }
    val start = System.nanoTime()
    repeat(runs) { r = kernel.run(case.size, scalars, onGpu(r), readBack = emptySet()) }
    kernel.run(case.size, scalars, onGpu(r), readBack = setOf(case.outputs.keys.first()))
    return (System.nanoTime() - start) / 1e6
}

/** Same rule as the browser runner: Ints must be equal, Floats within 1e-4 relative. Returns (match %, max error). */
private fun compare(expected: Map<String, Any>, result: GpuResult): Pair<Double, Double> {
    var total = 0L
    var bad = 0L
    var maxErr = 0.0
    for ((name, want) in expected) {
        when (want) {
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

private fun gpuData(buffers: Map<String, Any>): Map<String, GpuData> = buffers.mapValues { (_, v) ->
    when (v) {
        is FloatArray -> GpuData.of(v)
        is IntArray -> GpuData.of(v)
        else -> error("unsupported buffer $v")
    }
}

private suspend fun median(sample: suspend () -> Double): Double {
    repeat(2) { sample() }
    return List(5) { sample() }.sorted()[2]
}
