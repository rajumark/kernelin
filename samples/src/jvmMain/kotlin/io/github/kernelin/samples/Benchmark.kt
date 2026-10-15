package io.github.kernelin.samples

import io.github.kernelin.KernelinCpu
import io.github.kernelin.KernelinShaders
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.random.Random

/**
 * Times every sample kernel on the CPU and exports what the WebGPU runner needs:
 * the generated shaders, input buffers, expected outputs and the CPU timings.
 *
 * Usage: `./gradlew :samples:bench -Pout=<dir>`, then open runner.html served from <dir>.
 */
fun main(args: Array<String>) {
    val out = File(args.getOrElse(0) { "build/kernelin-bench" }).apply { mkdirs() }
    val cores = Runtime.getRuntime().availableProcessors()
    println("Kernelin CPU benchmark - ${System.getProperty("java.vm.name")} ${System.getProperty("java.version")}, $cores cores")

    val cases = listOf(
        vectorAddCase(16 shl 20),
        saxpyCase(16 shl 20),
        blurCase(16 shl 20),
        wavesCase(4 shl 20, rounds = 64),
        mandelbrotCase(2048, 2048, maxIter = 256),
    )
    val json = cases.joinToString(",\n", prefix = "[\n", postfix = "\n]\n") { it.export(out) }
    out.resolve("cases.json").writeText(json)
    out.resolve("cpu-info.json").writeText(
        "{ \"jvm\": \"${System.getProperty("java.vm.name")} ${System.getProperty("java.version")}\", \"cores\": $cores }\n",
    )
    println("Wrote ${cases.size} cases to ${out.absolutePath}")
}

private class Case(
    val name: String,
    val kernel: String,
    val size: Int,
    val scalars: Map<String, Number>,
    val inputs: Map<String, Any>,
    /** Buffers the kernel writes, as zeroed prototypes (an input may also be an output). */
    val outputs: Map<String, Any>,
    /** Runs the kernel once on [buffers] (fresh copies of the inputs plus zeroed outputs). */
    val run: (buffers: Map<String, Any>, parallel: Boolean) -> Unit,
) {
    lateinit var expected: Map<String, Any>
    var singleMs = 0.0
    var parallelMs = 0.0

    private fun freshBuffers(): Map<String, Any> =
        (outputs.filterKeys { it !in inputs } + inputs).mapValues { (_, v) -> copy(v) }

    fun measure() {
        singleMs = time { run(it, false) }
        parallelMs = time { run(it, true) }
        val buffers = freshBuffers()
        run(buffers, true)
        expected = outputs.keys.associateWith { buffers.getValue(it) }
        println("  %-12s 1 core: %9.2f ms   all cores: %8.2f ms".format(name, singleMs, parallelMs))
    }

    private fun time(body: (Map<String, Any>) -> Unit): Double {
        repeat(2) { body(freshBuffers()) }
        val samples = (0 until 5).map {
            val buffers = freshBuffers()
            val start = System.nanoTime()
            body(buffers)
            (System.nanoTime() - start) / 1e6
        }
        return samples.sorted()[samples.size / 2]
    }

    fun export(dir: File): String {
        measure()
        val fq = "io.github.kernelin.samples.$kernel"
        dir.resolve("$fq.wgsl").writeText(KernelinShaders.wgsl(fq))
        dir.resolve("$fq.json").writeText(KernelinShaders.manifest(fq))
        val inputFiles = inputs.map { (param, data) ->
            "\"$param\": \"${writeBin(dir, "$name.in.$param.bin", data)}\""
        }
        val expectedFiles = expected.map { (param, data) ->
            "\"$param\": \"${writeBin(dir, "$name.expected.$param.bin", data)}\""
        }
        val lengths = (outputs + inputs).map { (k, v) -> "\"$k\": ${length(v)}" }
        return """  {
    "name": "$name", "kernel": "$fq", "size": $size,
    "scalars": { ${scalars.map { (k, v) -> "\"$k\": $v" }.joinToString(", ")} },
    "lengths": { ${lengths.joinToString(", ")} },
    "inputs": { ${inputFiles.joinToString(", ")} },
    "expected": { ${expectedFiles.joinToString(", ")} },
    "cpu": { "singleMs": ${"%.3f".format(singleMs)}, "parallelMs": ${"%.3f".format(parallelMs)} }
  }"""
    }
}

private fun copy(v: Any): Any = when (v) {
    is FloatArray -> v.copyOf()
    is IntArray -> v.copyOf()
    else -> error("unsupported buffer $v")
}

private fun length(v: Any): Int = when (v) {
    is FloatArray -> v.size
    is IntArray -> v.size
    else -> error("unsupported buffer $v")
}

private fun writeBin(dir: File, name: String, data: Any): String {
    val bytes = ByteBuffer.allocate(length(data) * 4).order(ByteOrder.LITTLE_ENDIAN)
    when (data) {
        is FloatArray -> bytes.asFloatBuffer().put(data)
        is IntArray -> bytes.asIntBuffer().put(data)
    }
    dir.resolve(name).writeBytes(bytes.array())
    return name
}

private fun randomFloats(n: Int, seed: Int) = Random(seed).let { r -> FloatArray(n) { r.nextFloat() * 2f - 1f } }

private inline fun dispatch(size: Int, parallel: Boolean, noinline kernel: () -> Unit) =
    if (parallel) KernelinCpu.dispatchParallel(size, kernel) else KernelinCpu.dispatch(size, kernel)

private fun vectorAddCase(n: Int) = Case(
    "vectorAdd", "vectorAdd", n, emptyMap(),
    inputs = mapOf("a" to randomFloats(n, 1), "b" to randomFloats(n, 2)),
    outputs = mapOf("out" to FloatArray(n)),
) { b, p ->
    val a = b["a"] as FloatArray; val bb = b["b"] as FloatArray; val out = b["out"] as FloatArray
    dispatch(n, p) { vectorAdd(a, bb, out) }
}

private fun saxpyCase(n: Int) = Case(
    "saxpy", "saxpy", n, mapOf("alpha" to 2.5f),
    inputs = mapOf("x" to randomFloats(n, 3), "y" to randomFloats(n, 4)),
    outputs = mapOf("y" to FloatArray(0)),
) { b, p ->
    val x = b["x"] as FloatArray; val y = b["y"] as FloatArray
    dispatch(n, p) { saxpy(2.5f, x, y) }
}

private fun blurCase(n: Int) = Case(
    "blur", "blur", n, emptyMap(),
    inputs = mapOf("input" to randomFloats(n, 5)),
    outputs = mapOf("out" to FloatArray(n)),
) { b, p ->
    val input = b["input"] as FloatArray; val out = b["out"] as FloatArray
    dispatch(n, p) { blur(input, out) }
}

private fun wavesCase(n: Int, rounds: Int) = Case(
    "waves", "waves", n, mapOf("rounds" to rounds),
    inputs = mapOf("input" to randomFloats(n, 6)),
    outputs = mapOf("out" to FloatArray(n)),
) { b, p ->
    val input = b["input"] as FloatArray; val out = b["out"] as FloatArray
    dispatch(n, p) { waves(input, out, rounds) }
}

private fun mandelbrotCase(w: Int, h: Int, maxIter: Int) = Case(
    "mandelbrot", "mandelbrot", w * h, mapOf("width" to w, "height" to h, "maxIter" to maxIter),
    inputs = emptyMap(),
    outputs = mapOf("out" to IntArray(w * h)),
) { b, p ->
    val out = b["out"] as IntArray
    dispatch(w * h, p) { mandelbrot(out, w, h, maxIter) }
}
