package io.github.rajumark.kernelin.compiler

import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelinCpu
import org.junit.jupiter.api.Assumptions
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.file.Files
import java.util.concurrent.TimeUnit
import kotlin.math.abs
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * Differential testing: runs a kernel as plain Kotlin on the CPU and as the generated WGSL on a real GPU
 * (WebGPU through Deno), then compares every written buffer.
 *
 * Without Deno or a GPU adapter the GPU tests are skipped, unless KERNELIN_REQUIRE_GPU=1 (set in CI), where a
 * missing GPU fails the build so the tests can never silently stop running.
 */
object GpuDifferential {
    private val deno: File? = listOfNotNull(
        System.getenv("DENO"),
        "/opt/homebrew/bin/deno",
        "/usr/local/bin/deno",
        System.getProperty("user.home") + "/.deno/bin/deno",
    ).map(::File).firstOrNull { it.canExecute() }
        ?: System.getenv("PATH")?.split(File.pathSeparator)?.map { File(it, "deno") }?.firstOrNull { it.canExecute() }

    private val script = File(System.getProperty("kernelin.gpuRunner") ?: "src/test/js/gpu-run.mjs")

    private val required get() = System.getenv("KERNELIN_REQUIRE_GPU") == "1"

    /** Name of the GPU adapter, or null when no GPU can be used. */
    val adapter: String? by lazy {
        val d = deno ?: return@lazy null
        val out = runDeno(d, listOf("--probe"))
        Regex("\"adapter\":\"([^\"]*)\"").find(out.second)?.groupValues?.get(1)?.takeIf { out.first == 0 }
    }

    fun requireGpu() {
        if (adapter == null) {
            if (required) fail("KERNELIN_REQUIRE_GPU=1 but no GPU is available (deno: ${deno ?: "not found"})")
            Assumptions.assumeTrue(false, "No WebGPU adapter (install Deno for GPU tests)")
        }
    }

    class Result(val cpu: Map<String, Any>, val gpu: Map<String, Any>)

    /**
     * Compiles [source] (imports added), runs @Kernel [kernel] for ids `0 until size` on the CPU and the GPU,
     * and returns both sets of outputs. [args] holds every parameter by name: arrays are inputs (copied for each
     * side), Int/Float/Boolean are scalars.
     */
    fun run(source: String, kernel: String, size: Int, args: Map<String, Any>): Result {
        requireGpu()
        val c = KernelinCompiler.compile("import io.github.rajumark.kernelin.*\nimport kotlin.math.*\n\n$source")
        if (!c.ok) fail("Compilation failed:\n${c.output}")
        c.wgslFiles().forEach(Naga::validate)
        val manifest = KernelManifest.parse(c.manifest(kernel))

        // CPU: the very same Kotlin function, called once per id.
        val method = c.loadClass("KernelsKt").methods.single { it.name == kernel }
        val cpuArgs = manifest.params.map { p -> copy(args[p.name] ?: fail("missing argument ${p.name}")) }
        KernelinCpu.dispatch(size) { method.invoke(null, *cpuArgs.toTypedArray()) }
        val cpu = manifest.params.withIndex()
            .filter { (_, p) -> p.isWritten }
            .associate { (i, p) -> p.name to cpuArgs[i] }

        // GPU: the generated WGSL through WebGPU.
        val dir = Files.createTempDirectory("kernelin-gpu").toFile()
        val wgsl = dir.resolve("kernel.wgsl").apply { writeText(c.wgsl(kernel)) }
        val manifestFile = dir.resolve("kernel.json").apply { writeText(c.manifest(kernel)) }
        val buffers = manifest.buffers.joinToString(",\n") { p ->
            val data = args.getValue(p.name)
            val input = dir.resolve("${p.name}.in.bin").apply { writeBytes(bytes(data)) }
            "\"${p.name}\": { \"length\": ${length(data)}, \"input\": \"${input.path}\", \"output\": \"${dir.resolve("${p.name}.out.bin").path}\" }"
        }
        val scalars = manifest.scalars.joinToString(", ") { p -> "\"${p.name}\": ${args.getValue(p.name)}" }
        val job = dir.resolve("job.json").apply {
            writeText("{ \"wgsl\": \"${wgsl.path}\", \"manifest\": \"${manifestFile.path}\", \"size\": $size,\n" +
                "\"scalars\": { $scalars },\n\"buffers\": { $buffers } }")
        }
        val (exit, out) = runDeno(deno!!, listOf(job.path))
        if (exit != 0) fail("GPU run failed: $out\n--- shader ---\n${c.wgsl(kernel)}")
        val gpu = manifest.buffers.filter { it.isWritten }.associate { p ->
            val raw = ByteBuffer.wrap(dir.resolve("${p.name}.out.bin").readBytes()).order(ByteOrder.LITTLE_ENDIAN)
            p.name to if (p.type == "i32") IntArray(raw.remaining() / 4) { raw.getInt() } else FloatArray(raw.remaining() / 4) { raw.getFloat() }
        }
        return Result(cpu, gpu)
    }

    /**
     * Runs and asserts CPU == GPU. Floats may differ by `absTol + relTol * |cpu|`; up to [mismatchFraction] of
     * elements may differ beyond that (for chaotic float math like Mandelbrot, where GPUs fuse multiply-adds).
     */
    fun assertSame(
        source: String,
        kernel: String,
        size: Int,
        args: Map<String, Any>,
        absTol: Double = 1e-6,
        relTol: Double = 1e-5,
        mismatchFraction: Double = 0.0,
    ): Result {
        val r = run(source, kernel, size, args)
        for ((name, cpuValue) in r.cpu) {
            val gpuValue = r.gpu.getValue(name)
            val n = length(cpuValue)
            var bad = 0
            val examples = mutableListOf<String>()
            for (i in 0 until n) {
                val ok = when (cpuValue) {
                    is IntArray -> cpuValue[i] == (gpuValue as IntArray)[i]
                    is FloatArray -> {
                        val a = cpuValue[i]
                        val b = (gpuValue as FloatArray)[i]
                        (a.isNaN() && b.isNaN()) || a == b || abs(a - b) <= absTol + relTol * abs(a)
                    }
                    else -> error("unexpected $cpuValue")
                }
                if (!ok) {
                    bad++
                    if (examples.size < 8) {
                        examples += "[$i] cpu=${elementAt(cpuValue, i)} gpu=${elementAt(gpuValue, i)}"
                    }
                }
            }
            assertTrue(
                bad <= mismatchFraction * n,
                "'$name': $bad of $n elements differ between CPU and GPU (${adapter}):\n${examples.joinToString("\n")}",
            )
        }
        return r
    }

    private fun runDeno(deno: File, args: List<String>): Pair<Int, String> {
        val cmd = listOf(deno.path, "run", "--quiet", "--unstable-webgpu", "--allow-read", "--allow-write", "--allow-env", script.path) + args
        val p = ProcessBuilder(cmd).redirectErrorStream(true).start()
        val out = p.inputStream.bufferedReader().readText()
        if (!p.waitFor(120, TimeUnit.SECONDS)) {
            p.destroyForcibly()
            return -1 to "timeout: $out"
        }
        return p.exitValue() to out
    }

    private fun copy(v: Any): Any = when (v) {
        is IntArray -> v.copyOf()
        is FloatArray -> v.copyOf()
        else -> v
    }

    private fun length(v: Any) = when (v) {
        is IntArray -> v.size
        is FloatArray -> v.size
        else -> error("not an array: $v")
    }

    private fun elementAt(v: Any, i: Int): Any = when (v) {
        is IntArray -> v[i]
        is FloatArray -> v[i]
        else -> v
    }

    private fun bytes(v: Any): ByteArray {
        val buf = ByteBuffer.allocate(length(v) * 4).order(ByteOrder.LITTLE_ENDIAN)
        when (v) {
            is IntArray -> buf.asIntBuffer().put(v)
            is FloatArray -> buf.asFloatBuffer().put(v)
        }
        return buf.array()
    }
}
