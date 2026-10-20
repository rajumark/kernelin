package io.github.rajumark.kernelin.samples.android

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.widget.ScrollView
import android.widget.TextView
import io.github.rajumark.kernelin.webgpu.WebGpu
import java.io.File
import kotlinx.coroutines.runBlocking

/**
 * Runs the benchmark when opened and shows the results. It also writes them to
 * `/sdcard/Android/data/io.github.rajumark.kernelin.bench/files/results.json` and logs `KERNELIN_BENCH_DONE`.
 * Intent extra `scale` (Int, default 1) divides the sizes for a quick run. Intent extra `case` (Int) runs only the
 * case with that index (none when it's past the last one), so a script can give every case a fresh process and heap.
 */
class BenchActivity : Activity() {
    private lateinit var text: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        text = TextView(this).apply { setPadding(32, 48, 32, 32); textSize = 13f; setTextIsSelectable(true) }
        setContentView(ScrollView(this).apply { addView(text) })
        val scale = intent.getIntExtra("scale", 1).coerceAtLeast(1)
        val only = if (intent.hasExtra("case")) intent.getIntExtra("case", 0) else null
        Thread({ runBenchmark(scale, only) }, "Kernelin-bench").start()
    }

    private fun runBenchmark(scale: Int, only: Int?) = runBlocking {
        val out = File(getExternalFilesDir(null), "results.json").apply { delete() }
        val device = "${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE} (API ${Build.VERSION.SDK_INT}), " +
            "${Runtime.getRuntime().availableProcessors()} CPU cores"
        show("Kernelin benchmark\n$device\n")
        val gpu = try {
            WebGpu.create()
        } catch (e: Exception) {
            show("No GPU: ${e.message}\nTiming the CPU only.\n")
            null
        }
        gpu?.let { show("GPU: ${it.description}\n") }
        val results = mutableListOf<BenchResult>()
        try {
            val cases = benchCases(scale).let { all -> if (only == null) all else listOfNotNull(all.getOrNull(only)) }
            for (make in cases) {
                val r = runCase(make, gpu)
                results += r
                show(
                    "  GPU kernel ${ms(r.gpuKernelMs)}, GPU end-to-end ${ms(r.gpuEndToEndMs)}, " +
                        "CPU 1 core ${ms(r.cpuSingleMs)}, CPU all cores ${ms(r.cpuParallelMs)}" +
                        (r.matchPct?.let { ", match %.3f%%".format(it) } ?: "") + "\n",
                )
            }
        } catch (e: Throwable) {
            Log.e(TAG, "Benchmark failed", e)
            show("FAILED: $e")
        } finally {
            gpu?.close()
        }
        val json = "{\n  \"device\": \"$device\",\n  \"gpu\": ${gpu?.description?.let { "\"$it\"" } ?: "null"},\n" +
            "  \"scale\": $scale,\n  \"results\": [\n    ${results.joinToString(",\n    ") { it.json() }}\n  ]\n}\n"
        out.writeText(json)
        Log.i(TAG, "KERNELIN_BENCH_DONE ${out.absolutePath}")
        show("Done. Saved to ${out.absolutePath}")
    }

    /**
     * Creates and measures one case, then lets go of its arrays, so the next case's 64 MB arrays fit in a phone's app
     * heap. (scripts/bench-android-app.sh still gives every case its own process, so cases can't disturb each other.)
     */
    private suspend fun runCase(make: () -> BenchCase, gpu: WebGpu?): BenchResult {
        val case = make()
        show("${case.name} (${case.size} elements)...")
        return try {
            case.measure(gpu)
        } finally {
            case.release()
        }
    }


    private fun show(line: String) {
        Log.i(TAG, line)
        runOnUiThread { text.append(line + "\n") }
    }

    private fun ms(v: Double?) = if (v == null) "-" else "%.2f ms".format(v)

    private companion object {
        const val TAG = "KernelinBench"
    }
}
