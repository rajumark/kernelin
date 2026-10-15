@file:OptIn(ExperimentalWasmJsInterop::class, DelicateCoroutinesApi::class)

package io.github.rajumark.kernelin.samples

import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.kernelSource
import io.github.rajumark.kernelin.webgpu.GpuData
import io.github.rajumark.kernelin.webgpu.GpuKernel
import io.github.rajumark.kernelin.webgpu.WebGpu
import io.github.rajumark.kernelin.webgpu.toJsInt32Array
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.promise
import kotlin.js.Promise

// Functions the live demo page (docs/demo/index.html) calls. Both sides of the CPU vs GPU race are Kotlin:
// the GPU runs the WGSL Kernelin generated from mandelbrotView, the CPU runs mandelbrotView itself, compiled to Wasm.

private var kernel: GpuKernel? = null

/** Sets up WebGPU and the Mandelbrot kernel; resolves to a short GPU description. */
@JsExport
fun kernelinInit(): Promise<JsString> = GlobalScope.promise {
    val gpu = WebGpu.create()
    kernel = gpu.load(kernelSource(::mandelbrotView))
    gpu.description.toJsString()
}

/** The WGSL the compiler plugin embedded for mandelbrotView (available even without a GPU). */
@JsExport
fun kernelinWgsl(): String = kernelSource(::mandelbrotView).wgsl

/** Renders one frame on the GPU; resolves to `{ ms, data: Int32Array }`. */
@JsExport
fun gpuMandelbrot(width: Int, height: Int, maxIter: Int, centerX: Double, centerY: Double, scale: Double): Promise<JsAny> =
    GlobalScope.promise {
        val k = kernel ?: error("call kernelinInit() first")
        val size = width * height
        val result = k.run(
            size = size,
            scalars = mapOf(
                "width" to width, "height" to height, "maxIter" to maxIter,
                "centerX" to centerX.toFloat(), "centerY" to centerY.toFloat(), "scale" to scale.toFloat(),
            ),
            buffers = mapOf("out" to GpuData.empty(size)),
        )
        frame(result.millis, result.js("out"))
    }

/** Renders the same frame by calling the Kotlin kernel on the CPU (Wasm, 1 thread); returns `{ ms, data }`. */
@JsExport
fun cpuMandelbrot(width: Int, height: Int, maxIter: Int, centerX: Double, centerY: Double, scale: Double): JsAny {
    val size = width * height
    val out = IntArray(size)
    val cx = centerX.toFloat()
    val cy = centerY.toFloat()
    val s = scale.toFloat()
    val start = now()
    KernelinCpu.dispatch(size) { mandelbrotView(out, width, height, maxIter, cx, cy, s) }
    val millis = now() - start
    return frame(millis, out.toJsInt32Array())
}

private fun frame(ms: Double, data: JsAny): JsAny = js("({ ms, data })")

private fun now(): Double = js("performance.now()")
