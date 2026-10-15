package hello

import io.github.rajumark.kernelin.kernelSource
import io.github.rajumark.kernelin.webgpu.GpuData
import io.github.rajumark.kernelin.webgpu.WebGpu
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.GlobalScope
import kotlinx.coroutines.launch

/** In the browser: run the same kernel on the GPU and print the result to the console. */
@OptIn(DelicateCoroutinesApi::class)
fun main() {
    GlobalScope.launch {
        if (!WebGpu.isSupported) {
            println("WebGPU isn't available in this browser")
            return@launch
        }
        val gpu = WebGpu.create()
        val kernel = gpu.load(kernelSource(::squares))
        val result = kernel.run(size = 8, buffers = mapOf("out" to GpuData.empty(8)))
        println("GPU (${gpu.description}): ${result.ints("out").toList()}")
    }
}
