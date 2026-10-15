@file:OptIn(ExperimentalWasmJsInterop::class)

package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelSource
import kotlinx.coroutines.await
import kotlin.js.Promise

/**
 * Runs Kernelin kernels on the GPU from Kotlin/Wasm, through the browser's WebGPU API.
 *
 * ```
 * val gpu = WebGpu.create()
 * val kernel = gpu.load(kernelSource(::mandelbrot))
 * val result = kernel.run(size = w * h, scalars = mapOf("width" to w, ...), buffers = mapOf("out" to GpuData.empty(w * h)))
 * ```
 */
class WebGpu private constructor(internal val device: JsAny, val description: String) {
    companion object {
        /** True when this browser exposes WebGPU (`navigator.gpu`). */
        val isSupported: Boolean get() = gpuSupported()

        suspend fun create(): WebGpu {
            if (!isSupported) throw UnsupportedOperationException("WebGPU is not available in this browser")
            val device = requestDevice().await<JsAny>()
            return WebGpu(device, deviceDescription(device))
        }
    }

    /** Compiles the kernel's WGSL and builds its pipeline. Throws if the browser rejects the shader. */
    suspend fun load(source: KernelSource): GpuKernel {
        val module = createShaderModule(device, source.wgsl)
        val errors = compilationErrors(module).await<JsString>().toString()
        if (errors.isNotEmpty()) throw IllegalArgumentException("WGSL for ${source.name} was rejected: $errors")
        return GpuKernel(this, source, module)
    }
}

/** Input/output data for one kernel buffer. */
class GpuData private constructor(internal val js: JsAny?, internal val length: Int, internal val isInt: Boolean) {
    companion object {
        /** A zero-filled buffer of [length] elements, not uploaded (for outputs). */
        fun empty(length: Int) = GpuData(null, length, isInt = false)

        fun of(values: IntArray) = GpuData(values.toJsInt32Array(), values.size, isInt = true)

        fun of(values: FloatArray) = GpuData(values.toJsFloat32Array(), values.size, isInt = false)

        /** Wraps an existing JS `Int32Array` / `Float32Array` without copying. */
        fun ofJs(typedArray: JsAny) = GpuData(typedArray, typedArrayLength(typedArray), isInt = false)
    }
}

/** Results of one GPU run: every `read_write` buffer, as JS typed arrays. */
class GpuResult internal constructor(
    /** Milliseconds from upload until results were back in memory. */
    val millis: Double,
    private val outputs: Map<String, JsAny>,
) {
    /** The raw JS typed array (`Int32Array` or `Float32Array`); no copy. */
    fun js(name: String): JsAny = outputs[name] ?: error("No output buffer named '$name'")

    /** Copies an `i32` output into a Kotlin [IntArray]. */
    fun ints(name: String): IntArray = js(name).let { a -> IntArray(typedArrayLength(a)) { readInt(a, it) } }

    /** Copies an `f32` output into a Kotlin [FloatArray]. */
    fun floats(name: String): FloatArray = js(name).let { a -> FloatArray(typedArrayLength(a)) { readFloat(a, it).toFloat() } }
}

/** A compiled kernel. GPU buffers are reused between runs and only reallocated when sizes change. */
class GpuKernel internal constructor(private val gpu: WebGpu, val source: KernelSource, module: JsAny) {
    val manifest: KernelManifest = source.manifest
    private val device = gpu.device
    private val layout: JsAny
    private val pipeline: JsAny
    private val uniform: JsAny
    private val buffers = HashMap<String, JsAny>()
    private val staging = HashMap<String, JsAny>()
    private var bindGroup: JsAny? = null

    init {
        val entries = newArray()
        pushUniformEntry(entries, 0)
        for (b in manifest.buffers) pushStorageEntry(entries, b.binding, !b.isWritten)
        layout = createBindGroupLayout(device, entries)
        pipeline = createComputePipeline(device, module, layout, manifest.entryPoint)
        uniform = createBuffer(device, manifest.uniformBytes, USAGE_UNIFORM)
    }

    /**
     * Runs the kernel for ids `0 until size`. [scalars] holds every Int/Float parameter by name;
     * [buffers] holds every array parameter by name.
     */
    suspend fun run(size: Int, scalars: Map<String, Number> = emptyMap(), buffers: Map<String, GpuData>): GpuResult {
        val view = newDataView(manifest.uniformBytes)
        setUint32(view, 0, size)
        manifest.scalars.forEachIndexed { i, p ->
            val v = scalars[p.name] ?: throw IllegalArgumentException("Missing scalar parameter '${p.name}'")
            if (p.type == "f32") setFloat32(view, 4 + i * 4, v.toDouble()) else setInt32(view, 4 + i * 4, v.toInt())
        }
        writeBuffer(device, uniform, dataViewBuffer(view))

        val start = now()
        for (p in manifest.buffers) {
            val data = buffers[p.name] ?: throw IllegalArgumentException("Missing buffer parameter '${p.name}'")
            val buffer = ensure(this.buffers, p.name, data.length * 4, USAGE_STORAGE)
            data.js?.let { writeBuffer(device, buffer, it) }
        }
        val outputs = manifest.buffers.filter { it.isWritten }
        for (p in outputs) ensure(staging, p.name, bufferSize(this.buffers.getValue(p.name)), USAGE_STAGING)
        val group = bindGroup ?: run {
            val entries = newArray()
            pushBufferBinding(entries, 0, uniform)
            for (p in manifest.buffers) pushBufferBinding(entries, p.binding, this.buffers.getValue(p.name))
            createBindGroup(device, layout, entries).also { bindGroup = it }
        }

        val groups = (size + manifest.workgroupSize - 1) / manifest.workgroupSize
        val gx = minOf(groups, maxWorkgroupsPerDimension(device)).coerceAtLeast(1)
        val gy = (groups + gx - 1) / gx
        val copiesFrom = newArray()
        val copiesTo = newArray()
        for (p in outputs) {
            push(copiesFrom, this.buffers.getValue(p.name))
            push(copiesTo, staging.getValue(p.name))
        }
        submitDispatch(device, pipeline, group, gx, gy, copiesFrom, copiesTo)

        val results = LinkedHashMap<String, JsAny>()
        for (p in outputs) results[p.name] = readBack(staging.getValue(p.name), p.type == "i32").await<JsAny>()
        return GpuResult(now() - start, results)
    }

    private fun ensure(map: HashMap<String, JsAny>, name: String, bytes: Int, usage: Int): JsAny {
        val existing = map[name]
        if (existing != null && bufferSize(existing) == bytes) return existing
        existing?.let(::destroyBuffer)
        bindGroup = null
        return createBuffer(device, bytes, usage).also { map[name] = it }
    }
}

// ---------------------------------------------------------------------------------------------
// Thin WebGPU bindings. Each function is one JS expression; all the logic stays in Kotlin above.

private const val USAGE_UNIFORM = 0x40 or 0x08            // UNIFORM | COPY_DST
private const val USAGE_STORAGE = 0x80 or 0x04 or 0x08    // STORAGE | COPY_SRC | COPY_DST
private const val USAGE_STAGING = 0x01 or 0x08            // MAP_READ | COPY_DST

private fun gpuSupported(): Boolean = js("typeof navigator !== 'undefined' && !!navigator.gpu")

private fun requestDevice(): Promise<JsAny> = js("""(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice({ requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize } });
    const info = adapter.info || {};
    device.kernelinDescription = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'WebGPU';
    return device;
})()""")

private fun deviceDescription(device: JsAny): String = js("device.kernelinDescription")

private fun createShaderModule(device: JsAny, code: String): JsAny = js("device.createShaderModule({ code })")

private fun compilationErrors(module: JsAny): Promise<JsString> =
    js("module.getCompilationInfo().then(i => i.messages.filter(m => m.type === 'error').map(m => m.lineNum + ':' + m.message).join('; '))")

private fun newArray(): JsAny = js("[]")

private fun push(array: JsAny, value: JsAny): Unit = js("{ array.push(value); }")

private fun pushUniformEntry(entries: JsAny, binding: Int): Unit =
    js("{ entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }); }")

private fun pushStorageEntry(entries: JsAny, binding: Int, readOnly: Boolean): Unit =
    js("{ entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } }); }")

private fun pushBufferBinding(entries: JsAny, binding: Int, buffer: JsAny): Unit =
    js("{ entries.push({ binding, resource: { buffer } }); }")

private fun createBindGroupLayout(device: JsAny, entries: JsAny): JsAny = js("device.createBindGroupLayout({ entries })")

private fun createBindGroup(device: JsAny, layout: JsAny, entries: JsAny): JsAny = js("device.createBindGroup({ layout, entries })")

private fun createComputePipeline(device: JsAny, module: JsAny, layout: JsAny, entryPoint: String): JsAny =
    js("device.createComputePipeline({ layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }), compute: { module, entryPoint } })")

private fun createBuffer(device: JsAny, size: Int, usage: Int): JsAny = js("device.createBuffer({ size, usage })")

private fun bufferSize(buffer: JsAny): Int = js("buffer.size")

private fun destroyBuffer(buffer: JsAny): Unit = js("{ buffer.destroy(); }")

private fun writeBuffer(device: JsAny, buffer: JsAny, data: JsAny): Unit = js("{ device.queue.writeBuffer(buffer, 0, data); }")

private fun maxWorkgroupsPerDimension(device: JsAny): Int = js("device.limits.maxComputeWorkgroupsPerDimension")

private fun submitDispatch(device: JsAny, pipeline: JsAny, group: JsAny, gx: Int, gy: Int, from: JsAny, to: JsAny): Unit = js("""{
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    for (let i = 0; i < from.length; i++) encoder.copyBufferToBuffer(from[i], 0, to[i], 0, from[i].size);
    device.queue.submit([encoder.finish()]);
}""")

private fun readBack(staging: JsAny, isInt: Boolean): Promise<JsAny> = js("""staging.mapAsync(GPUMapMode.READ).then(() => {
    const copy = staging.getMappedRange().slice(0);
    staging.unmap();
    return isInt ? new Int32Array(copy) : new Float32Array(copy);
})""")

private fun newDataView(bytes: Int): JsAny = js("new DataView(new ArrayBuffer(bytes))")

private fun dataViewBuffer(view: JsAny): JsAny = js("view.buffer")

private fun setUint32(view: JsAny, offset: Int, value: Int): Unit = js("{ view.setUint32(offset, value >>> 0, true); }")

private fun setInt32(view: JsAny, offset: Int, value: Int): Unit = js("{ view.setInt32(offset, value, true); }")

private fun setFloat32(view: JsAny, offset: Int, value: Double): Unit = js("{ view.setFloat32(offset, value, true); }")

private fun now(): Double = js("performance.now()")

private fun typedArrayLength(array: JsAny): Int = js("array.length")

private fun readInt(array: JsAny, index: Int): Int = js("array[index]")

private fun readFloat(array: JsAny, index: Int): Double = js("array[index]")

private fun newInt32Array(length: Int): JsAny = js("new Int32Array(length)")

private fun newFloat32Array(length: Int): JsAny = js("new Float32Array(length)")

private fun writeInt(array: JsAny, index: Int, value: Int): Unit = js("{ array[index] = value; }")

private fun writeFloat(array: JsAny, index: Int, value: Double): Unit = js("{ array[index] = value; }")

/** Copies into a JS typed array (Wasm GC arrays live outside JS memory, so this is element by element). */
fun IntArray.toJsInt32Array(): JsAny = newInt32Array(size).also { a -> for (i in indices) writeInt(a, i, this[i]) }

fun FloatArray.toJsFloat32Array(): JsAny = newFloat32Array(size).also { a -> for (i in indices) writeFloat(a, i, this[i].toDouble()) }
