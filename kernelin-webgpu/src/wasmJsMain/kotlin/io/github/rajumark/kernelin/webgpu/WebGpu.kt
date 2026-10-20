@file:OptIn(ExperimentalWasmJsInterop::class, UnsafeWasmMemoryApi::class)

package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelLaunch
import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelSource
import kotlinx.coroutines.await
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlin.js.Promise
import kotlin.wasm.unsafe.Pointer
import kotlin.wasm.unsafe.UnsafeWasmMemoryApi
import kotlin.wasm.unsafe.wasmMemory
import kotlin.wasm.unsafe.withScopedMemoryAllocator

/**
 * Runs Kernelin kernels on the GPU from Kotlin/Wasm, through the browser's WebGPU API.
 *
 * ```
 * val gpu = WebGpu.create()
 * gpu.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }   // typed: checked by the compiler; ys now holds the result
 *
 * // or by name, with more control (data that stays on the GPU, what to read back):
 * val kernel = gpu.load(kernelSource(::mandelbrot))
 * val result = kernel.run(size = w * h, scalars = mapOf("width" to w, ...), buffers = mapOf("out" to GpuData.empty(w * h)))
 * ```
 *
 * The GPU can be lost at any time (driver reset, GPU unplugged, too many tabs): check [isLost] or set [onLost],
 * then create a new [WebGpu] and load the kernels again. [close] frees the device and every kernel loaded from it.
 */
public class WebGpu private constructor(internal val device: JsAny, public val description: String) : AutoCloseable {
    /** Entry point: check [isSupported], then [create]. */
    public companion object {
        /** True when this browser exposes WebGPU (`navigator.gpu`). */
        public val isSupported: Boolean get() = gpuSupported()

        /** Requests a high-performance GPU adapter and device. Throws if WebGPU is unavailable. */
        public suspend fun create(): WebGpu {
            if (!isSupported) throw UnsupportedOperationException("WebGPU is not available in this browser")
            val device = requestDevice().await<JsAny>()
            return WebGpu(device, deviceDescription(device))
        }
    }

    private val kernels = mutableListOf<GpuKernel>()

    /** Kernels loaded by [dispatch], by kernel name, so each is compiled once. */
    private val launched = HashMap<String, GpuKernel>()
    private val launchedLock = Mutex()

    /** True after [close]. */
    public var isClosed: Boolean = false
        private set

    /** Why the device was lost (for example `unknown: GPU process crashed`), or null while it works. */
    public var lostReason: String? = null
        private set

    /** True when the browser took the GPU device away. Every later [load] and [GpuKernel.run] throws [GpuException]. */
    public val isLost: Boolean get() = lostReason != null

    /** Called once with [lostReason] when the device is lost (not after [close]). */
    public var onLost: ((reason: String) -> Unit)? = null

    /**
     * Called with WebGPU errors that happen outside a [load] or [GpuKernel.run] (those throw [GpuException] instead).
     * The browser also logs them to the console.
     */
    public var onUncapturedError: ((message: String) -> Unit)? = null

    init {
        watchDevice(
            device,
            onLost = { reason ->
                lostReason = reason
                if (!isClosed) onLost?.invoke(reason)
            },
            onError = { message -> onUncapturedError?.invoke(message) },
        )
    }

    /** Compiles the kernel's WGSL and builds its pipeline. Throws if the browser rejects the shader. */
    public suspend fun load(source: KernelSource): GpuKernel {
        checkUsable()
        val label = source.name
        val module = createShaderModule(device, source.wgsl, label)
        val compileErrors = compilationErrors(module).await<JsString>().toString()
        if (compileErrors.isNotEmpty()) throw IllegalArgumentException("WGSL for ${source.name} was rejected: $compileErrors")

        val entries = newArray()
        pushUniformEntry(entries, 0)
        for (b in source.manifest.buffers) pushStorageEntry(entries, b.binding, !b.isWritten)
        val layout: JsAny
        val pipeline: Promise<JsAny>
        val uniform: JsAny
        val scopeErrors: Promise<JsString>
        pushErrorScopes(device)
        try {
            layout = createBindGroupLayout(device, entries, "$label: bind group layout")
            pipeline = createComputePipelineAsync(device, module, layout, source.manifest.entryPoint, "$label: pipeline")
            uniform = createBuffer(device, source.manifest.uniformBytes, USAGE_UNIFORM, "$label: scalars")
        } finally {
            scopeErrors = popErrorScopes(device)
        }
        val errors = scopeErrors.await<JsString>().toString()
        if (errors.isNotEmpty()) throw GpuException("WebGPU rejected the pipeline for ${source.name}: $errors")
        val ready = try {
            pipeline.await<JsAny>()
        } catch (e: Throwable) {
            throw GpuException("WebGPU could not build the pipeline for ${source.name}: ${e.message}", e)
        }
        return GpuKernel(this, source, layout, ready, uniform).also { kernels += it }
    }

    /**
     * Runs one kernel call for ids `0 until size`, with typed arguments:
     *
     * ```
     * gpu.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }
     * ```
     *
     * The block is a normal call to a `@Kernel` function, so the compiler checks parameter names and types, and
     * the IDE completes them. It does not run on the CPU: the Kernelin compiler plugin records the call (see
     * [KernelLaunch]), and the kernel runs on the GPU instead. Like `KernelinCpu.dispatch(n) { saxpy(...) }`, the
     * arrays the kernel writes hold the results afterwards; the returned [GpuResult] has them too.
     *
     * The kernel is loaded on the first call and reused after that. Throws like [load] and [GpuKernel.run].
     */
    public suspend fun dispatch(size: Int, kernel: KernelLaunch.() -> Unit): GpuResult =
        dispatch(LaunchShape.linear(size), KernelLaunch.capture(kernel))

    /**
     * Like [dispatch], for a `width x height` grid of threads: inside the kernel, `globalIdX()` is the column
     * (`0 until width`), `globalIdY()` the row and `globalId()` the row-major index `x + y * width`.
     *
     * ```
     * gpu.dispatch(width, height) { blur(src = pixels, dst = out, width = width) }
     * ```
     */
    public suspend fun dispatch(width: Int, height: Int, kernel: KernelLaunch.() -> Unit): GpuResult =
        dispatch(LaunchShape.grid(width, height, 1), KernelLaunch.capture(kernel))

    /** Like the 2D [dispatch], for a `width x height x depth` grid; `globalIdZ()` is the layer. */
    public suspend fun dispatch(width: Int, height: Int, depth: Int, kernel: KernelLaunch.() -> Unit): GpuResult =
        dispatch(LaunchShape.grid(width, height, depth), KernelLaunch.capture(kernel))

    internal suspend fun dispatch(shape: LaunchShape, launch: KernelLaunch): GpuResult {
        val loaded = launchedLock.withLock {
            launched[launch.source.name]?.takeUnless { it.isClosed } ?: load(launch.source).also { launched[launch.source.name] = it }
        }
        val arrays = launch.arrays
        val result = loaded.run(shape, launch.scalars, arrays.mapValues { (_, array) -> GpuData.ofArray(array) }, readBack = null)
        for (p in loaded.manifest.buffers) {
            if (!p.isWritten) continue
            when (val array = arrays.getValue(p.name)) {
                is IntArray -> result.ints(p.name).copyInto(array)
                is FloatArray -> result.floats(p.name).copyInto(array)
            }
        }
        return result
    }

    /** Frees every kernel loaded from this device, then the device itself. Safe to call more than once. */
    override fun close() {
        if (isClosed) return
        isClosed = true
        kernels.toList().forEach { it.close() }
        destroyDevice(device)
    }

    internal fun forget(kernel: GpuKernel) {
        kernels -= kernel
        if (launched[kernel.source.name] === kernel) launched -= kernel.source.name
    }

    internal fun checkUsable() {
        check(!isClosed) { "This WebGpu was closed; create a new one with WebGpu.create()" }
        lostReason?.let { throw GpuException("The GPU device was lost ($it). Create a new WebGpu and load the kernels again.") }
    }
}

/** WebGPU reported an error: the device was lost, ran out of memory, or rejected a pipeline or a command. */
public class GpuException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * Input/output data for one kernel buffer. [type] is the WGSL element type (`i32` or `f32`), or null when any
 * buffer type fits (an empty buffer, or a JS typed array of another kind).
 */
public class GpuData private constructor(
    internal val js: JsAny?,
    internal val length: Int,
    internal val type: String?,
    internal val gpuBuffer: GpuBuffer? = null,
) {
    /** Ways to create buffer data. */
    public companion object {
        /** A zero-filled buffer of [length] elements, not uploaded (for outputs). */
        public fun empty(length: Int): GpuData = GpuData(null, length, type = null)

        /** Copies an IntArray for upload (an `i32` buffer). */
        public fun of(values: IntArray): GpuData = GpuData(values.toJsInt32Array(), values.size, type = "i32")

        /** Copies a FloatArray for upload (an `f32` buffer). */
        public fun of(values: FloatArray): GpuData = GpuData(values.toJsFloat32Array(), values.size, type = "f32")

        /** Wraps an existing JS `Int32Array` / `Float32Array` without copying. */
        public fun ofJs(typedArray: JsAny): GpuData =
            GpuData(typedArray, typedArrayLength(typedArray), typedArrayType(typedArray).takeIf { it.isNotEmpty() })

        /**
         * Uses a buffer that is already on the GPU (from [GpuResult.buffer]): it is copied on the GPU, so the data
         * never comes back to the CPU. The buffer must come from a kernel of the same [WebGpu].
         */
        public fun of(buffer: GpuBuffer): GpuData = GpuData(null, buffer.length, buffer.type, buffer)

        /** An IntArray or FloatArray argument of a [KernelLaunch]. */
        internal fun ofArray(array: Any): GpuData = when (array) {
            is IntArray -> of(array)
            is FloatArray -> of(array)
            else -> throw IllegalArgumentException("Kernel arrays must be IntArray or FloatArray, not ${array::class.simpleName}")
        }
    }
}

/**
 * A kernel buffer as it was at the end of one run, still on the GPU. Pass it to another run with
 * [GpuData.of] to chain kernels without a round trip through the CPU.
 *
 * It stays valid until its kernel runs again or is closed; after that, using it throws [IllegalStateException].
 */
public class GpuBuffer internal constructor(
    private val kernel: GpuKernel,
    /** The kernel parameter this buffer belongs to. */
    public val name: String,
    /** Number of elements. */
    public val length: Int,
    internal val type: String,
    private val generation: Int,
) {
    /** False once the kernel ran again or was closed. */
    public val isValid: Boolean get() = !kernel.isClosed && kernel.generation == generation

    internal val gpu: WebGpu get() = kernel.gpu

    internal fun jsBuffer(): JsAny {
        check(!kernel.isClosed) { "GpuBuffer '$name' of ${kernel.source.name} can't be used: its kernel was closed" }
        check(kernel.generation == generation) {
            "GpuBuffer '$name' of ${kernel.source.name} is out of date: the kernel ran again, so the buffer holds " +
                "newer data. Use the buffer from the GpuResult of the latest run."
        }
        return kernel.gpuBuffer(name)
    }
}

/** Results of one GPU run: the `read_write` buffers that were read back, as JS typed arrays. */
public class GpuResult internal constructor(
    /** Milliseconds from upload until results were back in memory. */
    public val millis: Double,
    private val outputs: Map<String, JsAny>,
    private val buffers: Map<String, GpuBuffer>,
) {
    /** The raw JS typed array (`Int32Array` or `Float32Array`); no copy. */
    public fun js(name: String): JsAny = outputs[name] ?: throw IllegalArgumentException(
        if (name in buffers) {
            "Buffer '$name' was not read back (see readBack in GpuKernel.run); use buffer(\"$name\") to keep it on the GPU"
        } else {
            "No output buffer named '$name'"
        },
    )

    /** Copies an `i32` output into a Kotlin [IntArray] (one bulk copy). */
    public fun ints(name: String): IntArray = js(name).let { a -> copyFromJs(a, typedArrayLength(a)) { p, n -> IntArray(n) { (p + it * 4).loadInt() } } }

    /** Copies an `f32` output into a Kotlin [FloatArray] (one bulk copy). */
    public fun floats(name: String): FloatArray =
        js(name).let { a -> copyFromJs(a, typedArrayLength(a)) { p, n -> FloatArray(n) { Float.fromBits((p + it * 4).loadInt()) } } }

    /** The kernel's GPU buffer [name] after this run, to pass to another run with [GpuData.of] without a CPU copy. */
    public fun buffer(name: String): GpuBuffer = buffers[name] ?: throw IllegalArgumentException("No buffer parameter named '$name'")
}

/**
 * A compiled kernel. GPU buffers are reused between runs and only reallocated when sizes change.
 * Runs of one kernel never overlap: a second [run] waits until the first is done. [close] frees its GPU buffers.
 */
public class GpuKernel internal constructor(
    internal val gpu: WebGpu,
    public val source: KernelSource,
    private val layout: JsAny,
    private val pipeline: JsAny,
    private val uniform: JsAny,
) : AutoCloseable {
    /** The kernel's parameters and workgroup size. */
    public val manifest: KernelManifest = source.manifest
    private val device = gpu.device
    private val label = source.name
    private val buffers = HashMap<String, JsAny>()
    private val staging = HashMap<String, JsAny>()
    private var bindGroup: JsAny? = null
    private val mutex = Mutex()

    /** Counts runs, so a [GpuBuffer] from an older run can tell that its data was replaced. */
    internal var generation: Int = 0
        private set

    /** True after [close] (or [WebGpu.close]). */
    public var isClosed: Boolean = false
        private set

    /**
     * Runs the kernel for ids `0 until size`. [scalars] holds every Int/Float/Boolean parameter by name;
     * [buffers] holds every array parameter by name.
     *
     * [readBack] names the `read_write` buffers to copy back to the CPU; null (the default) means all of them.
     * Buffers that are not read back stay on the GPU: get them with [GpuResult.buffer].
     *
     * Throws [IllegalArgumentException] for wrong arguments, [IllegalStateException] after [close], and
     * [GpuException] when WebGPU reports an error or the device is lost.
     */
    public suspend fun run(
        size: Int,
        scalars: Map<String, Any> = emptyMap(),
        buffers: Map<String, GpuData>,
        readBack: Set<String>? = null,
    ): GpuResult = run(LaunchShape.linear(size), scalars, buffers, readBack)

    /**
     * Like [run], for a `width x height` grid of threads: inside the kernel, `globalIdX()` is the column
     * (`0 until width`), `globalIdY()` the row and `globalId()` the row-major index `x + y * width`.
     */
    public suspend fun run(
        width: Int,
        height: Int,
        scalars: Map<String, Any> = emptyMap(),
        buffers: Map<String, GpuData>,
        readBack: Set<String>? = null,
    ): GpuResult = run(LaunchShape.grid(width, height, 1), scalars, buffers, readBack)

    /** Like the 2D [run], for a `width x height x depth` grid; `globalIdZ()` is the layer. */
    public suspend fun run(
        width: Int,
        height: Int,
        depth: Int,
        scalars: Map<String, Any> = emptyMap(),
        buffers: Map<String, GpuData>,
        readBack: Set<String>? = null,
    ): GpuResult = run(LaunchShape.grid(width, height, depth), scalars, buffers, readBack)

    internal suspend fun run(
        shape: LaunchShape,
        scalars: Map<String, Any>,
        buffers: Map<String, GpuData>,
        readBack: Set<String>?,
    ): GpuResult = mutex.withLock {
        val size = shape.size
        checkOpen()
        checkRunArguments(
            source,
            scalars,
            buffers.mapValues { (_, data) -> BufferArgument(data.type, data.gpuBuffer == null || data.gpuBuffer.gpu === gpu) },
            readBack,
        )
        val bytes = uniformBytes(source, shape, scalars)
        val view = newDataView(bytes.size)
        for (offset in bytes.indices step 4) {
            var bits = 0
            for (i in 0 until 4) bits = bits or ((bytes[offset + i].toInt() and 0xff) shl (8 * i))
            setInt32(view, offset, bits)
        }
        // Buffers from earlier runs are read now, before this run replaces or reallocates them.
        val sources = manifest.buffers.mapNotNull { p -> buffers.getValue(p.name).gpuBuffer?.let { p.name to it.jsBuffer() } }.toMap()
        generation++
        val outputs = manifest.buffers.filter { it.isWritten && (readBack == null || it.name in readBack) }

        val start = now()
        val scopeErrors: Promise<JsString>
        pushErrorScopes(device)
        try {
            submit(size, view, buffers, sources, outputs)
        } finally {
            scopeErrors = popErrorScopes(device)
        }

        val stagings = newArray()
        for (p in outputs) push(stagings, staging.getValue(p.name))
        val reads = readBackAll(stagings, outputs.map { it.type == "i32" }.toJsBooleanArray())
        val errors = scopeErrors.await<JsString>().toString()
        if (errors.isNotEmpty()) {
            settle(reads).await<JsAny?>()
            throw GpuException("WebGPU rejected the run of ${source.name}: $errors")
        }
        val arrays = try {
            reads.await<JsAny>()
        } catch (e: Throwable) {
            if (isClosed || gpu.isClosed) throw IllegalStateException("${source.name} was closed during run()", e)
            gpu.lostReason?.let { throw GpuException("The GPU device was lost during run() of ${source.name} ($it)", e) }
            throw GpuException("Reading the results of ${source.name} failed: ${e.message}", e)
        }
        val results = LinkedHashMap<String, JsAny>()
        outputs.forEachIndexed { i, p -> results[p.name] = arrayAt(arrays, i) }
        val handles = manifest.buffers.associate { p ->
            p.name to GpuBuffer(this, p.name, bufferSize(this.buffers.getValue(p.name)) / 4, p.type, generation)
        }
        GpuResult(now() - start, results, handles)
    }

    /** Everything a run sends to the GPU, in one go (no suspension, so the error scopes around it see only this run). */
    private fun submit(size: Int, view: JsAny, inputs: Map<String, GpuData>, sources: Map<String, JsAny>, outputs: List<KernelManifest.Param>) {
        writeBuffer(device, uniform, dataViewBuffer(view))
        val retired = newArray()
        // Buffers reused from an earlier run still hold its data, so empty() ones are cleared on the GPU.
        val clears = newArray()
        val copyInFrom = newArray()
        val copyInTo = newArray()
        for (p in manifest.buffers) {
            val data = inputs.getValue(p.name)
            val buffer = ensure(this.buffers, p.name, data.length * 4, USAGE_STORAGE, "$label: ${p.name}", retired)
            val from = sources[p.name]
            when {
                from != null -> if (from != buffer) {
                    push(copyInFrom, from)
                    push(copyInTo, buffer)
                }
                data.js != null -> writeBuffer(device, buffer, data.js)
                else -> push(clears, buffer)
            }
        }
        for (p in outputs) {
            ensure(staging, p.name, bufferSize(this.buffers.getValue(p.name)), USAGE_STAGING, "$label: ${p.name} (read back)", retired)
        }
        val group = bindGroup ?: run {
            val entries = newArray()
            pushBufferBinding(entries, 0, uniform)
            for (p in manifest.buffers) pushBufferBinding(entries, p.binding, this.buffers.getValue(p.name))
            createBindGroup(device, layout, entries, "$label: bind group").also { bindGroup = it }
        }

        val (gx, gy) = workgroupGrid(size, manifest.workgroupSize, maxWorkgroupsPerDimension(device))
        val copiesFrom = newArray()
        val copiesTo = newArray()
        for (p in outputs) {
            push(copiesFrom, this.buffers.getValue(p.name))
            push(copiesTo, staging.getValue(p.name))
        }
        submitDispatch(device, pipeline, group, gx, gy, clears, copyInFrom, copyInTo, copiesFrom, copiesTo, "$label: run")
        // Destroyed after the submit: WebGPU frees them once the GPU work that uses them is done.
        destroyAll(retired)
    }

    /** The 0.2.0 signature, kept so code compiled against it still links. */
    @Deprecated("Kept for binary compatibility", level = DeprecationLevel.HIDDEN)
    public suspend fun run(size: Int, scalars: Map<String, Any> = emptyMap(), buffers: Map<String, GpuData>): GpuResult =
        run(size, scalars, buffers, readBack = null)

    /** Frees this kernel's GPU buffers. Later [run] calls throw [IllegalStateException]. Safe to call more than once. */
    override fun close() {
        if (isClosed) return
        isClosed = true
        for (b in buffers.values) destroyBuffer(b)
        for (b in staging.values) destroyBuffer(b)
        destroyBuffer(uniform)
        buffers.clear()
        staging.clear()
        bindGroup = null
        gpu.forget(this)
    }

    internal fun gpuBuffer(name: String): JsAny = buffers.getValue(name)

    private fun checkOpen() {
        check(!isClosed) { "GpuKernel ${source.name} was closed; load it again with WebGpu.load()" }
        gpu.checkUsable()
    }

    private fun ensure(map: HashMap<String, JsAny>, name: String, bytes: Int, usage: Int, label: String, retired: JsAny): JsAny {
        val existing = map[name]
        if (existing != null && bufferSize(existing) == bytes) return existing
        existing?.let { push(retired, it) }
        bindGroup = null
        return createBuffer(device, bytes, usage, label).also { map[name] = it }
    }
}

/** Copies into a JS `Int32Array` in one bulk copy (through Wasm linear memory). */
public fun IntArray.toJsInt32Array(): JsAny = withScopedMemoryAllocator { allocator ->
    val p = allocator.allocate(size * 4)
    for (i in indices) (p + i * 4).storeInt(this[i])
    int32ArrayFromMemory(wasmMemory, p.address.toInt(), size)
}

/** Copies into a JS `Float32Array` in one bulk copy (through Wasm linear memory). */
public fun FloatArray.toJsFloat32Array(): JsAny = withScopedMemoryAllocator { allocator ->
    val p = allocator.allocate(size * 4)
    for (i in indices) (p + i * 4).storeInt(this[i].toRawBits())
    float32ArrayFromMemory(wasmMemory, p.address.toInt(), size)
}

/** Kotlin arrays live in the Wasm GC heap, which JS can't see: the bytes go through linear memory in one copy. */
private inline fun <T> copyFromJs(typedArray: JsAny, length: Int, read: (Pointer, Int) -> T): T = withScopedMemoryAllocator { allocator ->
    val p = allocator.allocate(length * 4)
    copyToMemory(wasmMemory, typedArray, p.address.toInt())
    read(p, length)
}

private fun List<Boolean>.toJsBooleanArray(): JsAny = newArray().also { a -> forEach { pushBoolean(a, it) } }

// ---------------------------------------------------------------------------------------------
// Thin WebGPU bindings. Each function is one JS expression; all the logic stays in Kotlin above.

private const val USAGE_UNIFORM = 0x40 or 0x08            // UNIFORM | COPY_DST
private const val USAGE_STORAGE = 0x80 or 0x04 or 0x08    // STORAGE | COPY_SRC | COPY_DST
private const val USAGE_STAGING = 0x01 or 0x08            // MAP_READ | COPY_DST

private fun gpuSupported(): Boolean = js("typeof navigator !== 'undefined' && !!navigator.gpu")

private fun requestDevice(): Promise<JsAny> = js("""(async () => {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice({ label: 'Kernelin', requiredLimits: {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize } });
    const info = adapter.info || {};
    device.kernelinDescription = [info.vendor, info.architecture].filter(Boolean).join(' ') || 'WebGPU';
    return device;
})()""")

private fun deviceDescription(device: JsAny): String = js("device.kernelinDescription")

private fun watchDevice(device: JsAny, onLost: (String) -> Unit, onError: (String) -> Unit): Unit = js("""{
    device.lost.then(info => onLost(info.reason + ': ' + info.message));
    device.addEventListener('uncapturederror', e => onError(e.error.message));
}""")

private fun destroyDevice(device: JsAny): Unit = js("{ device.destroy(); }")

private fun pushErrorScopes(device: JsAny): Unit = js("{ device.pushErrorScope('out-of-memory'); device.pushErrorScope('validation'); }")

/** Pops what [pushErrorScopes] pushed; resolves to the errors joined with "; ", or "" when there were none. */
private fun popErrorScopes(device: JsAny): Promise<JsString> = js("""Promise.all([device.popErrorScope(), device.popErrorScope()])
    .then(([validation, memory]) => [validation && 'validation: ' + validation.message, memory && 'out of memory: ' + memory.message]
        .filter(Boolean).join('; '))""")

private fun createShaderModule(device: JsAny, code: String, label: String): JsAny = js("device.createShaderModule({ code, label })")

private fun compilationErrors(module: JsAny): Promise<JsString> =
    js("module.getCompilationInfo().then(i => i.messages.filter(m => m.type === 'error').map(m => m.lineNum + ':' + m.message).join('; '))")

private fun newArray(): JsAny = js("[]")

private fun push(array: JsAny, value: JsAny): Unit = js("{ array.push(value); }")

private fun pushBoolean(array: JsAny, value: Boolean): Unit = js("{ array.push(value); }")

private fun arrayAt(array: JsAny, index: Int): JsAny = js("array[index]")

private fun pushUniformEntry(entries: JsAny, binding: Int): Unit =
    js("{ entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }); }")

private fun pushStorageEntry(entries: JsAny, binding: Int, readOnly: Boolean): Unit =
    js("{ entries.push({ binding, visibility: GPUShaderStage.COMPUTE, buffer: { type: readOnly ? 'read-only-storage' : 'storage' } }); }")

private fun pushBufferBinding(entries: JsAny, binding: Int, buffer: JsAny): Unit =
    js("{ entries.push({ binding, resource: { buffer } }); }")

private fun createBindGroupLayout(device: JsAny, entries: JsAny, label: String): JsAny =
    js("device.createBindGroupLayout({ entries, label })")

private fun createBindGroup(device: JsAny, layout: JsAny, entries: JsAny, label: String): JsAny =
    js("device.createBindGroup({ layout, entries, label })")

private fun createComputePipelineAsync(device: JsAny, module: JsAny, layout: JsAny, entryPoint: String, label: String): Promise<JsAny> =
    js("""device.createComputePipelineAsync({ label, compute: { module, entryPoint },
        layout: device.createPipelineLayout({ bindGroupLayouts: [layout], label: label + ' layout' }) })""")

private fun createBuffer(device: JsAny, size: Int, usage: Int, label: String): JsAny = js("device.createBuffer({ size, usage, label })")

private fun bufferSize(buffer: JsAny): Int = js("buffer.size")

private fun destroyBuffer(buffer: JsAny): Unit = js("{ buffer.destroy(); }")

private fun destroyAll(buffers: JsAny): Unit = js("{ for (const b of buffers) b.destroy(); }")

private fun writeBuffer(device: JsAny, buffer: JsAny, data: JsAny): Unit = js("{ device.queue.writeBuffer(buffer, 0, data); }")

private fun maxWorkgroupsPerDimension(device: JsAny): Int = js("device.limits.maxComputeWorkgroupsPerDimension")

private fun submitDispatch(
    device: JsAny, pipeline: JsAny, group: JsAny, gx: Int, gy: Int, clears: JsAny,
    inFrom: JsAny, inTo: JsAny, outFrom: JsAny, outTo: JsAny, label: String,
): Unit = js("""{
    const encoder = device.createCommandEncoder({ label });
    for (let i = 0; i < inFrom.length; i++) encoder.copyBufferToBuffer(inFrom[i], 0, inTo[i], 0, inTo[i].size);
    for (const buffer of clears) encoder.clearBuffer(buffer);
    const pass = encoder.beginComputePass({ label });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    for (let i = 0; i < outFrom.length; i++) encoder.copyBufferToBuffer(outFrom[i], 0, outTo[i], 0, outFrom[i].size);
    device.queue.submit([encoder.finish({ label })]);
}""")

/** Maps every staging buffer and copies it out; resolves to an array of Int32Array / Float32Array. */
private fun readBackAll(stagings: JsAny, isInt: JsAny): Promise<JsAny> = js("""Promise.all(stagings.map((staging, i) =>
    staging.mapAsync(GPUMapMode.READ).then(() => {
        const copy = staging.getMappedRange().slice(0);
        staging.unmap();
        return isInt[i] ? new Int32Array(copy) : new Float32Array(copy);
    })))""")

/** Waits until [promise] is done, whether it worked or failed. */
private fun settle(promise: Promise<JsAny>): Promise<JsAny?> = js("promise.then(() => null, () => null)")

private fun newDataView(bytes: Int): JsAny = js("new DataView(new ArrayBuffer(bytes))")

private fun dataViewBuffer(view: JsAny): JsAny = js("view.buffer")

private fun setInt32(view: JsAny, offset: Int, value: Int): Unit = js("{ view.setInt32(offset, value, true); }")


private fun now(): Double = js("performance.now()")

private fun typedArrayLength(array: JsAny): Int = js("array.length")

private fun typedArrayType(array: JsAny): String =
    js("array instanceof Int32Array ? 'i32' : array instanceof Float32Array ? 'f32' : ''")

private fun int32ArrayFromMemory(memory: JsAny, address: Int, length: Int): JsAny =
    js("new Int32Array(memory.buffer.slice(address, address + length * 4))")

private fun float32ArrayFromMemory(memory: JsAny, address: Int, length: Int): JsAny =
    js("new Float32Array(memory.buffer.slice(address, address + length * 4))")

private fun copyToMemory(memory: JsAny, array: JsAny, address: Int): Unit =
    js("{ new Uint8Array(memory.buffer, address, array.byteLength).set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength)); }")

internal actual suspend fun openGpu(): GpuLauncher {
    val gpu = WebGpu.create()
    return object : GpuLauncher {
        override val description: String get() = gpu.description

        override suspend fun launch(shape: LaunchShape, launch: KernelLaunch): String? = try {
            gpu.dispatch(shape, launch)
            null
        } catch (e: GpuException) {
            gpu.lostReason ?: throw e
        }

        override fun close() = gpu.close()
    }
}
