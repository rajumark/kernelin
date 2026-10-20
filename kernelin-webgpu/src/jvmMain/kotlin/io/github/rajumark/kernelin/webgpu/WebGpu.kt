package io.github.rajumark.kernelin.webgpu

import com.github.xpenatan.jParser.api.NativeObject
import com.github.xpenatan.webgpu.JWebGPUBackend
import com.github.xpenatan.webgpu.JWebGPULoader
import com.github.xpenatan.webgpu.WGPU
import com.github.xpenatan.webgpu.WGPUAdapter
import com.github.xpenatan.webgpu.WGPUAdapterInfo
import com.github.xpenatan.webgpu.WGPUAdapterType
import com.github.xpenatan.webgpu.WGPUBindGroup
import com.github.xpenatan.webgpu.WGPUBindGroupDescriptor
import com.github.xpenatan.webgpu.WGPUBindGroupEntry
import com.github.xpenatan.webgpu.WGPUBindGroupLayout
import com.github.xpenatan.webgpu.WGPUBindGroupLayoutDescriptor
import com.github.xpenatan.webgpu.WGPUBindGroupLayoutEntry
import com.github.xpenatan.webgpu.WGPUBuffer
import com.github.xpenatan.webgpu.WGPUBufferBindingLayout
import com.github.xpenatan.webgpu.WGPUBufferBindingType
import com.github.xpenatan.webgpu.WGPUBufferDescriptor
import com.github.xpenatan.webgpu.WGPUBufferMapCallback
import com.github.xpenatan.webgpu.WGPUBufferUsage
import com.github.xpenatan.webgpu.WGPUCallbackMode
import com.github.xpenatan.webgpu.WGPUCommandBuffer
import com.github.xpenatan.webgpu.WGPUCommandBufferDescriptor
import com.github.xpenatan.webgpu.WGPUCommandEncoder
import com.github.xpenatan.webgpu.WGPUCommandEncoderDescriptor
import com.github.xpenatan.webgpu.WGPUComputePassDescriptor
import com.github.xpenatan.webgpu.WGPUComputePassEncoder
import com.github.xpenatan.webgpu.WGPUComputePipeline
import com.github.xpenatan.webgpu.WGPUComputePipelineDescriptor
import com.github.xpenatan.webgpu.WGPUDevice
import com.github.xpenatan.webgpu.WGPUDeviceDescriptor
import com.github.xpenatan.webgpu.WGPUErrorType
import com.github.xpenatan.webgpu.WGPUInstance
import com.github.xpenatan.webgpu.WGPULimits
import com.github.xpenatan.webgpu.WGPUMapAsyncStatus
import com.github.xpenatan.webgpu.WGPUMapMode
import com.github.xpenatan.webgpu.WGPUPipelineLayout
import com.github.xpenatan.webgpu.WGPUPipelineLayoutDescriptor
import com.github.xpenatan.webgpu.WGPUPowerPreference
import com.github.xpenatan.webgpu.WGPUQueue
import com.github.xpenatan.webgpu.WGPURequestAdapterCallback
import com.github.xpenatan.webgpu.WGPURequestAdapterOptions
import com.github.xpenatan.webgpu.WGPURequestAdapterStatus
import com.github.xpenatan.webgpu.WGPURequestDeviceCallback
import com.github.xpenatan.webgpu.WGPURequestDeviceStatus
import com.github.xpenatan.webgpu.WGPUShaderModule
import com.github.xpenatan.webgpu.WGPUShaderModuleDescriptor
import com.github.xpenatan.webgpu.WGPUShaderSourceWGSL
import com.github.xpenatan.webgpu.WGPUShaderStage
import com.github.xpenatan.webgpu.WGPUSType
import com.github.xpenatan.webgpu.WGPUUncapturedErrorCallback
import com.github.xpenatan.webgpu.WGPUVectorBindGroupEntry
import com.github.xpenatan.webgpu.WGPUVectorBindGroupLayout
import com.github.xpenatan.webgpu.WGPUVectorBindGroupLayoutEntry
import io.github.rajumark.kernelin.KernelLaunch
import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelSource
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executors
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.ExecutorCoroutineDispatcher
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import kotlinx.coroutines.yield

/**
 * Runs Kernelin kernels on the GPU of a desktop computer (macOS, Linux, Windows) from a plain JVM app, natively
 * through [jWebGPU](https://github.com/xpenatan/jWebGPU) and wgpu (Metal, Vulkan or Direct3D 12). No browser: the
 * same API as the Kotlin/Wasm and Android hosts.
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
 * Every GPU call runs on one background thread owned by this [WebGpu], so [load] and [GpuKernel.run] can be called
 * from any coroutine and any thread. The GPU can be lost (driver reset, GPU unplugged): check [isLost] or set
 * [onLost], then create a new [WebGpu] and load the kernels again. [close] frees the device and every kernel
 * loaded from it.
 */
public class WebGpu private constructor(
    private val instance: WGPUInstance,
    private val adapter: WGPUAdapter,
    internal val device: WGPUDevice,
    internal val queue: WGPUQueue,
    internal val maxPerDimension: Int,
    private val thread: ExecutorCoroutineDispatcher,
    private val gpuThread: Thread,
    /** The GPU, for example `Apple M5 (Metal)`. */
    public val description: String,
    private val state: DeviceState,
) : AutoCloseable {
    /** Entry point: check [isSupported], then [create]. */
    public companion object {
        /**
         * True on the desktops jWebGPU ships a native library for: macOS (Apple silicon and Intel), Linux x64 and
         * Windows x64. [create] can still fail when the computer has no GPU driver for Metal, Vulkan or Direct3D 12.
         */
        public val isSupported: Boolean get() = nativeLibraryFor(System.getProperty("os.name"), System.getProperty("os.arch")) != null

        /** Requests a high-performance GPU adapter and device. Throws [GpuException] when there is no usable GPU. */
        public suspend fun create(): WebGpu {
            if (!isSupported) {
                throw UnsupportedOperationException(
                    "Kernelin's JVM GPU host runs on macOS, Linux x64 and Windows x64, not on " +
                        "${System.getProperty("os.name")} ${System.getProperty("os.arch")}",
                )
            }
            val thread = Executors.newSingleThreadExecutor { r -> Thread(r, "Kernelin-WebGPU").apply { isDaemon = true } }
                .asCoroutineDispatcher()
            try {
                return withContext(thread) {
                    loadNativeLibrary()
                    val instance = WGPU.setupInstance()
                    if (!instance.isValid()) throw GpuException("Could not start WebGPU (wgpu)")
                    val state = DeviceState()
                    val adapter = try {
                        requestGpuAdapter(instance)
                    } catch (e: Throwable) {
                        instance.release()
                        throw e
                    }
                    val device = try {
                        requestGpuDevice(instance, adapter, state)
                    } catch (e: Throwable) {
                        adapter.release()
                        instance.release()
                        throw e
                    }
                    val limits = WGPULimits()
                    device.getLimits(limits)
                    val maxPerDimension = limits.maxComputeWorkgroupsPerDimension
                    limits.dispose()
                    val queue = WGPUQueue.native_new().also { it.native_copy(device.queue) }
                    WebGpu(instance, adapter, device, queue, maxPerDimension, thread, Thread.currentThread(), describe(adapter), state)
                }
            } catch (e: Throwable) {
                thread.close()
                throw e
            }
        }
    }

    private val kernels = mutableListOf<GpuKernel>()

    /** Kernels loaded by [dispatch], by kernel name, so each is compiled once. */
    private val launched = ConcurrentHashMap<String, GpuKernel>()
    private val launchedLock = Mutex()
    internal val dispatcher: CoroutineDispatcher get() = thread

    /** True after [close]. */
    @Volatile
    public var isClosed: Boolean = false
        private set

    /** Why the device was lost, or null while it works. */
    public val lostReason: String? get() = state.lostReason

    /** True when the GPU device was lost. Every later [load] and [GpuKernel.run] throws [GpuException]. */
    public val isLost: Boolean get() = lostReason != null

    /** Called once with [lostReason] when the device is lost (not after [close]), on Kernelin's GPU thread. */
    public var onLost: ((reason: String) -> Unit)?
        get() = state.onLost
        set(value) {
            state.onLost = value
        }

    /**
     * Called with WebGPU errors that happen outside a [load] or [GpuKernel.run] (those throw [GpuException] instead),
     * on Kernelin's GPU thread.
     */
    public var onUncapturedError: ((message: String) -> Unit)?
        get() = state.onUncapturedError
        set(value) {
            state.onUncapturedError = value
        }

    /** Compiles the kernel's WGSL and builds its pipeline. Throws if the GPU driver rejects the shader. */
    public suspend fun load(source: KernelSource): GpuKernel = onGpuThread {
        checkUsable()
        val label = source.name
        val manifest = source.manifest
        val natives = Natives()
        try {
            val module = WGPUShaderModule()
            // wgpu checks the WGSL while it creates the module and reports problems as a validation error.
            val compileErrors = capturingErrors {
                val wgsl = natives.add(WGPUShaderSourceWGSL()).apply { setCode(source.wgsl) }
                val chain = wgsl.chain.apply { setSType(WGPUSType.ShaderSourceWGSL) }
                val descriptor = natives.add(WGPUShaderModuleDescriptor()).apply {
                    setLabel(label)
                    setNextInChain(chain)
                }
                device.createShaderModule(descriptor, module)
            }
            if (compileErrors.isNotEmpty()) {
                if (module.isValid()) module.release()
                module.dispose()
                throw IllegalArgumentException("WGSL for ${source.name} was rejected: $compileErrors")
            }

            val layout = WGPUBindGroupLayout()
            val pipelineLayout = WGPUPipelineLayout()
            val pipeline = WGPUComputePipeline()
            val uniform = WGPUBuffer()
            val errors = capturingErrors {
                val entries = natives.add(WGPUVectorBindGroupLayoutEntry())
                entries.push_back(layoutEntry(natives, 0, WGPUBufferBindingType.Uniform))
                for (b in manifest.buffers) {
                    entries.push_back(layoutEntry(natives, b.binding, if (b.isWritten) WGPUBufferBindingType.Storage else WGPUBufferBindingType.ReadOnlyStorage))
                }
                device.createBindGroupLayout(
                    natives.add(WGPUBindGroupLayoutDescriptor()).apply {
                        setLabel("$label: bind group layout")
                        setEntries(entries)
                    },
                    layout,
                )
                val layouts = natives.add(WGPUVectorBindGroupLayout()).apply { push_back(layout) }
                device.createPipelineLayout(
                    natives.add(WGPUPipelineLayoutDescriptor()).apply {
                        setLabel("$label: pipeline layout")
                        setBindGroupLayouts(layouts)
                    },
                    pipelineLayout,
                )
                createBuffer(manifest.uniformBytes.toLong(), USAGE_UNIFORM, "$label: scalars", uniform)
                val descriptor = natives.add(WGPUComputePipelineDescriptor()).apply {
                    setLabel("$label: pipeline")
                    setLayout(pipelineLayout)
                    compute.setModule(module)
                    compute.setEntryPoint(manifest.entryPoint)
                }
                device.createComputePipeline(descriptor, pipeline)
            }
            pipelineLayout.release()
            pipelineLayout.dispose()
            module.release()
            module.dispose()
            if (errors.isNotEmpty() || !pipeline.isValid()) {
                for (o in listOf(layout, pipeline, uniform)) o.releaseHandle()
                throw GpuException("WebGPU rejected the pipeline for ${source.name}: ${errors.ifEmpty { "no pipeline" }}")
            }
            GpuKernel(this, source, layout, pipeline, uniform).also { kernels += it }
        } finally {
            natives.disposeAll()
        }
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

    /** Runs a launch that was already captured (also used by [Compute]). */
    internal suspend fun dispatch(shape: LaunchShape, launch: KernelLaunch): GpuResult {
        val loaded = launchedLock.withLock {
            launched[launch.source.name]?.takeUnless { it.isClosed } ?: load(launch.source).also { launched[launch.source.name] = it }
        }
        val arrays = launch.arrays
        val result = loaded.run(shape, launch.scalars, arrays.mapValues { (_, array) -> GpuData.ofArray(array) }, readBack = null)
        for (p in loaded.manifest.buffers) {
            if (!p.isWritten) continue
            when (val array = arrays.getValue(p.name)) {
                is IntArray -> result.bytes(p.name).asIntBuffer().get(array)
                is FloatArray -> result.bytes(p.name).asFloatBuffer().get(array)
            }
        }
        return result
    }

    /** Frees every kernel loaded from this device, then the device itself. Safe to call more than once. */
    override fun close() {
        if (isClosed) return
        isClosed = true
        state.closing = true
        val cleanup = Runnable {
            kernels.toList().forEach { it.closeOnGpuThread() }
            device.destroy()
            device.release()
            adapter.release()
            instance.release()
        }
        // wgpu objects belong to the GPU thread: clean up there, then stop it.
        if (Thread.currentThread() === gpuThread) cleanup.run() else thread.executor.execute(cleanup)
        thread.close()
    }

    internal fun forget(kernel: GpuKernel) {
        kernels -= kernel
        launched.remove(kernel.source.name, kernel)
    }

    internal fun checkUsable() {
        check(!isClosed) { "This WebGpu was closed; create a new one with WebGpu.create()" }
        lostReason?.let { throw GpuException("The GPU device was lost ($it). Create a new WebGpu and load the kernels again.") }
    }

    internal val isOnGpuThread: Boolean get() = Thread.currentThread() === gpuThread

    /** Runs [block] on the GPU thread, where every wgpu object is used. */
    internal suspend fun <T> onGpuThread(block: suspend WebGpu.() -> T): T {
        check(!isClosed) { "This WebGpu was closed; create a new one with WebGpu.create()" }
        return withContext(thread) { block() }
    }

    /** Waits for a wgpu callback: they are delivered from `processEvents`, so this polls until [result] is done. */
    internal suspend fun <T> pump(result: CompletableDeferred<T>): T = pumping(instance, result)

    /**
     * Runs [block] and returns the WebGPU errors it caused, joined with "; " ("" when there were none). jWebGPU has no
     * error scopes, but wgpu reports a validation or out-of-memory error during the call that caused it, so the
     * uncaptured-error callback collects them here while [block] runs.
     */
    internal suspend fun capturingErrors(block: suspend (errors: MutableList<String>) -> Unit): String {
        val errors = mutableListOf<String>()
        state.capture = errors
        try {
            block(errors)
            instance.processEvents()
        } finally {
            state.capture = null
        }
        return errors.joinToString("; ")
    }

    internal fun createBuffer(bytes: Long, usage: Int, label: String, into: WGPUBuffer) {
        val descriptor = WGPUBufferDescriptor()
        try {
            descriptor.setLabel(label)
            descriptor.setSize(bytes)
            synchronized(WGPUBufferUsage.CUSTOM) { descriptor.setUsage(WGPUBufferUsage.CUSTOM.setValue(usage)) }
            device.createBuffer(descriptor, into)
        } finally {
            descriptor.dispose()
        }
    }

    private fun layoutEntry(natives: Natives, binding: Int, type: WGPUBufferBindingType): WGPUBindGroupLayoutEntry =
        natives.add(WGPUBindGroupLayoutEntry()).apply {
            setBinding(binding)
            setVisibility(WGPUShaderStage.Compute)
            setBuffer(natives.add(WGPUBufferBindingLayout()).apply { setType(type) })
        }
}

/** The jWebGPU native library inside the jar for this OS and CPU, or null when there is none. */
internal fun nativeLibraryFor(osName: String, osArch: String): String? {
    val os = osName.lowercase()
    val arm = osArch.startsWith("aarch64") || osArch.startsWith("arm")
    val x64 = osArch == "amd64" || osArch == "x86_64"
    return when {
        os.startsWith("mac") && arm -> "mac arm64"
        os.startsWith("mac") && x64 -> "mac x64"
        os.startsWith("linux") && x64 -> "linux x64"
        os.startsWith("windows") && x64 -> "windows x64"
        else -> null
    }
}

private object NativeLibrary {
    @Volatile
    var error: Throwable? = null

    val loaded: Boolean by lazy {
        var ok = false
        JWebGPULoader.init(JWebGPUBackend.WGPU) { success, t ->
            ok = success
            error = t
        }
        ok
    }
}

/** Loads jWebGPU's native library once per process. */
private fun loadNativeLibrary() {
    if (!NativeLibrary.loaded) {
        throw GpuException("Could not load jWebGPU's native library: ${NativeLibrary.error?.message}", NativeLibrary.error)
    }
}

/**
 * The best adapter for compute: a real GPU first. A software adapter (like lavapipe or llvmpipe on a Linux server
 * without a GPU) is used only when nothing else is there; it runs kernels correctly, just slowly.
 */
private suspend fun requestGpuAdapter(instance: WGPUInstance): WGPUAdapter {
    val options = WGPURequestAdapterOptions()
    options.setPowerPreference(WGPUPowerPreference.HighPerformance)
    val result = CompletableDeferred<WGPUAdapter>()
    val callback = object : WGPURequestAdapterCallback() {
        override fun onCallback(status: WGPURequestAdapterStatus, adapter: WGPUAdapter, message: String?) {
            if (status == WGPURequestAdapterStatus.Success && !adapter.native_isNULL()) {
                result.complete(adapter)
            } else {
                result.completeExceptionally(GpuException("No GPU found for WebGPU ($status: ${message.orEmpty()})"))
            }
        }
    }
    try {
        instance.requestAdapter(options, WGPUCallbackMode.AllowProcessEvents, callback)
        return pumping(instance, result)
    } finally {
        callback.dispose()
        options.dispose()
    }
}

private suspend fun requestGpuDevice(instance: WGPUInstance, adapter: WGPUAdapter, state: DeviceState): WGPUDevice {
    // Big buffers: ask for everything the GPU allows, like the browser host does.
    val limits = WGPULimits()
    adapter.getLimits(limits)
    val descriptor = WGPUDeviceDescriptor()
    descriptor.setLabel("Kernelin")
    descriptor.setRequiredLimits(limits)
    val result = CompletableDeferred<WGPUDevice>()
    val callback = object : WGPURequestDeviceCallback() {
        override fun onCallback(status: WGPURequestDeviceStatus, device: WGPUDevice, message: String?) {
            if (status == WGPURequestDeviceStatus.Success && !device.native_isNULL()) {
                result.complete(device)
            } else {
                result.completeExceptionally(GpuException("Could not open the GPU device: ${message.orEmpty()}"))
            }
        }
    }
    try {
        adapter.requestDevice(descriptor, WGPUCallbackMode.AllowProcessEvents, callback, state.errorCallback)
        return pumping(instance, result)
    } finally {
        callback.dispose()
        descriptor.dispose()
        limits.dispose()
    }
}

private fun describe(adapter: WGPUAdapter): String {
    val info = WGPUAdapterInfo()
    try {
        adapter.getInfo(info)
        val gpuName = listOf(info.vendor.c_str(), info.architecture.c_str(), info.device.c_str())
            .map { it.orEmpty().trim() }.filter { it.isNotBlank() }.distinct().joinToString(" ")
            .ifEmpty { info.description.c_str().orEmpty().ifBlank { "WebGPU" } }
        val software = if (info.adapterType == WGPUAdapterType.CPU) ", software" else ""
        return "$gpuName (${info.backendType}$software)"
    } finally {
        info.dispose()
    }
}

/** Device-lost and error callbacks arrive before the [WebGpu] exists, so they write here. */
internal class DeviceState {
    @Volatile
    var lostReason: String? = null
    @Volatile
    var onLost: ((String) -> Unit)? = null
    @Volatile
    var onUncapturedError: ((String) -> Unit)? = null
    @Volatile
    var closing: Boolean = false

    /** Set while a load or run collects its errors (see [WebGpu.capturingErrors]); only used on the GPU thread. */
    var capture: MutableList<String>? = null

    /** jWebGPU 0.3 has no device-lost callback; wgpu reports a lost device as an error that says so. */
    val errorCallback: WGPUUncapturedErrorCallback = object : WGPUUncapturedErrorCallback() {
        override fun onCallback(errorType: WGPUErrorType, message: String?) {
            val text = "$errorType: ${message.orEmpty()}"
            if (message.orEmpty().contains("device lost", ignoreCase = true) || message.orEmpty().contains("DeviceLost")) lost(text)
            capture?.add(text) ?: uncaptured(text)
        }
    }

    fun lost(reason: String) {
        if (lostReason != null) return
        lostReason = reason
        if (!closing) onLost?.invoke(reason)
    }

    fun uncaptured(message: String) {
        onUncapturedError?.invoke(message)
    }
}

/** jWebGPU descriptors live in native memory: [disposeAll] frees the ones made for one call. */
internal class Natives {
    private val objects = mutableListOf<NativeObject>()

    fun <T : NativeObject> add(o: T): T = o.also { objects += it }

    fun disposeAll() {
        objects.asReversed().forEach { it.dispose() }
        objects.clear()
    }
}

/**
 * Waits for [result] while calling `processEvents` in a loop next to it (both on the current, single GPU thread).
 * It spins for the first millisecond, so short GPU work isn't slowed down, then checks every millisecond.
 */
private suspend fun <T> pumping(instance: WGPUInstance, result: CompletableDeferred<T>): T = coroutineScope {
    val poll = launch {
        val spinUntil = System.nanoTime() + 1_000_000
        while (true) {
            instance.processEvents()
            if (System.nanoTime() < spinUntil) yield() else delay(1)
        }
    }
    try {
        result.await()
    } finally {
        poll.cancel()
    }
}

/** Releases the WebGPU object (when one was made) and frees its jWebGPU wrapper, after a failed [WebGpu.load]. */
private fun NativeObject.releaseHandle() {
    when (this) {
        is WGPUBuffer -> if (isValid()) {
            destroy()
            release()
        }
        is WGPUBindGroupLayout -> if (isValid()) release()
        is WGPUComputePipeline -> if (isValid()) release()
    }
    dispose()
}

/** WebGPU reported an error: the device was lost, ran out of memory, or rejected a pipeline or a command. */
public class GpuException(message: String, cause: Throwable? = null) : RuntimeException(message, cause)

/**
 * Input/output data for one kernel buffer. [type] is the WGSL element type (`i32` or `f32`), or null for an empty
 * buffer, which fits any buffer type.
 */
public class GpuData private constructor(
    internal val bytes: ByteBuffer?,
    internal val length: Int,
    internal val type: String?,
    internal val gpuBuffer: GpuBuffer? = null,
) {
    /** Ways to create buffer data. */
    public companion object {
        /** A zero-filled buffer of [length] elements, not uploaded (for outputs). */
        public fun empty(length: Int): GpuData = GpuData(null, length, type = null)

        /** Copies an IntArray for upload (an `i32` buffer). */
        public fun of(values: IntArray): GpuData =
            GpuData(direct(values.size).also { it.asIntBuffer().put(values) }, values.size, type = "i32")

        /** Copies a FloatArray for upload (an `f32` buffer). */
        public fun of(values: FloatArray): GpuData =
            GpuData(direct(values.size).also { it.asFloatBuffer().put(values) }, values.size, type = "f32")

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

        private fun direct(elements: Int): ByteBuffer = ByteBuffer.allocateDirect(elements * 4).order(ByteOrder.LITTLE_ENDIAN)
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

    internal fun gpuBuffer(): GpuKernel.Storage {
        check(!kernel.isClosed) { "GpuBuffer '$name' of ${kernel.source.name} can't be used: its kernel was closed" }
        check(kernel.generation == generation) {
            "GpuBuffer '$name' of ${kernel.source.name} is out of date: the kernel ran again, so the buffer holds " +
                "newer data. Use the buffer from the GpuResult of the latest run."
        }
        return kernel.gpuBuffer(name)
    }
}

/** Results of one GPU run: the `read_write` buffers that were read back. */
public class GpuResult internal constructor(
    /** Milliseconds from upload until results were back in memory. */
    public val millis: Double,
    private val outputs: Map<String, ByteBuffer>,
    private val buffers: Map<String, GpuBuffer>,
) {
    /** The raw little-endian bytes of an output; no copy. */
    public fun bytes(name: String): ByteBuffer = (
        outputs[name] ?: throw IllegalArgumentException(
            if (name in buffers) {
                "Buffer '$name' was not read back (see readBack in GpuKernel.run); use buffer(\"$name\") to keep it on the GPU"
            } else {
                "No output buffer named '$name'"
            },
        )
        ).duplicate().order(ByteOrder.LITTLE_ENDIAN)

    /** Copies an `i32` output into a Kotlin [IntArray]. */
    public fun ints(name: String): IntArray = bytes(name).asIntBuffer().let { b -> IntArray(b.remaining()).also { b.get(it) } }

    /** Copies an `f32` output into a Kotlin [FloatArray]. */
    public fun floats(name: String): FloatArray = bytes(name).asFloatBuffer().let { b -> FloatArray(b.remaining()).also { b.get(it) } }

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
    private val layout: WGPUBindGroupLayout,
    private val pipeline: WGPUComputePipeline,
    private val uniform: WGPUBuffer,
) : AutoCloseable {
    /** The kernel's parameters and workgroup size. */
    public val manifest: KernelManifest = source.manifest
    private val device = gpu.device
    private val label = source.name
    private val buffers = HashMap<String, Storage>()
    private val staging = HashMap<String, Storage>()
    private var bindGroup: WGPUBindGroup? = null
    private val mutex = Mutex()

    /** A GPU buffer and its size in bytes (jWebGPU's size getter is a 32-bit int). */
    internal class Storage(val buffer: WGPUBuffer, val bytes: Int) {
        fun free() {
            buffer.destroy()
            buffer.release()
            buffer.dispose()
        }
    }

    /** Counts runs, so a [GpuBuffer] from an older run can tell that its data was replaced. */
    @Volatile
    internal var generation: Int = 0
        private set

    /** True after [close] (or [WebGpu.close]). */
    @Volatile
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
        for ((name, data) in buffers) {
            require(data.length <= MAX_ELEMENTS) { "Buffer '$name' has ${data.length} elements; the JVM GPU host takes at most $MAX_ELEMENTS (2 GB)" }
        }
        val uniformData = ByteBuffer.allocateDirect(manifest.uniformBytes).put(uniformBytes(source, shape, scalars)).also { it.flip() }
        gpu.onGpuThread {
            checkOpen()
            // Buffers from earlier runs are read now, before this run replaces or reallocates them.
            val sources = manifest.buffers.mapNotNull { p -> buffers.getValue(p.name).gpuBuffer?.let { p.name to it.gpuBuffer() } }.toMap()
            generation++
            val outputs = manifest.buffers.filter { it.isWritten && (readBack == null || it.name in readBack) }

            val start = System.nanoTime()
            val errors = capturingErrors { submit(size, uniformData, buffers, sources, outputs) }
            if (errors.isNotEmpty()) {
                gpu.lostReason?.let { throw GpuException("The GPU device was lost during run() of ${source.name} ($it)") }
                throw GpuException("WebGPU rejected the run of ${source.name}: $errors")
            }

            val results = LinkedHashMap<String, ByteBuffer>()
            for (p in outputs) {
                val s = staging.getValue(p.name)
                val mapped = CompletableDeferred<String?>()
                val callback = object : WGPUBufferMapCallback() {
                    override fun onCallback(status: WGPUMapAsyncStatus, message: String?) {
                        mapped.complete(if (status == WGPUMapAsyncStatus.Success) null else "$status: ${message.orEmpty()}")
                    }
                }
                val failure = try {
                    s.buffer.mapAsync(WGPUMapMode.Read, 0, s.bytes, WGPUCallbackMode.AllowProcessEvents, callback)
                    pump(mapped)
                } finally {
                    callback.dispose()
                }
                if (failure != null) {
                    if (isClosed || gpu.isClosed) throw IllegalStateException("${source.name} was closed during run()")
                    gpu.lostReason?.let { throw GpuException("The GPU device was lost during run() of ${source.name} ($it)") }
                    throw GpuException("Reading the results of ${source.name} failed: $failure")
                }
                val copy = ByteBuffer.allocateDirect(s.bytes).order(ByteOrder.LITTLE_ENDIAN)
                s.buffer.getConstMappedRange(0, s.bytes, copy)
                s.buffer.unmap()
                results[p.name] = copy
            }
            val handles = manifest.buffers.associate { p ->
                p.name to GpuBuffer(this@GpuKernel, p.name, this@GpuKernel.buffers.getValue(p.name).bytes / 4, p.type, generation)
            }
            GpuResult((System.nanoTime() - start) / 1e6, results, handles)
        }
    }

    /** Everything a run sends to the GPU, in one command buffer. */
    private fun submit(size: Int, uniformData: ByteBuffer, inputs: Map<String, GpuData>, sources: Map<String, Storage>, outputs: List<KernelManifest.Param>) {
        val queue = gpu.queue
        queue.writeBuffer(uniform, 0, uniformData, uniformData.remaining())
        val retired = mutableListOf<Storage>()
        // Buffers reused from an earlier run still hold its data, so empty() ones are cleared on the GPU.
        val clears = mutableListOf<Storage>()
        val copyIn = mutableListOf<Pair<Storage, Storage>>()
        for (p in manifest.buffers) {
            val data = inputs.getValue(p.name)
            val buffer = ensure(this.buffers, p.name, data.length * 4, USAGE_STORAGE, "$label: ${p.name}", retired)
            val from = sources[p.name]
            when {
                from != null -> if (from !== buffer) copyIn += from to buffer
                data.bytes != null -> queue.writeBuffer(buffer.buffer, 0, data.bytes, data.bytes.capacity())
                else -> clears += buffer
            }
        }
        for (p in outputs) {
            ensure(staging, p.name, this.buffers.getValue(p.name).bytes, USAGE_STAGING, "$label: ${p.name} (read back)", retired)
        }
        val group = bindGroup ?: createBindGroup().also { bindGroup = it }

        val (gx, gy) = workgroupGrid(size, manifest.workgroupSize, gpu.maxPerDimension)
        val natives = Natives()
        try {
            val encoder = natives.add(WGPUCommandEncoder())
            device.createCommandEncoder(natives.add(WGPUCommandEncoderDescriptor()).apply { setLabel("$label: run") }, encoder)
            for ((from, to) in copyIn) encoder.copyBufferToBuffer(from.buffer, 0, to.buffer, 0, to.bytes)
            for (b in clears) encoder.clearBuffer(b.buffer, 0, b.bytes)
            val pass = natives.add(WGPUComputePassEncoder())
            encoder.beginComputePass(natives.add(WGPUComputePassDescriptor()).apply { setLabel("$label: run") }, pass)
            pass.setPipeline(pipeline)
            pass.setBindGroup(0, group)
            pass.setDispatchWorkgroups(gx, gy, 1)
            pass.end()
            pass.release()
            for (p in outputs) {
                val from = this.buffers.getValue(p.name)
                encoder.copyBufferToBuffer(from.buffer, 0, staging.getValue(p.name).buffer, 0, from.bytes)
            }
            val commands = natives.add(WGPUCommandBuffer())
            encoder.finish(natives.add(WGPUCommandBufferDescriptor()), commands)
            queue.submit(commands)
            commands.release()
            encoder.release()
        } finally {
            natives.disposeAll()
        }
        // Destroyed after the submit: WebGPU frees them once the GPU work that uses them is done.
        retired.forEach { it.free() }
    }

    private fun createBindGroup(): WGPUBindGroup {
        val natives = Natives()
        try {
            val entries = natives.add(WGPUVectorBindGroupEntry())
            fun entry(binding: Int, buffer: WGPUBuffer, bytes: Int) {
                entries.push_back(
                    natives.add(WGPUBindGroupEntry()).apply {
                        setBinding(binding)
                        setBuffer(buffer)
                        setOffset(0)
                        setSize(bytes.toLong())
                    },
                )
            }
            entry(0, uniform, manifest.uniformBytes)
            for (p in manifest.buffers) buffers.getValue(p.name).let { entry(p.binding, it.buffer, it.bytes) }
            val group = WGPUBindGroup()
            device.createBindGroup(
                natives.add(WGPUBindGroupDescriptor()).apply {
                    setLabel("$label: bind group")
                    setLayout(layout)
                    setEntries(entries)
                },
                group,
            )
            return group
        } finally {
            natives.disposeAll()
        }
    }

    /** Frees this kernel's GPU buffers. Later [run] calls throw [IllegalStateException]. Safe to call more than once. */
    override fun close() {
        if (isClosed || gpu.isClosed || gpu.isOnGpuThread) return closeOnGpuThread()
        // wgpu objects belong to the GPU thread.
        runBlocking { gpu.onGpuThread { closeOnGpuThread() } }
    }

    internal fun closeOnGpuThread() {
        if (isClosed) return
        isClosed = true
        for (b in buffers.values) b.free()
        for (b in staging.values) b.free()
        Storage(uniform, manifest.uniformBytes).free()
        buffers.clear()
        staging.clear()
        releaseBindGroup()
        pipeline.release()
        pipeline.dispose()
        layout.release()
        layout.dispose()
        gpu.forget(this)
    }

    internal fun gpuBuffer(name: String): Storage = buffers.getValue(name)

    private fun releaseBindGroup() {
        bindGroup?.let {
            it.release()
            it.dispose()
        }
        bindGroup = null
    }

    private fun checkOpen() {
        check(!isClosed) { "GpuKernel ${source.name} was closed; load it again with WebGpu.load()" }
        gpu.checkUsable()
    }

    private fun ensure(map: HashMap<String, Storage>, name: String, bytes: Int, usage: Int, label: String, retired: MutableList<Storage>): Storage {
        val existing = map[name]
        if (existing != null && existing.bytes == bytes) return existing
        existing?.let { retired += it }
        if (map === buffers) releaseBindGroup()
        val buffer = WGPUBuffer()
        gpu.createBuffer(bytes.toLong(), usage, label, buffer)
        return Storage(buffer, bytes).also { map[name] = it }
    }
}

/** jWebGPU passes buffer sizes as 32-bit ints, so one buffer holds at most 2 GB. */
private const val MAX_ELEMENTS = Int.MAX_VALUE / 4

// Getters, not constants: jWebGPU reads enum values from its native library, which is loaded by WebGpu.create().
private val USAGE_UNIFORM get() = WGPUBufferUsage.Uniform.value or WGPUBufferUsage.CopyDst.value
private val USAGE_STORAGE get() = WGPUBufferUsage.Storage.value or WGPUBufferUsage.CopySrc.value or WGPUBufferUsage.CopyDst.value
private val USAGE_STAGING get() = WGPUBufferUsage.MapRead.value or WGPUBufferUsage.CopyDst.value

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
