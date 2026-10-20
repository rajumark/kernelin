package io.github.rajumark.kernelin.webgpu

import androidx.webgpu.BackendType
import androidx.webgpu.BufferBindingType
import androidx.webgpu.BufferUsage
import androidx.webgpu.CompilationMessageType
import androidx.webgpu.DeviceLostReason
import androidx.webgpu.ErrorFilter
import androidx.webgpu.FeatureLevel
import androidx.webgpu.ErrorType
import androidx.webgpu.GPU
import androidx.webgpu.GPUAdapter
import androidx.webgpu.GPUBindGroup
import androidx.webgpu.GPUBindGroupDescriptor
import androidx.webgpu.GPUBindGroupEntry
import androidx.webgpu.GPUBindGroupLayout
import androidx.webgpu.GPUBindGroupLayoutDescriptor
import androidx.webgpu.GPUBindGroupLayoutEntry
import androidx.webgpu.GPUBuffer
import androidx.webgpu.GPUBufferBindingLayout
import androidx.webgpu.GPUBufferDescriptor
import androidx.webgpu.GPUCommandEncoderDescriptor
import androidx.webgpu.GPUComputePassDescriptor
import androidx.webgpu.GPUComputePipeline
import androidx.webgpu.GPUComputePipelineDescriptor
import androidx.webgpu.GPUComputeState
import androidx.webgpu.GPUDevice
import androidx.webgpu.GPUDeviceDescriptor
import androidx.webgpu.GPUInstance
import androidx.webgpu.GPULimits
import androidx.webgpu.GPUPipelineLayoutDescriptor
import androidx.webgpu.GPURequestAdapterOptions
import androidx.webgpu.GPUShaderModuleDescriptor
import androidx.webgpu.GPUShaderSourceWGSL
import androidx.webgpu.MapMode
import androidx.webgpu.PowerPreference
import androidx.webgpu.ShaderStage
import androidx.webgpu.WebGpuException
import io.github.rajumark.kernelin.KernelLaunch
import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelSource
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.lang.ref.WeakReference
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.Executor
import java.util.concurrent.Executors
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
 * Runs Kernelin kernels on the GPU of an Android device, natively through AndroidX WebGPU (Dawn on Vulkan).
 * No browser and no WebView: the same API as the Kotlin/Wasm host.
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
 * from any coroutine, including the main thread. The GPU can be lost at any time (driver reset, the app going to
 * the background for long): check [isLost] or set [onLost], then create a new [WebGpu] and load the kernels again.
 * [close] frees the device and every kernel loaded from it.
 */
public class WebGpu private constructor(
    private val instance: GPUInstance,
    private val adapter: GPUAdapter,
    internal val device: GPUDevice,
    private val thread: ExecutorCoroutineDispatcher,
    private val gpuThread: Thread,
    /** The GPU, for example `qualcomm adreno-7xx (Vulkan)`. */
    public val description: String,
    private val state: DeviceState,
) : AutoCloseable {
    /** Entry point: check [isSupported], then [create]. */
    public companion object {
        /** True on Android 8.0 (API 26) and newer, which AndroidX WebGPU needs. [create] can still fail on a GPU without Vulkan. */
        public val isSupported: Boolean get() = android.os.Build.VERSION.SDK_INT >= 26

        /** Requests a high-performance GPU adapter and device. Throws [GpuException] when there is no usable GPU. */
        public suspend fun create(): WebGpu {
            if (!isSupported) throw UnsupportedOperationException("Kernelin's Android GPU host needs Android 8.0 (API 26) or newer")
            val thread = Executors.newSingleThreadExecutor { r -> Thread(r, "Kernelin-WebGPU").apply { isDaemon = true } }
                .asCoroutineDispatcher()
            try {
                return withContext(thread) {
                    System.loadLibrary("webgpu_c_bundled")
                    val instance = GPU.createInstance()
                    val state = DeviceState()
                    val adapter = requestGpuAdapter(instance)
                    val limits = adapter.getLimits()
                    val descriptor = GPUDeviceDescriptor(
                        deviceLostCallbackExecutor = Executor(Runnable::run),
                        uncapturedErrorCallbackExecutor = Executor(Runnable::run),
                        label = "Kernelin",
                        // Big buffers: ask for everything the GPU allows, like the browser host does.
                        requiredLimits = GPULimits(
                            maxStorageBufferBindingSize = limits.maxStorageBufferBindingSize,
                            maxBufferSize = limits.maxBufferSize,
                        ),
                        deviceLostCallback = { _, reason, message -> state.lost("${DeviceLostReason.toString(reason)}: $message") },
                        uncapturedErrorCallback = { _, type, message -> state.uncaptured("${ErrorType.toString(type)}: $message") },
                    )
                    val device = try {
                        pumping(instance) { adapter.requestDevice(descriptor) }
                    } catch (e: Exception) {
                        adapter.close()
                        instance.close()
                        throw GpuException("Could not open the GPU device: ${e.message}", e)
                    }
                    val info = adapter.getInfo()
                    val gpuName = listOf(info.vendor, info.architecture, info.device).filter { it.isNotBlank() }.joinToString(" ")
                        .ifEmpty { info.description.ifBlank { "WebGPU" } }
                    val name = "$gpuName (${BackendType.toString(info.backendType)})"
                    WebGpu(instance, adapter, device, thread, Thread.currentThread(), name, state)
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

    /** Why the device was lost (for example `Unknown: VK_ERROR_DEVICE_LOST`), or null while it works. */
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
        val module = device.createShaderModule(GPUShaderModuleDescriptor(label = label, shaderSourceWGSL = GPUShaderSourceWGSL(source.wgsl)))
        val compileErrors = pump { module.getCompilationInfo() }.messages
            .filter { it.type == CompilationMessageType.Error }
            .joinToString("; ") { "${it.lineNum}:${it.message}" }
        if (compileErrors.isNotEmpty()) throw IllegalArgumentException("WGSL for ${source.name} was rejected: $compileErrors")

        val layout: GPUBindGroupLayout
        val uniform: GPUBuffer
        val pipeline: GPUComputePipeline
        pushErrorScopes()
        try {
            val entries = buildList {
                add(GPUBindGroupLayoutEntry(binding = 0, visibility = ShaderStage.Compute, buffer = GPUBufferBindingLayout(type = BufferBindingType.Uniform)))
                for (b in manifest.buffers) {
                    val type = if (b.isWritten) BufferBindingType.Storage else BufferBindingType.ReadOnlyStorage
                    add(GPUBindGroupLayoutEntry(binding = b.binding, visibility = ShaderStage.Compute, buffer = GPUBufferBindingLayout(type = type)))
                }
            }
            layout = device.createBindGroupLayout(GPUBindGroupLayoutDescriptor(label = "$label: bind group layout", entries = entries.toTypedArray()))
            val pipelineLayout = device.createPipelineLayout(
                GPUPipelineLayoutDescriptor(label = "$label: pipeline layout", bindGroupLayouts = arrayOf(layout)),
            )
            uniform = device.createBuffer(GPUBufferDescriptor(usage = USAGE_UNIFORM, size = manifest.uniformBytes.toLong(), label = "$label: scalars"))
            pipeline = try {
                pump {
                    device.createComputePipelineAndAwait(
                        GPUComputePipelineDescriptor(
                            compute = GPUComputeState(module = module, entryPoint = manifest.entryPoint),
                            label = "$label: pipeline",
                            layout = pipelineLayout,
                        ),
                    )
                }
            } catch (e: Exception) {
                popErrorScopes()
                throw GpuException("WebGPU could not build the pipeline for ${source.name}: ${e.message}", e)
            }
        } catch (e: Throwable) {
            if (e !is GpuException) popErrorScopes()
            throw e
        }
        val errors = popErrorScopes()
        if (errors.isNotEmpty()) throw GpuException("WebGPU rejected the pipeline for ${source.name}: $errors")
        GpuKernel(this, source, layout, pipeline, uniform).also { kernels += it }
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
        // The arrays are read during this call, so they are uploaded as they are, without a snapshot copy first.
        val result = loaded.run(shape, launch.scalars, arrays.mapValues { (_, array) -> GpuData.borrow(array) }, readBack = null)
        for (p in loaded.manifest.buffers) {
            if (p.isWritten) result.readBack(p.name).readInto(arrays.getValue(p.name))
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
            uploads.forEach { it?.destroy() }
            device.destroy()
            adapter.close()
            instance.close()
        }
        // Dawn objects belong to the GPU thread: clean up there, then stop it.
        if (Thread.currentThread() === gpuThread) cleanup.run() else thread.executor.execute(cleanup)
        thread.close()
    }

    /**
     * Upload buffers that the CPU writes into directly (mapped), reused by every run: two, so the CPU fills one while
     * the GPU copies the other. Dawn's `writeBuffer` stages big uploads through a new buffer every time, which on a
     * phone (moto g57 power) takes ~140 ms for 64 MB; this takes ~25 ms.
     */
    private val uploads = arrayOfNulls<GPUBuffer>(2)
    private val uploadMapped = BooleanArray(2)
    private var nextUpload = 0
    /** Runs of different kernels can interleave on the GPU thread while one waits for a map, so uploads take turns. */
    private val uploadLock = Mutex()

    /** Copies [data] (CPU contents) into [target], in chunks of at most [UPLOAD_CHUNK] bytes. On the GPU thread. */
    internal suspend fun upload(target: GPUBuffer, data: GpuData, label: String): Unit = uploadLock.withLock {
        val total = data.length * 4L
        var offset = 0L
        try {
            while (offset < total) {
                val bytes = minOf(UPLOAD_CHUNK, total - offset)
                val i = nextUpload
                nextUpload = (i + 1) % uploads.size
                var staging = uploads[i]
                if (staging == null || staging.size < bytes) {
                    // Destroyed only once the GPU is done copying from it.
                    staging?.destroy()
                    staging = device.createBuffer(GPUBufferDescriptor(USAGE_UPLOAD, bytes, "Kernelin: upload", mappedAtCreation = true))
                    uploads[i] = staging
                    uploadMapped[i] = true
                } else if (!uploadMapped[i]) {
                    // Waits until the GPU has copied this buffer's previous chunk.
                    pump { staging.mapAndAwait(MapMode.Write, 0, staging.size) }
                    uploadMapped[i] = true
                }
                data.writeTo(staging.getMappedRange(0, bytes).order(ByteOrder.LITTLE_ENDIAN), offset.toInt(), bytes.toInt())
                staging.unmap()
                uploadMapped[i] = false
                val encoder = device.createCommandEncoder(GPUCommandEncoderDescriptor(label = "$label (upload)"))
                encoder.copyBufferToBuffer(staging, 0, target, offset, bytes)
                val commands = encoder.finish()
                device.queue.submit(arrayOf(commands))
                commands.close()
                encoder.close()
                offset += bytes
            }
        } catch (e: WebGpuException) {
            lostReason?.let { throw GpuException("The GPU device was lost while uploading $label ($it)", e) }
            throw GpuException("Uploading $label failed: ${e.message}", e)
        }
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

    /** Runs [block] on the GPU thread, where every Dawn object must be used. */
    internal suspend fun <T> onGpuThread(block: suspend WebGpu.() -> T): T {
        check(!isClosed) { "This WebGpu was closed; create a new one with WebGpu.create()" }
        return withContext(thread) { block() }
    }

    /** Waits for a Dawn callback: Dawn only delivers them from `processEvents`, so this polls until [block] is done. */
    internal suspend fun <T> pump(block: suspend () -> T): T = pumping(instance, block)

    internal fun pushErrorScopes() {
        device.pushErrorScope(ErrorFilter.OutOfMemory)
        device.pushErrorScope(ErrorFilter.Validation)
    }

    /** Pops what [pushErrorScopes] pushed; the errors joined with "; ", or "" when there were none. */
    internal suspend fun popErrorScopes(): String {
        val validation = popOne("validation")
        val memory = popOne("out of memory")
        return listOfNotNull(validation, memory).joinToString("; ")
    }

    private suspend fun popOne(kind: String): String? = try {
        val type = pump { device.popErrorScope() }
        if (type == ErrorType.NoError) null else "$kind: ${ErrorType.toString(type)}"
    } catch (e: Exception) {
        "$kind: ${e.message}"
    }
}

/**
 * Vulkan first. Phones (and emulators) whose Vulkan driver Dawn can't use get OpenGL ES 3.1 in WebGPU's
 * compatibility mode, which runs compute shaders too. Dawn's "Null" adapter accepts every command and runs none,
 * so it is never used.
 */
private suspend fun requestGpuAdapter(instance: GPUInstance): GPUAdapter {
    val failures = mutableListOf<String>()
    val attempts = listOf(
        GPURequestAdapterOptions(powerPreference = PowerPreference.HighPerformance, backendType = BackendType.Vulkan),
        GPURequestAdapterOptions(
            featureLevel = FeatureLevel.Compatibility,
            powerPreference = PowerPreference.HighPerformance,
            backendType = BackendType.OpenGLES,
        ),
    )
    for (options in attempts) {
        val name = BackendType.toString(options.backendType)
        val adapter = try {
            pumping(instance) { instance.requestAdapter(options) }
        } catch (e: Exception) {
            failures += "$name: ${e.message}"
            continue
        }
        if (adapter.getInfo().backendType == options.backendType) return adapter
        failures += "$name: got the ${BackendType.toString(adapter.getInfo().backendType)} adapter"
        adapter.close()
    }
    instance.close()
    throw GpuException("No GPU found for WebGPU (${failures.joinToString("; ")})")
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

    fun lost(reason: String) {
        if (lostReason != null) return
        lostReason = reason
        if (!closing) onLost?.invoke(reason)
    }

    fun uncaptured(message: String) {
        onUncapturedError?.invoke(message)
    }
}

/**
 * Runs [block] while calling `processEvents` in a loop next to it (both on the current, single GPU thread).
 * It spins for the first millisecond, so short GPU work isn't slowed down, then checks every millisecond.
 */
private suspend fun <T> pumping(instance: GPUInstance, block: suspend () -> T): T = coroutineScope {
    val poll = launch {
        val spinUntil = System.nanoTime() + 1_000_000
        while (true) {
            instance.processEvents()
            if (System.nanoTime() < spinUntil) yield() else delay(1)
        }
    }
    try {
        block()
    } finally {
        poll.cancel()
    }
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
    /** An IntArray or FloatArray read when the run uploads it, not copied before (see [borrow]). */
    private val array: Any? = null,
) {
    /** True when this data comes from the CPU and must be uploaded. */
    internal val hasContents: Boolean get() = bytes != null || array != null

    /** Writes [byteCount] bytes of the contents, from byte [from] on, into [into] (a mapped upload buffer). */
    internal fun writeTo(into: ByteBuffer, from: Int, byteCount: Int) {
        when {
            bytes != null -> into.put(bytes.duplicate().also { it.position(from); it.limit(from + byteCount) })
            array is FloatArray -> into.asFloatBuffer().put(array, from / 4, byteCount / 4)
            array is IntArray -> into.asIntBuffer().put(array, from / 4, byteCount / 4)
        }
    }

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

        /**
         * An IntArray or FloatArray argument of a [KernelLaunch], uploaded straight from the array without a snapshot
         * copy: [WebGpu.dispatch] reads it during the call, so it can't change in between.
         */
        internal fun borrow(array: Any): GpuData = when (array) {
            is IntArray -> GpuData(null, array.size, "i32", array = array)
            is FloatArray -> GpuData(null, array.size, "f32", array = array)
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

    internal fun gpuBuffer(): GPUBuffer {
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
    private val outputs: Map<String, ReadBack>,
    private val buffers: Map<String, GpuBuffer>,
) {
    /** The raw little-endian bytes of an output. The first call copies them out of GPU memory; later calls don't copy. */
    public fun bytes(name: String): ByteBuffer = readBack(name).bytes().duplicate().order(ByteOrder.LITTLE_ENDIAN)

    /** Copies an `i32` output into a Kotlin [IntArray]. */
    public fun ints(name: String): IntArray = readBack(name).read { b -> b.asIntBuffer().let { IntArray(it.remaining()).also(it::get) } }

    /** Copies an `f32` output into a Kotlin [FloatArray]. */
    public fun floats(name: String): FloatArray = readBack(name).read { b -> b.asFloatBuffer().let { FloatArray(it.remaining()).also(it::get) } }

    internal fun readBack(name: String): ReadBack = outputs[name] ?: throw IllegalArgumentException(
        if (name in buffers) {
            "Buffer '$name' was not read back (see readBack in GpuKernel.run); use buffer(\"$name\") to keep it on the GPU"
        } else {
            "No output buffer named '$name'"
        },
    )

    /** The kernel's GPU buffer [name] after this run, to pass to another run with [GpuData.of] without a CPU copy. */
    public fun buffer(name: String): GpuBuffer = buffers[name] ?: throw IllegalArgumentException("No buffer parameter named '$name'")
}

/**
 * One read-back output. It stays in the GPU's read-back buffer, mapped into memory, so [GpuResult.floats] and
 * [GpuResult.ints] copy it only once, straight into the Kotlin array. Before that buffer is reused or freed, the
 * kernel calls [keep], which copies the data to memory of its own if the result may still be read.
 */
internal class ReadBack(private var mapped: ByteBuffer?) {
    private var copy: ByteBuffer? = null

    /** Calls [block] with the data (little-endian, position 0), while [keep] can't run. */
    @Synchronized
    fun <T> read(block: (ByteBuffer) -> T): T = block(checkNotNull(copy ?: mapped).duplicate().order(ByteOrder.LITTLE_ENDIAN))

    /** Copies the data into [array], an IntArray or FloatArray of the same size. */
    fun readInto(array: Any): Unit = read { b ->
        when (array) {
            is IntArray -> b.asIntBuffer().get(array)
            is FloatArray -> b.asFloatBuffer().get(array)
        }
    }

    /** The data in memory of its own, copied out of the read-back buffer on the first call. */
    @Synchronized
    fun bytes(): ByteBuffer = copy ?: checkNotNull(mapped).let { m ->
        ByteBuffer.allocateDirect(m.capacity()).order(ByteOrder.LITTLE_ENDIAN).put(m.duplicate()).also { it.flip(); copy = it }
    }

    /** True when the data no longer needs the read-back buffer, because [bytes] copied it. */
    @Synchronized
    fun isCopied(): Boolean = (copy != null).also { if (it) mapped = null }

    /** Copies the data so the read-back buffer can be reused or freed. On the GPU thread. */
    @Synchronized
    fun keep() {
        bytes()
        mapped = null
    }
}

/**
 * A compiled kernel. GPU buffers are reused between runs and only reallocated when sizes change.
 * Runs of one kernel never overlap: a second [run] waits until the first is done. [close] frees its GPU buffers.
 */
public class GpuKernel internal constructor(
    internal val gpu: WebGpu,
    public val source: KernelSource,
    private val layout: GPUBindGroupLayout,
    private val pipeline: GPUComputePipeline,
    private val uniform: GPUBuffer,
) : AutoCloseable {
    /** The kernel's parameters and workgroup size. */
    public val manifest: KernelManifest = source.manifest
    private val device = gpu.device
    private val label = source.name
    private val buffers = HashMap<String, GPUBuffer>()
    /** Free read-back buffers, by parameter name. */
    private val staging = HashMap<String, GPUBuffer>()
    /** Read-back buffers lent to results that may still be read, oldest first. */
    private val lent = ArrayDeque<Lent>()
    private var bindGroup: GPUBindGroup? = null
    private val mutex = Mutex()
    private val maxPerDimension = device.getLimits().maxComputeWorkgroupsPerDimension

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
        val uniformData = ByteBuffer.allocateDirect(manifest.uniformBytes).put(uniformBytes(source, shape, scalars)).also { it.flip() }
        gpu.onGpuThread {
            checkOpen()
            // Buffers from earlier runs are read now, before this run replaces or reallocates them.
            val sources = manifest.buffers.mapNotNull { p -> buffers.getValue(p.name).gpuBuffer?.let { p.name to it.gpuBuffer() } }.toMap()
            generation++
            val outputs = manifest.buffers.filter { it.isWritten && (readBack == null || it.name in readBack) }

            val start = System.nanoTime()
            pushErrorScopes()
            try {
                submit(size, uniformData, buffers, sources, outputs)
            } catch (e: Throwable) {
                popErrorScopes()
                throw e
            }
            val errors = popErrorScopes()
            if (errors.isNotEmpty()) throw GpuException("WebGPU rejected the run of ${source.name}: $errors")

            val results = LinkedHashMap<String, ReadBack>()
            try {
                for (p in outputs) {
                    val s = staging.getValue(p.name)
                    pump { s.mapAndAwait(MapMode.Read, 0, s.size) }
                    // Stays mapped: the result reads it in place. The next run of this kernel uses another buffer.
                    val result = ReadBack(s.getConstMappedRange(0, s.size))
                    staging.remove(p.name)
                    lent.addLast(Lent(p.name, s, WeakReference(result)))
                    results[p.name] = result
                }
            } catch (e: Exception) {
                if (isClosed || gpu.isClosed) throw IllegalStateException("${source.name} was closed during run()", e)
                gpu.lostReason?.let { throw GpuException("The GPU device was lost during run() of ${source.name} ($it)", e) }
                throw GpuException("Reading the results of ${source.name} failed: ${e.message}", e)
            }
            val handles = manifest.buffers.associate { p ->
                p.name to GpuBuffer(this@GpuKernel, p.name, (this@GpuKernel.buffers.getValue(p.name).size / 4).toInt(), p.type, generation)
            }
            GpuResult((System.nanoTime() - start) / 1e6, results, handles)
        }
    }

    /** Everything a run sends to the GPU: the uploads (see [WebGpu.upload]), then one command buffer. */
    private suspend fun submit(size: Int, uniformData: ByteBuffer, inputs: Map<String, GpuData>, sources: Map<String, GPUBuffer>, outputs: List<KernelManifest.Param>) {
        device.queue.writeBuffer(uniform, 0, uniformData)
        val retired = mutableListOf<GPUBuffer>()
        reclaimReadBacks(retired)
        // Buffers reused from an earlier run still hold its data, so empty() ones are cleared on the GPU.
        val clears = mutableListOf<GPUBuffer>()
        val copyIn = mutableListOf<Pair<GPUBuffer, GPUBuffer>>()
        for (p in manifest.buffers) {
            val data = inputs.getValue(p.name)
            val buffer = ensure(this.buffers, p.name, data.length * 4L, USAGE_STORAGE, "$label: ${p.name}", retired)
            val from = sources[p.name]
            when {
                from != null -> if (from != buffer) copyIn += from to buffer
                data.hasContents -> gpu.upload(buffer, data, "$label: ${p.name}")
                else -> clears += buffer
            }
        }
        for (p in outputs) {
            val bytes = this.buffers.getValue(p.name).size
            if (staging[p.name] == null) takeOldestReadBack(p.name, bytes)?.let { staging[p.name] = it }
            ensure(staging, p.name, bytes, USAGE_STAGING, "$label: ${p.name} (read back)", retired)
        }
        val group = bindGroup ?: device.createBindGroup(
            GPUBindGroupDescriptor(
                layout = layout,
                label = "$label: bind group",
                entries = (
                    listOf(GPUBindGroupEntry(binding = 0, buffer = uniform)) +
                        manifest.buffers.map { p -> GPUBindGroupEntry(binding = p.binding, buffer = this.buffers.getValue(p.name)) }
                    ).toTypedArray(),
            ),
        ).also { bindGroup = it }

        val (gx, gy) = workgroupGrid(size, manifest.workgroupSize, maxPerDimension)
        val encoder = device.createCommandEncoder(GPUCommandEncoderDescriptor(label = "$label: run"))
        for ((from, to) in copyIn) encoder.copyBufferToBuffer(from, 0, to, 0, to.size)
        for (b in clears) encoder.clearBuffer(b)
        val pass = encoder.beginComputePass(GPUComputePassDescriptor(label = "$label: run"))
        pass.setPipeline(pipeline)
        pass.setBindGroup(0, group)
        pass.dispatchWorkgroups(gx, gy)
        pass.end()
        pass.close()
        for (p in outputs) {
            val from = this.buffers.getValue(p.name)
            encoder.copyBufferToBuffer(from, 0, staging.getValue(p.name), 0, from.size)
        }
        val commands = encoder.finish()
        device.queue.submit(arrayOf(commands))
        commands.close()
        encoder.close()
        // Destroyed after the submit: WebGPU frees them once the GPU work that uses them is done.
        retired.forEach { it.destroy() }
    }

    /**
     * Takes back the read-back buffers whose results were dropped or copied by [GpuResult.bytes]: unmapped, they are
     * free again (spares are destroyed).
     */
    private fun reclaimReadBacks(retired: MutableList<GPUBuffer>) {
        val iterator = lent.iterator()
        while (iterator.hasNext()) {
            val l = iterator.next()
            if (l.result.get()?.isCopied() == false) continue
            iterator.remove()
            l.buffer.unmap()
            if (staging[l.name] == null) staging[l.name] = l.buffer else retired += l.buffer
        }
    }

    /**
     * When results that are still around hold [MAX_LENT] read-back buffers of [name], the oldest one is copied to
     * memory of its own and its buffer reused, so a loop of runs can't pile up GPU memory before the garbage
     * collector notices that its results are gone.
     */
    private fun takeOldestReadBack(name: String, bytes: Long): GPUBuffer? {
        if (lent.count { it.name == name } < MAX_LENT) return null
        val oldest = lent.first { it.name == name }
        lent.remove(oldest)
        oldest.result.get()?.keep()
        oldest.buffer.unmap()
        if (oldest.buffer.size == bytes) return oldest.buffer
        oldest.buffer.destroy()
        return null
    }

    /** Frees this kernel's GPU buffers. Later [run] calls throw [IllegalStateException]. Safe to call more than once. */
    override fun close() {
        if (isClosed || gpu.isClosed || gpu.isOnGpuThread) return closeOnGpuThread()
        // Dawn objects belong to the GPU thread.
        runBlocking { gpu.onGpuThread { closeOnGpuThread() } }
    }

    internal fun closeOnGpuThread() {
        if (isClosed) return
        isClosed = true
        // Results of this kernel stay readable: their data is copied before the read-back buffers go.
        for (l in lent) {
            l.result.get()?.keep()
            l.buffer.destroy()
        }
        lent.clear()
        for (b in buffers.values) b.destroy()
        for (b in staging.values) b.destroy()
        uniform.destroy()
        buffers.clear()
        staging.clear()
        bindGroup = null
        gpu.forget(this)
    }

    internal fun gpuBuffer(name: String): GPUBuffer = buffers.getValue(name)

    private fun checkOpen() {
        check(!isClosed) { "GpuKernel ${source.name} was closed; load it again with WebGpu.load()" }
        gpu.checkUsable()
    }

    private fun ensure(map: HashMap<String, GPUBuffer>, name: String, bytes: Long, usage: Int, label: String, retired: MutableList<GPUBuffer>): GPUBuffer {
        val existing = map[name]
        if (existing != null && existing.size == bytes) return existing
        existing?.let { retired += it }
        bindGroup = null
        return device.createBuffer(GPUBufferDescriptor(usage = usage, size = bytes, label = label)).also { map[name] = it }
    }
}

private const val USAGE_UNIFORM = BufferUsage.Uniform or BufferUsage.CopyDst
private const val USAGE_STORAGE = BufferUsage.Storage or BufferUsage.CopySrc or BufferUsage.CopyDst
private const val USAGE_STAGING = BufferUsage.MapRead or BufferUsage.CopyDst
private const val USAGE_UPLOAD = BufferUsage.MapWrite or BufferUsage.CopySrc

/** Bytes per upload chunk: big enough that per-chunk overhead doesn't matter, small enough not to cost much memory. */
private const val UPLOAD_CHUNK = 16L shl 20

/** Read-back buffers per kernel parameter that results may hold mapped before the oldest is copied out. */
private const val MAX_LENT = 2

private class Lent(val name: String, val buffer: GPUBuffer, val result: WeakReference<ReadBack>)

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
