package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelLaunch
import io.github.rajumark.kernelin.KernelinCpu
import kotlin.coroutines.cancellation.CancellationException

/**
 * Runs kernels on the GPU when there is one, and on the CPU ([KernelinCpu]) when there isn't, with one call:
 *
 * ```
 * val compute = Compute.create()                        // the GPU if it can be opened, else the CPU
 * compute.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }   // ys holds the result either way
 * ```
 *
 * The launch block is a typed kernel call, checked by the compiler like `WebGpu.dispatch`. On the CPU the same call
 * runs for every id (on all cores on the JVM and Android), so code doesn't need a second path for devices without a
 * GPU. If the GPU device is lost later (a driver reset, the browser taking it away), the run that hit it and every
 * run after it use the CPU; [cpuReason] says why.
 *
 * Results are the same on both, within the documented CPU/GPU differences (Float rounding). Use [WebGpu] directly
 * when you need data that stays on the GPU between runs.
 */
public class Compute private constructor(
    private var gpu: GpuLauncher?,
    reason: String?,
) : AutoCloseable {
    /** Entry points. */
    public companion object {
        /**
         * Opens the GPU, or falls back to the CPU when there is none (no WebGPU, no adapter, an emulator, ...).
         * With [useGpu] false it always uses the CPU, which is handy for tests and for comparing results.
         */
        public suspend fun create(useGpu: Boolean = true): Compute {
            if (!useGpu) return Compute(null, "the CPU was chosen (useGpu = false)")
            return try {
                Compute(openGpu(), null)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Compute(null, "no GPU: ${e.message ?: e::class.simpleName}")
            }
        }
    }

    /** Why kernels run on the CPU, or null while they run on the GPU. */
    public var cpuReason: String? = reason
        private set

    /** True while kernels run on the GPU. */
    public val isGpu: Boolean get() = gpu != null

    /** Where kernels run, for logs: `GPU: <name>` or `CPU (<reason>)`. */
    public val description: String get() = gpu?.let { "GPU: ${it.description}" } ?: "CPU ($cpuReason)"

    /**
     * Runs one kernel call for ids `0 until size`, on the GPU or the CPU (see [isGpu]). Afterwards the arrays the
     * kernel writes hold the results, like with `KernelinCpu.dispatch(n) { saxpy(...) }`.
     *
     * Throws [IllegalArgumentException] for wrong arguments and the GPU host's exceptions for GPU errors other than
     * a lost device (those are bugs to fix, not missing hardware, so they aren't hidden by running on the CPU).
     */
    public suspend fun dispatch(size: Int, kernel: KernelLaunch.() -> Unit) {
        val launch = KernelLaunch.capture(kernel)
        require(size >= 0) { "size must be 0 or more, but was $size" }
        val current = gpu
        if (current != null) {
            val lost = current.launch(size, launch) ?: return
            fallBack("GPU device lost: $lost")
        }
        launch.runOnCpu(size)
    }

    /** Frees the GPU (if any). Later runs use the CPU. */
    override fun close() {
        if (gpu != null) fallBack("closed")
    }

    private fun fallBack(reason: String) {
        gpu?.close()
        gpu = null
        cpuReason = reason
    }
}

/** What [Compute] needs from a platform's GPU host. */
internal interface GpuLauncher : AutoCloseable {
    val description: String

    /** Runs [launch] and copies results into its arrays; returns null, or the reason when the device was lost. */
    suspend fun launch(size: Int, launch: KernelLaunch): String?
}

/** Opens this platform's GPU host. Throws when there is no usable GPU. */
internal expect suspend fun openGpu(): GpuLauncher
