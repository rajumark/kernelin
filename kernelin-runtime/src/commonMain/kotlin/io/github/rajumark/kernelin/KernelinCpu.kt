package io.github.rajumark.kernelin

/** Runs kernels on the CPU. Used for unit tests and as the benchmark baseline. */
public expect object KernelinCpu {
    internal fun currentId(): Int

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
    public fun dispatch(size: Int, kernel: () -> Unit)
}
