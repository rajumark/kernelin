package io.github.rajumark.kernelin

/** Runs kernels on the CPU. Used for unit tests and as the benchmark baseline. */
public expect object KernelinCpu {
    /** The current thread id; [caller] names the id function for the error outside a dispatch. */
    internal fun currentId(caller: String): Int

    /** Width of the current launch (the size, for a 1D launch). */
    internal fun currentWidth(): Int

    /** Height of the current launch (1, for a 1D launch). */
    internal fun currentHeight(): Int

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
    public fun dispatch(size: Int, kernel: () -> Unit)

    /**
     * Runs [kernel] once for every cell of a `width x height` grid, row by row, on the calling thread. Inside,
     * [globalIdX] is the column, [globalIdY] the row and [globalId] the row-major index `x + y * width`.
     */
    public fun dispatch(width: Int, height: Int, kernel: () -> Unit)

    /** Like the 2D [dispatch], for a `width x height x depth` grid; [globalIdZ] is the layer. */
    public fun dispatch(width: Int, height: Int, depth: Int, kernel: () -> Unit)
}
