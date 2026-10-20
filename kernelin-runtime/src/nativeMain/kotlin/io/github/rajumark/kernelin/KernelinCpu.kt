package io.github.rajumark.kernelin

import kotlin.native.concurrent.ThreadLocal

// One id per thread, so kernels dispatched from different threads never see each other's ids.
@ThreadLocal
private var current = -1

@ThreadLocal
private var currentWidth = 0

@ThreadLocal
private var currentHeight = 0

/** On Kotlin/Native, the current id is kept per thread. */
public actual object KernelinCpu {
    internal actual fun currentId(caller: String): Int {
        check(current >= 0) { "$caller() was called outside a Kernelin dispatch" }
        return current
    }

    internal actual fun currentWidth(): Int = currentWidth

    internal actual fun currentHeight(): Int = currentHeight

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
    public actual fun dispatch(size: Int, kernel: () -> Unit): Unit = run(size, size, 1, kernel)

    /** Runs [kernel] once for every cell of a `width x height` grid, row by row, on the calling thread. */
    public actual fun dispatch(width: Int, height: Int, kernel: () -> Unit): Unit =
        run(launchSize(width, height, 1), width, height, kernel)

    /** Runs [kernel] once for every cell of a `width x height x depth` grid, row by row, layer by layer. */
    public actual fun dispatch(width: Int, height: Int, depth: Int, kernel: () -> Unit): Unit =
        run(launchSize(width, height, depth), width, height, kernel)

    private fun run(size: Int, width: Int, height: Int, kernel: () -> Unit) {
        currentWidth = width
        currentHeight = height
        try {
            for (i in 0 until size) {
                current = i
                kernel()
            }
        } finally {
            current = -1
        }
    }
}
