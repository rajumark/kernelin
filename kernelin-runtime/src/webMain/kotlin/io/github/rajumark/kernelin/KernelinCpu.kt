package io.github.rajumark.kernelin

/** JS and Wasm run on one thread, so the current id is a plain field. */
public actual object KernelinCpu {
    private var current = -1
    private var width = 0
    private var height = 0

    internal actual fun currentId(caller: String): Int {
        check(current >= 0) { "$caller() was called outside a Kernelin dispatch" }
        return current
    }

    internal actual fun currentWidth(): Int = width

    internal actual fun currentHeight(): Int = height

    /** Runs [kernel] once for every id in `0 until size`. */
    public actual fun dispatch(size: Int, kernel: () -> Unit): Unit = run(size, size, 1, kernel)

    /** Runs [kernel] once for every cell of a `width x height` grid, row by row. */
    public actual fun dispatch(width: Int, height: Int, kernel: () -> Unit): Unit =
        run(launchSize(width, height, 1), width, height, kernel)

    /** Runs [kernel] once for every cell of a `width x height x depth` grid, row by row, layer by layer. */
    public actual fun dispatch(width: Int, height: Int, depth: Int, kernel: () -> Unit): Unit =
        run(launchSize(width, height, depth), width, height, kernel)

    private fun run(size: Int, width: Int, height: Int, kernel: () -> Unit) {
        this.width = width
        this.height = height
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
