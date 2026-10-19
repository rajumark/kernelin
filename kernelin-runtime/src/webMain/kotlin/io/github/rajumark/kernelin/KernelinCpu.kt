package io.github.rajumark.kernelin

/** JS and Wasm run on one thread, so the current id is a plain field. */
public actual object KernelinCpu {
    private var current = -1

    internal actual fun currentId(): Int {
        check(current >= 0) { "globalId() was called outside a Kernelin dispatch" }
        return current
    }

    /** Runs [kernel] once for every id in `0 until size`. */
    public actual fun dispatch(size: Int, kernel: () -> Unit) {
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
