package io.github.rajumark.kernelin

/** Wasm runs on one thread, so the current id is a plain field. */
actual object KernelinCpu {
    private var current = -1

    internal actual fun currentId(): Int {
        check(current >= 0) { "globalId() was called outside a Kernelin dispatch" }
        return current
    }

    actual fun dispatch(size: Int, kernel: () -> Unit) {
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
