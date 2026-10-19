package io.github.rajumark.kernelin

import kotlin.native.concurrent.ThreadLocal

// One id per thread, so kernels dispatched from different threads never see each other's ids.
@ThreadLocal
private var current = -1

/** On Kotlin/Native, the current id is kept per thread. */
public actual object KernelinCpu {
    internal actual fun currentId(): Int {
        check(current >= 0) { "globalId() was called outside a Kernelin dispatch" }
        return current
    }

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
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
