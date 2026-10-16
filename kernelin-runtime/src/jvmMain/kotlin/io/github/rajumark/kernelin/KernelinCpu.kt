package io.github.rajumark.kernelin

import java.util.concurrent.Executors
import java.util.concurrent.Future

/** On the JVM, kernels can also run on all cores with [dispatchParallel]. */
public actual object KernelinCpu {
    // A mutable cell per thread avoids boxing an Int on every globalId() call.
    private class IdCell(var value: Int = -1)

    private val current = ThreadLocal.withInitial { IdCell() }

    private val pool by lazy {
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()) { r ->
            Thread(r, "kernelin-cpu").apply { isDaemon = true }
        }
    }

    internal actual fun currentId(): Int {
        val id = current.get().value
        check(id >= 0) { "globalId() was called outside a Kernelin dispatch" }
        return id
    }

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
    public actual fun dispatch(size: Int, kernel: () -> Unit) {
        val cell = current.get()
        try {
            for (i in 0 until size) {
                cell.value = i
                kernel()
            }
        } finally {
            cell.value = -1
        }
    }

    /** Like [dispatch], but splits the ids across all CPU cores. */
    public fun dispatchParallel(size: Int, kernel: () -> Unit) {
        val workers = Runtime.getRuntime().availableProcessors().coerceAtMost(size.coerceAtLeast(1))
        val chunk = (size + workers - 1) / workers
        val futures = ArrayList<Future<*>>(workers)
        for (w in 0 until workers) {
            val start = w * chunk
            val end = minOf(size, start + chunk)
            if (start >= end) break
            futures += pool.submit {
                val cell = current.get()
                try {
                    for (i in start until end) {
                        cell.value = i
                        kernel()
                    }
                } finally {
                    cell.value = -1
                }
            }
        }
        futures.forEach { it.get() }
    }
}
