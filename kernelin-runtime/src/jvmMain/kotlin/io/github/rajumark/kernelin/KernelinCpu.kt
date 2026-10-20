package io.github.rajumark.kernelin

import java.util.concurrent.Executors
import java.util.concurrent.Future

/** On the JVM, kernels can also run on all cores with [dispatchParallel]. */
public actual object KernelinCpu {
    // A mutable cell per thread avoids boxing an Int on every globalId() call.
    private class IdCell(var value: Int = -1, var width: Int = 0, var height: Int = 0)

    private val current = ThreadLocal.withInitial { IdCell() }

    private val pool by lazy {
        Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors()) { r ->
            Thread(r, "kernelin-cpu").apply { isDaemon = true }
        }
    }

    internal actual fun currentId(caller: String): Int {
        val id = current.get().value
        check(id >= 0) { "$caller() was called outside a Kernelin dispatch" }
        return id
    }

    internal actual fun currentWidth(): Int = current.get().width

    internal actual fun currentHeight(): Int = current.get().height

    /** Runs [kernel] once for every id in `0 until size`, on the calling thread. */
    public actual fun dispatch(size: Int, kernel: () -> Unit): Unit = run(0, size, size, 1, kernel)

    /** Runs [kernel] once for every cell of a `width x height` grid, row by row, on the calling thread. */
    public actual fun dispatch(width: Int, height: Int, kernel: () -> Unit): Unit =
        run(0, launchSize(width, height, 1), width, height, kernel)

    /** Runs [kernel] once for every cell of a `width x height x depth` grid, row by row, layer by layer. */
    public actual fun dispatch(width: Int, height: Int, depth: Int, kernel: () -> Unit): Unit =
        run(0, launchSize(width, height, depth), width, height, kernel)

    /** Like [dispatch], but splits the ids across all CPU cores. */
    public fun dispatchParallel(size: Int, kernel: () -> Unit): Unit = runParallel(size, size, 1, kernel)

    /** Like the 2D [dispatch], but splits the cells across all CPU cores. */
    public fun dispatchParallel(width: Int, height: Int, kernel: () -> Unit): Unit =
        runParallel(launchSize(width, height, 1), width, height, kernel)

    /** Like the 3D [dispatch], but splits the cells across all CPU cores. */
    public fun dispatchParallel(width: Int, height: Int, depth: Int, kernel: () -> Unit): Unit =
        runParallel(launchSize(width, height, depth), width, height, kernel)

    private fun run(start: Int, end: Int, width: Int, height: Int, kernel: () -> Unit) {
        val cell = current.get()
        cell.width = width
        cell.height = height
        try {
            for (i in start until end) {
                cell.value = i
                kernel()
            }
        } finally {
            cell.value = -1
        }
    }

    private fun runParallel(size: Int, width: Int, height: Int, kernel: () -> Unit) {
        val workers = Runtime.getRuntime().availableProcessors().coerceAtMost(size.coerceAtLeast(1))
        val chunk = (size + workers - 1) / workers
        val futures = ArrayList<Future<*>>(workers)
        for (w in 0 until workers) {
            val start = w * chunk
            val end = minOf(size, start + chunk)
            if (start >= end) break
            futures += pool.submit { run(start, end, width, height, kernel) }
        }
        futures.forEach { it.get() }
    }
}
