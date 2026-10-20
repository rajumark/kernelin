package io.github.rajumark.kernelin

/**
 * [KernelinCpu.dispatch] of a `width x height x depth` grid on every CPU core where the platform has threads (JVM,
 * Android), else on this thread. A 1D launch is `size x 1 x 1`.
 */
internal expect fun dispatchOnAllCores(width: Int, height: Int, depth: Int, kernel: () -> Unit)
