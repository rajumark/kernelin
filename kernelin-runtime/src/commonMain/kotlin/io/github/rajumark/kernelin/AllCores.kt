package io.github.rajumark.kernelin

/** [KernelinCpu.dispatch] on every CPU core where the platform has threads (JVM, Android), else on this thread. */
internal expect fun dispatchOnAllCores(size: Int, kernel: () -> Unit)
