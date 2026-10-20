package io.github.rajumark.kernelin

internal actual fun dispatchOnAllCores(size: Int, kernel: () -> Unit) = KernelinCpu.dispatchParallel(size, kernel)
