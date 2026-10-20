package io.github.rajumark.kernelin

internal actual fun dispatchOnAllCores(width: Int, height: Int, depth: Int, kernel: () -> Unit) =
    KernelinCpu.dispatchParallel(width, height, depth, kernel)
