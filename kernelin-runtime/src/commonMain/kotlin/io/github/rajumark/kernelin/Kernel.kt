package io.github.rajumark.kernelin

/**
 * Marks a function as a GPU kernel.
 *
 * The Kernelin compiler plugin translates the function body to WGSL at compile time.
 * The function stays normal Kotlin, so it can also run on the CPU with [KernelinCpu].
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.BINARY)
public annotation class Kernel

/**
 * Index of the current thread inside a dispatch.
 *
 * On the GPU this is `global_invocation_id.x`. On the CPU it is set by [KernelinCpu.dispatch].
 * In a 2D or 3D launch it counts row by row: `globalIdX() + globalIdY() * width + globalIdZ() * width * height`,
 * which is the index of the thread's cell in a row-major array.
 */
public fun globalId(): Int = KernelinCpu.currentId("globalId")

/**
 * Column of the current thread: `0 until width` in a 2D or 3D launch like `KernelinCpu.dispatch(width, height)`.
 * In a 1D launch it equals [globalId].
 */
public fun globalIdX(): Int = KernelinCpu.currentId("globalIdX") % KernelinCpu.currentWidth()

/** Row of the current thread: `0 until height` in a 2D or 3D launch. Always 0 in a 1D launch. */
public fun globalIdY(): Int = KernelinCpu.currentId("globalIdY") / KernelinCpu.currentWidth() % KernelinCpu.currentHeight()

/** Layer of the current thread: `0 until depth` in a 3D launch. Always 0 in a 1D or 2D launch. */
public fun globalIdZ(): Int = KernelinCpu.currentId("globalIdZ") / (KernelinCpu.currentWidth() * KernelinCpu.currentHeight())

/** The thread count of a `width x height x depth` launch. Throws for negative sizes or more than Int.MAX_VALUE threads. */
internal fun launchSize(width: Int, height: Int, depth: Int): Int {
    require(width >= 0 && height >= 0 && depth >= 0) { "Launch sizes can't be negative, but got $width x $height x $depth" }
    val size = width.toLong() * height * depth
    require(size <= Int.MAX_VALUE) { "A launch has at most ${Int.MAX_VALUE} threads, but $width x $height x $depth is $size" }
    return size.toInt()
}
