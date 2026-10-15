package io.github.kernelin

/**
 * Marks a function as a GPU kernel.
 *
 * The Kernelin compiler plugin translates the function body to WGSL at compile time.
 * The function stays normal Kotlin, so it can also run on the CPU with [KernelinCpu].
 */
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.BINARY)
annotation class Kernel

/**
 * Index of the current thread inside a dispatch.
 *
 * On the GPU this is `global_invocation_id.x`. On the CPU it is set by [KernelinCpu.dispatch].
 */
fun globalId(): Int = KernelinCpu.currentId()
