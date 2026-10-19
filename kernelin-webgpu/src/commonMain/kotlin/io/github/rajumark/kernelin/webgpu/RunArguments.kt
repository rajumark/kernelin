package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelManifest
import io.github.rajumark.kernelin.KernelSource

/** What [checkRunArguments] needs to know about one buffer argument, on any platform. */
internal class BufferArgument(
    /** The WGSL element type of the data (`i32` / `f32`), or null when any buffer type fits. */
    val type: String?,
    /** False when the data is a GPU buffer of another device. */
    val sameDevice: Boolean = true,
)

/**
 * Checks the arguments of `GpuKernel.run` before anything reaches the GPU: unknown names, missing values and values
 * of the wrong type throw [IllegalArgumentException]. Shared by every GPU host, so they all reject the same calls.
 */
internal fun checkRunArguments(
    source: KernelSource,
    scalars: Map<String, Any>,
    buffers: Map<String, BufferArgument>,
    readBack: Set<String>?,
) {
    val manifest = source.manifest
    fun unknown(kind: String, name: String, known: List<KernelManifest.Param>): Nothing =
        throw IllegalArgumentException(
            "Unknown $kind parameter '$name' for ${source.name}; its $kind parameters are: " +
                known.joinToString { it.name }.ifEmpty { "(none)" },
        )
    scalars.keys.firstOrNull { name -> manifest.scalars.none { it.name == name } }?.let { unknown("scalar", it, manifest.scalars) }
    buffers.keys.firstOrNull { name -> manifest.buffers.none { it.name == name } }?.let { unknown("buffer", it, manifest.buffers) }
    for (p in manifest.scalars) {
        val v = scalars[p.name] ?: throw IllegalArgumentException("Missing scalar parameter '${p.name}'")
        val ok = when (p.type) {
            "f32" -> v is Number
            "bool" -> v is Boolean
            else -> v is Int
        }
        if (!ok) {
            throw IllegalArgumentException("Scalar parameter '${p.name}' is ${kotlinType(p.type)}, but got ${v::class.simpleName} ($v)")
        }
    }
    for (p in manifest.buffers) {
        val data = buffers[p.name] ?: throw IllegalArgumentException("Missing buffer parameter '${p.name}'")
        if (data.type != null && data.type != p.type) {
            throw IllegalArgumentException(
                "Buffer parameter '${p.name}' is ${kotlinType(p.type)}Array, but got ${kotlinType(data.type)}Array data",
            )
        }
        if (!data.sameDevice) throw IllegalArgumentException("Buffer parameter '${p.name}' got a GpuBuffer from another WebGpu device")
    }
    readBack?.firstOrNull { name -> manifest.buffers.none { it.name == name && it.isWritten } }?.let {
        throw IllegalArgumentException(
            "readBack: '$it' is not a buffer that ${source.name} writes; those are: " +
                manifest.buffers.filter { b -> b.isWritten }.joinToString { b -> b.name }.ifEmpty { "(none)" },
        )
    }
}

/** The Kotlin name of a WGSL parameter type, for error messages. */
internal fun kotlinType(wgsl: String): String = when (wgsl) {
    "f32" -> "Float"
    "bool" -> "Boolean"
    else -> "Int"
}

/**
 * Workgroups to dispatch for [size] threads, as (x, y): one dimension holds at most [maxPerDimension] workgroups,
 * so big runs spill into y. The shader turns (x, y) back into one id and skips ids at or above `size`.
 */
internal fun workgroupGrid(size: Int, workgroupSize: Int, maxPerDimension: Int): Pair<Int, Int> {
    val groups = (size + workgroupSize - 1) / workgroupSize
    val x = minOf(groups, maxPerDimension).coerceAtLeast(1)
    return x to (groups + x - 1) / x
}

/**
 * The bytes of the uniform buffer: the thread count, then each scalar parameter in manifest order, as
 * little-endian 32-bit values, padded to `manifest.uniformBytes`. [scalars] must have passed [checkRunArguments].
 */
internal fun uniformBytes(source: KernelSource, size: Int, scalars: Map<String, Any>): ByteArray {
    val manifest = source.manifest
    val bytes = ByteArray(manifest.uniformBytes)
    fun put(offset: Int, bits: Int) {
        for (i in 0 until 4) bytes[offset + i] = (bits ushr (8 * i)).toByte()
    }
    put(0, size)
    manifest.scalars.forEachIndexed { i, p ->
        val v = scalars.getValue(p.name)
        put(
            4 + i * 4,
            when (p.type) {
                "f32" -> (v as Number).toFloat().toRawBits()
                "bool" -> if (v as Boolean) 1 else 0
                else -> (v as Number).toInt()
            },
        )
    }
    return bytes
}
