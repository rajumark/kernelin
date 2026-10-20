package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertEquals

class KernelManifestTest {
    // Same format the compiler plugin writes.
    private val json = """
        {
          "kernel": "io.github.rajumark.kernelin.samples.saxpy",
          "entryPoint": "main",
          "workgroupSize": 64,
          "uniform": { "binding": 0, "fields": [
            { "name": "kernelin_size", "type": "u32" },
            { "name": "alpha", "type": "f32" }
          ] },
          "params": [
            { "name": "alpha", "kind": "scalar", "type": "f32" },
            { "name": "x", "kind": "buffer", "type": "f32", "binding": 1, "access": "read" },
            { "name": "y", "kind": "buffer", "type": "f32", "binding": 2, "access": "read_write" }
          ]
        }
    """.trimIndent()

    @Test
    fun parsesParamsInOrder() {
        val m = KernelManifest.parse(json)
        assertEquals("io.github.rajumark.kernelin.samples.saxpy", m.kernel)
        assertEquals(64, m.workgroupSize)
        assertEquals(listOf("alpha", "x", "y"), m.params.map { it.name })
        assertEquals(listOf("x", "y"), m.buffers.map { it.name })
        assertEquals(listOf(1, 2), m.buffers.map { it.binding })
        assertEquals(listOf(false, true), m.buffers.map { it.isWritten })
        assertEquals(16, m.uniformBytes)
    }

    @Test
    fun uniformIsPaddedTo16Bytes() {
        val seven = json.replace(
            "{ \"name\": \"alpha\", \"kind\": \"scalar\", \"type\": \"f32\" },",
            (1..6).joinToString("") { "{ \"name\": \"s$it\", \"kind\": \"scalar\", \"type\": \"i32\" }," },
        )
        assertEquals(32, KernelManifest.parse(seven).uniformBytes)
    }

    @Test
    fun gridIdsAddWidthAndHeightBeforeTheScalars() {
        val plain = KernelManifest.parse(json)
        assertEquals(false, plain.usesGridIds)
        assertEquals(4, plain.scalarOffset)
        val grid = KernelManifest.parse(json.replace("\"workgroupSize\": 64,", "\"workgroupSize\": 64, \"gridIds\": true,"))
        assertEquals(true, grid.usesGridIds)
        assertEquals(12, grid.scalarOffset)
        assertEquals(16, grid.uniformBytes)
        val twoScalars = KernelManifest("k", "main", 64, plain.params + plain.params.first(), usesGridIds = true)
        assertEquals(32, twoScalars.uniformBytes)
    }
}
