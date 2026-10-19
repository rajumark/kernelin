package io.github.rajumark.kernelin

import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame

/** [KernelLaunch.capture] on its own. The compiler tests cover the code the plugin generates. */
@OptIn(KernelinGeneratedApi::class)
class KernelLaunchTest {
    private val saxpy = KernelSource(
        "saxpy",
        "// wgsl",
        """
        { "kernel": "saxpy", "entryPoint": "main", "workgroupSize": 64,
          "params": [
            { "name": "alpha", "kind": "scalar", "type": "f32" },
            { "name": "x", "kind": "buffer", "type": "f32", "binding": 1, "access": "read" },
            { "name": "y", "kind": "buffer", "type": "f32", "binding": 2, "access": "read_write" },
            { "name": "on", "kind": "scalar", "type": "bool" }
          ] }
        """,
    )

    @Test
    fun recordedCallIsSplitIntoScalarsAndArrays() {
        val x = floatArrayOf(1f)
        val y = floatArrayOf(2f)
        val launch = KernelLaunch.capture {
            argument(0, 2f)
            argument(1, x)
            argument(2, y)
            argument(3, true)
            kernel(saxpy)
        }
        assertSame(saxpy, launch.source)
        assertEquals(mapOf<String, Any>("alpha" to 2f, "on" to true), launch.scalars)
        assertEquals(listOf("x", "y"), launch.arrays.keys.toList())
        assertSame(y, launch.arrays["y"])
    }

    @Test
    fun blockWithoutAKernelCallFailsClearly() {
        val e = assertFailsWith<IllegalStateException> { KernelLaunch.capture { } }
        assertContains(e.message!!, "No @Kernel call was recorded")
        // Without the plugin, the kernel body runs right here and its globalId() fails.
        val e2 = assertFailsWith<IllegalStateException> { KernelLaunch.capture { globalId() } }
        assertContains(e2.message!!, "compiler plugin must be applied")
        // Other errors in the block are the caller's own and pass through unchanged.
        val e3 = assertFailsWith<IllegalStateException> { KernelLaunch.capture { error("boom") } }
        assertEquals("boom", e3.message)
    }

    @Test
    fun twoKernelCallsOrMissingArgumentsAreRejected() {
        val e = assertFailsWith<IllegalStateException> {
            KernelLaunch.capture {
                kernel(saxpy)
                kernel(saxpy)
            }
        }
        assertContains(e.message!!, "exactly one")
        val e2 = assertFailsWith<IllegalStateException> {
            KernelLaunch.capture {
                argument(0, 1f)
                kernel(saxpy)
            }
        }
        assertContains(e2.message!!, "1 of 4 arguments")
    }
}
