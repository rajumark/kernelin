package io.github.rajumark.kernelin.compiler

import io.github.rajumark.kernelin.KernelLaunch
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertSame
import kotlin.test.assertTrue

/**
 * Typed launches: `gpu.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }`. The kernel call is checked by the normal
 * Kotlin compiler, and the plugin turns it into a [KernelLaunch] that records the shader and the arguments.
 * These tests compile real code, run it on the JVM and look at what was recorded.
 */
class KernelLaunchTest {
    private val kernels = """
        import io.github.rajumark.kernelin.*

        @Kernel
        fun saxpy(alpha: Float, x: FloatArray, y: FloatArray) {
            val i = globalId()
            y[i] = alpha * x[i] + y[i]
        }

        @Kernel
        fun mixed(on: Boolean, counts: IntArray, times: Int, out: FloatArray) {
            val i = globalId()
            if (on) out[i] = (counts[i] * times).toFloat()
        }

        @Kernel
        fun cells(out: IntArray) {
            out[globalId()] = globalIdX() + globalIdY() * 10 + globalIdZ() * 100
        }

        // Stands in for a GPU host such as WebGpu.dispatch.
        fun launch(block: KernelLaunch.() -> Unit): KernelLaunch = KernelLaunch.capture(block)
    """.trimIndent()

    private fun compile(caller: String): Compilation =
        KernelinCompiler.compileFiles(mapOf("Kernels.kt" to kernels, "Caller.kt" to "import io.github.rajumark.kernelin.*\n\n${caller.trimIndent()}\n"))

    private fun compileOk(caller: String): Compilation = compile(caller).also { assertTrue(it.ok, it.output) }

    // One class loader per compilation, so top-level values are the same objects in every call.
    private val callers = HashMap<Compilation, Class<*>>()

    private fun Compilation.call(function: String): Any? =
        callers.getOrPut(this) { loadClass("CallerKt") }.getMethod(function).invoke(null)

    @Test
    fun `a kernel call in a launch block records the shader and every argument by name`() {
        val c = compileOk(
            """
            val xs = floatArrayOf(1f, 2f)
            val ys = floatArrayOf(3f, 4f)
            fun positional() = launch { saxpy(2f, xs, ys) }
            fun named() = launch { saxpy(y = ys, alpha = 0.5f, x = xs) }
            fun arrays() = arrayOf(xs, ys)
            """,
        )
        val (xs, ys) = c.call("arrays") as Array<*>
        for ((function, alpha) in listOf("positional" to 2f, "named" to 0.5f)) {
            val launch = c.call(function) as KernelLaunch
            assertEquals("saxpy", launch.source.name)
            assertEquals(c.wgsl("saxpy"), launch.source.wgsl)
            assertEquals(mapOf<String, Any>("alpha" to alpha), launch.scalars)
            assertEquals(listOf("x", "y"), launch.arrays.keys.toList())
            // The same arrays, not copies: hosts write the results back into them.
            assertSame(xs, launch.arrays["x"])
            assertSame(ys, launch.arrays["y"])
        }
        // The kernel itself never ran on the CPU.
        assertEquals(listOf(3f, 4f), (ys as FloatArray).toList())
    }

    @Test
    fun `every parameter type is recorded and arguments keep Kotlin's evaluation order`() {
        val c = compileOk(
            """
            var order = ""
            fun <T> step(name: String, value: T): T { order += name; return value }
            fun run(): List<Any> {
                val launch = launch {
                    mixed(out = step("out", FloatArray(3)), times = step("times", 4), counts = step("counts", intArrayOf(1, 2, 3)), on = step("on", true))
                }
                return listOf(order, launch.scalars, launch.arrays.keys.toList(), (launch.arrays["counts"] as IntArray).toList())
            }
            """,
        )
        val (order, scalars, arrays, counts) = c.call("run") as List<*>
        assertEquals("outtimescountson", order)
        assertEquals(mapOf<String, Any>("on" to true, "times" to 4), scalars)
        assertEquals(listOf("counts", "out"), arrays)
        assertEquals(listOf(1, 2, 3), counts)
    }

    @Test
    fun `a recorded launch runs on the cpu with the same values, evaluated once`() {
        val c = compileOk(
            """
            var evaluations = 0
            fun run(): List<Any> {
                val ys = floatArrayOf(1f, 1f, 1f)
                val launch = launch { saxpy(alpha = 3f, x = floatArrayOf(1f, 2f, 3f).also { evaluations++ }, y = ys) }
                launch.runOnCpu(3)
                val outs = FloatArray(3)
                val counts = intArrayOf(1, 2, 3)
                launch { mixed(on = true, counts = counts, times = 5, out = outs) }.runOnCpu(3)
                val cells = IntArray(12)
                launch { cells(out = cells) }.runOnCpu(4, 3)
                val cells3D = IntArray(12)
                launch { cells(out = cells3D) }.runOnCpu(2, 3, 2)
                return listOf(ys.toList(), outs.toList(), evaluations, cells.toList(), cells3D.toList())
            }
            """,
        )
        val (ys, outs, evaluations, cells, cells3D) = c.call("run") as List<*>
        assertEquals(List(12) { it % 4 + it / 4 * 10 }, cells, "2D: x + 10 * y")
        assertEquals(List(12) { it % 2 + it / 2 % 3 * 10 + it / 6 * 100 }, cells3D, "3D: x + 10 * y + 100 * z")
        assertEquals(listOf(4f, 7f, 10f), ys)
        assertEquals(listOf(5f, 10f, 15f), outs)
        assertEquals(1, evaluations, "Arguments are evaluated once, not once per thread")
    }

    @Test
    fun `kernel calls outside launch blocks still run on the cpu`() {
        val c = compileOk(
            """
            fun cpu(): List<Float> {
                val ys = floatArrayOf(1f, 1f)
                KernelinCpu.dispatch(2) { saxpy(3f, floatArrayOf(1f, 2f), ys) }
                return ys.toList()
            }
            """,
        )
        assertEquals(listOf(4f, 7f), c.call("cpu"))
    }

    @Test
    fun `wrong names, types and counts are compile errors`() {
        val c = compile(
            """
            val xs = floatArrayOf(1f)
            fun typo() = launch { saxpy(alpah = 2f, x = xs, y = xs) }
            fun wrongType() = launch { saxpy(2.0, xs, xs) }
            fun wrongArray() = launch { saxpy(2f, intArrayOf(1), xs) }
            fun missing() = launch { saxpy(2f, xs) }
            """,
        )
        assertFalse(c.ok)
        val text = c.errors.joinToString("\n")
        assertContains(text, "alpah")
        assertContains(text, "Caller.kt:5:")
        assertContains(text, "Caller.kt:6:")
        assertContains(text, "Caller.kt:7:")
        assertFalse("Kernelin" in text, "Plain Kotlin errors, no extra Kernelin noise:\n$text")
    }

    @Test
    fun `a launch block holds exactly one kernel call`() {
        val c = compile(
            """
            val xs = floatArrayOf(1f)
            fun twoStatements() = launch {
                val alpha = 2f
                saxpy(alpha, xs, xs)
            }
            fun twoKernels() = launch {
                saxpy(1f, xs, xs)
                saxpy(2f, xs, xs)
            }
            fun notAKernel() = launch { println("hi") }
            fun empty() = launch { }
            """,
        )
        assertFalse(c.ok)
        val errors = c.errors.filter { "a launch block must be exactly one call to a @Kernel function" in it }
        assertEquals(4, errors.size, c.output)
        // Each error points at the statement that doesn't belong (or at the empty block).
        for (line in listOf(5, 10, 12, 13)) assertTrue(errors.any { "Caller.kt:$line:" in it }, "No error on line $line:\n${c.output}")
    }

    @Test
    fun `a kernel with translation errors is reported once, not again at the launch`() {
        val c = KernelinCompiler.compile(
            """
            import io.github.rajumark.kernelin.*

            @Kernel
            fun broken(out: IntArray) {
                out[globalId()] = 1 / 0
            }

            fun go() = KernelLaunch.capture { broken(IntArray(1)) }
            """.trimIndent(),
        )
        assertFalse(c.ok)
        assertEquals(1, c.errors.size, c.output)
        assertContains(c.errors.single(), "(in @Kernel broken)")
    }
}
