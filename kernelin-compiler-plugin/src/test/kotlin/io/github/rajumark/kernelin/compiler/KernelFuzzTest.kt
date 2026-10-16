package io.github.rajumark.kernelin.compiler

import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.fail

/**
 * Fuzzing: random kernels (see [KernelFuzzer]) must give identical results on the CPU and the GPU.
 *
 * Runs 40 kernels by default. More: `gradle :kernelin-compiler-plugin:test -Pkernelin.fuzz.count=1000`,
 * other seeds: `-Pkernelin.fuzz.seed=12345`. A failure prints the seed and the kernel's source.
 */
class KernelFuzzTest {
    @Test
    fun `random integer kernels give the same results on CPU and GPU`() {
        GpuDifferential.requireGpu()
        val count = System.getProperty("kernelin.fuzz.count")?.toInt() ?: 40
        val firstSeed = System.getProperty("kernelin.fuzz.seed")?.toLong() ?: 1L
        // Compile in batches: one compiler run for many kernels keeps the test fast.
        val failures = mutableListOf<String>()
        for (batch in (0 until count).chunked(50)) {
            val kernels = batch.map { k -> (firstSeed + k).let { seed -> seed to KernelFuzzer(seed).generate("fuzz$seed") } }
            val compiled = GpuDifferential.compile(kernels.joinToString("\n\n") { it.second })
            for ((seed, source) in kernels) {
                val n = 64
                val data = Random(seed).let { r ->
                    IntArray(n) { if (it % 7 == 0) listOf(0, -1, Int.MAX_VALUE, Int.MIN_VALUE).random(r) else r.nextInt() }
                }
                try {
                    GpuDifferential.assertSame(compiled, "fuzz$seed", n, mapOf("a" to data, "outI" to IntArray(n), "s" to seed.toInt() * 31))
                } catch (e: AssertionError) {
                    failures += "seed $seed: ${e.message}\n--- kernel ---\n$source"
                }
            }
        }
        if (failures.isNotEmpty()) fail("${failures.size} of $count random kernels differ:\n\n" + failures.joinToString("\n\n"))
    }
}
