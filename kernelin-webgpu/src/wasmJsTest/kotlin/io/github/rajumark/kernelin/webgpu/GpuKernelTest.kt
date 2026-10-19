@file:OptIn(ExperimentalWasmJsInterop::class)

package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelSource
import kotlinx.coroutines.test.runTest
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertFailsWith

/**
 * Tests for the host side of [GpuKernel]: argument checks and buffer handling. They run in Node against a small
 * fake WebGPU (see [installFakeGpu]) that keeps buffer contents in memory but never runs the shader, so what comes
 * back from a run is exactly what the host left in the buffer.
 */
class GpuKernelTest {
    private val source = KernelSource(
        "test.k",
        "// not compiled by the fake GPU",
        """
        { "kernel": "test.k", "entryPoint": "main", "workgroupSize": 64,
          "uniform": { "binding": 0, "fields": [] },
          "params": [
            { "name": "a", "kind": "buffer", "type": "f32", "binding": 1, "access": "read" },
            { "name": "out", "kind": "buffer", "type": "i32", "binding": 2, "access": "read_write" },
            { "name": "scale", "kind": "scalar", "type": "i32" },
            { "name": "on", "kind": "scalar", "type": "bool" }
          ] }
        """,
    )

    private val scalars = mapOf<String, Any>("scale" to 2, "on" to true)

    @BeforeTest
    fun setUp() = installFakeGpu()

    private suspend fun kernel() = WebGpu.create().load(source)

    @Test
    fun emptyOutputIsZeroOnEveryRun() = runTest {
        val k = kernel()
        val a = GpuData.of(floatArrayOf(1f, 2f, 3f))
        val first = k.run(3, scalars, mapOf("a" to a, "out" to GpuData.of(intArrayOf(5, 6, 7))))
        assertContentEquals(intArrayOf(5, 6, 7), first.ints("out"))
        // Same size, so the GPU buffer is reused: empty() must still start from zeros, as documented.
        val second = k.run(3, scalars, mapOf("a" to a, "out" to GpuData.empty(3)))
        assertContentEquals(intArrayOf(0, 0, 0), second.ints("out"))
    }

    @Test
    fun bufferOfTheWrongTypeIsRejected() = runTest {
        val k = kernel()
        val e = assertFailsWith<IllegalArgumentException> {
            k.run(3, scalars, mapOf("a" to GpuData.of(floatArrayOf(1f, 2f, 3f)), "out" to GpuData.of(floatArrayOf(1f, 2f, 3f))))
        }
        assertContains(e.message!!, "'out'")
        assertFailsWith<IllegalArgumentException> {
            k.run(3, scalars, mapOf("a" to GpuData.ofJs(newInt32ArrayForTest(3)), "out" to GpuData.empty(3)))
        }
    }

    @Test
    fun unknownNamesAreRejected() = runTest {
        val k = kernel()
        val buffers = mapOf("a" to GpuData.of(floatArrayOf(1f)), "out" to GpuData.empty(1))
        val e = assertFailsWith<IllegalArgumentException> { k.run(1, scalars + ("scal" to 3), buffers) }
        assertContains(e.message!!, "'scal'")
        val e2 = assertFailsWith<IllegalArgumentException> { k.run(1, scalars, buffers + ("outt" to GpuData.empty(1))) }
        assertContains(e2.message!!, "'outt'")
    }

    @Test
    fun scalarsOfTheWrongTypeAreRejected() = runTest {
        val k = kernel()
        val buffers = mapOf("a" to GpuData.of(floatArrayOf(1f)), "out" to GpuData.empty(1))
        // 2.5f would silently become 2 on an Int parameter.
        assertFailsWith<IllegalArgumentException> { k.run(1, mapOf("scale" to 2.5f, "on" to true), buffers) }
        assertFailsWith<IllegalArgumentException> { k.run(1, mapOf("scale" to 2, "on" to 1), buffers) }
    }
}

private fun newInt32ArrayForTest(length: Int): JsAny = js("new Int32Array(length)")

/** A tiny in-memory WebGPU: buffers hold real bytes; writes, clears and copies work; shaders never run. */
private fun installFakeGpu(): Unit = js("""{
    const device = {
        limits: { maxComputeWorkgroupsPerDimension: 65535 },
        queue: {
            writeBuffer(buffer, offset, data) {
                const src = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
                buffer.bytes.set(src, offset);
            },
            submit(commands) { for (const run of commands) run(); },
        },
        createShaderModule() { return { getCompilationInfo: async () => ({ messages: [] }) }; },
        createBindGroupLayout(d) { return d; },
        createPipelineLayout(d) { return d; },
        createComputePipeline(d) { return d; },
        createBindGroup(d) { return d; },
        createBuffer({ size, usage }) {
            const bytes = new Uint8Array(size);
            return { size, usage, bytes, destroy() {}, async mapAsync() {}, getMappedRange() { return bytes.buffer; }, unmap() {} };
        },
        createCommandEncoder() {
            const ops = [];
            return {
                beginComputePass() { return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} }; },
                clearBuffer(buffer) { ops.push(() => buffer.bytes.fill(0)); },
                copyBufferToBuffer(from, fromOffset, to, toOffset, size) {
                    ops.push(() => to.bytes.set(from.bytes.subarray(fromOffset, fromOffset + size), toOffset));
                },
                finish() { return () => ops.forEach(op => op()); },
            };
        },
    };
    const adapter = {
        limits: { maxStorageBufferBindingSize: 1 << 27, maxBufferSize: 1 << 28 },
        info: { vendor: 'fake' },
        requestDevice: async () => device,
    };
    globalThis.GPUShaderStage = { COMPUTE: 4 };
    globalThis.GPUMapMode = { READ: 1 };
    Object.defineProperty(globalThis, 'navigator', {
        value: { gpu: { requestAdapter: async () => adapter } }, configurable: true, writable: true,
    });
}""")
