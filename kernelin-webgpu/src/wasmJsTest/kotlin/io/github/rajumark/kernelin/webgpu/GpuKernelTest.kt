@file:OptIn(ExperimentalWasmJsInterop::class)

package io.github.rajumark.kernelin.webgpu

import io.github.rajumark.kernelin.KernelSource
import kotlinx.coroutines.async
import kotlinx.coroutines.await
import kotlinx.coroutines.test.runTest
import kotlin.js.Promise
import kotlin.test.BeforeTest
import kotlin.test.Test
import kotlin.test.assertContains
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Tests for the host side of [GpuKernel]: argument checks, buffer handling, errors and lifecycle. They run in Node
 * against a small fake WebGPU (see [installFakeGpu]) that keeps buffer contents in memory but never runs the shader,
 * so what comes back from a run is exactly what the host left in the buffer.
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

    private val inputs = mapOf("a" to GpuData.of(floatArrayOf(1f, 2f, 3f)), "out" to GpuData.of(intArrayOf(5, 6, 7)))

    @Test
    fun lostDeviceIsReportedAndStopsRuns() = runTest {
        val gpu = WebGpu.create()
        val k = gpu.load(source)
        var reason: String? = null
        gpu.onLost = { reason = it }
        assertFalse(gpu.isLost)
        fakeLoseDevice("GPU process crashed")
        tick()
        assertTrue(gpu.isLost)
        assertEquals("unknown: GPU process crashed", reason)
        val e = assertFailsWith<GpuException> { k.run(3, scalars, inputs) }
        assertContains(e.message!!, "GPU process crashed")
        assertFailsWith<GpuException> { gpu.load(source) }
    }

    @Test
    fun errorsInsideARunThrowGpuException() = runTest {
        val k = kernel()
        fakeErrorOnNextSubmit("validation", "Buffer 'test.k: out' is too small")
        val e = assertFailsWith<GpuException> { k.run(3, scalars, inputs) }
        assertContains(e.message!!, "validation: Buffer 'test.k: out' is too small")
        fakeErrorOnNextSubmit("out-of-memory", "no memory left")
        val e2 = assertFailsWith<GpuException> { k.run(3, scalars, inputs) }
        assertContains(e2.message!!, "out of memory: no memory left")
        // Every scope that was pushed was popped, and the kernel still works.
        assertEquals(0, fakeOpenErrorScopes())
        assertContentEquals(intArrayOf(5, 6, 7), k.run(3, scalars, inputs).ints("out"))
    }

    @Test
    fun errorsOutsideARunGoToOnUncapturedError() = runTest {
        val gpu = WebGpu.create()
        val errors = mutableListOf<String>()
        gpu.onUncapturedError = { errors += it }
        fakeRaise("validation", "stray error")
        assertEquals(listOf("stray error"), errors)
    }

    @Test
    fun rejectedPipelineThrowsGpuException() = runTest {
        val gpu = WebGpu.create()
        fakeFailPipeline("Entry point 'main' doesn't exist")
        val e = assertFailsWith<GpuException> { gpu.load(source) }
        assertContains(e.message!!, "Entry point 'main' doesn't exist")
        assertContains(e.message!!, "test.k")
        assertEquals(0, fakeOpenErrorScopes())
    }

    @Test
    fun closedKernelFreesItsBuffersAndRefusesToRun() = runTest {
        val k = kernel()
        k.run(3, scalars, inputs)
        assertEquals(0, fakeDestroyedBuffers())
        k.close()
        k.close()
        assertTrue(k.isClosed)
        // a, out, out's read-back copy and the scalars.
        assertEquals(4, fakeDestroyedBuffers())
        val e = assertFailsWith<IllegalStateException> { k.run(3, scalars, inputs) }
        assertContains(e.message!!, "closed")
    }

    @Test
    fun closedWebGpuDestroysTheDeviceAndItsKernels() = runTest {
        val gpu = WebGpu.create()
        var lost = false
        gpu.onLost = { lost = true }
        val k = gpu.load(source)
        k.run(3, scalars, inputs)
        gpu.close()
        tick()
        assertTrue(gpu.isClosed)
        assertTrue(k.isClosed)
        assertTrue(fakeDeviceDestroyed())
        assertFalse(lost, "close() is not a lost device")
        assertFailsWith<IllegalStateException> { k.run(3, scalars, inputs) }
        assertFailsWith<IllegalStateException> { gpu.load(source) }
    }

    @Test
    fun overlappingRunsWaitForEachOther() = runTest {
        val k = kernel()
        val a = GpuData.of(floatArrayOf(1f, 2f, 3f))
        // The fake fails like a browser when a staging buffer is mapped twice or copied into while mapping.
        val first = async { k.run(3, scalars, mapOf("a" to a, "out" to GpuData.of(intArrayOf(1, 2, 3)))) }
        val second = async { k.run(3, scalars, mapOf("a" to a, "out" to GpuData.of(intArrayOf(4, 5, 6)))) }
        assertContentEquals(intArrayOf(1, 2, 3), first.await().ints("out"))
        assertContentEquals(intArrayOf(4, 5, 6), second.await().ints("out"))
    }

    @Test
    fun everyGpuObjectIsLabeled() = runTest {
        kernel().run(3, scalars, inputs)
        val labels = fakeLabels().split("\n")
        val unlabeled = labels.filter { !it.contains("test.k") }
        assertTrue(unlabeled.isEmpty(), "Objects without the kernel name in their label: $unlabeled")
        for (expected in listOf("buffer test.k: a", "buffer test.k: out", "buffer test.k: out (read back)", "buffer test.k: scalars",
            "bindGroup test.k: bind group", "pipeline test.k: pipeline", "shaderModule test.k")) {
            assertContains(labels, expected)
        }
    }

    @Test
    fun readBackChoosesWhatComesBack() = runTest {
        val k = kernel()
        val result = k.run(3, scalars, inputs, readBack = emptySet())
        val e = assertFailsWith<IllegalArgumentException> { result.ints("out") }
        assertContains(e.message!!, "not read back")
        assertFalse(fakeLabels().contains("(read back)"), "No staging buffer is needed")
        assertContentEquals(intArrayOf(5, 6, 7), k.run(3, scalars, inputs, readBack = setOf("out")).ints("out"))
        val e2 = assertFailsWith<IllegalArgumentException> { k.run(3, scalars, inputs, readBack = setOf("a")) }
        assertContains(e2.message!!, "'a'")
    }

    @Test
    fun gpuBufferPassesDataToTheNextRunWithoutTheCpu() = runTest {
        val gpu = WebGpu.create()
        val k1 = gpu.load(source)
        val k2 = gpu.load(source)
        val first = k1.run(3, scalars, inputs, readBack = emptySet())
        val out = first.buffer("out")
        assertEquals(3, out.length)
        // Into another kernel: copied on the GPU.
        val second = k2.run(3, scalars, mapOf("a" to GpuData.of(floatArrayOf(0f, 0f, 0f)), "out" to GpuData.of(out)))
        assertContentEquals(intArrayOf(5, 6, 7), second.ints("out"))
        // Back into the same kernel (ping-pong style): the data is already in place.
        val third = k1.run(3, scalars, mapOf("a" to GpuData.of(first.buffer("a")), "out" to GpuData.of(out)))
        assertContentEquals(intArrayOf(5, 6, 7), third.ints("out"))
        // k1 ran again, so the old handle is out of date.
        assertFalse(out.isValid)
        val e = assertFailsWith<IllegalStateException> { k2.run(3, scalars, inputs + ("out" to GpuData.of(out))) }
        assertContains(e.message!!, "out of date")
        // Type and device are checked like any other data.
        assertFailsWith<IllegalArgumentException> { k2.run(3, scalars, inputs + ("a" to GpuData.of(third.buffer("out")))) }
        val other = WebGpu.create().load(source)
        assertFailsWith<IllegalArgumentException> { other.run(3, scalars, inputs + ("out" to GpuData.of(third.buffer("out")))) }
    }

    @Test
    fun bulkCopiesKeepEveryValue() {
        for (n in listOf(0, 1, 3, 1000)) {
            val ints = IntArray(n) { it * 7919 - 500_000 }.also { if (n > 2) { it[0] = Int.MIN_VALUE; it[1] = Int.MAX_VALUE } }
            val floats = FloatArray(n) { it * 0.37f - 9f }.also { if (n > 2) { it[0] = Float.MAX_VALUE; it[1] = -0f } }
            val result = GpuResult(0.0, mapOf("i" to ints.toJsInt32Array(), "f" to floats.toJsFloat32Array()), emptyMap())
            assertContentEquals(ints, result.ints("i"))
            assertContentEquals(floats.map { it.toRawBits() }, result.floats("f").map { it.toRawBits() })
            assertEquals(n, typedArrayLengthForTest(ints.toJsInt32Array()))
        }
        assertEquals(-5, readIntForTest(intArrayOf(9, -5).toJsInt32Array(), 1))
    }
}

private fun newInt32ArrayForTest(length: Int): JsAny = js("new Int32Array(length)")

private fun typedArrayLengthForTest(array: JsAny): Int = js("array.length")

private fun readIntForTest(array: JsAny, index: Int): Int = js("array[index]")

/** Lets pending JS callbacks (device.lost, timers) run. */
private suspend fun tick() = nextTick().await<JsAny?>()

private fun nextTick(): Promise<JsAny?> = js("new Promise(resolve => setTimeout(() => resolve(null), 0))")

private fun fakeLoseDevice(message: String): Unit = js("{ globalThis.fakeGpu.lose('unknown', message); }")

private fun fakeErrorOnNextSubmit(type: String, message: String): Unit = js("{ globalThis.fakeGpu.errorOnSubmit = { type, message }; }")

private fun fakeRaise(type: String, message: String): Unit = js("{ globalThis.fakeGpu.raise(type, message); }")

private fun fakeFailPipeline(message: String): Unit = js("{ globalThis.fakeGpu.failPipeline = message; }")

private fun fakeOpenErrorScopes(): Int = js("globalThis.fakeGpu.scopes.length")

private fun fakeDestroyedBuffers(): Int = js("globalThis.fakeGpu.objects.filter(o => o.kind === 'buffer' && o.destroyed).length")

private fun fakeDeviceDestroyed(): Boolean = js("globalThis.fakeGpu.deviceDestroyed")

private fun fakeLabels(): String = js("globalThis.fakeGpu.objects.map(o => o.kind + ' ' + (o.label || '')).join('\\n')")

/**
 * A tiny in-memory WebGPU: buffers hold real bytes; writes, clears and copies work; shaders never run.
 * Like a browser it has error scopes, device loss, labels, and fails when a buffer is mapped twice at once.
 * Test hooks live in `globalThis.fakeGpu`.
 */
private fun installFakeGpu(): Unit = js("""{
    const fake = { objects: [], scopes: [], listeners: [], errorOnSubmit: null, failPipeline: null, deviceDestroyed: false };
    globalThis.fakeGpu = fake;
    let resolveLost;
    const make = (kind, d, extra) => { const o = Object.assign({ kind, label: d && d.label }, extra); fake.objects.push(o); return o; };
    fake.raise = (type, message) => {
        for (let i = fake.scopes.length - 1; i >= 0; i--) {
            if (fake.scopes[i].filter === type) { fake.scopes[i].error ??= { message }; return; }
        }
        for (const listener of fake.listeners) listener({ error: { message } });
    };
    fake.lose = (reason, message) => resolveLost({ reason, message });
    const device = {
        limits: { maxComputeWorkgroupsPerDimension: 65535 },
        lost: new Promise(resolve => { resolveLost = resolve; }),
        addEventListener(type, listener) { if (type === 'uncapturederror') fake.listeners.push(listener); },
        pushErrorScope(filter) { fake.scopes.push({ filter, error: null }); },
        popErrorScope() {
            if (fake.scopes.length === 0) return Promise.reject(new Error('error scope stack is empty'));
            return Promise.resolve(fake.scopes.pop().error);
        },
        destroy() { fake.deviceDestroyed = true; resolveLost({ reason: 'destroyed', message: 'Device was destroyed.' }); },
        queue: {
            writeBuffer(buffer, offset, data) {
                const src = ArrayBuffer.isView(data) ? new Uint8Array(data.buffer, data.byteOffset, data.byteLength) : new Uint8Array(data);
                buffer.bytes.set(src, offset);
            },
            submit(commands) {
                if (fake.errorOnSubmit) { fake.raise(fake.errorOnSubmit.type, fake.errorOnSubmit.message); fake.errorOnSubmit = null; return; }
                for (const run of commands) run();
            },
        },
        createShaderModule(d) { return make('shaderModule', d, { getCompilationInfo: async () => ({ messages: [] }) }); },
        createBindGroupLayout(d) { return make('bindGroupLayout', d); },
        createPipelineLayout(d) { return make('pipelineLayout', d); },
        createComputePipelineAsync(d) {
            if (fake.failPipeline) { const message = fake.failPipeline; fake.failPipeline = null; return Promise.reject(new Error(message)); }
            return Promise.resolve(make('pipeline', d));
        },
        createBindGroup(d) { return make('bindGroup', d); },
        createBuffer({ size, usage, label }) {
            const bytes = new Uint8Array(size);
            const buffer = make('buffer', { label }, {
                size, usage, bytes, destroyed: false, mapState: 'unmapped',
                destroy() { this.destroyed = true; },
                mapAsync() {
                    if (this.mapState !== 'unmapped') return Promise.reject(new Error('Buffer ' + this.label + ' is already mapped or pending'));
                    this.mapState = 'pending';
                    return new Promise((resolve, reject) => setTimeout(() => {
                        if (this.destroyed || fake.deviceDestroyed) { this.mapState = 'unmapped'; reject(new Error('Device or buffer was destroyed')); }
                        else { this.mapState = 'mapped'; resolve(); }
                    }, 0));
                },
                getMappedRange() { return bytes.buffer; },
                unmap() { this.mapState = 'unmapped'; },
            });
            return buffer;
        },
        createCommandEncoder(d) {
            make('commandEncoder', d);
            const ops = [];
            return {
                beginComputePass(p) {
                    make('computePass', p);
                    return { setPipeline() {}, setBindGroup() {}, dispatchWorkgroups() {}, end() {} };
                },
                clearBuffer(buffer) { ops.push(() => buffer.bytes.fill(0)); },
                copyBufferToBuffer(from, fromOffset, to, toOffset, size) {
                    ops.push(() => {
                        if (from === to) return fake.raise('validation', 'copy from a buffer to itself');
                        if (to.mapState !== 'unmapped') return fake.raise('validation', 'copy into ' + to.label + ' while it is mapped');
                        to.bytes.set(from.bytes.subarray(fromOffset, fromOffset + size), toOffset);
                    });
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
