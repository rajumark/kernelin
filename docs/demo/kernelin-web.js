// Kernelin WebGPU host: runs a kernel using only the .wgsl + .json files the Kernelin compiler plugin generated.
//
//   const gpu = await KernelinGpu.create();
//   const kernel = await gpu.load('kernels/io.github.kernelin.samples.mandelbrotView');
//   const { outputs, ms } = await kernel.run({ size, scalars: { width, ... }, buffers: { out: width * height } });

export class KernelinGpu {
  static async create() {
    if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter found');
    const device = await adapter.requestDevice({ requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    } });
    return new KernelinGpu(adapter, device);
  }

  constructor(adapter, device) {
    this.adapter = adapter;
    this.device = device;
  }

  get info() {
    const i = this.adapter.info || {};
    return [i.vendor, i.architecture].filter(Boolean).join(' ');
  }

  /** `base` is the path without extension, e.g. `kernels/io.github.kernelin.samples.mandelbrotView`. */
  async load(base) {
    const [manifest, code] = await Promise.all([
      fetch(base + '.json').then((r) => r.json()),
      fetch(base + '.wgsl').then((r) => r.text()),
    ]);
    const module = this.device.createShaderModule({ code });
    const errors = (await module.getCompilationInfo()).messages.filter((m) => m.type === 'error');
    if (errors.length) throw new Error('WGSL rejected: ' + errors.map((e) => e.message).join('; '));
    return new KernelinKernel(this.device, manifest, module, code);
  }
}

class KernelinKernel {
  constructor(device, manifest, module, code) {
    this.device = device;
    this.manifest = manifest;
    this.code = code;
    this.bufferParams = manifest.params.filter((p) => p.kind === 'buffer');
    this.scalarParams = manifest.params.filter((p) => p.kind === 'scalar');
    this.layout = device.createBindGroupLayout({ entries: [
      { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
      ...this.bufferParams.map((b) => ({ binding: b.binding, visibility: GPUShaderStage.COMPUTE,
        buffer: { type: b.access === 'read' ? 'read-only-storage' : 'storage' } })),
    ] });
    this.pipeline = device.createComputePipeline({
      layout: device.createPipelineLayout({ bindGroupLayouts: [this.layout] }),
      compute: { module, entryPoint: manifest.entryPoint },
    });
    this.uniformBytes = Math.ceil((1 + this.scalarParams.length) * 4 / 16) * 16;
    this.uniform = device.createBuffer({ size: this.uniformBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.gpuBuffers = {};
    this.staging = {};
  }

  /** Reuses GPU buffers between runs; only reallocates when a size changes. */
  ensure(map, name, bytes, usage) {
    if (!map[name] || map[name].size !== bytes) {
      map[name]?.destroy();
      map[name] = this.device.createBuffer({ size: bytes, usage });
      this.bindGroup = null;
    }
    return map[name];
  }

  /**
   * `buffers` maps each array parameter to a typed array (uploaded) or a length (allocated, not uploaded).
   * Returns every read_write buffer, plus the time from submit until the results are back in JS.
   */
  async run({ size, scalars = {}, buffers }) {
    const dv = new DataView(new ArrayBuffer(this.uniformBytes));
    dv.setUint32(0, size, true);
    this.scalarParams.forEach((s, i) => {
      const v = scalars[s.name];
      if (v === undefined) throw new Error('Missing scalar parameter ' + s.name);
      if (s.type === 'f32') dv.setFloat32(4 + i * 4, v, true); else dv.setInt32(4 + i * 4, v, true);
    });
    this.device.queue.writeBuffer(this.uniform, 0, dv.buffer);

    const t0 = performance.now();
    for (const b of this.bufferParams) {
      const value = buffers[b.name];
      const length = typeof value === 'number' ? value : value.length;
      const buf = this.ensure(this.gpuBuffers, b.name, length * 4,
        GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST);
      if (typeof value !== 'number') this.device.queue.writeBuffer(buf, 0, value);
    }
    const outputs = this.bufferParams.filter((b) => b.access === 'read_write');
    for (const b of outputs) {
      this.ensure(this.staging, b.name, this.gpuBuffers[b.name].size, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
    }
    if (!this.bindGroup) {
      this.bindGroup = this.device.createBindGroup({ layout: this.layout, entries: [
        { binding: 0, resource: { buffer: this.uniform } },
        ...this.bufferParams.map((b) => ({ binding: b.binding, resource: { buffer: this.gpuBuffers[b.name] } })),
      ] });
    }

    const groups = Math.ceil(size / this.manifest.workgroupSize);
    const gx = Math.min(groups, this.device.limits.maxComputeWorkgroupsPerDimension);
    const gy = Math.ceil(groups / gx);
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(this.pipeline);
    pass.setBindGroup(0, this.bindGroup);
    pass.dispatchWorkgroups(gx, gy);
    pass.end();
    for (const b of outputs) encoder.copyBufferToBuffer(this.gpuBuffers[b.name], 0, this.staging[b.name], 0, this.gpuBuffers[b.name].size);
    this.device.queue.submit([encoder.finish()]);

    const result = {};
    for (const b of outputs) {
      const staging = this.staging[b.name];
      await staging.mapAsync(GPUMapMode.READ);
      const copy = staging.getMappedRange().slice(0);
      staging.unmap();
      result[b.name] = b.type === 'i32' ? new Int32Array(copy) : new Float32Array(copy);
    }
    return { outputs: result, ms: performance.now() - t0 };
  }
}
