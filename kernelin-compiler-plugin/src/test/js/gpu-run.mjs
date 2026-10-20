// Runs one Kernelin kernel on a real GPU through WebGPU (Deno), for the differential tests.
//
//   deno run --unstable-webgpu --allow-read --allow-write gpu-run.mjs --probe
//   deno run --unstable-webgpu --allow-read --allow-write gpu-run.mjs job.json
//
// job.json: { "wgsl": "<file>", "manifest": "<file>", "size": n, "width": w, "height": h, "scalars": { "name": value },
//             "buffers": { "name": { "length": n, "input": "<file or null>", "output": "<file>" } } }
// Buffers are little-endian i32 / f32 files. Every read_write buffer is written to its "output" file.

// Prefer a real GPU; CI machines only have a software one (lavapipe), which WebGPU calls a fallback adapter.
const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  ?? await navigator.gpu?.requestAdapter({ forceFallbackAdapter: true });
if (!adapter) {
  console.log(JSON.stringify({ ok: false, error: 'no WebGPU adapter' }));
  Deno.exit(2);
}
const info = adapter.info ?? {};
const adapterName = [info.vendor, info.architecture, info.device, info.description].filter(Boolean).join(' ') || 'unknown';

if (Deno.args[0] === '--probe') {
  console.log(JSON.stringify({ ok: true, adapter: adapterName }));
  Deno.exit(0);
}

const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    maxBufferSize: adapter.limits.maxBufferSize,
  },
});
const errors = [];
device.addEventListener?.('uncapturederror', (e) => errors.push(e.error.message));

const job = JSON.parse(await Deno.readTextFile(Deno.args[0]));
const manifest = JSON.parse(await Deno.readTextFile(job.manifest));
const code = await Deno.readTextFile(job.wgsl);

device.pushErrorScope('validation');
const module = device.createShaderModule({ code });
const compile = await module.getCompilationInfo();
const compileErrors = compile.messages.filter((m) => m.type === 'error');
if (compileErrors.length) {
  console.log(JSON.stringify({ ok: false, error: 'WGSL rejected: ' + compileErrors.map((m) => `${m.lineNum}:${m.message}`).join('; ') }));
  Deno.exit(1);
}

const buffers = manifest.params.filter((p) => p.kind === 'buffer');
const scalars = manifest.params.filter((p) => p.kind === 'scalar');
const layout = device.createBindGroupLayout({ entries: [
  { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
  ...buffers.map((b) => ({ binding: b.binding, visibility: GPUShaderStage.COMPUTE,
    buffer: { type: b.access === 'read' ? 'read-only-storage' : 'storage' } })),
] });
const pipeline = device.createComputePipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [layout] }),
  compute: { module, entryPoint: manifest.entryPoint },
});

// Uniform: kernelin_size (u32), plus kernelin_width and kernelin_height (u32) for kernels with grid ids, then each
// scalar in order, 4 bytes each, padded to 16 bytes. "width"/"height" are optional: a 1D launch is size x 1.
const header = manifest.gridIds ? 3 : 1;
const uniformBytes = Math.ceil((header + scalars.length) * 4 / 16) * 16;
const view = new DataView(new ArrayBuffer(uniformBytes));
view.setUint32(0, job.size, true);
if (manifest.gridIds) {
  view.setUint32(4, job.width ?? job.size, true);
  view.setUint32(8, job.height ?? 1, true);
}
scalars.forEach((s, i) => {
  const v = job.scalars[s.name];
  if (v === undefined) throw new Error('missing scalar ' + s.name);
  if (s.type === 'f32') view.setFloat32(header * 4 + i * 4, v, true);
  else if (s.type === 'bool') view.setUint32(header * 4 + i * 4, v ? 1 : 0, true);
  else view.setInt32(header * 4 + i * 4, v, true);
});
const uniform = device.createBuffer({ size: uniformBytes, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
device.queue.writeBuffer(uniform, 0, view.buffer);

const gpu = {};
for (const b of buffers) {
  const spec = job.buffers[b.name];
  if (!spec) throw new Error('missing buffer ' + b.name);
  const bytes = Math.max(4, spec.length * 4);
  gpu[b.name] = device.createBuffer({ size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  if (spec.input) device.queue.writeBuffer(gpu[b.name], 0, await Deno.readFile(spec.input));
}
const bindGroup = device.createBindGroup({ layout, entries: [
  { binding: 0, resource: { buffer: uniform } },
  ...buffers.map((b) => ({ binding: b.binding, resource: { buffer: gpu[b.name] } })),
] });

const groups = Math.ceil(job.size / manifest.workgroupSize);
const gx = Math.max(1, Math.min(groups, device.limits.maxComputeWorkgroupsPerDimension));
const gy = Math.max(1, Math.ceil(groups / gx));
const outputs = buffers.filter((b) => b.access === 'read_write');
const staging = Object.fromEntries(outputs.map((b) => [b.name,
  device.createBuffer({ size: gpu[b.name].size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST })]));

const encoder = device.createCommandEncoder();
if (job.size > 0) {
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(gx, gy);
  pass.end();
}
for (const b of outputs) encoder.copyBufferToBuffer(gpu[b.name], 0, staging[b.name], 0, gpu[b.name].size);
device.queue.submit([encoder.finish()]);

for (const b of outputs) {
  await staging[b.name].mapAsync(GPUMapMode.READ);
  const data = new Uint8Array(staging[b.name].getMappedRange().slice(0, job.buffers[b.name].length * 4));
  staging[b.name].unmap();
  await Deno.writeFile(job.buffers[b.name].output, data);
}
const validation = await device.popErrorScope();
if (validation) errors.push(validation.message);
console.log(JSON.stringify({ ok: errors.length === 0, adapter: adapterName, errors, dispatch: [gx, gy] }));
Deno.exit(errors.length === 0 ? 0 : 1);
