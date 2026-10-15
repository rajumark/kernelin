# Kernelin

Write GPU kernels in plain Kotlin. A K2 compiler plugin translates `@Kernel` functions to WGSL,
so they run on any GPU through WebGPU, and they still run on the CPU for unit tests.

```kotlin
@Kernel
fun saxpy(alpha: Float, x: FloatArray, y: FloatArray) {
    val i = globalId()
    y[i] += alpha * x[i]
}
```

becomes

```wgsl
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
...
  let i: i32 = i32(kernelin_id);
  y[i] += (kernelin_params.alpha * x[i]);
```

## Status

MVP in progress. On an Apple M5, the generated shaders match the Kotlin CPU results and run
up to 118× faster than all 10 CPU cores (see the [project report](https://rajumark.github.io/kernelin/) and `benchmarks/`).

**[Live demo →](https://rajumark.github.io/kernelin/demo/)** a fractal computed on your GPU by Kernelin-generated code.

## Run a kernel on the GPU from Kotlin

```kotlin
val gpu = WebGpu.create()                              // Kotlin/Wasm, any WebGPU browser
val kernel = gpu.load(kernelSource(::saxpy))           // the compiler embeds the WGSL here
val result = kernel.run(
    size = n,
    scalars = mapOf("alpha" to 2.5f),
    buffers = mapOf("x" to GpuData.of(x), "y" to GpuData.of(y)),
)
val y2 = result.floats("y")
```

## Modules

| Module | What it is |
|---|---|
| `kernelin-runtime` | Multiplatform (JVM + Wasm): `@Kernel`, `globalId()`, `KernelinCpu` (CPU runner), `kernelSource(::k)` |
| `kernelin-compiler-plugin` | K2 compiler plugin; `WgslKernelTranslator` turns Kotlin IR into WGSL and embeds it where `kernelSource(::k)` is called |
| `kernelin-webgpu` | Kotlin/Wasm WebGPU host: `WebGpu.create()`, `gpu.load(kernelSource(::k))`, `kernel.run(...)` |
| `kernelin-gradle-plugin` | One-line setup for users (not wired yet) |
| `samples` | Kernels in `commonMain` (JVM + Wasm), CPU tests and benchmark (JVM), live demo entry points (Wasm) |

## Build

Requires JDK 21. Tests validate every generated shader with [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
when it's installed (`cargo install naga-cli`).

```bash
gradle build
gradle :samples:bench -Pout=/path/to/folder   # then serve the folder and open runner.html
gradle :samples:exportDemo                     # rebuild the Kotlin/Wasm live demo into docs/demo
```
