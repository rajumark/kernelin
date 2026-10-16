# Kernelin

[![CI](https://github.com/rajumark/kernelin/actions/workflows/ci.yml/badge.svg)](https://github.com/rajumark/kernelin/actions/workflows/ci.yml)
[![Live demo](https://img.shields.io/badge/live%20demo-try%20it-2a78d6)](https://rajumark.github.io/kernelin/demo/)
[![Docs](https://img.shields.io/badge/docs-read-7f52ff)](https://rajumark.github.io/kernelin/docs.html)
[![Maven Central](https://img.shields.io/maven-central/v/io.github.rajumark/kernelin-runtime)](https://central.sonatype.com/artifact/io.github.rajumark/kernelin-runtime)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

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

**0.1.0 is released on Maven Central.** On an Apple M5, the generated shaders match the Kotlin CPU results and run
up to 118× faster than all 10 CPU cores (see the [website](https://rajumark.github.io/kernelin/#benchmarks) and `benchmarks/`).

**[Live demo →](https://rajumark.github.io/kernelin/demo/)** a fractal computed on your GPU by Kernelin-generated code.

**[Documentation →](https://rajumark.github.io/kernelin/docs.html)** guides, API reference and compiler internals. AI agents can read
[`docs.md`](https://rajumark.github.io/kernelin/docs.md) or [`llms.txt`](https://rajumark.github.io/kernelin/llms.txt); see also [`AGENTS.md`](AGENTS.md).

## Real-world projects

[kernelin-projects](https://github.com/rajumark/kernelin-projects) ([live](https://rajumark.github.io/kernelin-projects/)):
three apps built with Kernelin from Maven Central, each checked against exact answers and benchmarked.

- **Risk Engine** (simulation): Monte Carlo options pricing + greeks, 1M paths × 252 days × 4 scenarios in 46 ms on an Apple M5 GPU, 53× faster than 10 JVM cores.
- **Photo Studio** (image): live photo editing with edge-preserving denoise, 3.8 MP in 48 ms, +11.3 dB PSNR.
- **Motion Guard** (video): traffic camera motion vectors and vehicle speeds, 720p in 19 ms (live 30 fps), precision 0.995 vs ground truth.

## Use it in your project

Kernelin is on Maven Central. Make sure `mavenCentral()` is in your plugin repositories:

```kotlin
// settings.gradle.kts
pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}
```

```kotlin
// build.gradle.kts
plugins {
    kotlin("jvm") version "2.4.21"          // or kotlin("multiplatform")
    id("io.github.rajumark.kernelin") version "0.1.0"
}
```

The plugin runs the Kernelin compiler plugin on every compilation and adds
`io.github.rajumark:kernelin-runtime` (plus `kernelin-webgpu` to `wasmJsMain` in Multiplatform projects).
Kernelin 0.1.0 is built for **Kotlin 2.4.21**; the compiler plugin API changes between Kotlin versions.

| Artifact | What it is |
|---|---|
| `io.github.rajumark:kernelin-runtime:0.1.0` | `@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu` (JVM + Wasm) |
| `io.github.rajumark:kernelin-webgpu:0.1.0` | Kotlin/Wasm WebGPU host |
| `io.github.rajumark:kernelin-compiler-plugin:0.1.0` | the K2 compiler plugin (added by the Gradle plugin) |
| `io.github.rajumark:kernelin-gradle-plugin:0.1.0` | Gradle plugin `io.github.rajumark.kernelin` |

See [`examples/`](examples) for complete projects (`hello-jvm`, `hello-multiplatform`).

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
| `kernelin-gradle-plugin` | `id("io.github.rajumark.kernelin")`: one-line setup for JVM and Multiplatform projects |
| `samples` | Kernels in `commonMain` (JVM + Wasm), CPU tests and benchmark (JVM), live demo entry points (Wasm) |

## Build

Requires JDK 21. Tests validate every generated shader with [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga)
when it's installed (`cargo install naga-cli`).

```bash
gradle build
gradle :samples:bench -Pout=/path/to/folder   # then serve the folder and open runner.html
scripts/bench-android.sh /path/to/folder      # same GPU benchmark in Chrome on a USB-connected Android phone
scripts/bench-android-cpu.sh                 # the CPU side on the phone (Kotlin on ART)
gradle :samples:exportDemo                     # rebuild the Kotlin/Wasm live demo into docs/demo
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md), the compatibility policy in [VERSIONING.md](VERSIONING.md), and
[SECURITY.md](SECURITY.md) for reporting vulnerabilities. AI agents: start with [AGENTS.md](AGENTS.md).
