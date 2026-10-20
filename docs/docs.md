# Kernelin documentation

Kernelin lets you write GPU compute kernels in plain Kotlin. A K2 compiler plugin translates every function marked `@Kernel` into a WGSL compute shader. The shader runs on any GPU through WebGPU, and the original function stays normal Kotlin, so it also runs on the CPU for unit tests.

> This page describes **Kernelin 0.3.0**, built for **Kotlin 2.4.21**. Features marked **(0.3.0)** are new in this release; **(0.3.0)** marks what is on `main` but not yet on Maven Central. The same text is available as plain Markdown at [`docs.md`](docs.md) for tools and AI agents.

```kotlin
@Kernel
fun saxpy(alpha: Float, x: FloatArray, y: FloatArray) {
    val i = globalId()
    y[i] += alpha * x[i]
}
```

| | |
|---|---|
| **What it is** | A Kotlin compiler plugin + a small runtime + a WebGPU host |
| **Input** | Top-level Kotlin functions marked `@Kernel` |
| **Output** | A WGSL compute shader and a JSON manifest per kernel, embedded in your program |
| **Runs on** | Any WebGPU GPU (Metal, Vulkan, Direct3D 12) from Kotlin/Wasm; the CPU on JVM, Android, JS, Wasm and Native |
| **License** | Apache 2.0 |

## Quick start

### 1. Add the plugin

Kernelin is published on Maven Central. Make sure `mavenCentral()` is in your plugin repositories:

```kotlin
// settings.gradle.kts
pluginManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
}
```

Then apply the Gradle plugin next to the Kotlin plugin:

```kotlin
// build.gradle.kts
plugins {
    kotlin("jvm") version "2.4.21"          // or kotlin("multiplatform"), or an Android plugin
    id("io.github.rajumark.kernelin") version "0.3.0"
}
```

That's all. The Gradle plugin runs the compiler plugin on every Kotlin compilation and adds the runtime library for you.
See [Supported platforms](#supported-platforms) for Android and the other Kotlin targets.

### 2. Write a kernel

```kotlin
import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId

/** Makes every pixel a bit brighter, keeping values between 0 and 1. */
@Kernel
fun brighten(pixels: FloatArray, amount: Float) {
    val i = globalId()
    pixels[i] = (pixels[i] + amount).coerceIn(0f, 1f)
}
```

A kernel describes the work of **one thread**. `globalId()` tells the thread which element it owns. When you run the kernel with `size = n`, it runs once for every id from `0` to `n - 1`, in parallel on the GPU.

### 3. Run it on the CPU and look at the shader

```kotlin
import io.github.rajumark.kernelin.KernelinCpu
import io.github.rajumark.kernelin.kernelSource

fun main() {
    val pixels = floatArrayOf(0.1f, 0.5f, 0.95f)
    KernelinCpu.dispatch(pixels.size) { brighten(pixels, 0.1f) }
    println(pixels.toList())               // [0.2, 0.6, 1.0]

    val source = kernelSource(::brighten)  // filled in by the compiler
    println(source.wgsl)                   // the generated WGSL compute shader
}
```

### 4. Run it on the GPU

GPU execution uses WebGPU from **Kotlin/Wasm** in the browser. See [Run on the GPU](#run-on-the-gpu).

Complete, runnable projects live in [`examples/`](https://github.com/rajumark/kernelin/tree/main/examples): `hello-jvm` and `hello-multiplatform`.

## Installation

### Artifacts

| Artifact | What it is |
|---|---|
| `io.github.rajumark:kernelin-runtime:0.3.0` | `@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu`, `KernelManifest`, random numbers (JVM, JS, Wasm, Native; see below) |
| `io.github.rajumark:kernelin-webgpu:0.3.0` | Kotlin/Wasm WebGPU host: `WebGpu`, `GpuKernel`, `GpuData`, `GpuResult` |
| `io.github.rajumark:kernelin-compiler-plugin:0.3.0` | The K2 compiler plugin (added by the Gradle plugin) |
| `io.github.rajumark:kernelin-gradle-plugin:0.3.0` | Gradle plugin with id `io.github.rajumark.kernelin` |

### What the Gradle plugin does

- Runs `kernelin-compiler-plugin` on every Kotlin compilation in the project.
- **Kotlin JVM projects:** adds `kernelin-runtime` to `implementation`.
- **Kotlin Multiplatform projects:** adds `kernelin-runtime` to `commonMain`, and `kernelin-webgpu` to `wasmJsMain` when a `wasmJs` target exists.
  This includes Android targets made with `com.android.kotlin.multiplatform.library`.
- **Android projects (0.3.0):** adds `kernelin-runtime` to `implementation`, both with the Kotlin Android
  plugin (`org.jetbrains.kotlin.android`) and with AGP 9's built-in Kotlin (only `com.android.application` or
  `com.android.library`). With built-in Kotlin, make sure the Kotlin Gradle plugin is version 2.4.21 (for example
  `classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:2.4.21")` in `buildscript`).

### Supported platforms

| Platform | Kernels run on the CPU (`KernelinCpu`) | `kernelSource(::k)` (WGSL) | GPU host in Kernelin |
|---|---|---|---|
| JVM | yes, also on all cores with `dispatchParallel` | yes | no (use the WGSL with your own WebGPU host) |
| Android (JVM artifact) | yes | yes | yes: `kernelin-webgpu` on Android 8.0+, native WebGPU through AndroidX WebGPU **(main)** |
| Kotlin/Wasm (`wasmJs`) | yes | yes | yes: `kernelin-webgpu` |
| Kotlin/JS (`js`, browser and Node.js) **(0.3.0)** | yes | yes | no |
| Kotlin/Native **(0.3.0)**: `macosArm64`, `iosArm64`, `iosSimulatorArm64`, `iosX64`, `tvosArm64`, `tvosSimulatorArm64`, `watchosArm32`, `watchosArm64`, `watchosSimulatorArm64`, `linuxX64`, `linuxArm64`, `mingwX64` | yes, on the calling thread | yes | no |

Kernels in `commonMain` compile for every target above. The compiler plugin runs on all of them, so a kernel that
the GPU can't run is a compile error everywhere, not only on the Wasm target.

### Gradle settings **(0.3.0)**

```kotlin
// build.gradle.kts
kernelin {
    enabled = false   // default true; turns the compiler plugin off (kernels still run on the CPU)
}
```

With `enabled = false`, `@Kernel` functions are not translated and `kernelSource(::k)` throws at run time.

### Kotlin Multiplatform setup

Put kernels in `commonMain` so every target shares them: the JVM, JS and Native targets run them on the CPU, and the Wasm
target runs them on the GPU.

```kotlin
// build.gradle.kts
plugins {
    kotlin("multiplatform") version "2.4.21"
    id("io.github.rajumark.kernelin") version "0.3.0"
}

kotlin {
    jvmToolchain(21)
    jvm()
    @OptIn(org.jetbrains.kotlin.gradle.ExperimentalWasmDsl::class)
    wasmJs {
        browser()
        binaries.executable()
    }
}
```

### Requirements

| | Version |
|---|---|
| Kotlin | **2.4.21** exactly. Compiler plugin APIs change between Kotlin versions. |
| JDK | 21 to build |
| GPU runtime | A browser with WebGPU (Chrome, Edge, Safari 26+, Firefox with WebGPU) |

## Core concepts

### One function, many threads

A kernel is a function that runs once per **thread**. You never write the loop over all elements; the GPU (or `KernelinCpu`) does that. Inside the kernel, `globalId()` returns this thread's index.

```kotlin
@Kernel
fun squares(out: IntArray) {
    val i = globalId()   // 0, 1, 2, ... size - 1
    out[i] = i * i
}
```

### Parameters become bindings

| Kotlin parameter | On the GPU |
|---|---|
| `FloatArray` | `array<f32>` storage buffer |
| `IntArray` | `array<i32>` storage buffer |
| `Int` | `i32` field in a uniform struct |
| `Float` | `f32` field in a uniform struct |
| `Boolean` **(0.2.0)** | `u32` field (0 or 1) in the uniform struct, read as `bool` |

Kernelin looks at how each array is used. Arrays you only read become `read` buffers; arrays you write become `read_write` buffers. Only `read_write` buffers are copied back from the GPU.

### Results go into arrays

A kernel returns `Unit`. To produce output, write into an array parameter. Each thread should write to its own element (usually `out[globalId()]`); two threads writing the same element is a race.

### Bounds are checked for you

The GPU launches threads in groups of 64, so the last group may have extra threads. The generated shader returns early when `globalId() >= size`, so you don't need that check yourself.

## What you can write in a kernel

Kernels are a strict subset of Kotlin: the parts that map directly onto GPU hardware. Anything outside this subset is a **compile error** with file, line and a hint (see [Compiler errors](#compiler-errors)).

### Supported

| Area | Supported |
|---|---|
| Types | `Int`, `Float`, `Boolean` for locals; `FloatArray`, `IntArray`, `Int`, `Float` for kernel parameters; `Boolean` **(0.2.0)** |
| Variables | `val`, `var`, with or without a type |
| Operators | `+ - * / %`, unary `-`, `++ --`, `+= -= *= /= %=`, comparisons, `&& \|\| !`, `and or xor inv shl shr ushr` (since 0.2.0, shift counts use their low 5 bits, exactly like Kotlin, on every GPU) |
| Conversions | `toFloat()`, `toInt()`; mixed `Int`/`Float` math is promoted like Kotlin |
| Conditions | `if`/`else`, `when` (with or without a subject), both as statements and as values |
| Loops | `for` over Int ranges: `a until b`, `a..b`, `a..<b`, `a downTo b`, `step n`, `array.indices`; `while`, `do-while`, `break`, `continue`. **(Unreleased)**: ranges that end at or near `Int.MAX_VALUE` / `Int.MIN_VALUE` (like `0..Int.MAX_VALUE`) and runtime steps stop exactly like Kotlin; one `step` per range |
| More loops **(0.2.0)** | `repeat(n) { }` (with `return@repeat`), labeled `break@outer` / `continue@outer` at any depth |
| Ranges **(0.2.0)** | `x in a..b`, `until`, `..<`, `downTo`, `x !in ...`, `when (x) { in a..b -> }` for Int and Float (`step` inside `in` is an error) |
| Arrays | `a[i]`, `a[i] = v`, `a[i] += v`, `a.size`, `a.indices` |
| Math | `kotlin.math` Float versions: `sqrt sin cos tan asin acos atan atan2 sinh cosh tanh exp ln log2 log10 pow hypot floor ceil round truncate abs sign min max` |
| Clamping | `coerceIn`, `coerceAtLeast`, `coerceAtMost` |
| Float rules **(Unreleased)** | `min`, `max`, `coerceIn`, `coerceAtLeast`, `coerceAtMost`, `sign` and `compareTo` give Kotlin's exact results for NaN and `-0f` (`min(-0f, 0f)` is `-0f`, `min(x, NaN)` is NaN, `NaN.compareTo(x)` is 1); `hypot` doesn't overflow for big values, and `log10` is accurate near 1 |
| Constants | `PI.toFloat()`, `E.toFloat()`, `1.5.toFloat()`, `1.5.toInt()` (a Double constant converted right away) |
| Constant math **(0.2.0)** | All-constant expressions are computed at build time with Kotlin's rules (wrapping, Infinity/NaN), e.g. `-75 shl 28`; `Float.POSITIVE_INFINITY`, `NEGATIVE_INFINITY` and `NaN` constants are allowed |
| Float checks **(0.2.0)** | `isNaN()`, `isInfinite()`, `isFinite()`; `Float.toInt()` matches Kotlin exactly (NaN → 0, too large → `Int.MAX_VALUE`, too small → `Int.MIN_VALUE`) |
| 2D/3D thread ids **(main)** | `globalIdX()`, `globalIdY()`, `globalIdZ()` with `dispatch(width, height)` / `dispatch(width, height, depth)` launches; `globalId()` is then the row-major index `x + y * width` (see [2D and 3D launches](#example-2d-and-3d-launches-main)) |
| Random numbers **(main)** | `randomInt(seed, index, stream)`, `randomFloat(...)` in `[0, 1)`, `randomIntBelow(bound, seed, index, stream)`, `randomNormal(...)`; the same bits on the CPU and the GPU (see [Random numbers](#random-numbers-main)) |
| Helpers | Your own top-level functions and extension functions with `Int`/`Float`/`Boolean` parameters, returning `Int`, `Float`, `Boolean` or `Unit`. They become WGSL `fn`s. **(0.2.0)**: helpers can also take the kernel's `FloatArray`/`IntArray` parameters. |

### Not supported (compile error)

| Not allowed | Why | Write instead |
|---|---|---|
| `Double`, `Long`, `UInt` | Most GPUs have no 64-bit math; unsigned types not yet mapped | `Float` (`1.0f`), `Int` |
| `String`, `println` | GPUs have no text | Write numbers into an output array |
| Lambdas, `forEach {}`, `map {}` (`repeat {}` works) | No closures on the GPU | A `for` loop |
| Recursion | No call stack on the GPU | A loop |
| `throw`, `try/catch` | No exceptions | Return early or write an error code |
| Nullable types, `is`, `as` | No objects on the GPU | Plain values |
| Classes, objects, collections | Only scalars and arrays exist | Several arrays (one per field) |
| Library calls (except `kotlin.math` and Kernelin's random functions) | Their code isn't available to translate | Your own helper function |
| Default args, generics in helpers | Not supported yet | Pass every argument |
| Structs / data classes, shared memory, atomics | Not supported yet | Flat arrays, one writer per element |
| A non-`Unit` return from a kernel | Results go into arrays | Write into an array parameter |
| Calling another `@Kernel` | Kernels are entry points | Move shared code into a helper |

### Example: everything together

```kotlin
import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId
import kotlin.math.cos
import kotlin.math.sin
import kotlin.math.sqrt

fun smooth(x: Float): Float = x / (1f + sqrt(x * x))   // helper → WGSL fn

@Kernel
fun waves(input: FloatArray, out: FloatArray, rounds: Int) {
    val i = globalId()
    var v = input[i]
    for (r in 0 until rounds) {
        v = smooth(sin(v) * cos(v * 0.5f) + r * 0.001f)
    }
    out[i] = v
}
```

### Example: neighbours and bounds

```kotlin
/** 1D box blur with radius 4: each output is the average of 9 neighbours. */
@Kernel
fun blur(input: FloatArray, out: FloatArray) {
    val i = globalId()
    val n = input.size
    var sum = 0f
    var count = 0
    for (d in -4..4) {
        val j = i + d
        if (j >= 0 && j < n) {
            sum += input[j]
            count++
        }
    }
    out[i] = sum / count
}
```

### Example: 2D and 3D launches **(main)**

Image and grid kernels can run as a `width x height` (or `width x height x depth`) grid of threads. Inside,
`globalIdX()` is the column, `globalIdY()` the row and `globalIdZ()` the layer. `globalId()` is the row-major index
`x + y * width + z * width * height`, so it is the cell's index in a flat array.

```kotlin
@Kernel
fun blur(src: FloatArray, dst: FloatArray, width: Int, height: Int) {
    val x = globalIdX()
    val y = globalIdY()
    var sum = 0f
    var count = 0
    for (dy in -1..1) for (dx in -1..1) {
        if (x + dx in 0 until width && y + dy in 0 until height) {
            sum += src[(y + dy) * width + x + dx]
            count++
        }
    }
    dst[globalId()] = sum / count
}

KernelinCpu.dispatch(width, height) { blur(pixels, out, width, height) }        // CPU
gpu.dispatch(width, height) { blur(src = pixels, dst = out, width = width, height = height) }   // GPU
gpu.load(kernelSource(::blur)).run(width, height, scalars, buffers)            // GPU, with names
```

- The 3D forms are `dispatch(width, height, depth)` and `run(width, height, depth, ...)`; `dispatchParallel` has
  them too.
- Threads run in the same order everywhere: row by row, layer by layer. CPU and GPU give the same ids.
- In a 1D launch (`dispatch(n)`), `globalIdX()` equals `globalId()` and `globalIdY()` / `globalIdZ()` are 0.
- A launch has at most `Int.MAX_VALUE` threads; a negative size or a bigger launch throws `IllegalArgumentException`.
  Big grids are fine on the GPU: like 1D launches, they are spread over the GPU's workgroup limit (65535 per
  dimension) and the extra threads are skipped.
- Kernels that use these ids have two more uniform fields, `kernelin_width` and `kernelin_height`, after
  `kernelin_size`. `KernelManifest.usesGridIds` and `KernelManifest.scalarOffset` tell custom hosts where the
  scalars start. Kernels that don't use them keep the old layout.

### Example: 2D data by hand

Before 2D launches (and in 0.3.0), store 2D data in a flat array and compute the row and column from the id.

```kotlin
@Kernel
fun mandelbrot(out: IntArray, width: Int, height: Int, maxIter: Int) {
    val id = globalId()
    val px = id % width
    val py = id / width
    val x0 = px.toFloat() / width * 3.5f - 2.5f
    val y0 = py.toFloat() / height * 2.0f - 1.0f
    var x = 0f
    var y = 0f
    var iter = 0
    while (x * x + y * y <= 4f && iter < maxIter) {
        val xt = x * x - y * y + x0
        y = 2f * x * y + y0
        x = xt
        iter++
    }
    out[id] = iter
}
```

## Run on the CPU

`KernelinCpu` runs a kernel as normal Kotlin. Use it in unit tests and as a reference result.

```kotlin
val a = FloatArray(1_000) { it.toFloat() }
val b = FloatArray(1_000) { 1f }
val out = FloatArray(1_000)

// One thread after another on the calling thread (JVM and Wasm).
KernelinCpu.dispatch(out.size) { vectorAdd(a, b, out) }

// JVM only: split the ids across all CPU cores.
KernelinCpu.dispatchParallel(out.size) { vectorAdd(a, b, out) }

// 2D and 3D grids (main): see "Example: 2D and 3D launches".
KernelinCpu.dispatch(width, height) { blur(pixels, out, width, height) }
```

Calling `globalId()` (or `globalIdX()`, `globalIdY()`, `globalIdZ()`) outside a dispatch throws `IllegalStateException`.

### Unit testing a kernel

```kotlin
class SaxpyTest {
    @Test
    fun saxpyOnCpu() {
        val x = floatArrayOf(1f, 2f, 3f)
        val y = floatArrayOf(10f, 20f, 30f)
        KernelinCpu.dispatch(3) { saxpy(2f, x, y) }
        assertContentEquals(floatArrayOf(12f, 24f, 36f), y)
    }
}
```

## Run on the GPU

GPU execution is provided by `kernelin-webgpu` on the **Kotlin/Wasm** (`wasmJs`) target, in any browser with WebGPU,
and **(main)** on **Android** 8.0+ natively, with no browser or WebView (see [On Android](#on-android-main)).
The API is the same on both.

```kotlin
import io.github.rajumark.kernelin.kernelSource
import io.github.rajumark.kernelin.webgpu.GpuData
import io.github.rajumark.kernelin.webgpu.WebGpu

suspend fun runSaxpy(x: FloatArray, y: FloatArray): FloatArray {
    check(WebGpu.isSupported) { "WebGPU isn't available in this browser" }
    val gpu = WebGpu.create()                       // pick the GPU, once per app
    val kernel = gpu.load(kernelSource(::saxpy))    // compile the pipeline, once per kernel
    val result = kernel.run(
        size = x.size,                              // number of threads
        scalars = mapOf("alpha" to 2.5f),           // every Int/Float parameter, by name
        buffers = mapOf(                            // every array parameter, by name
            "x" to GpuData.of(x),
            "y" to GpuData.of(y),
        ),
    )
    return result.floats("y")                       // read_write buffers come back
}
```

### Typed launches **(main)**

`gpu.dispatch` runs a kernel with a normal Kotlin call, so the compiler checks it:

```kotlin
val gpu = WebGpu.create()
gpu.dispatch(x.size) { saxpy(alpha = 2.5f, x = x, y = y) }   // y now holds the result
```

- The block is **one call to a `@Kernel` function**. The compiler checks the parameter names, types and count like
  any other call, and the IDE completes them. A typo (`alpah = 2.5f`) or a wrong type (`2.5` instead of `2.5f`,
  an `IntArray` for a `FloatArray`) is a compile error, not a run-time error.
- The kernel does **not** run on the CPU here. The compiler plugin turns the call into a `KernelLaunch` (the shader
  plus the argument values), and `dispatch` runs it on the GPU.
- It works like `KernelinCpu.dispatch(n) { saxpy(2.5f, x, y) }`: afterwards, the arrays the kernel writes hold the
  results. `dispatch` also returns the `GpuResult` (with `millis` and the GPU buffers).
- The kernel is loaded on the first `dispatch` and reused after that.
- Anything other than one kernel call in the block is a compile error:
  `a launch block must be exactly one call to a @Kernel function`. Compute other values before the block.
- The kernel must come from the same module, like `kernelSource(::k)`.

Use `gpu.load(...)` and `run(...)` with names (below) when you need more control: data that stays on the GPU
(`GpuData.of(buffer)`), `GpuData.empty(n)` outputs, or choosing what to read back.

### GPU or CPU in one call **(main)**

`Compute` runs kernels on the GPU when the device has one, and on the CPU (`KernelinCpu`) when it doesn't, so you
don't write two code paths:

```kotlin
val compute = Compute.create()                         // the GPU if it can be opened, else the CPU
compute.dispatch(n) { saxpy(alpha = 2.5f, x = x, y = y) }  // y holds the result either way
println(compute.description)                           // "GPU: apple metal-3 (Metal)" or "CPU (no GPU: ...)"
```

- The launch block is a typed kernel call, checked by the compiler, like `WebGpu.dispatch`.
- On the CPU, the same call runs for every id: on all cores on the JVM and Android, on the calling thread on
  Wasm, JS and Native. Each argument is evaluated once, not once per thread.
- `Compute.create()` uses the CPU when there is no WebGPU, no GPU adapter (for example an Android emulator) or the
  device can't be opened. `cpuReason` says why; `isGpu` says where kernels run now.
- If the GPU device is **lost** later, the run that hit it is redone on the CPU, and later runs use the CPU too.
  Other GPU errors (a rejected pipeline, out of memory) still throw: they are bugs to fix, not missing hardware.
- `Compute.create(useGpu = false)` always uses the CPU, which is handy in tests and for comparing results.
- Results match within the [differences between CPU and GPU](#differences-between-cpu-and-gpu) (Float rounding).
- For data that stays on the GPU between runs, use `WebGpu` directly.

### On Android **(main)**

On Android, `kernelin-webgpu` runs kernels through [AndroidX WebGPU](https://developer.android.com/jetpack/androidx/releases/webgpu)
(Google's Dawn on Vulkan, or on OpenGL ES 3.1 when Vulkan can't be used). The Gradle plugin doesn't add it for
Android yet, so add the dependency yourself:

```kotlin
dependencies {
    implementation("io.github.rajumark:kernelin-webgpu:<next version>")
}
```

```kotlin
val gpu = WebGpu.create()                           // throws GpuException when there is no usable GPU
val result = gpu.load(kernelSource(::saxpy)).run(x.size, mapOf("alpha" to 2.5f), mapOf("x" to GpuData.of(x), "y" to GpuData.of(y)))
val y2 = result.floats("y")
```

- `load` and `run` are `suspend` and can be called from any coroutine, including the main thread: every GPU call
  runs on one background thread that the `WebGpu` owns. `close()` stops it.
- `GpuResult.bytes(name)` is a little-endian `java.nio.ByteBuffer` (no copy); `floats` / `ints` copy into arrays.
- Fall back to `KernelinCpu` when `WebGpu.create()` throws `GpuException`. **Emulators** offer no real GPU to
  WebGPU (only Dawn's "Null" adapter, which runs nothing), so `create()` throws there; test GPU code on a phone.
- Needs `minSdk` 26. The library adds Dawn's native code for `arm64-v8a`, `armeabi-v7a`, `x86` and `x86_64`
  (several MB per ABI), so use App Bundles or ABI splits.
- Benchmark on your phone: `scripts/bench-android-app.sh` builds and runs the `samples-android` app, which times
  the sample kernels on the phone's GPU and CPU and checks that the results match.

### Rules for `run`

- `size` is the number of threads (usually the length of the output array). For a 2D or 3D grid, pass
  `run(width, height, ...)` or `run(width, height, depth, ...)` instead **(main)**.
- `scalars` must contain **every** `Int` and `Float` parameter by its Kotlin name (and `Boolean`, since 0.2.0; the type is `Map<String, Any>`, source compatible with 0.1.0, but Wasm libraries built against 0.1.0 that call `run()` need a recompile).
- `buffers` must contain **every** array parameter by its Kotlin name.
- `run` checks the arguments before anything reaches the GPU: an unknown name, an `Int` parameter given a `Float`
  (or any other wrong scalar type), and `FloatArray` data for an `IntArray` parameter (or the other way round) throw
  `IllegalArgumentException` (since 0.3.0; before, they were silently ignored or reinterpreted).
- Use `GpuData.empty(n)` for pure outputs, so nothing is uploaded. It starts from zeros on every run (since 0.3.0;
  before, a second run of the same size kept the previous run's values).
- Only arrays the kernel writes (`read_write`) are copied back. Read them with `result.floats(name)` or `result.ints(name)`.
  Pass `readBack = setOf("out")` to copy back only some of them, or `emptySet()` for none (since 0.3.0).
- `result.millis` is the time from upload until the results are back.
- Runs of one `GpuKernel` never overlap: if you call `run` again before the first one finished, the second waits
  (since 0.3.0; before, the two runs shared GPU buffers and one of them failed).
- When WebGPU reports an error (out of GPU memory, a rejected command, a lost device), `run` and `load` throw
  `GpuException` with the browser's message (since 0.3.0; before, errors only appeared in the browser console).
  Every GPU buffer, pipeline and bind group has a label like `my.pkg.blur: out`, so browser messages name the
  kernel and the parameter.

### Keep data on the GPU **(0.3.0)**

Copying data to the CPU and back is often slower than the kernel itself. To run several kernels one after the
other, keep the data on the GPU: skip the read back with `readBack`, take the buffer with `result.buffer(name)`
and pass it to the next run with `GpuData.of(buffer)`. The data is copied on the GPU, never through the CPU.

```kotlin
var cells = GpuData.of(start)                                  // uploaded once
repeat(100) {
    val step = life.run(n, mapOf("width" to w), mapOf("cells" to cells, "next" to GpuData.empty(n)), readBack = emptySet())
    cells = GpuData.of(step.buffer("next"))                    // stays on the GPU
}
val last = life.run(n, mapOf("width" to w), mapOf("cells" to cells, "next" to GpuData.empty(n)))
val result = last.ints("next")                                 // one copy back at the end
```

A `GpuBuffer` is the kernel's own buffer, so it is valid only until that kernel runs again or is closed (check
`isValid`); using an older one throws `IllegalStateException`. Pass it on before the next run of its kernel, as
above. It must come from a kernel of the same `WebGpu`.

### Lost devices and cleaning up **(0.3.0)**

The browser can take the GPU away at any time (a driver update, a GPU reset, too many tabs). Then `gpu.isLost` is
true, `gpu.onLost` is called with the reason, and every `load` and `run` throws `GpuException`. Create a new
`WebGpu` and load the kernels again:

```kotlin
gpu.onLost = { reason -> console.log("GPU lost: $reason"); scope.launch { gpu = WebGpu.create(); kernel = gpu.load(source) } }
gpu.onUncapturedError = { message -> console.log("WebGPU error: $message") }  // errors outside load and run
```

`kernel.close()` frees the kernel's GPU buffers; `gpu.close()` frees every kernel and the device. Both are
`AutoCloseable`, so `use { }` works. A `run` after `close` throws `IllegalStateException`.

### Performance tips

- **Create once, run many.** `WebGpu.create()` and `gpu.load(...)` are slow; `kernel.run(...)` is fast. Keep the `GpuKernel` and call `run` again; GPU buffers are reused when sizes match.
- **Do enough work per element.** Copying data to the GPU and back costs time. Light kernels (one add per element) can be slower end-to-end than the CPU; heavy math wins big (see the benchmarks on the home page).
- **Use `GpuData.ofJs(typedArray)`** to pass an existing JS `Float32Array` without a copy, and `result.js(name)` to get the output typed array without a copy.
- **Keep data on the GPU** between kernels: `readBack = emptySet()` and `GpuData.of(result.buffer(name))` (see
  [Keep data on the GPU](#keep-data-on-the-gpu-030)). Read back only what you need, once, at the end.
- **Copies between Kotlin arrays and the GPU are bulk copies** (since 0.3.0): `GpuData.of(FloatArray)`,
  `result.floats(name)` and `toJsFloat32Array()` copy 4M floats in about 3 ms in Node (before: about 20 ms,
  one element at a time). `ofJs` / `js` still avoid even that copy.

### Differences between CPU and GPU

The same kernel gives the same results on the CPU and the GPU, with these known differences:

| Case | On the CPU (Kotlin) | On the GPU (WGSL) |
|---|---|---|
| Float rounding in `sin`, `exp`, `pow`, … | IEEE 754 double-backed results | 32-bit, may differ in the last bits. Compare with a tolerance |
| `for (i in a..b step s)` with `s` 0 or less | Throws `IllegalArgumentException` | **No iterations** (since 0.3.0; a constant step of 0 or less is a compile error). Before, the loop never ended |
| `min`, `max`, `coerceIn`, `sign`, `compareTo` with NaN or `-0f` | NaN wins in `min`/`max`, `-0f < 0f`, NaN is the largest in `compareTo` | **Same as Kotlin** (since 0.3.0). WGSL leaves these unspecified, so 0.2.0 and older may differ |
| `hypot`, `log10` | Computed in Double, rounded once | **Within 2 ULPs (`hypot`) and 4 ULPs (`log10`)** (since 0.3.0). Before, `hypot` overflowed to Infinity for values above about 1e19 |
| `pow` with a negative or zero base | `(-2f).pow(3f) == -8f`, `(-2f).pow(0.5f)` is NaN | **Same as Kotlin** (since 0.3.0). WGSL leaves these undefined, so 0.2.0 and older may return NaN on Vulkan GPUs |
| Float division by zero (`x / 0f`) | `Infinity` or `NaN` | **Not guaranteed.** WGSL leaves it undefined; some GPUs (for example llvmpipe, used in CI) return an ordinary number. Check the divisor yourself |
| `Infinity` / `NaN` that arrive in a buffer | Normal Kotlin behaviour | **Same as Kotlin** (since 0.2.0): `isNaN()`, `isInfinite()`, `isFinite()` and `toInt()` check the raw bits |
| `Float.toInt()` on NaN or out-of-range values | NaN → 0, saturates at `Int.MIN_VALUE` / `Int.MAX_VALUE` | **Same as Kotlin** (since 0.2.0). On 0.1.0 the result is GPU-specific |
| Array index out of bounds | Throws `ArrayIndexOutOfBoundsException` | No error: the access is clamped silently. Test on the CPU to catch it |
| Integer division by zero | Throws `ArithmeticException` | No error: returns a defined value. Since 0.2.0, a **constant** division like `5 / 0` is a compile error |
| `randomNormal` **(main)** | Computed with `ln`, `sqrt`, `cos` | Within about 1e-5 of the CPU, not bit for bit. `randomInt`, `randomFloat` and `randomIntBelow` are **bit-identical** |
| `randomIntBelow` with `bound` 0 or less **(main)** | Throws `IllegalArgumentException` | Returns 0 |
| Two threads writing the same element | `dispatch`: the highest id wins; `dispatchParallel`: a race | Undefined order. Give each thread its own element |

Rule of thumb: don't rely on producing `Infinity` or `NaN` inside a kernel. Guard divisions (`if (d != 0f) x / d else 0f`).

### Other ways to use the shader

`kernelSource(::k)` gives you the WGSL and manifest on every platform, so any WebGPU host can run it (for example wgpu, Dawn, or a native app). On the JVM the plugin also writes the files into the compiled resources: `KernelinShaders.wgsl("com.example.saxpy")`.

## Compiler errors

Unsupported code fails the build, like any Kotlin error. The message always starts with `Kernelin:` and ends with the kernel it belongs to.

```text
e: Kernels.kt:1:48 Kernelin: recursion isn't allowed on the GPU ('fact' calls itself) (in @Kernel factorials)
```

Common messages and fixes:

| Message (shortened) | Fix |
|---|---|
| `Double isn't supported on most GPUs; use Float (add an f suffix: 1.0f)` | Use Float literals: `0.5f` |
| `Long isn't supported on the GPU; use Int` | Use `Int` |
| `strings aren't supported on the GPU` | Remove strings / `println` |
| `lambdas aren't supported in kernels yet ... use a for loop` | Replace `repeat {}` / `forEach {}` with `for` |
| `recursion isn't allowed on the GPU` | Rewrite as a loop |
| `a @Kernel function must return Unit` | Write the result into an array |
| `parameter 'p' has type X; kernels accept FloatArray, IntArray, Int, Float and Boolean` | Change the parameter type |
| `variable 'x' has type X; only Int, Float and Boolean are supported` | Use `Int`, `Float` or `Boolean` |
| `nullable values aren't supported on the GPU` | Remove `?.`, `?:` and `!!` |
| `creating objects or arrays isn't supported in kernels` | Pass arrays in as kernel parameters |
| `'kotlin.collections.sum' can't run on the GPU` | Write the loop yourself |
| `for loops must iterate over an Int range (until, .., ..<, downTo, step, indices)` | Use an Int range |
| `step must be positive, but it is 0 (Kotlin throws IllegalArgumentException)` | Use a step of 1 or more |
| `only one step per range is supported; combine the steps into one` | Write one `step` |
| `an if/when used as a value needs an else branch` | Add `else` |
| `integer division by zero (Kotlin would throw ArithmeticException here)` **(0.2.0)** | Fix the constant divisor |
| `kernelSource() was not replaced: apply the Kernelin compiler plugin` (at runtime) | Apply the Gradle plugin to that module |
| `a launch block must be exactly one call to a @Kernel function` **(main)** | Write `gpu.dispatch(n) { k(...) }` with only the call; compute other values before it |
| `pass a value for every kernel parameter ('p' has none)` **(main)** | Default values can't be launched; pass every argument |
| `No @Kernel call was recorded` (at runtime) **(main)** | Apply the Gradle plugin to the module with the `dispatch` call |

### Errors in your IDE **(0.2.0)**

Kernelin checks kernels in the K2 frontend too (FIR checkers), so IntelliJ IDEA and Android Studio underline unsupported code **while you type**, with the same messages as the build. The frontend checks cover the common mistakes: kernel signature, parameter and variable types, `Double`/`Long`/`String` values, lambdas, library calls, object creation, exceptions, casts and nullable values. Everything else (for example recursion or unsupported loop shapes) is still checked when you build.

**(Unreleased)** The same checks run on your helper functions that a kernel calls (and the helpers they call), plus the helper rules (return and parameter types, top-level, no type parameters). A helper in the kernel's file gets the error on its own code, ending with `(in helper 'h', called from @Kernel k)`. A helper in another file gets one error on the call in the kernel, naming the helper's file, because the IDE shows a file's errors only while it checks that file.

To see them in the IDE:

1. Use the IDE's K2 mode (the default in IntelliJ IDEA 2025.1+ and recent Android Studio).
2. The IDE runs only its bundled compiler plugins by default. Open **Help → Find Action → Registry…** and set `kotlin.k2.only.bundled.compiler.plugins.enabled` to `false`, then reload the Gradle project.
3. Use a Kotlin plugin in the IDE that matches Kotlin 2.4.21, because the compiler plugin API changes between versions.

Frontend errors stop the build before the translator runs. If a kernel has both kinds of mistakes, the remaining errors show up after you fix the first ones.

With the compiler's `-verbose` flag, the build log shows `Kernelin: translated @Kernel com.example.saxpy to WGSL (N lines)` for each kernel (since 0.3.0; before, every build printed it).

## API reference

All runtime APIs are in package `io.github.rajumark.kernelin`; the GPU host is in `io.github.rajumark.kernelin.webgpu`.

### `@Kernel`

```kotlin
@Target(AnnotationTarget.FUNCTION)
@Retention(AnnotationRetention.BINARY)
annotation class Kernel
```

Marks a top-level function as a GPU kernel. The function must return `Unit`, have no type parameters, no receiver, and not be `suspend`.

### `globalId()`

```kotlin
fun globalId(): Int
```

The index of the current thread, from `0` to `size - 1`. On the GPU it is the global invocation id; on the CPU it is set by `KernelinCpu`.
In a 2D or 3D launch it is the row-major index `x + y * width + z * width * height`.

### `globalIdX()`, `globalIdY()`, `globalIdZ()` **(main)**

```kotlin
fun globalIdX(): Int   // column: 0 until width
fun globalIdY(): Int   // row:    0 until height (0 in a 1D launch)
fun globalIdZ(): Int   // layer:  0 until depth  (0 in a 1D or 2D launch)
```

The position of the current thread in a `dispatch(width, height)` or `dispatch(width, height, depth)` launch. In a
1D launch, `globalIdX()` equals `globalId()`. See [2D and 3D launches](#example-2d-and-3d-launches-main).

### Random numbers **(main)**

```kotlin
fun randomInt(seed: Int, index: Int, stream: Int = 0): Int            // all 32 bits random
fun randomFloat(seed: Int, index: Int, stream: Int = 0): Float        // in [0, 1), never 1
fun randomIntBelow(bound: Int, seed: Int, index: Int, stream: Int = 0): Int  // in 0 until bound
fun randomNormal(seed: Int, index: Int, stream: Int = 0): Float       // mean 0, standard deviation 1
```

Random numbers for kernels, in `io.github.rajumark.kernelin`. They keep no state: the same `(seed, index, stream)` always gives the same number. Use `globalId()` as the `index`, and number the draws of one thread with `stream` (0, 1, 2, ...). So the results don't depend on how the work is split into threads, and a run is easy to repeat: pass the `seed` as a kernel parameter.

```kotlin
@Kernel
fun monteCarloPi(seed: Int, hits: IntArray) {
    val i = globalId()
    val x = randomFloat(seed, i, 0)
    val y = randomFloat(seed, i, 1)
    hits[i] = if (x * x + y * y < 1f) 1 else 0
}
```

- `randomInt`, `randomFloat` and `randomIntBelow` give **exactly the same bits** on the CPU (every platform) and the GPU. The math is 32-bit unsigned, where Kotlin's `Int` wraps the same way. `randomFloat` is the top 24 bits times 2^-24, which is exact in a Float.
- `randomNormal` uses the Box-Muller method (`sqrt`, `ln`, `cos`), so the GPU is within about 1e-5 of the CPU, not bit for bit. Its values stay within about ±5.8.
- `randomIntBelow` needs a positive `bound`: the CPU throws, the GPU returns 0. It is very slightly biased (at most `bound / 2^32`).
- For one `seed` and `stream`, different indexes always give different `randomInt` values.
- The bits come from three rounds of the PCG hash (Jarzynski and Olano, 2020). Good for simulations, noise and sampling, not for cryptography.
- They also work in your helper functions and on the CPU outside a kernel.

### `kernelSource(kernel)`

```kotlin
fun kernelSource(kernel: KFunction<*>): KernelSource
```

Returns the generated shader for a kernel: `kernelSource(::saxpy)`. The compiler replaces the call with the shader text at compile time, so it works on every platform without files or reflection.

### `KernelSource`

| Member | Type | Description |
|---|---|---|
| `name` | `String` | Fully qualified kernel name, e.g. `com.example.saxpy` |
| `wgsl` | `String` | The WGSL compute shader, entry point `main` |
| `manifestJson` | `String` | JSON description of parameters and bindings |
| `manifest` | `KernelManifest` | The parsed manifest |

### `KernelManifest`

| Member | Description |
|---|---|
| `kernel` | Fully qualified kernel name |
| `entryPoint` | Always `"main"` |
| `workgroupSize` | Threads per workgroup (64) |
| `params` | All parameters in Kotlin order |
| `buffers` / `scalars` | Parameters filtered by kind |
| `uniformBytes` | Size of the uniform buffer in bytes |
| `usesGridIds` **(main)** | `true` when the kernel reads `globalIdX/Y/Z`; the uniform then has `kernelin_width` and `kernelin_height` (u32) after `kernelin_size` |
| `scalarOffset` **(main)** | Byte offset of the first scalar in the uniform: 4, or 12 when `usesGridIds` |
| `Param.name`, `kind`, `type`, `binding`, `access` | `kind` is `buffer` or `scalar`; `type` is `f32`, `i32` or `u32` (plus `bool` since 0.2.0); `access` is `read` or `read_write` |

### `KernelinCpu`

| Function | Platforms | Description |
|---|---|---|
| `dispatch(size) { ... }` | all (JVM, JS, Wasm, Native) | Runs the block once per id in `0 until size` on the calling thread |
| `dispatchParallel(size) { ... }` | JVM | Same, split across all CPU cores |
| `dispatch(width, height) { ... }`, `dispatch(width, height, depth) { ... }` **(main)** | all | A 2D / 3D grid, row by row; read the position with `globalIdX/Y/Z()` |
| `dispatchParallel(width, height)`, `dispatchParallel(width, height, depth)` **(main)** | JVM | Same, split across all CPU cores |

### `KernelLaunch` **(main)**

One kernel call recorded by the compiler plugin from a launch block (a lambda with a `KernelLaunch` receiver, such
as the one `WebGpu.dispatch` takes). For GPU hosts; users just write the block.

| Member | Description |
|---|---|
| `KernelLaunch.capture(block)` | Runs the block and returns the call. Throws `IllegalStateException` if it holds no kernel call (plugin not applied) |
| `source` | The kernel's `KernelSource` |
| `scalars` | `Int` / `Float` / `Boolean` arguments by parameter name |
| `arrays` | `IntArray` / `FloatArray` arguments by parameter name (the same array objects) |
| `runOnCpu(size)` | Runs the same call on the CPU for ids `0 until size` (all cores on the JVM and Android) |
| `runOnCpu(width, height)`, `runOnCpu(width, height, depth)` | The same for a 2D / 3D grid |

### `Compute` **(main)**

Runs kernels on the GPU when there is one, else on the CPU. In `kernelin-webgpu` (Kotlin/Wasm and Android).

| Member | Description |
|---|---|
| `suspend Compute.create(useGpu = true)` | Opens the GPU, or uses the CPU when there is none (or when `useGpu` is false). Never throws for missing hardware |
| `suspend dispatch(size) { kernel(...) }` | Runs one typed kernel call on the GPU or the CPU; the arrays the kernel writes hold the results |
| `suspend dispatch(width, height) { ... }`, `dispatch(width, height, depth) { ... }` | The same for a 2D / 3D grid (see `globalIdX()`) |
| `isGpu` | `true` while kernels run on the GPU |
| `cpuReason` | Why kernels run on the CPU (no GPU, lost device, `useGpu = false`, closed), or `null` |
| `description` | `GPU: <name>` or `CPU (<reason>)`, for logs |
| `close()` | Frees the GPU; later runs use the CPU |

### `KernelinShaders` (JVM)

| Function | Description |
|---|---|
| `wgsl(fqName)` | Reads `kernelin/<fqName>.wgsl` from the classpath |
| `manifest(fqName)` | Reads `kernelin/<fqName>.json` from the classpath |

### `WebGpu` (Kotlin/Wasm, and Android **(main)**)

| Member | Description |
|---|---|
| `WebGpu.isSupported` | `true` when the browser exposes `navigator.gpu`; on Android, on API 26+ |
| `suspend WebGpu.create()` | Requests a GPU device. Throws `UnsupportedOperationException` without WebGPU |
| `description` | Human-readable GPU description |
| `suspend load(source)` | Compiles the shader and pipeline. Throws if the browser rejects the WGSL, `GpuException` if it rejects the pipeline |
| `suspend dispatch(size) { k(...) }` | Typed launch: runs one `@Kernel` call on the GPU, writes the results into the arrays the kernel writes, returns the `GpuResult`. Loads the kernel once **(main)** |
| `suspend dispatch(width, height) { k(...) }`, `dispatch(width, height, depth) { k(...) }` **(main)** | The same for a 2D / 3D grid of threads (see `globalIdX()`) |
| `isLost`, `lostReason` | Whether the browser took the device away, and why **(since 0.3.0)** |
| `onLost` | Called once with the reason when the device is lost (not after `close()`) **(since 0.3.0)** |
| `onUncapturedError` | Called with WebGPU errors that happen outside `load` and `run` **(since 0.3.0)** |
| `close()`, `isClosed` | Frees every kernel and the device; `AutoCloseable` **(since 0.3.0)** |

### `GpuKernel`

| Member | Description |
|---|---|
| `source`, `manifest` | What this kernel was built from |
| `suspend run(size, scalars, buffers, readBack = null)` | Uploads inputs, dispatches `size` threads, reads back the `read_write` buffers named in `readBack` (null: all). Runs of one kernel wait for each other. Throws `GpuException` on WebGPU errors |
| `suspend run(width, height, scalars, buffers, readBack = null)`, `run(width, height, depth, ...)` **(main)** | The same for a 2D / 3D grid of threads |
| `close()`, `isClosed` | Frees the kernel's GPU buffers; later runs throw `IllegalStateException`. `AutoCloseable` **(since 0.3.0)** |

### `GpuData`

| Factory | Description |
|---|---|
| `GpuData.of(FloatArray)` / `GpuData.of(IntArray)` | Upload these values |
| `GpuData.empty(length)` | Output only, nothing uploaded; zero-filled on every run |
| `GpuData.ofJs(typedArray)` | Wrap an existing JS typed array without copying |
| `GpuData.of(GpuBuffer)` | Use a buffer already on the GPU (from `result.buffer(name)`); copied on the GPU **(since 0.3.0)** |

### `GpuResult`

| Member | Description |
|---|---|
| `millis` | Time from upload until results were back |
| `floats(name)` / `ints(name)` | Copy an output buffer into a Kotlin array |
| `js(name)` | The raw JS typed array, no copy |
| `buffer(name)` | The kernel's `GpuBuffer` for parameter `name` after this run, still on the GPU **(since 0.3.0)** |

### `GpuBuffer` **(0.3.0)**

| Member | Description |
|---|---|
| `name`, `length` | The kernel parameter and its number of elements |
| `isValid` | `false` once its kernel ran again or was closed; then passing it to `run` throws `IllegalStateException` |

### `GpuException` **(0.3.0)**

Thrown by `load` and `run` when WebGPU reports an error: a validation error, out of GPU memory, a rejected
pipeline, or a lost device. The message is the browser's, with the kernel name.

## How the compiler works

```text
@Kernel fun (Kotlin source)
   │  K2 frontend: parse, resolve, type-check
   ▼
Kotlin IR  ──►  KernelinIrGenerationExtension
                  ├─ WgslKernelTranslator: IR tree → WGSL text + manifest
                  ├─ errors → normal compiler diagnostics (file:line)
                  └─ kernelSource(::k) calls → replaced with KernelSource(name, wgsl, manifest)
   ▼
Platform backends (JVM, Wasm, …) compile the rest as usual
```

1. **Find kernels.** After type checking, the plugin's `IrGenerationExtension` visits every function annotated with `@Kernel`.
2. **Translate.** `WgslKernelTranslator` walks the IR tree before any platform lowering, so loops, `&&`, `++` and `+=` still have their source shape. Each expression maps to a WGSL expression of type `i32`, `f32` or `bool`. Helper functions are translated once and emitted as WGSL `fn`s.
3. **Report.** Anything outside the supported subset becomes an error at the exact source location. The build fails before a bad shader can exist.
4. **Embed.** Every `kernelSource(::k)` call is replaced with a `KernelSource(...)` constructor call holding the WGSL and the manifest as string constants. On the JVM, the files are also written into the compiled resources under `kernelin/`.

### Generated shader layout

```wgsl
struct KernelinParams {
  kernelin_size: u32,          // number of threads, set by the host
  alpha: f32,                  // one field per Int/Float parameter, in Kotlin order
}

@group(0) @binding(0) var<uniform> kernelin_params: KernelinParams;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // arrays: bindings 1, 2, 3...
@group(0) @binding(2) var<storage, read_write> y: array<f32>;    // written arrays are read_write

var<private> kernelin_id: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) kernelin_gid: vec3<u32>, @builtin(num_workgroups) kernelin_nwg: vec3<u32>) {
  kernelin_id = kernelin_gid.x + kernelin_gid.y * kernelin_nwg.x * 64u;
  if (kernelin_id >= kernelin_params.kernelin_size) {
    return;
  }
  let i: i32 = i32(kernelin_id);
  y[i] += (kernelin_params.alpha * x[i]);
}
```

- Workgroup size is **64**. Large dispatches use a 2D grid of workgroups (x and y), which the id formula folds back into one index.
- Binding `0` is always the uniform. The uniform is padded to a multiple of 16 bytes.
- Names that collide with WGSL keywords are renamed safely.

### Manifest format

```json
{
  "kernel": "com.example.saxpy",
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
```

To run a kernel from any WebGPU host: write `size` and the scalars (little-endian, 4 bytes each, in `uniform.fields` order) into the uniform buffer, bind every buffer at its `binding`, dispatch `ceil(size / 64)` workgroups, and read back the `read_write` buffers.

## Project structure

| Module | What it is |
|---|---|
| `kernelin-runtime` | Multiplatform (JVM, JS, Wasm, Native): `@Kernel`, `globalId()`, `KernelinCpu`, `kernelSource`, `KernelManifest` |
| `kernelin-compiler-plugin` | K2 compiler plugin; `WgslKernelTranslator` turns Kotlin IR into WGSL |
| `kernelin-webgpu` | Kotlin/Wasm WebGPU host |
| `kernelin-gradle-plugin` | `id("io.github.rajumark.kernelin")` |
| `samples` | Sample kernels, CPU tests, benchmark, live demo |
| `examples` | Standalone `hello-jvm` and `hello-multiplatform` projects |
| `benchmarks` | Raw benchmark results (JSON) |
| `docs` | This website, the live demo and these docs |

## Building from source

Requires JDK 21.

```bash
git clone https://github.com/rajumark/kernelin.git
cd kernelin
gradle build
```

Tests check every generated shader with the [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) validator when it is installed:

```bash
cargo install naga-cli
```

On main, GPU differential tests also run kernels as Kotlin on the CPU and as WGSL on a real GPU (WebGPU through Deno) and compare the results. Install Deno to run them locally; without it they are skipped. CI runs them on a software GPU (llvmpipe through Vulkan, via Deno); they must pass, and CI fails if no GPU is found.

```bash
brew install deno
scripts/check-incremental.sh                   # checks incremental builds (JVM, JS, Wasm) use the current shader
```

Quality checks, all part of `gradle build`:

| Check | What it does |
|---|---|
| Fuzzer (`KernelFuzzTest`) | Random valid Int kernels (with exact Float values like NaN and `-0f`, and loops near `Int.MAX_VALUE`) must give identical results on CPU and GPU. 40 per build, 300 fresh ones on every CI run; run more with `-Pkernelin.fuzz.count=1000` (`-Pkernelin.fuzz.seed=N` to repeat a run) |
| ABI validation | The public API of every module is recorded in `api/` and checked by `checkKotlinAbi`; explicit API mode and KDoc on all public API |
| Coverage | Kover, currently 86%; the build fails below 80%. Report: `gradle koverHtmlReport` |

88 tests in total: 52 compiler tests (including 20 GPU tests, 9 IDE-check tests and the fuzzer), 9 runtime, 15 WebGPU host, 7 Gradle plugin, 5 samples.

Other tasks:

```bash
gradle :samples:bench -Pout=/path/to/folder   # benchmark; then serve the folder and open runner.html
scripts/bench-android.sh /path/to/folder      # same GPU benchmark in Chrome on a USB-connected Android phone
scripts/bench-android-cpu.sh                 # the CPU side on the phone (Kotlin on ART)
gradle :samples:exportDemo                     # rebuild the live demo into docs/demo
```

## Contributing and versioning

- [CONTRIBUTING.md](https://github.com/rajumark/kernelin/blob/main/CONTRIBUTING.md): how to set up, and the rule for adding a new Kotlin construct: translator change, WGSL test, GPU-vs-CPU test, fuzzer case and a row in the tables on this page.
- [VERSIONING.md](https://github.com/rajumark/kernelin/blob/main/VERSIONING.md): semantic versioning. Before 1.0, minor releases may change the public API (always listed in the changelog); patch releases never do. Each release supports exactly one Kotlin version. Public API = the runtime and WebGPU classes, the Gradle plugin id and the supported Kotlin subset; the exact WGSL text is not.
- Security: report vulnerabilities privately at [github.com/rajumark/kernelin/security/advisories/new](https://github.com/rajumark/kernelin/security/advisories/new), not in public issues. Policy: [SECURITY.md](https://github.com/rajumark/kernelin/blob/main/SECURITY.md).
- Issue and pull request templates are in `.github/`.

## Troubleshooting

**`kernelSource() was not replaced`** — The module calling `kernelSource` doesn't have the compiler plugin. Apply `id("io.github.rajumark.kernelin")` to that module.

**The plugin fails with an API error** — Your Kotlin version is not 2.4.21. Kernelin 0.3.0 only supports Kotlin 2.4.21.

**`WebGPU is not available in this browser`** — Use a recent Chrome, Edge or Safari, or enable WebGPU in Firefox. Check `WebGpu.isSupported` before calling `create()`.

**`Missing scalar parameter` / `Missing buffer parameter`** — `run` needs every parameter by its exact Kotlin name.

**`Unknown scalar parameter` / `Unknown buffer parameter`** — A name in `scalars` or `buffers` isn't a parameter of the kernel (often a typo). The message lists the real names.

**`Scalar parameter 'x' is Int, but got Float` / `Buffer parameter 'x' is IntArray, but got FloatArray data`** — Pass values of the parameter's Kotlin type, e.g. `GpuData.of(IntArray)` for an `IntArray` parameter.

**`GpuException: The GPU device was lost`** — The browser reset the GPU. Create a new `WebGpu` and load the kernels again; set `gpu.onLost` to find out right away.

**`GpuException: WebGPU rejected the run ... out of memory`** — The buffers don't fit in GPU memory. Use smaller sizes, or `close()` kernels you no longer need.

**`GpuBuffer 'x' ... is out of date`** — Its kernel ran again, so the buffer now holds newer data. Use `buffer(name)` from the latest result.

**GPU and CPU results differ slightly** — GPUs use 32-bit floats with slightly different rounding for some math functions. Compare floats with a tolerance. See [Differences between CPU and GPU](#differences-between-cpu-and-gpu), especially division by zero.

**Results are all zero** — Make sure the output array is written in the kernel (so it becomes `read_write`) and that `size` is not zero.

## FAQ

**Is this a DSL?** No. Kernels are ordinary Kotlin functions. There is no builder API and no strings of shader code.

**Why WGSL and WebGPU?** WebGPU is the one modern GPU API that runs on Metal, Vulkan and Direct3D 12, in browsers and natively. WGSL is its shading language.

**Can I run kernels on Android or desktop JVM GPUs?** The bundled GPU host is Kotlin/Wasm in the browser, which also works in Chrome on Android (tested on a Snapdragon phone with an Adreno GPU: all sample kernels match the CPU). The shader and manifest are available on every platform through `kernelSource`, so any native WebGPU host (wgpu, Dawn) can run them.

**Does it support Kotlin/Native or JS?** The runtime targets JVM and Wasm.

**Is it fast?** On an Apple M5, heavy kernels run up to 118× faster than the same Kotlin on all 10 CPU cores. Light kernels are limited by copying data.

## For AI agents

This section is written for AI coding agents (and the humans who instruct them). It is also served as plain text:

- [`llms.txt`](llms.txt) — short index of the docs for LLMs
- [`docs.md`](docs.md) — this whole page as Markdown

### Rules for writing a Kernelin kernel

Follow every rule. If a rule can't be met, don't write a kernel; write normal Kotlin instead.

1. Annotate with `@Kernel` (`import io.github.rajumark.kernelin.Kernel`). The function is **top-level**, returns **`Unit`**, has no receiver, no type parameters, and is not `suspend`.
2. Parameters are only `FloatArray`, `IntArray`, `Int`, `Float`, `Boolean`.
3. Start with `val i = globalId()` (`import io.github.rajumark.kernelin.globalId`). Each thread handles element `i`. Never write a loop over all elements.
4. Write results into an array parameter at index `i`. Never return a value from the kernel.
5. Use only `Int`, `Float`, `Boolean` locals. Always write Float literals with `f`: `0.5f`, `1f`. Never use `Double`, `Long`, `String`, nullable types, classes or collections.
6. No lambdas except `repeat(n) { }`: no `forEach {}`, `map {}`. Use `for (k in 0 until n)`, `while`, or `do-while`.
7. Math only from `kotlin.math` with Float arguments (`sqrt`, `sin`, `cos`, `exp`, `pow`, `abs`, `min`, `max`, …). For π use `PI.toFloat()`.
8. Helper functions must be top-level, take only `Int`/`Float`/`Boolean`, return `Int`/`Float`/`Boolean`/`Unit`, not recurse, have no default arguments. They may take the kernel's own array parameters.
9. 2D data is a flat, row-major array. On main, launch a grid with `dispatch(width, height)` and read `globalIdX()` / `globalIdY()` (`globalId()` is then `x + y * width`); with 0.3.0, use `x = i % width`, `y = i / width`.
10. Guard neighbour reads: `if (j >= 0 && j < a.size)`. The bounds check against `size` is automatic only for `globalId()` itself.
11. Never divide by a value that can be zero without a check: `x / 0f` is undefined on the GPU (it is not guaranteed to be `Infinity` or `NaN`).
12. For random numbers use Kernelin's `randomFloat(seed, i, stream)`, `randomInt`, `randomIntBelow` or `randomNormal` (`import io.github.rajumark.kernelin.*`), with a different `stream` for every draw in one thread. Don't use `kotlin.random` and don't write your own hash.

### Template

```kotlin
import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId

@Kernel
fun myKernel(input: FloatArray, out: FloatArray, factor: Float, count: Int) {
    val i = globalId()
    var acc = 0f
    for (k in 0 until count) {
        acc += input[i] * factor
    }
    out[i] = acc
}
```

### Testing template

```kotlin
import io.github.rajumark.kernelin.KernelinCpu
import kotlin.test.Test
import kotlin.test.assertEquals

class MyKernelTest {
    @Test
    fun runsOnCpu() {
        val input = floatArrayOf(1f, 2f, 3f)
        val out = FloatArray(3)
        KernelinCpu.dispatch(out.size) { myKernel(input, out, 2f, 1) }
        assertEquals(listOf(2f, 4f, 6f), out.toList())
    }
}
```

### GPU template (Kotlin/Wasm)

```kotlin
val gpu = WebGpu.create()
// Typed (main): checked by the compiler; `out` holds the result afterwards.
gpu.dispatch(n) { myKernel(input = input, out = out, factor = 2f, count = 1) }

// By name (0.3.0):
val kernel = gpu.load(kernelSource(::myKernel))
val result = kernel.run(
    size = n,
    scalars = mapOf("factor" to 2f, "count" to 1),
    buffers = mapOf("input" to GpuData.of(input), "out" to GpuData.empty(n)),
)
val out = result.floats("out")
```

### Snippet for your project's `AGENTS.md` or `CLAUDE.md`

```text
This project uses Kernelin 0.3.0 (Kotlin 2.4.21) to run @Kernel functions on the GPU.
Docs for agents: https://rajumark.github.io/kernelin/docs.md
When writing a @Kernel: top-level, returns Unit, params only FloatArray/IntArray/Int/Float/Boolean,
start with val i = globalId(), write results to out[i], Float literals with f,
no Double/Long/String/lambdas (except repeat)/recursion/classes; math only from kotlin.math.
Test every kernel on the CPU with KernelinCpu.dispatch(size) { kernel(...) }.
```

### Self-check before you finish

- [ ] The build passes (Kernelin errors start with `Kernelin:` and give file:line).
- [ ] A CPU unit test runs the kernel with `KernelinCpu.dispatch` and checks the output.
- [ ] Every `run(...)` call names all scalars and all buffers exactly as in the Kotlin signature.

## Roadmap

**Next (towards 1.0)**

- Workgroup shared memory and atomics (for matrix multiply, reductions and histograms); 2D/3D ids are done on main
- Vector types
- Android: kernels already run and match on a phone GPU in Chrome; next, AndroidX WebGPU natively
- Done in 0.2.0: IDE errors while you type (FIR checkers), `Boolean` parameters, arrays in helpers, `repeat`, labeled `break`/`continue`, `in` ranges, `isNaN`, Kotlin-exact constant math and shifts; GPU-vs-CPU differential tests, a fuzzer, ABI checks and coverage gates

**Later (2.0 and beyond)**

- Data classes as GPU structs, including arrays of structs
- Built-in `Vec2/3/4` and `Mat2/3/4` types with operators; `UInt`, `Short`, `Byte`
- More backends next to WGSL: SPIR-V (Vulkan), MSL (Metal) and CUDA

See the [changelog](https://github.com/rajumark/kernelin/blob/main/CHANGELOG.md) for released changes and the "Unreleased" list of everything on main since the last release.
