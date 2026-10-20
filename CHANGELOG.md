# Changelog

## Unreleased

- Automatic CPU fallback: `Compute.create()` opens the GPU when there is one and uses the CPU (`KernelinCpu`)
  when there isn't, and `compute.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }` runs the typed kernel call on
  whichever it has, so apps need one code path. A GPU device that is lost later switches to the CPU, and the run
  that hit it is redone there. `cpuReason`, `isGpu` and `description` say where kernels run and why. On the JVM and
  Android the CPU path uses all cores. Under the hood, the compiler plugin now also records a CPU call in each
  `KernelLaunch` (`runOnCpu(size)`), with every argument evaluated once.
- Typed launches: `gpu.dispatch(n) { saxpy(alpha = 2f, x = xs, y = ys) }` runs a kernel on the GPU with a normal
  Kotlin call, so parameter names, types and count are checked at compile time (and completed by the IDE) instead
  of at run time. The arrays the kernel writes hold the results afterwards, like with `KernelinCpu.dispatch`. The
  compiler plugin records the call in a new `KernelLaunch` (runtime) that GPU hosts read with
  `KernelLaunch.capture`. A launch block that is not exactly one kernel call is a compile error, shown in the IDE
  too. The map API (`gpu.load(...).run(size, scalars, buffers)`) still works.
- 2D and 3D thread ids: `globalIdX()`, `globalIdY()` and `globalIdZ()`, with `dispatch(width, height)` and
  `dispatch(width, height, depth)` launches on `KernelinCpu` (also `dispatchParallel`) and on the GPU (typed
  `gpu.dispatch(width, height) { ... }`, `Compute.dispatch(width, height) { ... }` and `GpuKernel.run(width, height, ...)`,
  on Kotlin/Wasm and Android; `KernelLaunch.runOnCpu(width, height)`).
  Image and grid kernels no longer compute `id % width` and `id / width` by hand. Threads run row by row, so
  `globalId()` is the row-major index `x + y * width` and CPU and GPU see the same ids. Big grids use the same
  workgroup spreading as 1D launches (at most 65535 workgroups per dimension). Kernels that use these ids get
  `kernelin_width` / `kernelin_height` in their uniform; the new `KernelManifest.usesGridIds` and
  `KernelManifest.scalarOffset` tell custom hosts where the scalars start. 1D kernels and their uniform are
  unchanged.
- Built-in random numbers for kernels: `randomInt(seed, index, stream)`, `randomFloat` (in `[0, 1)`),
  `randomIntBelow(bound, seed, index, stream)` and `randomNormal` (Box-Muller). They are counter based and keep no
  state, so pass `globalId()` as the index and a different `stream` for each draw. `randomInt`, `randomFloat` and
  `randomIntBelow` give the same bits on the CPU (every platform) and the GPU; `randomNormal` agrees to about 1e-5.
  Before, apps had to write their own hash.
- `kernelin-webgpu` now runs on **Android** 8.0+ (API 26), natively through AndroidX WebGPU (Dawn on Vulkan, or
  OpenGL ES 3.1), without a browser or WebView. Same API as the Kotlin/Wasm host: `WebGpu.create()`, typed `dispatch`, `load`, `run`,
  data kept on the GPU, lost devices, `close()`. Every GPU call runs on one background thread owned by the `WebGpu`.
- `kernelin-webgpu` now also runs on the **desktop JVM** (macOS, Linux x64, Windows x64), natively through
  [jWebGPU](https://github.com/xpenatan/jWebGPU) and wgpu (Metal, Vulkan, Direct3D 12), without a browser. Same API
  as the other hosts. Works on JDK 21+. JVM GPU tests compare every result with `KernelinCpu` on the real GPU (on
  lavapipe in CI), and `gradle :samples:benchGpu` runs the benchmark on the computer's GPU.
- The argument checks of `GpuKernel.run` moved to common code, so every GPU host rejects the same calls.
- New `samples-android` app and `scripts/bench-android-app.sh`: the benchmark kernels on a phone's GPU (native, not
  Chrome) and CPU, with a GPU-vs-CPU result check. Android GPU tests: `gradle :kernelin-webgpu:connectedAndroidDeviceTest`
  (skipped on emulators, which have no real GPU for WebGPU). Tested on a moto g57 power (Adreno 710, Vulkan): all
  device tests pass and the 5 benchmark kernels match the CPU; GPU kernel times equal Chrome's on the same phone.
  The script runs every benchmark case in its own app process, so 64 MB arrays fit in a phone's app heap. New
  device tests check that runs and closed kernels don't keep their callers' memory alive.
- Faster Android GPU runs: fewer copies of the data per run. Uploads now go through two small reused buffers that
  the CPU writes into directly, instead of Dawn's `writeBuffer`, which staged every big upload through a new buffer
  (~140 ms for 64 MB on the moto g57 power, now ~25 ms). Results stay in GPU read-back memory, and `floats()` /
  `ints()` copy them once, straight into the Kotlin array (before: a copy into a direct buffer first). Typed
  `gpu.dispatch { }` uploads straight from your arrays and writes results straight back into them, with no extra
  snapshot copy. End-to-end on the moto g57 power (64 MB arrays): vectorAdd 385 → 148 ms, saxpy 324 → 129 ms, blur
  279 → 155 ms, waves 192 → 97 ms, mandelbrot 97 → 73 ms, now faster than Chrome on the same phone for all 5 kernels.
  The API is unchanged: `GpuData.of` still copies the array when you call it, and results keep their data after
  later runs and after `close()`. `GpuResult.bytes(name)` copies the output out of GPU memory on its first call.

## 0.3.0 (2026-10-19)

More platforms:
- `kernelin-runtime` now also targets Kotlin/JS (browser and Node.js) and Kotlin/Native (`macosArm64`, `iosArm64`,
  `iosSimulatorArm64`, `iosX64`, `tvosArm64`, `tvosSimulatorArm64`, `watchosArm32`, `watchosArm64`,
  `watchosSimulatorArm64`, `linuxX64`, `linuxArm64`, `mingwX64`). Multiplatform projects with these targets can keep
  kernels in `commonMain`, run them with `KernelinCpu` and get their WGSL with `kernelSource(::k)`.
- The Gradle plugin adds `kernelin-runtime` to Android projects too: with the Kotlin Android plugin, with AGP 9's
  built-in Kotlin (`com.android.application` / `com.android.library` alone) and with
  `com.android.kotlin.multiplatform.library`. Before, Android projects got the compiler plugin but no runtime.
- New `kernelin { enabled = false }` Gradle setting turns the compiler plugin off.
- The Gradle plugin has functional tests (Gradle TestKit) that build JVM, Multiplatform and Android projects.

Fixes (the GPU now matches Kotlin in all of these):
- An `if` / `when` used as a value ran every branch on the GPU (it became `select()`), so a helper that writes an
  array ran even in branches Kotlin skips. Branches with side effects or setup code now become a real `if`.
- `a && b` / `a || b`: setup code for `b` (for example a temporary for `helper(...) in 0..5`) ran even when `a`
  already decided the result. It now runs only when Kotlin evaluates `b`.
- `Float.pow` with a negative base (`(-2f).pow(3f) == -8f`) or a zero base: WGSL leaves these undefined, and
  Vulkan GPUs return NaN. Kernelin now handles the sign and the special cases itself, like Kotlin.
- `for` loops whose range ends at or near `Int.MAX_VALUE` / `Int.MIN_VALUE` (like `0..Int.MAX_VALUE`, or
  `a..b step s` from data) wrapped around and never ended on the GPU. Such loops now count their steps up front,
  like Kotlin's `IntProgression`. A step of 0 or less (Kotlin throws) runs no iterations instead of looping
  forever; a constant one is a compile error, and so is a second `step` on the same range.
- `min`, `max`, `coerceIn`, `coerceAtLeast`, `coerceAtMost`, `sign` and `compareTo` on Float: WGSL leaves NaN and
  the sign of zero unspecified. They now follow Kotlin exactly (`min(-0f, 0f)` is `-0f`, `max(1f, NaN)` is NaN,
  `sign(-0f)` is `-0f`, `NaN.compareTo(1f)` is 1), checked bit for bit on the GPU. Constant `-0f < 0f` is now
  `false` (it was folded with `compareTo`).
- `hypot` overflowed to Infinity for values above about 1e19 and gave 0 below about 1e-19; it is now scaled and
  within 2 ULPs of Kotlin. `log10` lost most digits near 1; it is now within 4 ULPs of Kotlin.
- The kernel manifest (`.json`) put kernel and parameter names in without escaping, so a backticked name with a
  quote made invalid JSON.

IDE checks:
- The frontend (FIR) checks now also run on the helper functions a kernel calls, so unsupported code in a helper is
  underlined while you type, not only reported by the build. Helpers in the kernel's file get the error on their own
  code; helpers in other files get it on the call in the kernel.

Build:
- The `Kernelin: translated @Kernel ... to WGSL` line for each kernel now shows only with the compiler's
  `-verbose` flag.

Build fixes:
- Incremental builds: when only the file calling `kernelSource(::k)` was recompiled, without the file holding
  `@Kernel k` (for example after a library it uses changed), the build failed with "'k' must be a @Kernel from this
  module". `kernelSource()` now uses the shader saved when `k` was last compiled. On JS, Wasm and Native the Gradle
  plugin saves shaders in `build/kernelin/<target>/<compilation>`; the JVM keeps them next to the classes.
  `scripts/check-incremental.sh` now checks this on the JVM, Kotlin/JS and Kotlin/Wasm, and CI runs it.

WebGPU host:
- `GpuData.empty(n)` is zero-filled on every run, as documented. Before, a run that reused the GPU buffer of an
  earlier run of the same size started from that run's results.
- `GpuKernel.run` rejects unknown parameter names, scalars of the wrong type (a `Float` for an `Int`) and
  buffer data of the wrong type (`FloatArray` for an `IntArray`) with a clear `IllegalArgumentException`,
  before anything reaches the GPU. Before, they were ignored, truncated or reinterpreted bit by bit.
- WebGPU errors are no longer silent: `load` and `run` check WebGPU's error scopes (validation, out of memory) and
  throw the new `GpuException` with the browser's message. The pipeline is built with `createComputePipelineAsync`.
- Lost devices: `WebGpu.isLost`, `lostReason` and an `onLost` callback; `load` and `run` on a lost device throw
  `GpuException`. `onUncapturedError` reports errors outside `load` and `run`.
- `close()` on `GpuKernel` (frees its buffers) and on `WebGpu` (frees every kernel and the device); both are
  `AutoCloseable`. `run` after `close` throws `IllegalStateException`.
- Two overlapping `run` calls on one kernel no longer share buffers (one of them failed); the second now waits.
- Every buffer, pipeline and bind group is labeled with the kernel and parameter name (`my.pkg.blur: out`), so
  browser errors name them.
- Keep data on the GPU: `run(..., readBack = setOf("out"))` copies back only those buffers, and
  `GpuData.of(result.buffer("out"))` passes a buffer to the next run with a GPU-side copy. The old `run` signature
  still compiles and links.
- Bulk copies between Kotlin arrays and JS typed arrays (`GpuData.of`, `floats`, `ints`, `toJsFloat32Array`,
  `toJsInt32Array`): about 8x faster to JS and 4x faster back for 4M floats.

## 0.2.0 (2026-10-16)

IDE support:
- Errors while you type: K2 frontend (FIR) checks report unsupported code in `@Kernel` functions, so IntelliJ IDEA
  and Android Studio underline it with the same messages as the build. Third-party compiler plugins need the IDE
  registry key `kotlin.k2.only.bundled.compiler.plugins.enabled` set to `false`. They only report what the
  translator rejects anyway (checked with 2,000 fuzzed kernels). Frontend errors stop the build before the
  translator, so other errors in the same kernel appear after you fix them.
- New, clearer messages for nullable values (`?.`, `?:`, `!!`) and for creating objects or arrays in a kernel.

New in kernels:
- `x in a..b`, `until`, `..<`, `downTo`, `!in`, and `when (x) { in a..b -> }`, for Int and Float.
- `repeat(n) { }`, including `return@repeat` and `break` / `continue` to an outer loop from inside it.
- Labeled `break@outer` / `continue@outer` across any number of nested loops.
- `Boolean` kernel parameters.
- `FloatArray` / `IntArray` parameters in your own helper functions.
- `PI.toFloat()`, `E.toFloat()` and other `Double` constants converted right away.
- `Float.isNaN()`, `isInfinite()`, `isFinite()`.

Fixes (the GPU now matches Kotlin in all of these):
- Helper functions named like a WGSL built-in (`uniform`, `storage`, `read_write`, `fract`, `dot`, ...) are renamed.
  Before, Chrome rejected the shader ("cannot use function 'uniform' as address space").
- Calls to helpers that return `Unit` were left out of the shader.
- `Float.toInt()`: NaN gives 0, out-of-range values give `Int.MIN_VALUE` / `Int.MAX_VALUE`.
- Shift counts use only their low 5 bits, as in Kotlin, on every GPU.
- Constant expressions are computed with Kotlin's rules (wrapping, Infinity/NaN). Before, WGSL rejected
  shaders where an all-constant expression overflowed, e.g. `-75 shl 28`.
- JVM: `kernelSource()` reads the shader shipped next to the kernel, so an incremental build never keeps a
  stale copy.
- Constant integer division by zero is a compile error (Kotlin would throw).

API:
- `GpuKernel.run(scalars = …)` takes `Map<String, Any>` (was `Map<String, Number>`) so Booleans fit. Source
  compatible; Wasm libraries compiled against 0.1.0 that call `run()` need a recompile.
- The public API of every module is now checked on every build (`api/` files), documented, and declared
  explicitly (explicit API mode).

Testing:
- Benchmarked on an Android phone (moto g57 power, Adreno GPU, Chrome): all 5 kernels match the CPU, up to 33×
  faster than all 8 CPU cores. `scripts/bench-android.sh` and `scripts/bench-android-cpu.sh` reproduce it.
- GPU-vs-CPU differential tests (WebGPU through Deno) for every feature, in CI on a software GPU.
- A fuzzer: random valid kernels must give identical results on CPU and GPU (40 per build, 2,000 before this
  release, 300 more on every CI run).
- Code coverage with Kover (86%), and the build fails below 80%.

## 0.1.0 (2026-10-15)

First release.

- `@Kernel` functions written in plain Kotlin are translated to WGSL compute shaders by a K2 compiler plugin.
- Supported in kernels: `Int`, `Float`, `Boolean`; `FloatArray` / `IntArray` buffers; `Int` / `Float` parameters;
  all arithmetic, comparison, logic and bit operators; `if` / `when` (also as values); `for` over ranges,
  `while`, `do-while`, `break`, `continue`; `kotlin.math`; your own helper and extension functions.
- Clear compile errors with file and line for code the GPU can't run (strings, `Double`, `Long`, lambdas,
  recursion, exceptions, casts, nullable types, library calls).
- `kernelSource(::kernel)` embeds the generated shader in your program on every platform.
- `kernelin-runtime` for JVM and Wasm, with `KernelinCpu` to run the same kernels on the CPU.
- `kernelin-webgpu`: run kernels on the GPU from Kotlin/Wasm through WebGPU.
- Gradle plugin `io.github.rajumark.kernelin` for Kotlin JVM and Multiplatform projects.
- Built for Kotlin 2.4.21.
