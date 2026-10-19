# Changelog

## Unreleased

Fixes (the GPU now matches Kotlin in all of these):
- An `if` / `when` used as a value ran every branch on the GPU (it became `select()`), so a helper that writes an
  array ran even in branches Kotlin skips. Branches with side effects or setup code now become a real `if`.
- `a && b` / `a || b`: setup code for `b` (for example a temporary for `helper(...) in 0..5`) ran even when `a`
  already decided the result. It now runs only when Kotlin evaluates `b`.
- `Float.pow` with a negative base (`(-2f).pow(3f) == -8f`) or a zero base: WGSL leaves these undefined, and
  Vulkan GPUs return NaN. Kernelin now handles the sign and the special cases itself, like Kotlin.

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
