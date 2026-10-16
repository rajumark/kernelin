# Changelog

## Unreleased

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
