# Changelog

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
