# AGENTS.md

Guide for AI coding agents working in this repository. For *using* Kernelin in another project, read
[docs/docs.md](docs/docs.md) (section "For AI agents"), also served at https://rajumark.github.io/kernelin/docs.md.

## What this is

Kernelin is a K2 compiler plugin that turns plain Kotlin `@Kernel` functions into WGSL compute shaders.

| Module | Role |
|---|---|
| `kernelin-runtime` | Multiplatform (JVM + Wasm): `@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu`, `KernelManifest` |
| `kernelin-compiler-plugin` | `KernelinIrGenerationExtension` finds kernels; `WgslKernelTranslator` turns IR into WGSL |
| `kernelin-webgpu` | Kotlin/Wasm WebGPU host (`WebGpu`, `GpuKernel`, `GpuData`) |
| `kernelin-gradle-plugin` | Plugin id `io.github.rajumark.kernelin` |
| `samples`, `examples` | Sample kernels, benchmark, live demo; standalone example projects |
| `docs` | GitHub Pages site: `index.html` (landing), `docs.html` + `docs.md` (documentation), `llms.txt`, `demo/` |

## Build and test

- JDK 21 and Kotlin 2.4.21. Use the installed `gradle` (`gradle build`).
- Compiler tests validate every generated shader with `naga` when it is on the PATH (`cargo install naga-cli`).
- CI (`.github/workflows/ci.yml`) builds, tests and runs both examples.

## Rules

- A new supported Kotlin construct needs: translation in `WgslKernelTranslator`, a test in `WgslTranslationTest`
  (WGSL text + naga), a CPU test if behaviour matters, and an update to the supported table in `docs/docs.md`.
- Unsupported code must produce a clear compile error with a fix hint, never wrong WGSL.
- `docs/docs.md` is the single source of the documentation; `docs/docs.html` renders it. Keep `docs/llms.txt` in sync.
- Don't use "Kotlin" in artifact or package names (trademark); packages are `io.github.rajumark.kernelin`.
- Maven Central versions are permanent: never re-publish an existing version.
