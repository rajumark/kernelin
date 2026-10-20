# AGENTS.md

Guide for AI coding agents working in this repository. For *using* Kernelin in another project, read
[docs/docs.md](docs/docs.md) (section "For AI agents"), also served at https://rajumark.github.io/kernelin/docs.md.

## What this is

Kernelin is a K2 compiler plugin that turns plain Kotlin `@Kernel` functions into WGSL compute shaders.

| Module | Role |
|---|---|
| `kernelin-runtime` | Multiplatform (JVM, JS, Wasm, Native): `@Kernel`, `globalId()`, `kernelSource()`, `KernelinCpu`, `KernelManifest` |
| `kernelin-compiler-plugin` | `KernelinIrGenerationExtension` finds kernels; `WgslKernelTranslator` turns IR into WGSL |
| `kernelin-webgpu` | WebGPU host (`WebGpu`, `GpuKernel`, `GpuData`) for Kotlin/Wasm, Android (AndroidX WebGPU) and the desktop JVM (jWebGPU, wgpu); shared argument checks in `commonMain` |
| `kernelin-gradle-plugin` | Plugin id `io.github.rajumark.kernelin`; TestKit tests in `src/functionalTest` (`functionalTest` task) |
| `samples`, `samples-android`, `examples` | Sample kernels, benchmark, live demo; Android benchmark app; standalone example projects |
| `docs` | GitHub Pages site: `index.html` (landing), `docs.html` + `docs.md` (documentation), `llms.txt`, `demo/` |

## Build and test

- JDK 21 and Kotlin 2.4.21. Use the installed `gradle` (`gradle build`).
- Compiler tests validate every generated shader with `naga` when it is on the PATH (`cargo install naga-cli`).
- GPU tests need Deno (`brew install deno`); without it they are skipped locally. CI runs them on llvmpipe.
- CI (`.github/workflows/ci.yml`) builds, tests, fuzzes and runs both examples; a macOS job runs the Apple targets.
- The Gradle plugin's functional tests build Android projects only when `ANDROID_HOME` points to an Android SDK.
- The build needs an Android SDK (`ANDROID_HOME`) for the Android target of `kernelin-webgpu` and `samples-android`.
- JVM GPU tests: `gradle :kernelin-webgpu:jvmTest` on this computer's GPU (skipped without one; CI runs them on lavapipe).
- Android GPU tests: `gradle :kernelin-webgpu:connectedAndroidDeviceTest` on a phone (emulators have no real GPU for WebGPU, so they skip).

## Rules

- Follow [CONTRIBUTING.md](CONTRIBUTING.md). A new supported Kotlin construct needs: translation in
  `WgslKernelTranslator`, a WGSL test (text + naga), a GPU-vs-CPU test, a fuzzer case, and a row in the
  supported table in `docs/docs.md`. Add it to the "Unreleased" section of `CHANGELOG.md`.
- Public API changes follow [VERSIONING.md](VERSIONING.md); `checkKotlinAbi` fails the build on an accidental
  change (update the `api/` dumps only on purpose). Coverage must stay at 80% or more.
- Unsupported code must produce a clear compile error with a fix hint, never wrong WGSL.
- `docs/docs.md` is the single source of the documentation; `docs/docs.html` renders it. Keep `docs/llms.txt` in sync.
- Don't use "Kotlin" in artifact or package names (trademark); packages are `io.github.rajumark.kernelin`.
- Maven Central versions are permanent: never re-publish an existing version.
