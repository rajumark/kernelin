# Versioning and compatibility

Kernelin follows [semantic versioning](https://semver.org): `MAJOR.MINOR.PATCH`.

## Before 1.0 (now)

- `0.x` minor releases (0.1 → 0.2) may change the public API, but every change is listed in
  [CHANGELOG.md](CHANGELOG.md) with what to do.
- Patch releases (0.1.0 → 0.1.1) never change the public API.
- A kernel that compiles and runs correctly keeps doing so. If a release changes what a kernel computes, that
  is a bug fix and it's called out in the changelog.

## From 1.0

- No breaking changes to the public API in minor or patch releases. Removals go through at least one minor
  release with `@Deprecated` first.

## What is public API

- `kernelin-runtime`: `@Kernel`, `globalId()`, `kernelSource()`, `KernelSource`, `KernelManifest`, `KernelinCpu`,
  `KernelinShaders` (JVM).
- `kernelin-webgpu`: `WebGpu`, `GpuKernel`, `GpuData`, `GpuResult` and the typed-array helpers.
- The Gradle plugin id `io.github.rajumark.kernelin` and its behaviour.
- The supported Kotlin subset in kernels, and the error messages' meaning (not their exact wording).

The binary API of every published module is recorded in its `api/` folder and checked on every build
(`checkKotlinAbi`), so an accidental change can't be released.

Not public: anything `internal`, the compiler plugin's classes, and the exact WGSL text Kernelin generates.

## Kotlin versions

The Kotlin compiler plugin API changes between Kotlin releases, so each Kernelin release is built for one
Kotlin version (0.1.0 to 0.4.0: Kotlin 2.4.21). New Kotlin versions are supported in new Kernelin releases.
