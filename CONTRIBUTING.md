# Contributing to Kernelin

Thanks for helping. Kernelin turns Kotlin into GPU code, so a wrong translation is worse than no translation:
every change must keep the CPU and the GPU giving the same results.

## Set up

- JDK 21, Gradle 9.6 (or `./gradlew`).
- [Deno](https://deno.com) (`brew install deno`) to run the GPU tests. Without it they are skipped locally;
  CI always runs them.
- Optional: [naga](https://github.com/gfx-rs/wgpu/tree/trunk/naga) (`cargo install naga-cli`) validates every
  generated shader in the tests.

```bash
gradle build                      # compile, all tests, API check, coverage rule
gradle koverHtmlReport            # coverage report: build/reports/kover/html/index.html
scripts/check-incremental.sh gradle
```

## Rules for a change

1. **A new Kotlin construct in kernels** needs:
   - the translation in `WgslKernelTranslator`;
   - a WGSL test in `WgslTranslationTest`;
   - a GPU-vs-CPU test in `GpuDifferentialTest` (the result must be identical, or within a documented
     tolerance for float math);
   - if it can be generated randomly, a case in `KernelFuzzer`;
   - an entry in the supported table in `docs/docs.md`.
2. **Unsupported code** must fail to compile with a clear message and a hint. Never emit WGSL that behaves
   differently from Kotlin.
3. **CPU/GPU differences** that can't be avoided (float rounding, undefined behaviour in WGSL) are documented
   in `docs/docs.md` ("Differences between CPU and GPU").
4. **Public API** changes show up in the `api/` files. Run `gradle updateKotlinAbi` and commit the result; the
   reviewer checks it against [VERSIONING.md](VERSIONING.md).
5. Run the fuzzer harder before a release: `gradle :kernelin-compiler-plugin:test --tests '*KernelFuzzTest*'
   -Pkernelin.fuzz.count=1000`.

## Commits and pull requests

- One topic per pull request, with a short description of what changes for users.
- CI must be green: build, tests (including GPU tests on a software GPU), API check, coverage, both examples
  and the incremental-build check.
