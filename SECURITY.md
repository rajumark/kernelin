# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Please don't describe a security problem in a public issue. Report it privately on GitHub:
**Security → Report a vulnerability** on https://github.com/rajumark/kernelin
(direct link: https://github.com/rajumark/kernelin/security/advisories/new).

Include what you found, how to reproduce it and the Kernelin and Kotlin versions. You'll get an answer within
7 days.

## Scope

Kernelin runs inside the Kotlin compiler at build time and, at runtime, sends compute work to the GPU through
WebGPU. Relevant reports include generated shaders reading or writing outside their buffers in a way WebGPU
doesn't contain, the compiler plugin reading or writing files outside the build's output folders, and the
published artifacts or their signatures not matching this repository.
