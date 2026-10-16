# Security policy

## Supported versions

Security fixes go into the latest release.

## Reporting a vulnerability

Please don't describe a security problem in a public issue. Use GitHub's private vulnerability reporting
(**Security → Report a vulnerability** on https://github.com/rajumark/kernelin) when it's available, or open an
issue titled "Security contact request" without any details and you'll be contacted privately.

Include what you found, how to reproduce it and the Kernelin and Kotlin versions. You'll get an answer within
7 days.

## Scope

Kernelin runs inside the Kotlin compiler at build time and, at runtime, sends compute work to the GPU through
WebGPU. Relevant reports include generated shaders reading or writing outside their buffers in a way WebGPU
doesn't contain, the compiler plugin reading or writing files outside the build's output folders, and the
published artifacts or their signatures not matching this repository.
