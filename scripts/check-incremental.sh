#!/usr/bin/env bash
# Regression check: after changing only a kernel's body, an incremental build must use the new shader,
# even where kernelSource(::kernel) is called from another (not recompiled) file.
# Usage: scripts/check-incremental.sh [gradle command, default: <repo>/gradlew]
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gradle="${1:-$repo/gradlew}"
dir="$(mktemp -d)"
trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir/src/main/kotlin/ic" "$dir/gradle"
cp "$repo/gradle/gradle-daemon-jvm.properties" "$dir/gradle/"
cp "$repo/examples/hello-jvm/gradle.properties" "$dir/"
cat > "$dir/settings.gradle.kts" <<KTS
rootProject.name = "ic-check"
pluginManagement {
    includeBuild("$repo")
    repositories { gradlePluginPortal(); mavenCentral() }
}
dependencyResolutionManagement { repositories { mavenCentral() } }
includeBuild("$repo")
KTS
cat > "$dir/build.gradle.kts" <<'KTS'
plugins {
    kotlin("jvm") version "2.4.21"
    id("io.github.rajumark.kernelin")
    application
}
kotlin { jvmToolchain(21) }
application { mainClass.set("ic.MainKt") }
KTS
kernel() {
  cat > "$dir/src/main/kotlin/ic/Kernels.kt" <<KT
package ic

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId

@Kernel
fun k(out: IntArray) {
    out[globalId()] = $1
}
KT
}
cat > "$dir/src/main/kotlin/ic/Main.kt" <<'KT'
package ic

import io.github.rajumark.kernelin.kernelSource

fun main() = println(kernelSource(::k).wgsl.lines().first { "out[" in it }.trim())
KT
run() { (cd "$dir" && "$gradle" run -q --console=plain 2>&1 | tail -1); }
kernel 111
first="$(run)"
kernel 222
second="$(run)"
echo "first build:  $first"
echo "second build: $second"
[[ "$first" == *"= 111i;"* ]] || { echo "FAIL: first build has the wrong shader"; exit 1; }
[[ "$second" == *"= 222i;"* ]] || { echo "FAIL: incremental build kept a stale shader"; exit 1; }
echo "OK: incremental builds pick up kernel changes"
