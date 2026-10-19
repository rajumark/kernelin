#!/usr/bin/env bash
# Regression check for incremental builds, on the JVM and on the klib targets (Kotlin/JS and Kotlin/Wasm):
# 1. after changing only a kernel's body, the program must use the new shader, even where
#    kernelSource(::kernel) is called from another (not recompiled) file;
# 2. after a library the caller uses changes (as when kernelin-webgpu changes), only the file that calls
#    kernelSource(::kernel) is recompiled, not the file holding the @Kernel; it must still get the shader.
# Usage: scripts/check-incremental.sh [gradle command, default: <repo>/gradlew] [targets, default: "jvm js wasmJs"]
# Set KEEP=1 to keep the generated project for debugging.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gradle="${1:-$repo/gradlew}"
targets="${2:-jvm js wasmJs}"
dir="$(mktemp -d)"
[[ -n "${KEEP:-}" ]] && echo "project: $dir" || trap 'rm -rf "$dir"' EXIT
mkdir -p "$dir/src/commonMain/kotlin/ic" "$dir/lib" "$dir/gradle"
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
include(":lib")
KTS
cat > "$dir/build.gradle.kts" <<'KTS'
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    kotlin("multiplatform") version "2.4.21"
    id("io.github.rajumark.kernelin")
}
kotlin {
    jvmToolchain(21)
    jvm { mainRun { mainClass.set("ic.MainKt") } }
    js { nodejs(); binaries.executable() }
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs { nodejs(); binaries.executable() }
    sourceSets.commonMain.dependencies { implementation(project(":lib")) }
}
KTS
cat > "$dir/lib/build.gradle.kts" <<'KTS'
import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins { kotlin("multiplatform") }
kotlin {
    jvmToolchain(21)
    jvm()
    js { nodejs() }
    @OptIn(ExperimentalWasmDsl::class)
    wasmJs { nodejs() }
}
KTS
kernel() {
  cat > "$dir/src/commonMain/kotlin/ic/Kernels.kt" <<KT
package ic

import io.github.rajumark.kernelin.Kernel
import io.github.rajumark.kernelin.globalId

@Kernel
fun k(out: IntArray) {
    out[globalId()] = $1
}
KT
}
# Like the samples: the kernel in commonMain, the code calling kernelSource() in each platform's source set.
for t in jvm js wasmJs; do
  mkdir -p "$dir/src/${t}Main/kotlin/ic"
  cat > "$dir/src/${t}Main/kotlin/ic/Main.kt" <<'KT'
package ic

import io.github.rajumark.kernelin.kernelSource
import lib.label

fun main() = println(label() + " " + kernelSource(::k).wgsl.lines().first { "out[" in it }.trim())
KT
done
# Main.kt uses label(); changing its signature makes the build recompile Main.kt but not Kernels.kt.
library() {
  for t in jvm js wasmJs; do
    mkdir -p "$dir/lib/src/${t}Main/kotlin/lib"
    cat > "$dir/lib/src/${t}Main/kotlin/lib/Label.kt" <<KT
package lib

fun label($1): String = "shader$2"
KT
  done
}
run_task() {
  case "$1" in
    jvm) echo :jvmRun ;;
    *) echo ":${1}NodeDevelopmentRun" ;;
  esac
}
# Prints the program's line, or the build's error.
run() {
  local out
  if out="$(cd "$dir" && "$gradle" "$(run_task "$1")" -q --console=plain 2>&1)"; then
    grep -m1 '^shader' <<<"$out" || echo "no output: $out"
  else
    echo "BUILD FAILED: $(grep -m1 -E '^e: |Kernelin:' <<<"$out" || grep -m1 -A3 'What went wrong' <<<"$out")"
  fi
}
expect() {
  echo "$1 $2: $3"
  [[ "$3" == "shader$4 "*"= $5i;"* ]] || { echo "FAIL: $1 $2 should print shader$4 with = $5i;"; exit 1; }
}
kernel 111
library "" ""
for t in $targets; do expect "$t" "first build" "$(run "$t")" "" 111; done
kernel 222
for t in $targets; do expect "$t" "kernel changed" "$(run "$t")" "" 222; done
library "suffix: String = \":\"" "\$suffix"
for t in $targets; do expect "$t" "library changed" "$(run "$t")" ":" 222; done
echo "OK: incremental builds pick up kernel changes ($targets)"
