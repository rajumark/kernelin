#!/usr/bin/env bash
# Runs the same Kotlin CPU benchmark on a USB-connected Android phone (ART), so the phone's GPU numbers
# from scripts/bench-android.sh can be compared with the phone's own CPU.
# Usage: scripts/bench-android-cpu.sh [gradle command, default: gradle]
# Needs adb and the Android SDK build-tools (d8); set ANDROID_HOME if it isn't ~/Library/Android/sdk.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
gradle="${1:-gradle}"
sdk="${ANDROID_HOME:-$HOME/Library/Android/sdk}"
d8="$(ls -d "$sdk"/build-tools/*/d8 | sort -V | tail -1)"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

classpath=()
while IFS= read -r line; do [ -n "$line" ] && classpath+=("$line"); done \
  < <(cd "$repo" && "$gradle" -q :samples:printBenchClasspath)
# d8 takes jars and .class files; turn class directories into jars first.
inputs=()
for entry in "${classpath[@]}"; do
  if [ -d "$entry" ]; then
    jar="$work/$(echo "$entry" | md5 -q 2>/dev/null || echo "$entry" | md5sum | cut -c1-32).jar"
    (cd "$entry" && zip -qr "$jar" .)
    inputs+=("$jar")
  else
    inputs+=("$entry")
  fi
done
"$d8" --release --min-api 26 --output "$work" "${inputs[@]}" 2>&1 | grep -v -i "warning" || true
(cd "$work" && zip -q bench.jar classes*.dex)

adb push "$work/bench.jar" /data/local/tmp/kernelin-bench.jar >/dev/null
echo "Running on $(adb shell getprop ro.product.model) ($(adb shell getprop ro.soc.model)), $(adb shell nproc) cores"
adb shell dalvikvm -Xmx1g -cp /data/local/tmp/kernelin-bench.jar \
  io.github.rajumark.kernelin.samples.BenchmarkKt --time-only
adb shell rm -f /data/local/tmp/kernelin-bench.jar
