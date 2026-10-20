#!/usr/bin/env bash
# Runs the benchmark app (samples-android) on a USB-connected Android phone: the sample kernels on the phone's GPU
# through native WebGPU (AndroidX WebGPU, no browser) and on its CPU, then saves the results.
# Usage: scripts/bench-android-app.sh [output json, default build/bench-android-app.json] [scale, default 1]
# scale divides the sizes for a quick run. Needs adb, JDK 21 and an Android SDK (ANDROID_HOME).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
out="${1:-$repo/build/bench-android-app.json}"
scale="${2:-1}"
pkg=io.github.rajumark.kernelin.bench

(cd "$repo" && gradle :samples-android:assembleRelease -q --console=plain)
adb install -r -g "$repo/samples-android/build/outputs/apk/release/samples-android-release.apk" >/dev/null
adb shell svc power stayon usb
adb shell input keyevent KEYCODE_WAKEUP

# Every case gets a fresh process (intent extra "case"), so one case's 64 MB arrays can't crowd the next one out of
# the app heap. The run with no result line is past the last case.
echo "Running on $(adb shell getprop ro.product.model | tr -d '\r') - keep the phone unlocked with the app in front."
header=""
results=()
for case in $(seq 0 99); do
  adb logcat -c
  adb shell am force-stop "$pkg"
  adb shell am start -W -n "$pkg/io.github.rajumark.kernelin.samples.android.BenchActivity" --ei scale "$scale" --ei case "$case" >/dev/null
  done=""
  for _ in $(seq 1 900); do
    if adb logcat -d -s KernelinBench:E | grep -q "Benchmark failed"; then
      adb logcat -d -s KernelinBench:* AndroidRuntime:E | tail -40
      exit 1
    fi
    if adb logcat -d -s KernelinBench:I | grep -q KERNELIN_BENCH_DONE; then done=1; break; fi
    sleep 1
  done
  [ -n "$done" ] || { echo "Timed out after 15 minutes" >&2; exit 1; }
  json="$(adb shell cat "/sdcard/Android/data/$pkg/files/results.json" | tr -d '\r')"
  line="$(printf '%s\n' "$json" | grep '^    {"name"' || true)"
  [ -n "$line" ] || break
  [ -n "$header" ] || header="$(printf '%s\n' "$json" | sed -n '1,/"results"/p')"
  results+=("$line")
  adb logcat -d -s KernelinBench:I | sed -n 's/.*KernelinBench: //p' | grep -v -e KERNELIN_BENCH_DONE -e '^Done\.' -e '^$' |
    { [ "$case" -eq 0 ] && cat || grep -v -e '^Kernelin benchmark' -e 'CPU cores$' -e '^GPU: '; }
done
[ ${#results[@]} -gt 0 ] || { echo "No benchmark results" >&2; exit 1; }
mkdir -p "$(dirname "$out")"
{
  printf '%s\n' "$header"
  last=$((${#results[@]} - 1))
  for i in "${!results[@]}"; do
    if [ "$i" -lt "$last" ]; then printf '%s,\n' "${results[$i]}"; else printf '%s\n' "${results[$i]}"; fi
  done
  printf '  ]\n}\n'
} > "$out"
echo "Saved $out"
