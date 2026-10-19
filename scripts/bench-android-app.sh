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
adb logcat -c
adb shell am force-stop "$pkg"
adb shell am start -n "$pkg/io.github.rajumark.kernelin.samples.android.BenchActivity" --ei scale "$scale" >/dev/null

echo "Running on $(adb shell getprop ro.product.model | tr -d '\r') - keep the phone unlocked with the app in front."
for _ in $(seq 1 900); do
  if adb logcat -d -s KernelinBench:I | grep -q KERNELIN_BENCH_DONE; then
    mkdir -p "$(dirname "$out")"
    adb shell cat "/sdcard/Android/data/$pkg/files/results.json" | tr -d '\r' > "$out"
    adb logcat -d -s KernelinBench:I | sed -n 's/.*KernelinBench: //p'
    echo "Saved $out"
    exit 0
  fi
  if adb logcat -d -s KernelinBench:E | grep -q "Benchmark failed"; then
    adb logcat -d -s KernelinBench:* AndroidRuntime:E | tail -40
    exit 1
  fi
  sleep 1
done
echo "Timed out after 15 minutes" >&2
exit 1
