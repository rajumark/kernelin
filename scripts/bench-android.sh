#!/usr/bin/env bash
# Runs the WebGPU benchmark in Chrome on a USB-connected Android phone and saves the results.
# The phone loads the runner from this computer through `adb reverse`, so no Wi-Fi or hosting is needed.
# Usage: scripts/bench-android.sh <bench data dir from `gradle :samples:bench -Pout=<dir>`> [port, default 8765]
# Writes <dir>/results.json. Needs adb, python3, and Chrome on the phone (WebGPU: Android 12+).
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
dir="$(cd "$1" && pwd)"
port="${2:-8765}"
[ -f "$dir/cases.json" ] || { echo "No cases.json in $dir; run gradle :samples:bench -Pout=$dir first" >&2; exit 1; }

cp "$repo/samples/web/runner.html" "$dir/"
rm -f "$dir/results.json"

python3 - "$dir" "$port" <<'PY' &
import http.server, os, sys
root, port = sys.argv[1], int(sys.argv[2])
os.chdir(root)
class Handler(http.server.SimpleHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        with open("results.json", "wb") as f:
            f.write(body)
        self.send_response(204)
        self.end_headers()
    def log_message(self, *args):
        pass
http.server.ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
PY
server=$!
trap 'kill $server 2>/dev/null; adb reverse --remove tcp:$port 2>/dev/null || true' EXIT

adb reverse "tcp:$port" "tcp:$port" >/dev/null
adb shell svc power stayon usb
adb shell input keyevent KEYCODE_WAKEUP
adb shell am start -a android.intent.action.VIEW -d "http://localhost:$port/runner.html?t=$(date +%s)" \
  com.android.chrome >/dev/null

echo "Running on $(adb shell getprop ro.product.model) - keep the phone unlocked with Chrome in front."
for _ in $(seq 1 600); do
  [ -s "$dir/results.json" ] && break
  sleep 1
done
adb shell svc power stayon false
[ -s "$dir/results.json" ] || { echo "No results after 10 minutes" >&2; exit 1; }
cat "$dir/results.json"
echo
