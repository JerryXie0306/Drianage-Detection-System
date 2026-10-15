#!/bin/bash
# Double-click: fully automatic recognition from a live camera (no uploading, no clicking).
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "The .venv environment is missing. See README."; read -n 1 -s -r -p "Press any key to close"; exit 1; }

SRC=$(grep -v '^[[:space:]]*#' camera.txt 2>/dev/null | grep -v '^[[:space:]]*$' | sed -n 1p | tr -d '\r')
INTERVAL=$(grep -v '^[[:space:]]*#' camera.txt 2>/dev/null | grep -v '^[[:space:]]*$' | sed -n 2p | tr -d '\r')
[ -n "$SRC" ] || SRC=0
[ -n "$INTERVAL" ] || INTERVAL=5

PORT=$("$PY" - <<'PYEOF'
import socket
for p in range(8000, 8100):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
PYEOF
)
[ -n "$PORT" ] || { echo "No free port between 8000 and 8099."; read -n 1 -s -r -p "Press any key to close"; exit 1; }
URL="http://127.0.0.1:$PORT"

REGIONS="regions/drain.json"
[ -f "$REGIONS" ] || REGIONS="regions/example_fisheye.json"
ARGS=(--port "$PORT" --interval "$INTERVAL" --backend segment --regions "$REGIONS")
[ -f models/seg_head.joblib ] && ARGS+=(--head models/seg_head.joblib)
case "$SRC" in
  folder:*) ARGS+=(--watch "${SRC#folder:}") ;;
  *)        ARGS+=(--camera "$SRC") ;;
esac

echo "Live source : $SRC   (analysis every ${INTERVAL}s)"
echo "Regions     : $REGIONS   Trained head: $([ -f models/seg_head.joblib ] && echo yes || echo 'no (zero-shot)')"
echo "Edit camera.txt to change the camera. Starting at $URL  (close this window or Ctrl+C to stop)"
if [ -z "$DG_NO_BROWSER" ]; then
  ( for _ in $(seq 1 90); do
      curl -s -o /dev/null "$URL/api/status" && { open "$URL"; break; }
      sleep 1
    done ) &
fi

"$PY" -m drainguard.server "${ARGS[@]}"

echo
echo "DrainGuard stopped."
[ -t 0 ] && read -n 1 -s -r -p "Press any key to close"
exit 0
