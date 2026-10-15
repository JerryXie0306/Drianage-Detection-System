#!/bin/bash
# Double-click to start the DrainGuard control app (opens your browser when it is ready).
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="/opt/anaconda3/bin/python3"
[ -x "$PY" ] || PY="python3"

PORT=$("$PY" - <<'EOF'
import socket
for p in range(8000, 8100):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
EOF
)
[ -n "$PORT" ] || { echo "No free port between 8000 and 8099."; read -n 1 -s -r -p "Press any key to close"; exit 1; }
URL="http://127.0.0.1:$PORT"

ARGS=(--port "$PORT")
if "$PY" -c "import torch, transformers" 2>/dev/null; then
  REGIONS="regions/drain.json"
  [ -f "$REGIONS" ] || REGIONS="regions/example_fisheye.json"
  ARGS+=(--backend segment --regions "$REGIONS")
  [ -f models/seg_head.joblib ] && ARGS+=(--head models/seg_head.joblib)
  echo "Mode: segmentation (regions: $REGIONS, trained head: $([ -f models/seg_head.joblib ] && echo yes || echo 'no - zero-shot'))"
  if ls data/seg/images/* >/dev/null 2>&1; then
    ARGS+=(--frames data/seg/images --interval 8)
    echo "Replaying the images in data/seg/images. Use 'Upload photo' in the app to test your own."
  fi
else
  echo "Mode: colour heuristic (torch not installed - see README for the segmentation backend)"
  [ -d data/synthetic ] || "$PY" -m drainguard.synthetic --out data/synthetic --per-class 8
  ARGS+=(--frames data/synthetic --interval 2)
fi

echo "Starting DrainGuard at $URL   (close this window or press Ctrl+C to stop)"
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
