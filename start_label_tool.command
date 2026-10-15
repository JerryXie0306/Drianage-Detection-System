#!/bin/bash
# Double-click to start the labelling / region-calibration tool (opens your browser when it is ready).
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "The .venv environment is missing (needs torch for pre-labelling). See README."; read -n 1 -s -r -p "Press any key to close"; exit 1; }

PORT=$("$PY" - <<'EOF'
import socket
for p in range(8770, 8800):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", p)); s.close(); print(p); break
    except OSError:
        s.close()
EOF
)
[ -n "$PORT" ] || { echo "No free port between 8770 and 8799."; read -n 1 -s -r -p "Press any key to close"; exit 1; }
URL="http://127.0.0.1:$PORT"

mkdir -p data/seg/images data/seg/masks
ARGS=(--images data/seg/images --masks data/seg/masks --regions regions/drain.json --predict --port "$PORT")
[ -f models/seg_head.joblib ] && ARGS+=(--head models/seg_head.joblib)

echo "Put your photos in: $(pwd)/data/seg/images"
echo "Masks are saved to: $(pwd)/data/seg/masks   Regions: regions/drain.json"
echo "Starting label tool at $URL   (close this window or press Ctrl+C to stop)"
if [ -z "$DG_NO_BROWSER" ]; then
  ( for _ in $(seq 1 90); do
      curl -s -o /dev/null "$URL/api/list" && { open "$URL"; break; }
      sleep 1
    done ) &
fi

"$PY" -m drainguard.label "${ARGS[@]}"

echo
echo "Label tool stopped."
[ -t 0 ] && read -n 1 -s -r -p "Press any key to close"
exit 0
