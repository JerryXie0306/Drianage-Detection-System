#!/bin/bash
# Double-click to start the DrainGuard control app (opens your browser when it is ready).
# Reads motor.txt: with no motor address in it the cover is SIMULATED.
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || PY="/opt/anaconda3/bin/python3"
[ -x "$PY" ] || PY="python3"

"$PY" launch.py

echo
echo "DrainGuard stopped."
[ -t 0 ] && read -n 1 -s -r -p "Press any key to close"
exit 0
