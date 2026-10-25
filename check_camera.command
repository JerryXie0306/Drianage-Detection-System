#!/bin/bash
# Double-click: tests that the camera in camera.txt can be reached, and saves a test picture.
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"
[ -x "$PY" ] || { echo "Please double-click setup_and_run.command first."; [ -t 0 ] && read -n 1 -s -r -p "Press any key to close"; exit 1; }
"$PY" check_camera.py
echo
[ -t 0 ] && read -n 1 -s -r -p "Press any key to close"
exit 0
