#!/bin/bash
# Double-click: fully automatic recognition from the camera set in camera.txt (no uploading, no clicking).
cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
[ -x "$PY" ] || { echo "Please double-click setup_and_run.command first. / 请先双击 setup_and_run.command。"; [ -t 0 ] && read -n 1 -s -r -p "Press any key to close"; exit 1; }

"$PY" launch.py --live

echo
echo "DrainGuard stopped."
[ -t 0 ] && read -n 1 -s -r -p "Press any key to close"
exit 0
