#!/usr/bin/env bash
# Generates synthetic demo frames (if missing) and starts the control app on http://127.0.0.1:8000
set -e
cd "$(dirname "$0")"
if [ ! -d data/synthetic ]; then
  python3 -m drainguard.synthetic --out data/synthetic --per-class 8
fi
exec python3 -m drainguard.server --frames data/synthetic --interval 2 "$@"
