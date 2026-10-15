#!/bin/bash
# Double-click: starts DrainGuard with fully automatic live recognition (camera source set in camera.txt).
exec "$(cd "$(dirname "$0")" && pwd)/start_live.command"
