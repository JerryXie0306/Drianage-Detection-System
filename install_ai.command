#!/bin/bash
# OPTIONAL. Adds the AI segmentation backend (debris % per type) and live-camera support.
# Downloads a few hundred MB now, and the CLIPSeg model (~600 MB) the first time the app runs. Needs internet.
cd "$(dirname "$0")" || exit 1

pause() { [ -t 0 ] && read -n 1 -s -r -p "Press any key to close"; echo; }

if [ ! -x .venv/bin/python ]; then
  echo "Please double-click setup_and_run.command first."
  pause; exit 1
fi

echo "Installing the AI components..."
.venv/bin/python -m pip install --disable-pip-version-check --timeout 60 --retries 5 torch transformers opencv-python-headless scikit-learn \
  && echo "Done. Double-click setup_and_run.command; it will now use the AI backend." \
  || echo "Install failed. Check your internet connection and try again."
pause
