@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DrainGuard AI install

rem OPTIONAL. Adds the AI segmentation backend (debris % per type) and live-camera support.
rem Large download now, and the CLIPSeg model (about 600 MB) the first time the app runs.
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup_and_run.bat first.
  pause
  exit /b 1
)

echo Installing the AI components, this is a big download...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --timeout 60 --retries 5 torch transformers opencv-python-headless scikit-learn
if errorlevel 1 (
  echo Install failed. Check your internet connection and try again.
) else (
  echo Done. Double-click setup_and_run.bat; it will now use the AI backend.
)
pause
