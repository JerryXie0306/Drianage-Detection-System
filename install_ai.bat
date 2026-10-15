@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DrainGuard AI install

rem OPTIONAL. Adds the AI segmentation backend (debris % per type) and live-camera support.
rem Large download now, and the CLIPSeg model (about 600 MB) the first time the app runs.
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup_and_run.bat first. / 请先双击 setup_and_run.bat。
  pause
  exit /b 1
)

echo Installing the AI components, this is a big download... / 正在安装 AI 组件（较大，请耐心等待）...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --timeout 60 --retries 5 torch transformers opencv-python-headless scikit-learn
if errorlevel 1 (
  echo Install failed. Check your internet connection and try again. / 安装失败，请检查网络后重试。
) else (
  echo Done. Double-click setup_and_run.bat; it will now use the AI backend. / 完成，再双击 setup_and_run.bat 即可。
)
pause
