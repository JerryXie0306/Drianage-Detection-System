@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DrainGuard camera check

rem Double-click: tests that the camera in camera.txt can be reached, and saves a test picture.
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup_and_run.bat first. / 请先双击 setup_and_run.bat。
  pause
  exit /b 1
)
".venv\Scripts\python.exe" check_camera.py
echo.
pause
