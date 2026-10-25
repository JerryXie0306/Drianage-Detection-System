@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DrainGuard live

rem Double-click: fully automatic recognition from the camera set in camera.txt.
if not exist ".venv\Scripts\python.exe" (
  echo Please double-click setup_and_run.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" launch.py --live
echo.
echo DrainGuard stopped.
pause
