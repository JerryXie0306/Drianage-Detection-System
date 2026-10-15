@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
title DrainGuard

rem Double-click me. First run installs what DrainGuard needs (a few minutes, needs internet), then starts it.
set "VPY=.venv\Scripts\python.exe"
if exist "%VPY%" goto deps

set "PY="
where py >nul 2>nul
if not errorlevel 1 (
  py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
  if not errorlevel 1 set "PY=py -3"
)
if not defined PY (
  where python >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if not errorlevel 1 set "PY=python"
  )
)
if not defined PY goto nopython

echo Creating environment...
%PY% -m venv .venv
if errorlevel 1 goto fail

:deps
"%VPY%" -c "import numpy, PIL, scipy" >nul 2>nul
if errorlevel 1 (
  echo First run: installing components, about 1-3 minutes...
  echo 首次运行，正在安装组件，请稍等 1-3 分钟...
  "%VPY%" -m pip install --disable-pip-version-check --timeout 60 --retries 5 -q numpy Pillow scipy
  if errorlevel 1 goto fail
)

echo Starting DrainGuard... / 正在启动...
"%VPY%" launch.py
echo.
echo DrainGuard stopped.
pause
exit /b 0

:nopython
start "" "https://www.python.org/downloads/windows/"
echo.
echo Python 3 was not found on this computer.
echo   1. On the web page that just opened, download and install Python.
echo      IMPORTANT: tick "Add python.exe to PATH" on the first installer screen.
echo   2. Then double-click setup_and_run.bat again.
echo 未找到 Python 3。请在刚打开的网页下载安装 Python（安装第一页勾选 Add python.exe to PATH），然后再次双击本文件。
echo.
pause
exit /b 1

:fail
echo.
echo Something went wrong. Please check your internet connection and double-click again.
echo 出错了，请检查网络后再次双击本文件。
echo.
pause
exit /b 1
