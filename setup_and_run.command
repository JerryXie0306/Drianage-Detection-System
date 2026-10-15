#!/bin/bash
# Double-click me. First run installs what DrainGuard needs (a few minutes, needs internet), then starts it.
# Next runs start in seconds.
cd "$(dirname "$0")" || exit 1

pause() { [ -t 0 ] && read -n 1 -s -r -p "Press any key to close / 按任意键关闭"; echo; }
fail() { echo; echo "$1"; echo; pause; exit 1; }

# macOS marks files downloaded from the internet; clear that so the other launchers open without warnings.
xattr -dr com.apple.quarantine . 2>/dev/null
chmod +x ./*.command 2>/dev/null

ok_python() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' >/dev/null 2>&1; }

find_python() {
  for p in /opt/homebrew/bin/python3 /usr/local/bin/python3 \
           /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
           /opt/anaconda3/bin/python3 "$HOME/anaconda3/bin/python3" "$HOME/miniconda3/bin/python3"; do
    [ -x "$p" ] && ok_python "$p" && { echo "$p"; return; }
  done
  # /usr/bin/python3 is only a stub until Apple's developer tools are installed; running it would pop up an installer.
  if xcode-select -p >/dev/null 2>&1 && [ -x /usr/bin/python3 ] && ok_python /usr/bin/python3; then
    echo /usr/bin/python3
  fi
}

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ] || ! "$VENV_PY" -c 'import sys' >/dev/null 2>&1; then
  PY=$(find_python)
  if [ -z "$PY" ]; then
    open "https://www.python.org/downloads/macos/" 2>/dev/null
    fail "Python 3 was not found on this Mac.
  1. On the web page that just opened, download and install Python (the big yellow button).
  2. Then double-click this file again.
未找到 Python 3。请在刚打开的网页下载并安装 Python，然后再次双击本文件。"
  fi
  echo "Using Python: $PY"
  rm -rf .venv
  "$PY" -m venv .venv || fail "Could not create the .venv environment. / 创建环境失败。"
fi

if ! "$VENV_PY" -c 'import numpy, PIL, scipy' >/dev/null 2>&1; then
  echo "First run: installing components (about 1-3 minutes)... / 首次运行，正在安装组件，请稍等 1-3 分钟..."
  "$VENV_PY" -m pip install --disable-pip-version-check --timeout 60 --retries 5 -q numpy Pillow scipy \
    || fail "Install failed. Please check your internet connection and double-click again. / 安装失败，请检查网络后重试。"
fi

echo "Starting DrainGuard... / 正在启动..."
exec ./start_app.command
