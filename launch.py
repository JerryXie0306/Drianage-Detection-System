"""Cross-platform launcher used by setup_and_run.bat: picks a free port, chooses the backend,
starts the server and opens the browser. Set DG_NO_BROWSER=1 to skip opening the browser."""

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))


def free_port() -> int:
    for p in range(8000, 8100):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            return p
        except OSError:
            continue
        finally:
            s.close()
    raise SystemExit("No free port between 8000 and 8099.")


def open_when_ready(url: str):
    for _ in range(90):
        try:
            urllib.request.urlopen(url + "/api/status", timeout=1).read()
            webbrowser.open(url)
            return
        except Exception:
            time.sleep(1)


def main():
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    args = ["--port", str(port)]
    try:
        import torch, transformers  # noqa: F401
        regions = "regions/drain.json" if Path("regions/drain.json").exists() else "regions/example_fisheye.json"
        args += ["--backend", "segment", "--regions", regions]
        if Path("models/seg_head.joblib").exists():
            args += ["--head", "models/seg_head.joblib"]
        frames = Path("data/seg/images")
        if frames.is_dir() and any(frames.iterdir()):
            args += ["--frames", str(frames), "--interval", "8"]
        print(f"Mode: segmentation (regions: {regions})")
    except ImportError:
        print("Mode: colour heuristic (AI components not installed; run install_ai for the segmentation backend)")
        if not Path("data/synthetic").is_dir():
            subprocess.run([sys.executable, "-m", "drainguard.synthetic", "--out", "data/synthetic", "--per-class", "8"], check=True)
        args += ["--frames", "data/synthetic", "--interval", "2"]

    print(f"Starting DrainGuard at {url}   (close this window or press Ctrl+C to stop)")
    if not os.environ.get("DG_NO_BROWSER"):
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    from drainguard.server import main as server_main
    sys.argv = ["drainguard.server"] + args
    server_main()


if __name__ == "__main__":
    main()
