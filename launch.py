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


def read_camera_txt():
    lines = []
    p = Path("camera.txt")
    for line in p.read_text().splitlines() if p.exists() else []:
        line = line.strip()
        if line and not line.startswith("#"):
            lines.append(line)
    return (lines[0] if lines else "0"), (lines[1] if len(lines) > 1 else "5")


def live_args():
    src, interval = read_camera_txt()
    args = ["--interval", interval]
    if src.startswith("snapshot:"):
        args += ["--snapshot", src[len("snapshot:"):]]
    elif src.startswith("folder:"):
        args += ["--watch", src[len("folder:"):]]
    else:
        try:
            import cv2  # noqa: F401
        except ImportError:
            raise SystemExit("OpenCV is needed to read a webcam or video stream. Run install_ai once, "
                             "or use 'snapshot:http://IP/capture' in camera.txt (needs no OpenCV).")
        args += ["--camera", src]
    print(f"Live source: {src}   (analysis every {interval} s)")
    return args


def main():
    live = "--live" in sys.argv
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    args = ["--port", str(port)]
    if live:
        args += live_args()
        try:
            import torch, transformers  # noqa: F401
            regions = "regions/drain.json" if Path("regions/drain.json").exists() else "regions/example_fisheye.json"
            args += ["--backend", "segment", "--regions", regions]
            if Path("models/seg_head.joblib").exists():
                args += ["--head", "models/seg_head.joblib"]
            print("Mode: segmentation")
        except ImportError:
            print("Mode: colour heuristic (run install_ai for the AI segmentation backend)")
        return run(args, url)
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

    run(args, url)


def run(args, url):
    print(f"Starting DrainGuard at {url}   (close this window or press Ctrl+C to stop)")
    if not os.environ.get("DG_NO_BROWSER"):
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    from drainguard.server import main as server_main
    sys.argv = ["drainguard.server"] + args
    server_main()


if __name__ == "__main__":
    main()
