"""Cross-platform launcher used by setup_and_run.bat: picks a free port, chooses the backend,
starts the server and opens the browser. Set DG_NO_BROWSER=1 to skip opening the browser."""

import atexit
import os
import signal
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


PIDFILE = ROOT / ".drainguard.pid"


def replace_previous_instance():
    """Only one DrainGuard at a time: two copies would both command the same motor."""
    if os.name == "posix" and PIDFILE.exists():
        try:
            old = int(PIDFILE.read_text().strip())
            cmd = subprocess.run(["ps", "-p", str(old), "-o", "command="], capture_output=True, text=True).stdout
            if old != os.getpid() and "launch.py" in cmd:
                print("Another DrainGuard is still running: stopping it first.")
                os.kill(old, signal.SIGINT)  # lets it send /stop to the motor on its way out
                for _ in range(30):
                    time.sleep(0.1)
                    if subprocess.run(["ps", "-p", str(old)], capture_output=True).returncode != 0:
                        break
                else:
                    os.kill(old, signal.SIGTERM)
                    time.sleep(1)
        except Exception:
            pass
    PIDFILE.write_text(str(os.getpid()))
    atexit.register(lambda: PIDFILE.unlink(missing_ok=True))


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


def motor_args():
    p = Path("motor.txt")
    lines = [l.strip() for l in (p.read_text().splitlines() if p.exists() else [])
             if l.strip() and not l.strip().startswith("#")]
    if not lines:
        print("Cover: SIMULATED (no motor set in motor.txt)")
        return []
    args = ["--motor-url", lines[0], "--start-mode", "manual"]  # a real motor never starts in AUTO by itself
    if len(lines) > 1:
        args += ["--open-ms", lines[1]]
    if len(lines) > 2:
        args += ["--close-ms", lines[2]]
    if len(lines) > 3:
        args += ["--open-dir", lines[3]]
    print("Starting in MANUAL mode: press AUTO in the app when you want the system to open the cover by itself.")
    print(f"Cover: REAL MOTOR at {lines[0]}  (open {lines[1] if len(lines) > 1 else 3000} ms, "
          f"close {lines[2] if len(lines) > 2 else 3000} ms, opens by {lines[3] if len(lines) > 3 else 'forward'})")
    return args


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
    print(f"Live source: {src}   (analysis every {interval} s, or press 'Capture now' in the app)")
    try:
        slow = float(interval) >= 120
    except ValueError:
        slow = False
    if slow:
        args += ["--open-frames", "2", "--close-frames", "2"]
        print("Slow schedule: the cover opens after 2 blocked pictures in a row (instead of 3) and closes after 2 clear ones.")
    return args


def main():
    live = "--live" in sys.argv
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    args = ["--port", str(port)] + motor_args()
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
    replace_previous_instance()
    print(f"Starting DrainGuard at {url}   (close this window or press Ctrl+C to stop)")
    if not os.environ.get("DG_NO_BROWSER"):
        threading.Thread(target=open_when_ready, args=(url,), daemon=True).start()

    from drainguard.server import main as server_main
    sys.argv = ["drainguard.server"] + args
    server_main()


if __name__ == "__main__":
    main()
