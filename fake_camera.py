"""Pretend to be an ESP32-CAM so you can test DrainGuard without the hardware.

    python fake_camera.py                 # clear -> partial -> blocked, cycling every 20 s
    python fake_camera.py --state blocked # always blocked

Then put this in camera.txt:   snapshot:http://127.0.0.1:8081/capture
(or the MJPEG stream:          http://127.0.0.1:8081/stream)
"""

import argparse
import io
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from drainguard import synthetic  # noqa: E402

CYCLE = ["clear", "partial", "blocked"]


class State:
    def __init__(self, fixed, period):
        self.fixed, self.period, self.t0 = fixed, period, time.time()
        self.rng = random.Random(1)
        self.lock = threading.Lock()

    def label(self):
        if self.fixed:
            return self.fixed
        return CYCLE[int((time.time() - self.t0) // self.period) % len(CYCLE)]

    def jpeg(self):
        with self.lock:
            img = synthetic.make_image(self.label(), self.rng).resize((640, 480))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return buf.getvalue()


def make_handler(state):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/capture":
                body = state.jpeg()
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                try:
                    while True:
                        b = state.jpeg()
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(b))
                        self.wfile.write(b + b"\r\n")
                        time.sleep(0.2)
                except (BrokenPipeError, ConnectionResetError):
                    pass
            else:
                body = b"fake ESP32-CAM: use /capture or /stream"
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--state", choices=CYCLE, help="always show this state instead of cycling")
    ap.add_argument("--period", type=float, default=20.0, help="seconds per state when cycling")
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(State(a.state, a.period)))
    srv.daemon_threads = True
    print(f"Fake ESP32-CAM on http://127.0.0.1:{a.port}   /capture (one picture)   /stream (video)   Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
