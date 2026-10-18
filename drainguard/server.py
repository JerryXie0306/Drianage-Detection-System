"""HTTP server (stdlib only) exposing the control app + JSON API.

    python -m drainguard.server --frames data/synthetic --port 8000

No authentication: bind to localhost (default) unless you are on a trusted network.
"""

import argparse
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional

from PIL import Image

from .actuator import SerialActuator, SimulatedActuator
from .config import ActuatorConfig, ControllerConfig, DetectorConfig
from .controller import CoverController
from .detector import Detector, annotate

WEB_DIR = Path(__file__).parent / "web"
MAX_UPLOAD = 10 * 1024 * 1024
MAX_SIDE = 1600  # longest side of the images shown in the app
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


class App:
    def __init__(self, detector: Detector, controller: CoverController,
                 frames_dir: Optional[Path] = None, interval: float = 2.0, source=None):
        self.detector = detector
        self.controller = controller
        self.frames_dir = frames_dir
        self.source = source  # live CameraSource / FolderWatcher: analysed automatically, no uploads needed
        self.interval = interval
        self.replay_playing = frames_dir is not None or source is not None
        self._detect_lock = threading.Lock()  # one analysis at a time (model inference is not re-entrant)
        self.frame_id = 0
        self.frame_jpeg: Optional[bytes] = None  # annotated / analysis image
        self.orig_jpeg: Optional[bytes] = None  # the image as received, for side-by-side comparison
        self.frame_size = None
        self.last_analysis_s: Optional[float] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()

    @staticmethod
    def _encode(img: Image.Image, scale: float) -> bytes:
        if scale < 1.0:
            img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=88)
        return buf.getvalue()

    def process_frame(self, img: Image.Image):
        img = img.convert("RGB")
        t0 = time.monotonic()
        with self._detect_lock:
            det = self.detector.detect(img)
            analysed = annotate(img, det, self.detector.cfg.roi)
        scale = min(1.0, MAX_SIDE / max(img.size))  # same factor for both images -> identical geometry
        orig_b, ana_b = self._encode(img, scale), self._encode(analysed, scale)
        with self._lock:
            self.orig_jpeg, self.frame_jpeg = orig_b, ana_b
            self.frame_size = img.size
            self.last_analysis_s = time.monotonic() - t0
            self.frame_id += 1
        self.controller.on_detection(det)
        return det

    def start_background(self):
        threading.Thread(target=self._poll_loop, daemon=True).start()
        if self.frames_dir:
            threading.Thread(target=self._replay_loop, daemon=True).start()
        if self.source is not None:
            self.source.start()
            threading.Thread(target=self._source_loop, daemon=True).start()

    def stop(self):
        self._stop.set()
        if self.source is not None:
            self.source.stop()

    def _poll_loop(self):
        while not self._stop.wait(0.5):
            self.controller.poll()

    def _replay_loop(self):
        files = sorted(p for p in Path(self.frames_dir).rglob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            print(f"[replay] no images found in {self.frames_dir}")
            return
        i = 0
        while not self._stop.wait(self.interval):
            if not self.replay_playing:
                continue
            try:
                self.process_frame(Image.open(files[i % len(files)]))
            except Exception as e:  # keep the demo alive on a bad file
                print(f"[replay] skipped {files[i % len(files)]}: {e}")
            i += 1

    def _source_loop(self):
        """Analyse the newest frame every `interval` s, measured from the START of each analysis."""
        last_id = -1
        next_at = time.monotonic()
        while not self._stop.wait(max(0.0, next_at - time.monotonic())):
            next_at = time.monotonic() + self.interval
            if not self.replay_playing:
                continue
            fid, img, _ = self.source.latest()
            if img is None or fid == last_id:
                continue
            last_id = fid
            try:
                self.process_frame(img)
            except Exception as e:
                print(f"[source] frame skipped: {e}")

    def status(self) -> dict:
        s = self.controller.status()
        s["frame_id"] = self.frame_id
        s["frame"] = {"size": self.frame_size, "analysis_s": self.last_analysis_s}
        s["replay"] = {"available": self.frames_dir is not None or self.source is not None,
                       "playing": self.replay_playing,
                       "kind": "live" if self.source is not None else "replay",
                       "source": getattr(self.source, "name", None),
                       "source_error": getattr(self.source, "error", None)}
        s["drain_id"] = "SB-01"
        return s


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _send(self, code: int, body: bytes, ctype: str):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _body(self) -> bytes:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_UPLOAD:
                raise ValueError("body too large")
            return self.rfile.read(n)

        def _json_body(self) -> dict:
            try:
                return json.loads(self._body() or b"{}")
            except json.JSONDecodeError:
                raise ValueError("invalid JSON")

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(app.status())
            elif path == "/api/frame_original.jpg":
                if app.orig_jpeg is None:
                    self._json({"error": "no frame yet"}, 404)
                else:
                    self._send(200, app.orig_jpeg, "image/jpeg")
            elif path == "/api/frame.jpg":
                if app.frame_jpeg is None:
                    self._json({"error": "no frame yet"}, 404)
                else:
                    self._send(200, app.frame_jpeg, "image/jpeg")
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                if path == "/api/frame":
                    img = Image.open(io.BytesIO(self._body()))
                    img.load()
                    app.replay_playing = False  # an operator upload takes over from the replay
                    det = app.process_frame(img)
                    self._json({"label": det.label, "confidence": det.confidence, "coverage": det.coverage})
                elif path == "/api/command":
                    self._json(app.controller.command(str(self._json_body().get("action", "")), "operator"))
                elif path == "/api/mode":
                    self._json(app.controller.set_mode(str(self._json_body().get("mode", ""))))
                elif path == "/api/flood":
                    self._json(app.controller.set_flood_risk(bool(self._json_body().get("active"))))
                elif path == "/api/replay":
                    if app.frames_dir is None and app.source is None:
                        self._json({"ok": False, "msg": "no replay source"}, 400)
                    else:
                        app.replay_playing = bool(self._json_body().get("playing"))
                        self._json({"ok": True, "playing": app.replay_playing})
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, OSError) as e:
                self._json({"error": str(e)}, 400)

    return Handler


def make_server(app: App, host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), make_handler(app))
    srv.daemon_threads = True
    return srv


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--frames", help="folder of images replayed as a simulated camera")
    ap.add_argument("--interval", type=float, default=2.0, help="seconds between analysed frames")
    ap.add_argument("--camera", help="LIVE source analysed automatically: webcam index (0), RTSP/HTTP stream URL, or video file")
    ap.add_argument("--watch", help="LIVE source: analyse the newest image appearing in this folder")
    ap.add_argument("--snapshot", help="LIVE source: URL that returns one JPEG per request, e.g. ESP32-CAM http://IP/capture")
    ap.add_argument("--open-frames", type=int, default=3, help="consecutive blocked frames before auto-open")
    ap.add_argument("--close-frames", type=int, default=10, help="consecutive clear frames before auto-close")
    ap.add_argument("--backend", choices=["heuristic", "model", "segment"], default="heuristic")
    ap.add_argument("--model", help="path to trained model (backend=model)")
    ap.add_argument("--regions", help="regions JSON (grate / sump polygons) for backend=segment")
    ap.add_argument("--head", help="trained segmentation head (backend=segment); zero-shot if omitted")
    ap.add_argument("--serial", help="serial port of the real actuator, e.g. /dev/tty.usbmodem1101")
    ap.add_argument("--actuator-delay", type=float, default=2.0, help="simulated open/close time (s)")
    ap.add_argument("--roi", type=float, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), default=(0, 0, 1, 1))
    a = ap.parse_args()

    segmenter = None
    if a.backend == "segment":
        from .segment import Segmenter
        segmenter = Segmenter(head_path=a.head, regions_path=a.regions)
    detector = Detector(a.backend, a.model, DetectorConfig(roi=tuple(a.roi)), segmenter=segmenter)
    actuator = SerialActuator(a.serial) if a.serial else SimulatedActuator(ActuatorConfig(delay_s=a.actuator_delay))
    controller = CoverController(actuator, ControllerConfig(open_frames=a.open_frames, close_frames=a.close_frames))
    source = None
    if a.camera:
        from .camera import CameraSource
        source = CameraSource(a.camera)
    elif a.snapshot:
        from .camera import SnapshotSource
        source = SnapshotSource(a.snapshot)
    elif a.watch:
        from .camera import FolderWatcher
        source = FolderWatcher(a.watch)
    app = App(detector, controller, Path(a.frames) if a.frames else None, a.interval, source)
    app.start_background()
    srv = make_server(app, a.host, a.port)
    print(f"DrainGuard running on http://{a.host}:{a.port}  "
          f"(backend={a.backend}, actuator={actuator.name}{' [SIMULATED]' if actuator.simulated else ''})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        app.stop()


if __name__ == "__main__":
    main()
