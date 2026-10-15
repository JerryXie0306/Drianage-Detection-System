"""Browser tool for labelling images (and calibrating the grate / sump regions).

    python -m drainguard.label --images data/seg/images --masks data/seg/masks --regions regions/drain.json
    python -m drainguard.label ... --predict --head models/seg_head.joblib   # pre-label with the model, then correct

Masks are saved as 8-bit PNGs at the image's own resolution: 0 = unlabelled, 1..6 = classes
(see seg_metrics.CLASSES). Region polygons are saved to --regions as normalised coordinates.
Binds to localhost only.
"""

import argparse
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .regions import save_regions, validate_regions
from .seg_metrics import CLASSES, COLORS

WEB = Path(__file__).parent / "web" / "label.html"
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}
MAX_BODY = 40 * 1024 * 1024


class LabelApp:
    def __init__(self, images: Path, masks: Path, regions_path: str = None, predict=False, head=None):
        self.images, self.masks = images, masks
        self.regions_path = regions_path
        self.masks.mkdir(parents=True, exist_ok=True)
        self.want_predict, self.head = predict, head
        self._seg = None
        self._lock = threading.Lock()

    def names(self):
        return sorted(p.name for p in self.images.iterdir() if p.suffix.lower() in IMG_EXT)

    def safe_image(self, name: str) -> Path:
        p = (self.images / Path(name).name)
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            raise FileNotFoundError(name)
        return p

    def mask_path(self, name: str) -> Path:
        return self.masks / (Path(name).stem + ".png")

    def regions(self):
        if self.regions_path and Path(self.regions_path).exists():
            return json.loads(Path(self.regions_path).read_text())
        return {}

    def predict_mask(self, name: str) -> bytes:
        from .segment import Segmenter

        with self._lock:
            if self._seg is None:
                self._seg = Segmenter(head_path=self.head, regions_path=self.regions_path
                                      if self.regions_path and Path(self.regions_path).exists() else None)
            img = Image.open(self.safe_image(name)).convert("RGB")
            out = self._seg.segment(img)
        m = Image.fromarray(out.label_map, "L").resize(img.size, Image.NEAREST)
        buf = io.BytesIO()
        m.save(buf, format="PNG")
        return buf.getvalue()


def make_handler(app: LabelApp):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def do_GET(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            name = (q.get("name") or [""])[0]
            try:
                if u.path == "/":
                    self._send(200, WEB.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/list":
                    self._json({"images": [{"name": n, "labelled": app.mask_path(n).exists()} for n in app.names()],
                                "classes": [{"name": c, "id": i + 1, "color": COLORS[c]} for i, c in enumerate(CLASSES)],
                                "regions": app.regions(), "can_predict": app.want_predict,
                                "regions_file": bool(app.regions_path)})
                elif u.path == "/api/image":
                    p = app.safe_image(name)
                    ctype = "image/png" if p.suffix.lower() == ".png" else "image/jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "image/webp"
                    self._send(200, p.read_bytes(), ctype)
                elif u.path == "/api/mask":
                    mp = app.mask_path(name)
                    if not mp.exists():
                        self._json({"error": "no mask"}, 404)
                    else:
                        self._send(200, mp.read_bytes(), "image/png")
                elif u.path == "/api/predict":
                    if not app.want_predict:
                        self._json({"error": "start with --predict"}, 400)
                    else:
                        self._send(200, app.predict_mask(name), "image/png")
                else:
                    self._json({"error": "not found"}, 404)
            except FileNotFoundError:
                self._json({"error": "not found"}, 404)

        def do_POST(self):
            u = urlparse(self.path)
            name = (parse_qs(u.query).get("name") or [""])[0]
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY:
                return self._json({"error": "too large"}, 400)
            body = self.rfile.read(n)
            try:
                if u.path == "/api/save":
                    size = Image.open(app.safe_image(name)).size
                    rgba = Image.open(io.BytesIO(body)).convert("RGBA")
                    if rgba.size != size:
                        return self._json({"error": f"mask size {rgba.size} != image size {size}"}, 400)
                    idx = np.asarray(rgba)[..., 0]  # index is stored in the red channel
                    if idx.max() > len(CLASSES):
                        return self._json({"error": "invalid class index"}, 400)
                    Image.fromarray(idx.astype(np.uint8), "L").save(app.mask_path(name))
                    self._json({"ok": True, "labelled_pixels": int((idx > 0).sum())})
                elif u.path == "/api/regions":
                    if not app.regions_path:
                        return self._json({"error": "start with --regions FILE"}, 400)
                    regs = json.loads(body)
                    validate_regions(regs)  # validate BEFORE touching the file
                    save_regions(app.regions_path, {k: v for k, v in regs.items()
                                                    if k == "_open_ratio" or isinstance(v, list)})
                    self._json({"ok": True})
                else:
                    self._json({"error": "not found"}, 404)
            except (ValueError, FileNotFoundError, OSError) as e:
                self._json({"error": str(e)}, 400)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/seg/images")
    ap.add_argument("--masks", default="data/seg/masks")
    ap.add_argument("--regions", help="regions JSON to display and edit (created on save)")
    ap.add_argument("--predict", action="store_true", help="enable model pre-labelling (loads CLIPSeg)")
    ap.add_argument("--head", help="trained head for pre-labelling")
    ap.add_argument("--port", type=int, default=8770)
    a = ap.parse_args()
    app = LabelApp(Path(a.images), Path(a.masks), a.regions, a.predict, a.head)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(app))
    srv.daemon_threads = True
    print(f"Label tool: http://127.0.0.1:{a.port}   ({len(app.names())} images in {a.images})")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
