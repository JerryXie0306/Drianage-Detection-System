"""Live frame sources, so recognition runs automatically with nobody uploading photos.

CameraSource  - anything OpenCV can open: a webcam index ("0"), an RTSP/HTTP stream URL, or a video file
                (files loop, which is handy for demos). A background thread keeps only the LATEST frame so
                slow analysis never works on stale, buffered video. Reconnects automatically if the stream drops.
FolderWatcher - picks up the newest image written to a folder (cameras / phones that save snapshots).
"""

import threading
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def _default_capture(source):
    import cv2

    return cv2.VideoCapture(source)


class CameraSource:
    def __init__(self, source: str, capture_factory: Callable = _default_capture,
                 reconnect_s: float = 3.0, to_image: Optional[Callable] = None):
        self.source = int(source) if str(source).strip().isdigit() else str(source)
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.name = f"camera {self.source}" if isinstance(self.source, int) else str(self.source)
        self._factory = capture_factory
        self._reconnect_s = reconnect_s
        self._to_image = to_image or self._bgr_to_pil
        self._lock = threading.Lock()
        self._frame: Optional[Image.Image] = None
        self._frame_id = 0
        self._frame_time = 0.0
        self.error: Optional[str] = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @staticmethod
    def _bgr_to_pil(frame) -> Image.Image:
        import cv2

        return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def _run(self):
        while not self._stop.is_set():
            cap = None
            try:
                cap = self._factory(self.source)
                if not cap.isOpened():
                    raise RuntimeError("cannot open source (is camera permission granted? is the URL right?)")
                self.error = None
                fps = cap.get(5) if self.is_file else 0  # CAP_PROP_FPS
                delay = 1.0 / fps if fps and fps > 1 else 0.03
                fails = 0
                while not self._stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        if self.is_file:  # loop the demo video
                            cap.set(1, 0)  # CAP_PROP_POS_FRAMES
                            fails += 1
                            if fails > 3:
                                raise RuntimeError("video file has no readable frames")
                            continue
                        raise RuntimeError("stream stopped delivering frames")
                    fails = 0
                    img = self._to_image(frame)
                    with self._lock:
                        self._frame, self._frame_id, self._frame_time = img, self._frame_id + 1, time.time()
                    if self.is_file:
                        time.sleep(delay)
            except Exception as e:  # keep trying: cameras get unplugged, Wi-Fi drops
                self.error = str(e)
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            self._stop.wait(self._reconnect_s)

    def latest(self) -> Tuple[int, Optional[Image.Image], float]:
        """(frame_id, image, age_seconds). image is None until the first frame arrives."""
        with self._lock:
            age = time.time() - self._frame_time if self._frame is not None else float("inf")
            return self._frame_id, self._frame, age


class FolderWatcher:
    """Reports the newest image in a folder; only a *new* file/modification counts as a new frame."""

    def __init__(self, folder: str):
        self.folder = Path(folder)
        self.name = f"folder {folder}"
        self.error: Optional[str] = None
        self._seen = None
        self._id = 0

    def start(self):
        return self

    def stop(self):
        pass

    def latest(self) -> Tuple[int, Optional[Image.Image], float]:
        try:
            files = [p for p in self.folder.iterdir() if p.suffix.lower() in IMG_EXT]
        except OSError as e:
            self.error = str(e)
            return self._id, None, float("inf")
        self.error = None
        if not files:
            return self._id, None, float("inf")
        newest = max(files, key=lambda p: p.stat().st_mtime_ns)
        key = (newest.name, newest.stat().st_mtime_ns)
        age = time.time() - newest.stat().st_mtime
        if key == self._seen:
            return self._id, None, age  # nothing new
        try:
            img = Image.open(newest)
            img.load()
        except OSError:  # file still being written - try again next tick
            return self._id, None, age
        self._seen, self._id = key, self._id + 1
        return self._id, img, age
