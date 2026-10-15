import json
import random
import threading
import time
import urllib.request

import pytest
from PIL import Image

from drainguard import synthetic
from drainguard.actuator import SimulatedActuator
from drainguard.camera import CameraSource, FolderWatcher
from drainguard.config import ActuatorConfig, ControllerConfig
from drainguard.controller import CoverController
from drainguard.detector import Detector
from drainguard.server import App, make_server


class FakeCap:
    """Stands in for cv2.VideoCapture."""

    def __init__(self, frames, opened=True, fps=0):
        self.frames, self.opened, self.i, self.fps = frames, opened, 0, fps
        self.released = False

    def isOpened(self):
        return self.opened

    def read(self):
        if self.i >= len(self.frames):
            return False, None
        self.i += 1
        return True, self.frames[self.i - 1]

    def get(self, prop):
        return self.fps

    def set(self, prop, value):
        self.i = 0
        return True

    def release(self):
        self.released = True


def wait(cond, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def src_with(frames_fn, **kw):
    return CameraSource("0", capture_factory=frames_fn, to_image=lambda f: f, reconnect_s=0.05, **kw)


def test_camera_source_keeps_latest_frame():
    imgs = [Image.new("RGB", (8, 8), (i, i, i)) for i in range(5)]
    cam = src_with(lambda s: FakeCap(imgs)).start()
    try:
        assert wait(lambda: cam.latest()[1] is not None)
        fid, img, age = cam.latest()
        assert fid >= 1 and age < 1
    finally:
        cam.stop()


def test_camera_reconnects_after_failures():
    attempts = {"n": 0}
    img = Image.new("RGB", (8, 8))

    def factory(source):
        attempts["n"] += 1
        return FakeCap([img] * 50, opened=attempts["n"] >= 3)

    cam = src_with(factory).start()
    try:
        assert wait(lambda: cam.latest()[1] is not None)
        assert attempts["n"] >= 3
    finally:
        cam.stop()


def test_camera_reports_error_when_it_cannot_open():
    cam = src_with(lambda s: FakeCap([], opened=False)).start()
    try:
        assert wait(lambda: cam.error is not None)
        assert "cannot open" in cam.error and cam.latest()[1] is None
    finally:
        cam.stop()


def test_video_file_loops(tmp_path):
    f = tmp_path / "clip.mp4"
    f.write_bytes(b"x")  # exists, so it is treated as a file source
    imgs = [Image.new("RGB", (8, 8), (i * 40, 0, 0)) for i in range(3)]
    cam = CameraSource(str(f), capture_factory=lambda s: FakeCap(imgs, fps=1000), to_image=lambda x: x).start()
    try:
        assert wait(lambda: cam.latest()[0] > 6)  # more frames than the clip holds -> it looped
    finally:
        cam.stop()


def test_folder_watcher_reports_only_new_images(tmp_path):
    w = FolderWatcher(str(tmp_path))
    assert w.latest()[1] is None
    Image.new("RGB", (8, 8), (10, 10, 10)).save(tmp_path / "a.png")
    fid1, img1, _ = w.latest()
    assert img1 is not None
    assert w.latest()[1] is None  # same file again: nothing new
    time.sleep(0.01)
    Image.new("RGB", (8, 8), (200, 10, 10)).save(tmp_path / "b.png")
    fid2, img2, _ = w.latest()
    assert img2 is not None and fid2 == fid1 + 1


def test_folder_watcher_ignores_half_written_file(tmp_path):
    (tmp_path / "c.png").write_bytes(b"\x89PNG not finished")
    assert FolderWatcher(str(tmp_path)).latest()[1] is None


def test_automatic_end_to_end_no_upload_no_clicks():
    """Camera frames in -> detection -> controller -> cover opens, with nobody touching anything."""
    rng = random.Random(5)
    frames = [synthetic.make_image("blocked", rng) for _ in range(6)]
    cam = src_with(lambda s: FakeCap(frames * 30))
    act = SimulatedActuator(ActuatorConfig(delay_s=0.05, jitter_s=0))
    ctrl = CoverController(act, ControllerConfig(open_frames=2))
    app = App(Detector(), ctrl, interval=0.05, source=cam)
    app.start_background()
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        def status():
            return json.loads(urllib.request.urlopen(base + "/api/status").read())

        assert wait(lambda: status()["cover_state"] == "open", 5)
        s = status()
        assert s["mode"] == "auto" and s["detection"]["label"] == "blocked"
        assert s["replay"]["kind"] == "live" and s["camera_online"] is True
    finally:
        srv.shutdown()
        app.stop()


def test_pausing_live_analysis_stops_processing():
    rng = random.Random(6)
    frames = [synthetic.make_image("clear", rng) for _ in range(5)]
    cam = src_with(lambda s: FakeCap(frames * 50))
    app = App(Detector(), CoverController(SimulatedActuator(ActuatorConfig(delay_s=0.01, jitter_s=0))),
              interval=0.03, source=cam)
    app.start_background()
    try:
        assert wait(lambda: app.frame_id >= 2)
        app.replay_playing = False
        time.sleep(0.15)
        n = app.frame_id
        time.sleep(0.2)
        assert app.frame_id == n
    finally:
        app.stop()
