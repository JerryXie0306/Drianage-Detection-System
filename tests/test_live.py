import json
import random
import threading
import time
import urllib.error
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


def test_snapshot_source_delivers_frames_and_survives_bad_pictures(rng):
    from drainguard.camera import SnapshotSource

    imgs = [synthetic.make_image("blocked", rng) for _ in range(3)]
    calls = {"n": 0}

    def fetch():
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("image file is truncated")  # ESP32-CAMs sometimes send half a JPEG
        return imgs[min(calls["n"] - 1, 2)]

    src = SnapshotSource("http://cam/capture", period_s=0.02, retry_s=0.02, fetch=fetch).start()
    try:
        end = time.time() + 3
        while src.latest()[0] < 2 and time.time() < end:
            time.sleep(0.02)
        fid, img, age = src.latest()
        assert fid >= 2 and img is not None and age < 2
        assert calls["n"] >= 3  # kept going after the bad picture
        assert src.error is None
    finally:
        src.stop()


def test_snapshot_source_reports_unreachable_camera():
    from drainguard.camera import SnapshotSource

    def fetch():
        raise OSError("timed out")

    src = SnapshotSource("http://192.168.1.50/capture", period_s=0.02, retry_s=0.02, fetch=fetch).start()
    try:
        end = time.time() + 2
        while src.error is None and time.time() < end:
            time.sleep(0.02)
        assert src.error and "192.168.1.50" in src.error
        assert src.latest()[1] is None
    finally:
        src.stop()


def test_snapshot_source_is_idle_until_asked(rng):
    from drainguard.camera import SnapshotSource

    calls = []
    img = synthetic.make_image("blocked", rng)
    src = SnapshotSource("http://cam/capture", fetch=lambda: calls.append(1) or img).start()
    time.sleep(0.3)
    assert calls == [] and src.latest()[1] is None  # no background polling: a slow schedule leaves the camera alone
    fid, got, _ = src.grab()
    assert calls == [1] and got is not None and fid == 1
    assert src.grab()[0] == 2


def test_snapshot_grab_failure_gives_no_stale_picture(rng):
    from drainguard.camera import SnapshotSource

    state = {"ok": True}
    img = synthetic.make_image("clear", rng)

    def fetch():
        if not state["ok"]:
            raise OSError("timed out")
        return img

    src = SnapshotSource("http://cam/capture", fetch=fetch)
    assert src.grab()[1] is not None
    state["ok"] = False
    fid, got, age = src.grab()
    assert got is None and "timed out" in src.error  # never re-analyse an old picture as if it were new


def test_capture_now_endpoint_and_slow_schedule_status(rng):
    from drainguard.camera import SnapshotSource

    img = synthetic.make_image("blocked", rng)
    src = SnapshotSource("http://cam/capture", fetch=lambda: img)
    ctrl = CoverController(SimulatedActuator(ActuatorConfig(delay_s=0.05, jitter_s=0.0)),
                           ControllerConfig(open_frames=2, camera_online_s=900))
    app = App(Detector(), ctrl, interval=300, source=src)
    app.start_background()
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        def post(path):
            req = urllib.request.Request(base + path, data=b"", method="POST")
            return json.loads(urllib.request.urlopen(req).read())

        end = time.time() + 5
        while app.frame_id < 1 and time.time() < end:  # the schedule analyses once at start
            time.sleep(0.05)
        assert post("/api/capture")["label"] == "blocked"
        assert app.frame_id == 2  # the manual capture counted as a frame; the 5-minute schedule did not fire again
        s = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert s["camera_online"] is True and s["replay"]["interval_s"] == 300 and 200 < s["replay"]["next_in_s"] <= 300
        end = time.time() + 3
        while s["cover_state"] != "open" and time.time() < end:  # two blocked pictures in a row open the cover
            time.sleep(0.05)
            s = json.loads(urllib.request.urlopen(base + "/api/status").read())
        assert s["cover_state"] == "open"
    finally:
        srv.shutdown()
        app.stop()


def test_capture_without_a_live_camera_is_rejected():
    app = App(Detector(), CoverController(SimulatedActuator(ActuatorConfig(delay_s=0.01))))
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        req = urllib.request.Request(f"http://127.0.0.1:{srv.server_address[1]}/api/capture", data=b"", method="POST")
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400
    finally:
        srv.shutdown()
