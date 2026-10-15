import io
import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from PIL import Image

from drainguard import synthetic
from drainguard.actuator import SimulatedActuator
from drainguard.config import ActuatorConfig, ControllerConfig
from drainguard.controller import CoverController
from drainguard.detector import Detector
from drainguard.server import App, make_server


@pytest.fixture
def server():
    act = SimulatedActuator(ActuatorConfig(delay_s=0.05, jitter_s=0.0))
    ctrl = CoverController(act, ControllerConfig(open_frames=2))
    app = App(Detector(), ctrl)
    app.start_background()
    srv = make_server(app, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", app
    srv.shutdown()
    app.stop()


def call(base, path, body=None, raw=None):
    data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
    req = urllib.request.Request(base + path, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            payload = r.read()
            return r.status, (json.loads(payload) if r.headers["Content-Type"] == "application/json" else payload)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def jpeg(label, rng):
    buf = io.BytesIO()
    synthetic.make_image(label, rng).save(buf, format="JPEG")
    return buf.getvalue()


def wait_state(base, state, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        s = call(base, "/api/status")[1]
        if s["cover_state"] == state:
            return s
        time.sleep(0.05)
    raise AssertionError(f"never reached {state}: {s['cover_state']}")


def test_index_and_status(server):
    base, _ = server
    code, html = call(base, "/")
    assert code == 200 and b"Drain Control App" in html
    s = call(base, "/api/status")[1]
    assert s["cover_state"] == "closed" and s["mode"] == "auto" and s["actuator"]["simulated"] is True


def test_blocked_frames_trigger_auto_open_end_to_end(server, rng):
    base, _ = server
    for _ in range(2):
        code, res = call(base, "/api/frame", raw=jpeg("blocked", rng))
        assert code == 200 and res["label"] == "blocked"
    s = wait_state(base, "open")
    assert s["last_delay_s"] is not None and s["detection"]["label"] == "blocked"
    code, jpg = call(base, "/api/frame.jpg")
    assert code == 200 and Image.open(io.BytesIO(jpg)).size == (320, 240)


def test_manual_commands_and_mode(server):
    base, _ = server
    assert call(base, "/api/command", {"action": "open"})[1]["ok"]
    s = wait_state(base, "open")
    assert s["mode"] == "manual"
    assert call(base, "/api/command", {"action": "close"})[1]["ok"]
    wait_state(base, "closed")
    assert call(base, "/api/mode", {"mode": "auto"})[1]["ok"]


def test_flood_signal_opens(server):
    base, _ = server
    call(base, "/api/flood", {"active": True})
    assert wait_state(base, "open")["flood_risk"] is True


def test_bad_inputs_are_rejected(server):
    base, _ = server
    assert call(base, "/api/frame", raw=b"not an image")[0] == 400
    assert call(base, "/api/command", {"action": "explode"})[1]["ok"] is False
    assert call(base, "/api/frame.jpg")[0] == 404
    assert call(base, "/nope")[0] == 404
    assert call(base, "/api/replay", {"playing": True})[0] == 400


def test_original_and_analysis_images_match_for_side_by_side(server, rng):
    base, _ = server
    assert call(base, "/api/frame_original.jpg")[0] == 404  # nothing analysed yet
    big = synthetic.make_image("partial", rng).resize((2400, 1800))  # larger than the display cap
    buf = io.BytesIO()
    big.save(buf, format="JPEG")
    assert call(base, "/api/frame", raw=buf.getvalue())[0] == 200
    orig = Image.open(io.BytesIO(call(base, "/api/frame_original.jpg")[1]))
    ana = Image.open(io.BytesIO(call(base, "/api/frame.jpg")[1]))
    assert orig.size == ana.size == (1600, 1200)  # same geometry, capped at 1600 px on the long side
    s = call(base, "/api/status")[1]
    assert s["frame"]["size"] == [2400, 1800] and s["frame"]["analysis_s"] >= 0
