import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from drainguard.actuator import HttpActuator
from drainguard.config import ControllerConfig
from drainguard.controller import CoverController


@pytest.fixture
def motor():
    got = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            got.append(self.path)
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"ok")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", got
    srv.shutdown()


def wait(cond, timeout=3):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_open_and_close_use_timed_urls_in_opposite_directions(motor):
    url, got = motor
    act = HttpActuator(url, open_ms=100, close_ms=150, margin_s=0.02)
    res = []
    act.send("open", lambda ok, st: res.append((ok, st)))
    assert wait(lambda: res)
    act.send("close", lambda ok, st: res.append((ok, st)))
    assert wait(lambda: len(res) == 2)
    assert got == ["/timed?dir=forward&ms=100", "/timed?dir=reverse&ms=150"]
    assert res == [(True, "open"), (True, "closed")]


def test_open_direction_can_be_reversed(motor):
    url, got = motor
    act = HttpActuator(url, open_ms=50, close_ms=50, open_dir="reverse", margin_s=0.01)
    done = []
    act.send("open", lambda ok, st: done.append(ok))
    assert wait(lambda: done)
    assert got == ["/timed?dir=reverse&ms=50"]


def test_done_is_reported_only_after_the_motor_has_had_time_to_finish(motor):
    url, _ = motor
    act = HttpActuator(url, open_ms=300, margin_s=0.0)
    t0, done = time.time(), []
    act.send("open", lambda ok, st: done.append(time.time() - t0))
    assert wait(lambda: done)
    assert done[0] >= 0.29


def test_unreachable_motor_reports_failure_not_success():
    act = HttpActuator("http://127.0.0.1:9", open_ms=50, http_timeout=0.5)
    res = []
    act.send("open", lambda ok, st: res.append(ok))
    assert wait(lambda: res)
    assert res == [False]


@pytest.mark.parametrize("kwargs", [{"open_ms": 0}, {"open_ms": 15001}, {"close_ms": -5}, {"open_dir": "up"}])
def test_unsafe_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HttpActuator("http://x", **kwargs)


def test_controller_drives_the_motor_end_to_end_and_faults_when_it_is_unreachable(motor):
    url, got = motor
    ctrl = CoverController(HttpActuator(url, open_ms=80, close_ms=80, margin_s=0.01), ControllerConfig(confirm_timeout=5))
    assert ctrl.command("open")["ok"]
    assert wait(lambda: ctrl.status()["cover_state"] == "open")
    assert got == ["/timed?dir=forward&ms=80"]

    bad = CoverController(HttpActuator("http://127.0.0.1:9", open_ms=50, http_timeout=0.5), ControllerConfig())
    bad.command("open")
    assert wait(lambda: bad.status()["cover_state"] == "fault")
