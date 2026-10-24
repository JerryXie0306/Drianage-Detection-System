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
    assert [g for g in got if g.startswith("/timed")] == ["/timed?dir=forward&ms=100", "/timed?dir=reverse&ms=150"]
    assert res == [(True, "open"), (True, "closed")]


def test_open_direction_can_be_reversed(motor):
    url, got = motor
    act = HttpActuator(url, open_ms=50, close_ms=50, open_dir="reverse", margin_s=0.01)
    done = []
    act.send("open", lambda ok, st: done.append(ok))
    assert wait(lambda: done)
    assert [g for g in got if g.startswith("/timed")] == ["/timed?dir=reverse&ms=50"]


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


@pytest.mark.parametrize("kwargs", [{"open_ms": 0}, {"open_ms": 30001}, {"close_ms": -5}, {"open_dir": "up"}])
def test_unsafe_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        HttpActuator("http://x", **kwargs)


def test_controller_drives_the_motor_end_to_end_and_faults_when_it_is_unreachable(motor):
    url, got = motor
    ctrl = CoverController(HttpActuator(url, open_ms=80, close_ms=80, margin_s=0.01), ControllerConfig(confirm_timeout=5))
    assert ctrl.command("open")["ok"]
    assert wait(lambda: ctrl.status()["cover_state"] == "open")
    assert [g for g in got if g.startswith("/timed")] == ["/timed?dir=forward&ms=80"]

    bad = CoverController(HttpActuator("http://127.0.0.1:9", open_ms=50, http_timeout=0.5), ControllerConfig())
    bad.command("open")
    assert wait(lambda: bad.status()["cover_state"] == "fault")


def test_board_status_is_recorded_after_a_move(motor):
    url, got = motor
    act = HttpActuator(url, open_ms=50, margin_s=0.01)
    done = []
    act.send("open", lambda ok, st: done.append(ok))
    assert wait(lambda: done)
    assert got == ["/timed?dir=forward&ms=50", "/status"] and act.last_status == "ok"


def test_start_mode_manual_means_blocked_pictures_do_not_move_the_motor(motor):
    from drainguard.detector import Detection

    url, got = motor
    ctrl = CoverController(HttpActuator(url, open_ms=50, margin_s=0.01), ControllerConfig(open_frames=1))
    ctrl.set_mode("manual")
    for _ in range(3):
        ctrl.on_detection(Detection(label="blocked", confidence=0.99, coverage=0.7))
    time.sleep(0.2)
    assert got == [] and ctrl.status()["cover_state"] == "closed"


def test_a_board_that_answers_only_after_the_move_is_not_mistaken_for_a_failure():
    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/timed"):
                time.sleep(0.6)  # answers when the move is done, longer than http_timeout
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"OK")

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        act = HttpActuator(f"http://127.0.0.1:{srv.server_address[1]}", open_ms=700, http_timeout=0.3, margin_s=0.01)
        res = []
        act.send("open", lambda ok, st: res.append(ok))
        assert wait(lambda: res, timeout=5)
        assert res == [True]
    finally:
        srv.shutdown()


def _board_ctrl(url, **cfg):
    return CoverController(HttpActuator(url, open_ms=50, close_ms=50, margin_s=0.01), ControllerConfig(**cfg))


def test_water_level_auto_needs_a_closed_cover_and_calls_auto(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.assume_state("open")
    res = c.set_mode("level")
    assert res["ok"] is False and "CLOSED" in res["msg"] and "/auto" not in got  # never enable it on an open cover
    c.assume_state("closed")
    assert c.set_mode("level")["ok"] and c.status()["mode"] == "level" and "/auto" in got
    assert c.status()["actuator"]["board_auto"] is True


def test_board_status_is_mirrored_in_water_level_mode(motor):
    url, _ = motor
    c = _board_ctrl(url)
    c.set_mode("level")
    c.observe_board("FORWARD | AUTO | OPENING")
    assert c.status()["cover_state"] == "opening"
    c.observe_board("STOPPED | AUTO | OPEN")
    assert c.status()["cover_state"] == "open"
    c.observe_board("REVERSE | AUTO | CLOSING")
    c.observe_board("STOPPED | AUTO | CLOSED")
    assert c.status()["cover_state"] == "closed" and c.status()["mode"] == "level"
    c.observe_board("STOPPED | MANUAL")  # someone used the board's own page
    assert c.status()["mode"] == "manual"
    c.observe_board("STOPPED | AUTO | CLOSED")  # board was put into AUTO from its own page
    assert c.status()["mode"] == "level"


def test_camera_detections_are_ignored_in_water_level_mode(motor):
    from drainguard.detector import Detection

    url, got = motor
    c = _board_ctrl(url, open_frames=1)
    c.set_mode("level")
    for _ in range(3):
        c.on_detection(Detection(label="blocked", confidence=0.99, coverage=0.7))
    time.sleep(0.2)
    assert [g for g in got if g.startswith("/timed")] == []  # the board decides, not the camera


def test_operator_button_cancels_water_level_auto_and_manual_sends_stop(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.set_mode("level")
    assert c.command("open")["ok"] and c.status()["mode"] == "manual"
    assert wait(lambda: any(g.startswith("/timed") for g in got))
    c2 = _board_ctrl(url)
    c2.set_mode("level")
    n = got.count("/stop")
    c2.set_mode("manual")
    assert got.count("/stop") == n + 1 and c2.status()["mode"] == "manual"  # leaving AUTO tells the board to stop


def test_stop_while_moving_sends_stop_and_the_old_move_never_reports_done(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.actuator.open_ms = 2000  # a long move
    assert c.command("open")["ok"]
    assert wait(lambda: any(g.startswith("/timed") for g in got))
    t0 = time.time()
    assert c.emergency_stop()["ok"]
    assert "/stop" in got and c.status()["cover_state"] == "stopped"
    time.sleep(2.3)  # the old move's timer would have fired by now
    assert c.status()["cover_state"] == "stopped"  # ...but it must not overwrite the state with "open"
    assert time.time() - t0 > 2


def test_after_stop_a_new_command_does_not_wait_for_the_old_move(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.actuator.open_ms = 5000
    c.command("open")
    assert wait(lambda: any(g.startswith("/timed") for g in got))
    c.emergency_stop()
    c.actuator.close_ms = 50
    t0 = time.time()
    assert c.command("close")["ok"]  # allowed from "stopped"
    assert wait(lambda: c.status()["cover_state"] == "closed", timeout=2)
    assert time.time() - t0 < 2  # not 5 s: the stopped move released the motor straight away
    assert [g for g in got if g.startswith("/timed")][-1].startswith("/timed?dir=reverse")


def test_stop_before_a_queued_move_starts_cancels_it(motor):
    url, got = motor
    act = HttpActuator(url, open_ms=300, close_ms=300, margin_s=0.01)
    done = []
    act.send("open", lambda ok, st: done.append(("open", ok)))  # holds the motor lock while it waits
    assert wait(lambda: any(g.startswith("/timed") for g in got))
    act.send("close", lambda ok, st: done.append(("close", ok)))  # queued behind it
    act.stop()
    time.sleep(0.8)
    assert done == [] and len([g for g in got if g.startswith("/timed")]) == 1  # the queued close never ran


def test_stop_works_in_every_mode_and_leaves_auto(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.set_mode("level")
    n = got.count("/stop")
    assert c.emergency_stop()["ok"]
    assert got.count("/stop") == n + 1 and c.status()["mode"] == "manual"
    c.set_mode("auto")
    c.emergency_stop()
    assert c.status()["mode"] == "manual"  # camera AUTO must not re-open it straight after a STOP


def test_stop_when_idle_changes_nothing_but_still_sends_stop(motor):
    url, got = motor
    c = _board_ctrl(url)
    c.assume_state("closed")
    c.emergency_stop()
    assert "/stop" in got and c.status()["cover_state"] == "closed"


def test_stopped_position_is_remembered_and_restored(tmp_path):
    from drainguard.actuator import Actuator
    from drainguard.server import attach_position_memory

    class Idle(Actuator):
        simulated = False

        def send(self, action, on_done):
            pass

    f = tmp_path / "pos.json"
    a = CoverController(Idle())
    attach_position_memory(a, f)
    a.command("open")
    a.emergency_stop()
    b = CoverController(Idle())
    attach_position_memory(b, f)
    assert b.status()["cover_state"] == "stopped"  # a restart must not pretend the position is known
