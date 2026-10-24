import pytest

from drainguard.actuator import Actuator
from drainguard.config import ControllerConfig
from drainguard.controller import CoverController
from drainguard.detector import Detection


class FakeActuator(Actuator):
    """Records commands; the test decides when (and whether) the cover 'finishes moving'."""

    name = "fake"

    def __init__(self):
        self.sent = []
        self.pending = None

    def send(self, action, on_done):
        self.sent.append(action)
        self.pending = (action, on_done)

    def complete(self, ok=True):
        action, cb = self.pending
        self.pending = None
        cb(ok, "open" if action == "open" else "closed")


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


BLOCKED = Detection("blocked", 0.92, 0.5)
CLEAR = Detection("clear", 0.95, 0.0)


@pytest.fixture
def setup():
    act, clock = FakeActuator(), Clock()
    ctrl = CoverController(act, ControllerConfig(min_open_seconds=30, close_frames=3), clock=clock)
    return ctrl, act, clock


def test_auto_open_needs_debounced_blocked_frames(setup):
    ctrl, act, _ = setup
    ctrl.on_detection(BLOCKED)
    ctrl.on_detection(BLOCKED)
    assert act.sent == []  # 2 of 3 frames
    ctrl.on_detection(BLOCKED)
    assert act.sent == ["open"] and ctrl.state == "opening"


def test_isolated_blocked_frame_resets_streak(setup):
    ctrl, act, _ = setup
    for det in (BLOCKED, BLOCKED, CLEAR, BLOCKED, BLOCKED):
        ctrl.on_detection(det)
    assert act.sent == []


def test_low_confidence_does_not_trigger(setup):
    ctrl, act, _ = setup
    for _ in range(5):
        ctrl.on_detection(Detection("blocked", 0.60, 0.5))
    assert act.sent == []


def test_partial_does_not_open_by_default(setup):
    ctrl, act, _ = setup
    for _ in range(5):
        ctrl.on_detection(Detection("partial", 0.95, 0.2))
    assert act.sent == []


def test_manual_mode_ignores_detections(setup):
    ctrl, act, _ = setup
    ctrl.set_mode("manual")
    for _ in range(5):
        ctrl.on_detection(BLOCKED)
    assert act.sent == []


def test_operator_command_switches_to_manual(setup):
    ctrl, act, _ = setup
    assert ctrl.command("open")["ok"]
    assert ctrl.mode == "manual" and act.sent == ["open"]


def test_flood_risk_opens_immediately_without_detection(setup):
    ctrl, act, _ = setup
    ctrl.set_flood_risk(True)
    assert act.sent == ["open"]


def test_delay_is_recorded_on_confirmation(setup):
    ctrl, act, clock = setup
    ctrl.command("open")
    clock.t += 3.4
    act.complete()
    assert ctrl.state == "open" and ctrl.last_delay == pytest.approx(3.4)


def test_no_second_command_while_moving(setup):
    ctrl, act, _ = setup
    ctrl.command("open")
    assert not ctrl.command("close")["ok"]
    assert act.sent == ["open"]


def test_already_open_is_rejected(setup):
    ctrl, act, _ = setup
    ctrl.command("open")
    act.complete()
    assert not ctrl.command("open")["ok"]


def test_auto_close_after_clear_frames_and_min_open_time(setup):
    ctrl, act, clock = setup
    ctrl.command("open")
    act.complete()
    ctrl.set_mode("auto")
    for _ in range(3):
        ctrl.on_detection(CLEAR)
    assert act.sent == ["open"]  # still within min_open_seconds
    clock.t += 31
    ctrl.on_detection(CLEAR)
    assert act.sent == ["open", "close"]


def test_stays_open_while_flood_risk(setup):
    ctrl, act, clock = setup
    ctrl.set_flood_risk(True)
    act.complete()
    clock.t += 100
    for _ in range(10):
        ctrl.on_detection(CLEAR)
    assert act.sent == ["open"]


def test_timeout_leads_to_fault_and_allows_retry(setup):
    ctrl, act, clock = setup
    ctrl.command("open")
    clock.t += 11
    ctrl.poll()
    assert ctrl.state == "fault"
    assert ctrl.command("open")["ok"]  # operator can retry from fault


def test_actuator_failure_leads_to_fault(setup):
    ctrl, act, _ = setup
    ctrl.command("open")
    act.complete(ok=False)
    assert ctrl.state == "fault"


def test_late_confirmation_of_old_command_is_ignored(setup):
    ctrl, act, clock = setup
    ctrl.command("open")
    stale_cb = act.pending[1]
    clock.t += 11
    ctrl.poll()  # -> fault
    ctrl.command("open")  # new command id
    stale_cb(True, "open")  # confirmation of the old one arrives
    assert ctrl.state == "opening"


def test_invalid_inputs(setup):
    ctrl, _, _ = setup
    assert not ctrl.command("explode")["ok"]
    assert not ctrl.set_mode("turbo")["ok"]


def test_assume_state_moves_nothing_and_unblocks_the_right_command():
    from drainguard.actuator import SimulatedActuator
    from drainguard.config import ActuatorConfig
    from drainguard.controller import CoverController

    sent = []

    class Spy(SimulatedActuator):
        def send(self, action, on_done):
            sent.append(action)
            super().send(action, on_done)

    c = CoverController(Spy(ActuatorConfig(delay_s=0.01, jitter_s=0.0)))
    assert c.command("close")["msg"] == "already closed"  # the app believes closed, the cover is really open
    assert c.assume_state("open")["ok"] and c.status()["cover_state"] == "open" and sent == []
    assert c.command("close")["ok"] and sent == ["close"]
    assert c.assume_state("bogus")["ok"] is False


def test_assume_state_is_refused_while_moving(monkeypatch):
    from drainguard.actuator import Actuator
    from drainguard.controller import CoverController

    class Never(Actuator):
        def send(self, action, on_done):
            pass

    c = CoverController(Never())
    c.command("open")
    assert c.assume_state("closed") == {"ok": False, "msg": "cover is moving"}


def test_position_is_remembered_across_restarts(tmp_path):
    from drainguard.actuator import SimulatedActuator
    from drainguard.config import ActuatorConfig
    from drainguard.controller import CoverController
    from drainguard.server import attach_position_memory

    f = tmp_path / "pos.json"
    a = CoverController(SimulatedActuator(ActuatorConfig(delay_s=0.01, jitter_s=0.0)))
    attach_position_memory(a, f)
    a.assume_state("open")
    b = CoverController(SimulatedActuator(ActuatorConfig(delay_s=0.01, jitter_s=0.0)))
    attach_position_memory(b, f)  # a fresh process
    assert b.status()["cover_state"] == "open"
    assert any("Restored" in e["msg"] for e in b.status()["events"])
