"""Decision logic + cover state machine: detect -> decide -> command -> open (+ feedback)."""

import threading
import time
from collections import deque
from typing import Callable, Optional

from .actuator import Actuator
from .config import ControllerConfig
from .detector import Detection


class CoverController:
    def __init__(self, actuator: Actuator, config: Optional[ControllerConfig] = None,
                 clock: Callable[[], float] = time.monotonic):
        self.actuator = actuator
        self.cfg = config or ControllerConfig()
        self.clock = clock
        self.lock = threading.RLock()

        self.state = "closed"  # closed | opening | open | closing | fault
        self.mode = "auto"  # auto | manual
        self.flood_risk = False
        self.last_detection: Optional[Detection] = None
        self.last_frame_wall: Optional[float] = None
        self._prev_label: Optional[str] = None

        self._cmd = None  # {"id", "action", "t", "source"}
        self._cmd_id = 0
        self._open_streak = 0
        self._clear_streak = 0
        self._opened_at: Optional[float] = None
        self.last_delay: Optional[float] = None
        self.delays = []  # command -> confirmed seconds (opens and closes)
        self.events = deque(maxlen=200)

    # ---------- helpers ----------
    def log(self, msg: str):
        self.events.append({"t": time.time(), "msg": msg})

    def _is_blocked(self, det: Detection) -> bool:
        if det.confidence < self.cfg.confidence_threshold:
            return False
        return det.label == "blocked" or (self.cfg.open_on_partial and det.label == "partial")

    def _send(self, action: str, reason: str):
        self._cmd_id += 1
        cid = self._cmd_id
        self._cmd = {"id": cid, "action": action, "t": self.clock(), "source": reason}
        self.state = "opening" if action == "open" else "closing"
        self.log(f"{action.upper()} sent ({reason})")
        self.actuator.send(action, lambda ok, st, cid=cid: self._on_done(cid, ok, st))

    def _on_done(self, cid: int, ok: bool, state: str):
        with self.lock:
            if not self._cmd or self._cmd["id"] != cid:
                return
            delay = self.clock() - self._cmd["t"]
            if ok:
                self.state = state
                self.last_delay = delay
                self.delays.append(delay)
                if state == "open":
                    self._opened_at = self.clock()
                    self._clear_streak = 0
                self.log(f"Cover {state.upper()} confirmed ({delay:.1f} s after command)")
            else:
                self.state = "fault"
                self.log("Actuator reported failure -> FAULT")
            self._cmd = None

    # ---------- public API ----------
    def poll(self):
        """Call periodically: detects actuator timeouts."""
        with self.lock:
            if self._cmd and self.clock() - self._cmd["t"] > self.cfg.confirm_timeout:
                self.log(f"No confirmation for {self._cmd['action'].upper()} within "
                         f"{self.cfg.confirm_timeout:.0f} s -> FAULT")
                self.state = "fault"
                self._cmd = None

    def command(self, action: str, source: str = "operator") -> dict:
        action = action.lower()
        if action not in ("open", "close"):
            return {"ok": False, "msg": "action must be open or close"}
        with self.lock:
            if self.state in ("opening", "closing"):
                return {"ok": False, "msg": "cover is moving"}
            if (action == "open" and self.state == "open") or (action == "close" and self.state == "closed"):
                return {"ok": False, "msg": f"already {self.state}"}
            if source == "operator" and self.mode == "auto":
                self.mode = "manual"
                self.log("Operator command -> MANUAL mode")
            self._send(action, source)
            return {"ok": True, "msg": f"{action} sent"}

    def set_mode(self, mode: str) -> dict:
        if mode not in ("auto", "manual"):
            return {"ok": False, "msg": "mode must be auto or manual"}
        with self.lock:
            if mode != self.mode:
                self.mode = mode
                self._open_streak = self._clear_streak = 0
                self.log(f"Mode set to {mode.upper()}")
            if mode == "auto":
                self._auto_step()
            return {"ok": True, "msg": f"mode {mode}"}

    def set_flood_risk(self, active: bool) -> dict:
        with self.lock:
            if active != self.flood_risk:
                self.flood_risk = bool(active)
                self.log("Flood risk signal " + ("ON" if active else "OFF"))
            self._auto_step()
            return {"ok": True, "msg": f"flood_risk {self.flood_risk}"}

    def on_detection(self, det: Detection):
        with self.lock:
            self.last_detection = det
            self.last_frame_wall = time.time()
            if det.label != self._prev_label:
                self.log(f"Frame: {det.label.upper()} {det.confidence * 100:.0f}% "
                         f"(debris {det.coverage * 100:.0f}%)")
                self._prev_label = det.label
            self._auto_step()

    def _auto_step(self):
        if self.mode != "auto":
            return
        det = self.last_detection
        blocked = bool(det and self._is_blocked(det))
        want_open = self.flood_risk or blocked

        if self.state == "closed":
            if self.flood_risk:
                self._send("open", "auto: flood risk")
                return
            self._open_streak = self._open_streak + 1 if blocked else 0
            if self._open_streak >= self.cfg.open_frames:
                self._send("open", f"auto: {det.label.upper()} {det.confidence * 100:.0f}%")
                self._open_streak = 0
        elif self.state == "open":
            self._clear_streak = 0 if want_open else self._clear_streak + 1
            held = self.clock() - (self._opened_at or 0)
            if self._clear_streak >= self.cfg.close_frames and held >= self.cfg.min_open_seconds:
                self._send("close", "auto: debris cleared, no flood risk")
                self._clear_streak = 0

    def status(self) -> dict:
        self.poll()
        with self.lock:
            det = self.last_detection
            age = None if self.last_frame_wall is None else time.time() - self.last_frame_wall
            return {
                "cover_state": self.state,
                "mode": self.mode,
                "flood_risk": self.flood_risk,
                "threshold": self.cfg.confidence_threshold,
                "detection": None if det is None else {
                    "label": det.label, "confidence": det.confidence, "coverage": det.coverage,
                    "bbox": det.bbox, "backend": det.backend, "extra": det.extra,
                },
                "last_frame_age_s": age,
                "camera_online": age is not None and age < 15,
                "last_delay_s": self.last_delay,
                "pending": None if not self._cmd else self._cmd["action"],
                "actuator": {"name": self.actuator.name, "simulated": self.actuator.simulated},
                "events": list(self.events)[-12:],
            }
