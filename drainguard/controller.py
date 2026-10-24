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

        self.state = "closed"  # closed | opening | open | closing | stopped (position unknown) | fault
        self.mode = "auto"  # auto (camera decides) | manual | level (the motor board's own water-level AUTO)
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
        self.board_raw: Optional[str] = None  # last text the motor board reported
        self.on_position: Optional[Callable[[str], None]] = None  # called with "open"/"closed" whenever the position is known

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
                if self.on_position:
                    self.on_position(state)
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
            if source == "operator" and self.mode in ("auto", "level"):
                was_level = self.mode == "level"
                self.mode = "manual"
                self.log("Operator command -> MANUAL mode" + (" (the board's water-level AUTO is cancelled)" if was_level else ""))
            self._send(action, source)
            return {"ok": True, "msg": f"{action} sent"}

    def emergency_stop(self, source: str = "operator") -> dict:
        """STOP now, in any mode and any state. Cancels whatever is pending; if it was moving the position is unknown."""
        stop = getattr(self.actuator, "stop", None)
        if stop:
            stop()  # the network call goes first and outside the lock, so a slow board never delays the STOP
        with self.lock:
            moving = self.state in ("opening", "closing")
            self._cmd = None
            self._open_streak = self._clear_streak = 0
            was = self.mode
            if was in ("auto", "level"):
                self.mode = "manual"
            if moving:
                self.state = "stopped"
                if self.on_position:
                    self.on_position("stopped")
            self.log(f"STOP pressed ({source})" + (" -> MANUAL mode" if was in ("auto", "level") else "")
                     + (". Position unknown: use 'It is OPEN / CLOSED'." if moving else ""))
            return {"ok": True, "msg": "stopped"}

    def assume_state(self, state: str, source: str = "operator") -> dict:
        """Tell the controller where the cover REALLY is. Nothing moves. For hardware with no position sensor."""
        if state not in ("open", "closed"):
            return {"ok": False, "msg": "state must be open or closed"}
        with self.lock:
            if self.state in ("opening", "closing"):
                return {"ok": False, "msg": "cover is moving"}
            self.state = state
            self._cmd = None
            self._open_streak = self._clear_streak = 0
            if state == "open":
                self._opened_at = self.clock()
            self.log(f"Cover position set to {state.upper()} by {source} (no movement)")
            if self.on_position:
                self.on_position(state)
            return {"ok": True, "msg": f"cover marked {state}"}

    def set_mode(self, mode: str) -> dict:
        if mode not in ("auto", "manual", "level"):
            return {"ok": False, "msg": "mode must be auto, manual or level"}
        if mode == "level":
            return self._enable_level()
        leaving_level = False
        with self.lock:
            if mode != self.mode:
                leaving_level = self.mode == "level"
                self.mode = mode
                self._open_streak = self._clear_streak = 0
                self.log(f"Mode set to {mode.upper()}")
            if mode == "auto":
                self._auto_step()
        if leaving_level and hasattr(self.actuator, "stop"):
            self.actuator.stop()  # the board only leaves its own AUTO when told to stop
            self.log("Sent STOP to the motor board to leave water-level AUTO. If the cover was moving, check its position "
                     "and use 'It is OPEN / CLOSED'.")
        return {"ok": True, "msg": f"mode {mode}"}

    def _enable_level(self) -> dict:
        if not getattr(self.actuator, "has_board_auto", False):
            return {"ok": False, "msg": "this actuator has no water-level AUTO"}
        with self.lock:
            if self.mode == "level":
                return {"ok": True, "msg": "already in water-level AUTO"}
            if self.state != "closed":
                return {"ok": False, "msg": "the cover must be CLOSED before water-level AUTO is enabled "
                                            "(if it is closed but shows otherwise, use 'It is CLOSED')"}
        ok, msg = self.actuator.enable_auto()
        with self.lock:
            if ok:
                self.mode = "level"
                self._open_streak = self._clear_streak = 0
                self.log("Mode set to WATER-LEVEL AUTO: the motor board opens and closes the cover by itself")
            else:
                self.log(msg)
            return {"ok": ok, "msg": msg}

    def observe_board(self, raw: Optional[str]):
        """Mirror what the motor board says about itself: "MOTION | MODE | POSITION" (position only while in AUTO)."""
        with self.lock:
            self.board_raw = raw
            if not raw:
                return
            toks = [t.strip().upper() for t in raw.split("|")]
            board_mode = toks[1] if len(toks) > 1 else ""
            pos = toks[2] if len(toks) > 2 else ""
            if board_mode == "AUTO" and self.mode != "level":
                self.mode = "level"
                self.log("The motor board is in water-level AUTO")
            elif board_mode == "MANUAL" and self.mode == "level":
                self.mode = "manual"
                self.log("The motor board left water-level AUTO")
            if self.mode != "level" or self._cmd:
                return
            new = ("opening" if pos.startswith("OPENING") else "closing" if pos.startswith("CLOSING")
                   else "open" if pos.startswith("OPEN") else "closed" if pos.startswith("CLOS") else None)
            if new and new != self.state:
                self.state = new
                self.log(f"Motor board reports the cover is {new.upper()}")
                if new in ("open", "closed") and self.on_position:
                    self.on_position(new)

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
                "camera_online": age is not None and age < self.cfg.camera_online_s,
                "last_delay_s": self.last_delay,
                "pending": None if not self._cmd else self._cmd["action"],
                "actuator": {"name": self.actuator.name, "simulated": self.actuator.simulated,
                             "board_status": self.board_raw or getattr(self.actuator, "last_status", None),
                             "board_auto": bool(getattr(self.actuator, "has_board_auto", False))},
                "events": list(self.events)[-12:],
            }
