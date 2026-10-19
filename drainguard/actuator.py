"""Actuator interfaces. The controller only needs: send(action, on_done)."""

import random
import threading
import time
from typing import Callable, Optional

from .config import ActuatorConfig

DoneCallback = Callable[[bool, str], None]  # (ok, resulting_state: "open"|"closed")


class Actuator:
    name = "base"
    simulated = True

    def send(self, action: str, on_done: DoneCallback) -> None:
        raise NotImplementedError


class SimulatedActuator(Actuator):
    """Software stand-in for the team's hardware. Results from it are NOT hardware results."""

    name = "simulated"
    simulated = True

    def __init__(self, config: Optional[ActuatorConfig] = None, rng: Optional[random.Random] = None):
        self.cfg = config or ActuatorConfig()
        self.rng = rng or random.Random()

    def send(self, action: str, on_done: DoneCallback) -> None:
        delay = max(0.0, self.cfg.delay_s + self.rng.uniform(-self.cfg.jitter_s, self.cfg.jitter_s))
        if self.rng.random() < self.cfg.fail_rate:
            return  # silent failure: controller should time out into "fault"
        state = "open" if action == "open" else "closed"
        t = threading.Timer(delay, on_done, args=(True, state))
        t.daemon = True
        t.start()


class SerialActuator(Actuator):
    """Talks to a microcontroller (e.g. Arduino) over a serial port.

    Protocol (one ASCII line each way):
      host -> device:  "OPEN\\n" | "CLOSE\\n"
      device -> host:  "OPENED\\n" | "CLOSED\\n" once the cover has finished moving
    Not tested against real hardware yet - adjust to whatever the hardware teammates implement.
    """

    name = "serial"
    simulated = False

    def __init__(self, port: str, baud: int = 115200, timeout: float = 15.0):
        import serial  # pip install pyserial

        self.ser = serial.Serial(port, baud, timeout=timeout)
        self.lock = threading.Lock()

    def send(self, action: str, on_done: DoneCallback) -> None:
        def run():
            with self.lock:
                self.ser.reset_input_buffer()
                self.ser.write((action.upper() + "\n").encode())
                line = self.ser.readline().decode(errors="ignore").strip().upper()
            expected = "OPENED" if action == "open" else "CLOSED"
            if line == expected:
                on_done(True, "open" if action == "open" else "closed")

        threading.Thread(target=run, daemon=True).start()


class HttpActuator(Actuator):
    """ESP32 motor controller driven by URL, e.g. http://172.20.10.2/timed?dir=forward&ms=3000.

    Open = run the motor `open_dir` for `open_ms`, close = the opposite way for `close_ms`. Timed moves stop
    by themselves on the ESP32. There is NO position sensor, so "open" means "the motor ran for that long",
    not "a sensor saw the cover open". The cover must start physically closed.
    """

    simulated = False
    MAX_MS = 15000  # hard cap: never ask for a longer run than this

    def __init__(self, base_url: str, open_ms: int = 3000, close_ms: int = 3000, open_dir: str = "forward",
                 http_timeout: float = 5.0, margin_s: float = 0.5):
        if open_dir not in ("forward", "reverse"):
            raise ValueError("open_dir must be 'forward' or 'reverse'")
        for ms in (open_ms, close_ms):
            if not 0 < ms <= self.MAX_MS:
                raise ValueError(f"motor run time must be between 1 and {self.MAX_MS} ms")
        self.base = base_url.rstrip("/")
        if not self.base.startswith("http"):
            self.base = "http://" + self.base
        self.open_ms, self.close_ms = int(open_ms), int(close_ms)
        self.open_dir = open_dir
        self.close_dir = "reverse" if open_dir == "forward" else "forward"
        self.timeout, self.margin = http_timeout, margin_s
        self.name = "motor over Wi-Fi (timed, no position sensor)"
        self.lock = threading.Lock()

    def _get(self, path: str) -> None:
        import urllib.request

        with urllib.request.urlopen(self.base + path, timeout=self.timeout) as r:
            if not 200 <= r.status < 300:
                raise OSError(f"motor answered HTTP {r.status}")

    def stop(self) -> None:
        try:
            self._get("/stop")
        except Exception:
            pass

    def send(self, action: str, on_done: DoneCallback) -> None:
        opening = action == "open"
        ms = self.open_ms if opening else self.close_ms
        direction = self.open_dir if opening else self.close_dir

        def run():
            with self.lock:
                try:
                    self._get(f"/timed?dir={direction}&ms={ms}")
                except Exception:
                    self.stop()
                    on_done(False, "open" if opening else "closed")
                    return
                time.sleep(ms / 1000.0 + self.margin)
            on_done(True, "open" if opening else "closed")

        threading.Thread(target=run, daemon=True).start()
