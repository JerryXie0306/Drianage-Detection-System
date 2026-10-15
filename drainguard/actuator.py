"""Actuator interfaces. The controller only needs: send(action, on_done)."""

import random
import threading
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
