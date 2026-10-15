"""Measure command -> cover-confirmed delay  (Test 2: "App command -> cover open time (bench)").

    python -m drainguard.latency --cycles 20                    # SIMULATED actuator (software check only)
    python -m drainguard.latency --cycles 20 --serial /dev/tty.usbmodemXXXX   # real hardware

With the simulated actuator the delay is just the configured value plus jitter, so it validates the
software path (state machine, feedback) but is NOT a hardware measurement.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .actuator import SerialActuator, SimulatedActuator
from .config import ActuatorConfig, ControllerConfig
from .controller import CoverController
from .detector import Detection


def wait_for(ctrl: CoverController, state: str, timeout: float) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        ctrl.poll()
        if ctrl.state == state:
            return True
        if ctrl.state == "fault":
            return False
        time.sleep(0.01)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cycles", type=int, default=20)
    ap.add_argument("--serial")
    ap.add_argument("--sim-delay", type=float, default=0.5, help="simulated delay (s); kept short for quick runs")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    simulated = a.serial is None
    act = SimulatedActuator(ActuatorConfig(delay_s=a.sim_delay, jitter_s=a.sim_delay * 0.2)) if simulated \
        else SerialActuator(a.serial)
    ctrl = CoverController(act, ControllerConfig(confirm_timeout=20.0, min_open_seconds=0))

    opens, closes, decision_ms, failures = [], [], [], 0
    for _ in range(a.cycles):
        # software decision latency: blocked frame in -> OPEN command out (auto mode)
        ctrl.cfg.open_frames = 1
        ctrl.mode = "auto"
        t0 = time.perf_counter()
        ctrl.on_detection(Detection("blocked", 0.95, 0.5))
        decision_ms.append((time.perf_counter() - t0) * 1000)
        if wait_for(ctrl, "open", 30):
            opens.append(ctrl.last_delay)
        else:
            failures += 1
            break
        ctrl.command("close", "bench")
        if wait_for(ctrl, "closed", 30):
            closes.append(ctrl.last_delay)
        else:
            failures += 1
            break

    def stats(name, xs, unit="s", scale=1.0):
        if not xs:
            return f"{name}: no data"
        v = np.array(xs) * scale
        return (f"{name}: n={len(v)} mean={v.mean():.3f}{unit} median={np.median(v):.3f}{unit} "
                f"p95={np.percentile(v, 95):.3f}{unit} max={v.max():.3f}{unit}")

    tag = "SIMULATED actuator (software path only, not a hardware result)" if simulated else f"hardware on {a.serial}"
    print(f"[{tag}]  cycles requested={a.cycles}  failures={failures}")
    print(stats("decision (frame -> command sent)", decision_ms, "ms"))
    print(stats("command -> OPEN confirmed", opens))
    print(stats("command -> CLOSED confirmed", closes))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"latency_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cycle", "open_s", "close_s", "decision_ms", "simulated"])
        for i in range(max(len(opens), len(closes))):
            w.writerow([i + 1, opens[i] if i < len(opens) else "", closes[i] if i < len(closes) else "",
                        round(decision_ms[i], 3) if i < len(decision_ms) else "", simulated])
    print(f"saved {path}")


if __name__ == "__main__":
    main()
