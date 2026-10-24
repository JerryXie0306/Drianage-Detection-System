"""Take labelled test photos from the camera in camera.txt.

    python collect_samples.py clear 5      # saves 5 photos to data/test/clear/
    python collect_samples.py partial 5
    python collect_samples.py blocked 5

Put the debris in place first, then run it. Then evaluate with:
    python -m drainguard.evaluate --data data/test --backend segment --regions regions/drain.json
"""

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def camera_source() -> str:
    p = ROOT / "camera.txt"
    for line in p.read_text().splitlines() if p.exists() else []:
        line = line.strip()
        if line and not line.startswith("#"):
            return line
    raise SystemExit("no camera in camera.txt")


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("clear", "partial", "blocked"):
        raise SystemExit(__doc__)
    label = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    src = camera_source()
    if not src.startswith("snapshot:"):
        raise SystemExit("this tool needs a 'snapshot:http://IP/capture' camera in camera.txt")
    url = src[len("snapshot:"):]
    out = ROOT / "data" / "test" / label
    out.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        with urllib.request.urlopen(url, timeout=12) as r:
            data = r.read()
        name = out / f"{label}_{time.strftime('%H%M%S')}_{i + 1}.jpg"
        name.write_bytes(data)
        print("saved", name.relative_to(ROOT))
        time.sleep(1.0)  # let the ESP32-CAM breathe; move or re-arrange the debris between photos for more variety


if __name__ == "__main__":
    main()
