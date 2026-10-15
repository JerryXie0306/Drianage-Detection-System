"""Synthetic grate images for SMOKE TESTS and the demo only.

They are generated from the same colour assumptions the heuristic detector uses, so accuracy on them
means nothing. Real evidence for the project log must come from real photos (data/test/...).
"""

import argparse
import math
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

W, H = 320, 240


def _grate(rng: random.Random) -> Image.Image:
    base = rng.randint(135, 165)
    img = Image.new("RGB", (W, H), (base, base, base - 6))
    d = ImageDraw.Draw(img)
    d.rectangle((30, 40, W - 30, H - 40), fill=(55, 58, 60))
    for x in range(40, W - 40, 18):
        d.rectangle((x, 46, x + 8, H - 46), fill=(120, 124, 128))
    return img


def _blob(d: ImageDraw.ImageDraw, rng: random.Random, kind: str):
    cx, cy = rng.randint(40, W - 40), rng.randint(50, H - 50)
    r = rng.randint(20, 42)
    if kind == "green":
        col = (rng.randint(40, 90), rng.randint(110, 170), rng.randint(30, 70))
    else:
        col = (rng.randint(95, 130), rng.randint(65, 90), rng.randint(35, 55))
    pts = []
    for k in range(9):
        ang = 2 * math.pi * k / 9
        rad = r * rng.uniform(0.6, 1.0)
        pts.append((cx + rad * math.cos(ang), cy + 0.7 * rad * math.sin(ang)))
    d.polygon(pts, fill=col)


def make_image(label: str, rng: random.Random) -> Image.Image:
    img = _grate(rng)
    d = ImageDraw.Draw(img)
    n = {"clear": 0, "partial": rng.randint(9, 12), "blocked": rng.randint(55, 70)}[label]
    for _ in range(n):
        _blob(d, rng, rng.choice(["green", "brown"]))
    img = img.filter(ImageFilter.GaussianBlur(0.6))
    return img


def generate(out_dir: Path, per_class: int = 20, seed: int = 0):
    rng = random.Random(seed)
    for label in ("clear", "partial", "blocked"):
        d = out_dir / label
        d.mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            make_image(label, rng).save(d / f"{label}_{i:03d}.jpg", quality=92)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    generate(Path(a.out), a.per_class, a.seed)
    print(f"wrote synthetic images to {a.out} (smoke tests / demo only)")
