"""Evaluate the detector on labelled photos  (Test 1: "N images, clear vs blocked, varied lighting -> accuracy").

Expected layout:  <data>/clear/*.jpg  <data>/partial/*.jpg  <data>/blocked/*.jpg

    python -m drainguard.evaluate --data data/test
    python -m drainguard.evaluate --data data/test --conditions normal,dark,blur

Conditions other than "normal" are SIMULATED by editing the photos (darken, blur, noise). They are a
quick robustness check, not a substitute for photographing real wet / low-light drains.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from .config import DetectorConfig
from .detector import LABELS, Detector

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def apply_condition(img: Image.Image, cond: str, rng: np.random.Generator) -> Image.Image:
    if cond == "normal":
        return img
    if cond == "dark":
        return ImageEnhance.Brightness(img).enhance(0.35)
    if cond == "bright":
        return ImageEnhance.Brightness(img).enhance(1.6)
    if cond == "lowcontrast":
        return ImageEnhance.Contrast(img).enhance(0.5)
    if cond == "blur":
        return img.filter(ImageFilter.GaussianBlur(3))
    if cond == "noise":
        a = np.asarray(img, dtype=np.float32) + rng.normal(0, 18, (img.height, img.width, 3))
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    raise ValueError(f"unknown condition {cond}")


def load_dataset(root: Path):
    items = []
    for label in LABELS:
        d = root / label
        if d.is_dir():
            items += [(p, label) for p in sorted(d.iterdir()) if p.suffix.lower() in IMG_EXT]
    return items


def metrics(y_true, y_pred):
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[idx[t], idx[p]] += 1
    per = {}
    for l in LABELS:
        i = idx[l]
        tp, fp, fn = cm[i, i], cm[:, i].sum() - cm[i, i], cm[i, :].sum() - cm[i, i]
        prec = tp / (tp + fp) if tp + fp else float("nan")
        rec = tp / (tp + fn) if tp + fn else float("nan")
        per[l] = {"precision": prec, "recall": rec, "support": int(cm[i, :].sum())}
    n = cm.sum()
    # the dangerous error: a blocked drain reported as clear
    missed = cm[idx["blocked"], idx["clear"]] / cm[idx["blocked"], :].sum() if cm[idx["blocked"], :].sum() else float("nan")
    return {"n": int(n), "accuracy": float(np.trace(cm) / n) if n else float("nan"),
            "blocked_called_clear_rate": float(missed), "confusion": cm.tolist(), "per_class": per}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--backend", choices=["heuristic", "model"], default="heuristic")
    ap.add_argument("--model")
    ap.add_argument("--conditions", default="normal")
    ap.add_argument("--roi", type=float, nargs=4, default=(0, 0, 1, 1))
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    if "synthetic" in a.data.lower():
        print("WARNING: synthetic images share the detector's colour assumptions - these numbers are "
              "NOT evidence of real-world accuracy.\n")
    items = load_dataset(Path(a.data))
    if not items:
        raise SystemExit(f"no images found under {a.data}/(clear|partial|blocked)/")

    det = Detector(a.backend, a.model, DetectorConfig(roi=tuple(a.roi)))
    rng = np.random.default_rng(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    summary, rows = {}, []

    for cond in [c.strip() for c in a.conditions.split(",") if c.strip()]:
        y_true, y_pred = [], []
        for path, label in items:
            img = apply_condition(Image.open(path).convert("RGB"), cond, rng)
            d = det.detect(img)
            y_true.append(label)
            y_pred.append(d.label)
            rows.append({"condition": cond, "file": str(path), "true": label, "pred": d.label,
                         "confidence": round(d.confidence, 3), "coverage": round(d.coverage, 3)})
        summary[cond] = metrics(y_true, y_pred)

    with open(out / f"eval_{stamp}.json", "w") as f:
        json.dump({"data": a.data, "backend": a.backend, "summary": summary}, f, indent=2)
    with open(out / f"eval_{stamp}_predictions.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"backend={a.backend}  images={len(items)}  data={a.data}\n")
    print("| condition | n | accuracy | blocked called clear |")
    print("|---|---|---|---|")
    for cond, m in summary.items():
        print(f"| {cond} | {m['n']} | {m['accuracy'] * 100:.1f}% | {m['blocked_called_clear_rate'] * 100:.1f}% |")
    first = next(iter(summary.values()))
    print("\nconfusion matrix (rows=true, cols=pred; order clear/partial/blocked), first condition:")
    for l, row in zip(LABELS, first["confusion"]):
        print(f"  {l:8s} {row}")
    print(f"\nsaved to {out}/eval_{stamp}.json and _predictions.csv")


if __name__ == "__main__":
    main()
