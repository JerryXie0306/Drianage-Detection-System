"""Evaluate segmentation + blockage percentage against hand-labelled masks.

    python -m drainguard.evaluate_seg --images data/seg/test/images --masks data/seg/test/masks \
        --regions regions/sample_fisheye.json --head models/seg_head.joblib

Reports, for zero-shot and (if --head) the trained head:
  * per-class IoU and pixel accuracy over labelled pixels
  * blocked-percentage error inside the grate region: predicted vs the value computed from the ground-truth
    mask (mean absolute error in percentage points, bias, per-image table)
Use photos that were NOT used to train the head.
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from .regions import load_open_ratio, load_regions, region_masks
from .seg_data import cached_logits, list_pairs, load_mask, load_rgb
from .seg_metrics import CLASSES, confusion_matrix, iou_from_confusion, region_stats
from .segment import ClipSegExtractor, Segmenter


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--masks", required=True)
    ap.add_argument("--regions")
    ap.add_argument("--head")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()

    pairs = list_pairs(a.images, a.masks)
    if not pairs:
        raise SystemExit("no labelled pairs found")
    ex = ClipSegExtractor()
    seg = Segmenter(extractor=ex, head_path=a.head, regions_path=a.regions)
    regions = load_regions(a.regions)
    open_ratio = load_open_ratio(a.regions)
    S = ex.size
    modes = [("zero-shot", False)] + ([("head", True)] if a.head else [])

    results = {}
    rows = []
    for mode, use_head in modes:
        cm = np.zeros((6, 6), dtype=np.int64)
        errs = []
        for ip, mp in pairs:
            lg = cached_logits(ex, ip, a.cache)
            rgb = load_rgb(ip, S)
            gt = load_mask(mp, S)
            pred, _ = seg.decide(lg, rgb, use_head=use_head)
            masks = region_masks(regions, (S, S))
            cm += confusion_matrix(gt, pred, masks["grate"] if regions else None)
            g = region_stats(gt, masks["grate"], open_ratio)
            p = region_stats(pred, masks["grate"], open_ratio)
            if g["blocked_pct"] is not None and p["blocked_pct"] is not None:
                errs.append(p["blocked_pct"] - g["blocked_pct"])
                rows.append({"mode": mode, "image": ip.name, "gt_blocked_pct": round(g["blocked_pct"], 1),
                             "pred_blocked_pct": round(p["blocked_pct"], 1),
                             "error_pts": round(p["blocked_pct"] - g["blocked_pct"], 1)})
        iou = iou_from_confusion(cm)
        e = np.array(errs) if errs else np.array([np.nan])
        results[mode] = {
            "iou": iou,
            "mean_iou": float(np.nanmean(list(iou.values()))),
            "pixel_accuracy": float(np.trace(cm) / cm.sum()) if cm.sum() else float("nan"),
            "blocked_pct_mae": float(np.mean(np.abs(e))),
            "blocked_pct_bias": float(np.mean(e)),
            "n_images": len(pairs),
            "confusion": cm.tolist(),
        }

    print(f"{len(pairs)} labelled images   scope: {'grate region' if regions else 'whole frame'}\n")
    print("| mode | mean IoU | pixel acc | blocked% MAE (pts) | blocked% bias (pts) |")
    print("|---|---|---|---|---|")
    for m, r in results.items():
        print(f"| {m} | {r['mean_iou'] * 100:.1f}% | {r['pixel_accuracy'] * 100:.1f}% | "
              f"{r['blocked_pct_mae']:.1f} | {r['blocked_pct_bias']:+.1f} |")
    print("\nper-class IoU (%):")
    for m, r in results.items():
        print(f"  {m:9s}", "  ".join(f"{c}={v * 100:.1f}" if not np.isnan(v) else f"{c}=n/a" for c, v in r["iou"].items()))

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    (out / f"seg_eval_{stamp}.json").write_text(json.dumps(results, indent=2))
    if rows:
        with open(out / f"seg_eval_{stamp}_blocked.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"\nsaved to {out}/seg_eval_{stamp}.json")


if __name__ == "__main__":
    main()
