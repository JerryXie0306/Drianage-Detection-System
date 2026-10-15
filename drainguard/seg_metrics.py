"""Class definitions and the blockage formulas, shared by prediction, labelling and evaluation.

Mask value convention (PNG, uint8): 0 = unlabelled/ignore, 1..6 = the classes below.
"""

from typing import Dict, Optional

import numpy as np

CLASSES = ["vegetation", "mud", "litter", "grate_bars", "open_slot", "wall"]
CLASS_IDS = {name: i + 1 for i, name in enumerate(CLASSES)}  # mask value for each class
DEBRIS = ("vegetation", "mud", "litter")
COLORS = {  # RGB used for overlays and the labelling tool
    "vegetation": (40, 200, 60),
    "mud": (150, 90, 30),
    "litter": (235, 40, 40),
    "grate_bars": (50, 120, 255),
    "open_slot": (255, 255, 255),
    "wall": (130, 130, 130),
}


def region_stats(label_map: np.ndarray, region: Optional[np.ndarray], open_ratio: float = 1.0,
                 min_open_pct: float = 2.0) -> Dict[str, float]:
    """Per-class share (%) of a region, plus the headline numbers.

    blocked_pct   = debris pixels / (open_ratio * region pixels), capped at 100. Independent of lighting.
                    open_ratio is the share of the grate polygon that is open when the grate is clear
                    (bars take the rest); 1.0 = report plain debris coverage of the polygon.
    obstruction_pct = debris / (debris + open_slot): light-based cross-check. Only given when at least
                    min_open_pct of the region shows light through slots (otherwise it is meaningless,
                    e.g. at night), else None.
    """
    if region is None:
        region = np.ones(label_map.shape, dtype=bool)
    valid = region & (label_map > 0)
    n = int(valid.sum())
    if n == 0:
        return {"pixels": 0, "blocked_pct": None, "obstruction_pct": None, "debris_pct": None,
                "composition_pct": {}, "class_pct": {}}
    counts = {c: int(((label_map == CLASS_IDS[c]) & valid).sum()) for c in CLASSES}
    class_pct = {c: 100.0 * counts[c] / n for c in CLASSES}
    debris = sum(counts[c] for c in DEBRIS)
    openable = debris + counts["open_slot"]
    has_light = class_pct["open_slot"] >= min_open_pct
    return {
        "pixels": n,
        "class_pct": class_pct,
        "debris_pct": 100.0 * debris / n,
        "blocked_pct": min(100.0, 100.0 * debris / (open_ratio * n)),
        "obstruction_pct": 100.0 * debris / openable if (openable and has_light) else None,
        "composition_pct": {c: (100.0 * counts[c] / debris if debris else 0.0) for c in DEBRIS},
    }


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, region: Optional[np.ndarray] = None) -> np.ndarray:
    """6x6 matrix (rows = true, cols = predicted) over labelled pixels (gt > 0)."""
    valid = gt > 0
    if region is not None:
        valid &= region
    cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    t = gt[valid].astype(int) - 1
    p = np.clip(pred[valid].astype(int) - 1, 0, len(CLASSES) - 1)
    np.add.at(cm, (t, p), 1)
    return cm


def iou_from_confusion(cm: np.ndarray) -> Dict[str, float]:
    out = {}
    for i, c in enumerate(CLASSES):
        tp = cm[i, i]
        denom = cm[i, :].sum() + cm[:, i].sum() - tp
        out[c] = float(tp / denom) if denom else float("nan")
    return out
