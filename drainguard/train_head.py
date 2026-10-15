"""Train the pixel classifier on top of CLIPSeg logits from your labelled photos.

    python -m drainguard.train_head --images data/seg/images --masks data/seg/masks --out models/seg_head.joblib

Honest evaluation: leave-one-IMAGE-out cross-validation (never train and test on the same photo), compared
with the zero-shot baseline on the same held-out images. Needs >= 3 labelled images for that report.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .seg_data import cached_logits, list_pairs, load_mask, load_rgb
from .seg_metrics import CLASSES, confusion_matrix, iou_from_confusion
from .segment import PROMPT_LIST, ClipSegExtractor, Segmenter, pixel_features


def sample_pixels(feat, labels, per_class, rng):
    idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        idx.append(rng.choice(ci, min(per_class, len(ci)), replace=False))
    idx = np.concatenate(idx)
    return feat[idx], labels[idx]


def make_model(seed):
    return HistGradientBoostingClassifier(max_iter=200, max_depth=6, learning_rate=0.08,
                                          class_weight="balanced", random_state=seed)


def fmt_iou(iou):
    return "  ".join(f"{c}={v * 100:5.1f}" if not np.isnan(v) else f"{c}=  n/a" for c, v in iou.items())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default="data/seg/images")
    ap.add_argument("--masks", default="data/seg/masks")
    ap.add_argument("--out", default="models/seg_head.joblib")
    ap.add_argument("--use-position", action="store_true", help="add x,y position features (fixed camera only)")
    ap.add_argument("--per-class", type=int, default=6000, help="pixels sampled per class per image")
    ap.add_argument("--cache", default=".cache")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pairs = list_pairs(a.images, a.masks)
    if not pairs:
        raise SystemExit("no labelled pairs found (need images/<name>.jpg and masks/<name>.png)")
    ex = ClipSegExtractor()
    zs = Segmenter(extractor=ex)
    rng = np.random.default_rng(a.seed)

    data = []
    for ip, mp in pairs:
        lg = cached_logits(ex, ip, a.cache)
        rgb = load_rgb(ip, ex.size)
        mask = load_mask(mp, ex.size)
        feat = pixel_features(lg, rgb, a.use_position)
        data.append({"name": ip.name, "lg": lg, "rgb": rgb, "mask": mask, "feat": feat})
    print(f"{len(data)} labelled images")
    totals = np.zeros(len(CLASSES), dtype=int)
    for d in data:
        totals += np.bincount(d["mask"][d["mask"] > 0].ravel() - 1, minlength=len(CLASSES))
    print("labelled pixels per class:", dict(zip(CLASSES, totals.tolist())))
    missing = [c for c, n in zip(CLASSES, totals) if n == 0]
    if missing:
        print(f"WARNING: no labelled pixels for {missing} - the head cannot learn those classes")

    if len(data) >= 3:
        cm_head = np.zeros((6, 6), dtype=np.int64)
        cm_zs = np.zeros((6, 6), dtype=np.int64)
        for i, held in enumerate(data):
            Xs, ys = [], []
            for j, d in enumerate(data):
                if j == i:
                    continue
                lab = d["mask"].ravel()
                m = lab > 0
                X, y = sample_pixels(d["feat"][m], lab[m], a.per_class, rng)
                Xs.append(X)
                ys.append(y)
            clf = make_model(a.seed).fit(np.concatenate(Xs), np.concatenate(ys))
            pred = clf.predict(held["feat"]).reshape(held["mask"].shape)
            cm_head += confusion_matrix(held["mask"], pred)
            zl, _ = zs.decide(held["lg"], held["rgb"], use_head=False)
            cm_zs += confusion_matrix(held["mask"], zl)
        iz, ih = iou_from_confusion(cm_zs), iou_from_confusion(cm_head)
        print("\nleave-one-image-out IoU (%):")
        print("  zero-shot:", fmt_iou(iz))
        print("  head     :", fmt_iou(ih))
        print(f"  mean IoU  zero-shot {np.nanmean(list(iz.values())) * 100:.1f}   head {np.nanmean(list(ih.values())) * 100:.1f}")
    else:
        print("\nfewer than 3 labelled images: skipping cross-validation (no honest accuracy estimate yet)")

    Xs, ys = [], []
    for d in data:
        lab = d["mask"].ravel()
        m = lab > 0
        X, y = sample_pixels(d["feat"][m], lab[m], a.per_class, rng)
        Xs.append(X)
        ys.append(y)
    clf = make_model(a.seed).fit(np.concatenate(Xs), np.concatenate(ys))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "prompts": PROMPT_LIST, "use_position": a.use_position}, a.out)
    print(f"\nsaved head to {a.out}")


if __name__ == "__main__":
    main()
