"""Train the optional scikit-learn classifier on your own labelled photos.

    python -m drainguard.train --data data/train --out models/clf.joblib

Keep photos used for training OUT of data/test/, otherwise the evaluation is meaningless.
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

from .config import DetectorConfig
from .detector import features
from .evaluate import load_dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/train")
    ap.add_argument("--out", default="models/clf.joblib")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = load_dataset(Path(a.data))
    if not items:
        raise SystemExit(f"no images found under {a.data}/(clear|partial|blocked)/")
    cfg = DetectorConfig()
    X = np.array([features(Image.open(p).convert("RGB"), cfg) for p, _ in items])
    y = np.array([l for _, l in items])
    classes, counts = np.unique(y, return_counts=True)
    print("class counts:", dict(zip(classes, counts.tolist())))

    clf = RandomForestClassifier(n_estimators=300, random_state=a.seed, class_weight="balanced")
    folds = min(5, int(counts.min()))
    if folds >= 2:
        scores = cross_val_score(clf, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=a.seed))
        print(f"{folds}-fold CV accuracy: {scores.mean() * 100:.1f}% (+/- {scores.std() * 100:.1f})")
    else:
        print("too few images per class for cross-validation - add more photos")
    clf.fit(X, y)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, a.out)
    print(f"saved model to {a.out}")


if __name__ == "__main__":
    main()
