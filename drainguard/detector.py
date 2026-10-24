"""Blockage detection for a drain inlet image: clear / partial / blocked.

Two backends:
  heuristic - colour-based debris coverage (vegetation + soil/mud) inside the grate ROI.
              Works with no training data; confidence is a margin score, NOT a calibrated probability.
  model     - scikit-learn classifier trained on your own labelled photos (see train.py).
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .config import DetectorConfig

LABELS = ("clear", "partial", "blocked")


@dataclass
class Detection:
    label: str
    confidence: float
    coverage: float
    bbox: Optional[tuple] = None  # (x0, y0, x1, y1) in original image pixels
    backend: str = "heuristic"
    extra: dict = field(default_factory=dict)
    overlay: Optional[Image.Image] = field(default=None, repr=False)  # segment backend: label-map overlay


def _hsv(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("HSV"), dtype=np.float32)
    h = a[..., 0] * (360.0 / 255.0)
    s = a[..., 1] / 255.0
    v = a[..., 2] / 255.0
    return np.stack([h, s, v], axis=-1)


def debris_masks(img: Image.Image, min_component_frac: float = 0.003):
    """Return (vegetation_mask, soil_mask, combined_mask) as boolean arrays."""
    hsv = _hsv(img)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    veg = (h >= 40) & (h <= 170) & (s > 0.22) & (v > 0.12)
    soil = (h >= 8) & (h < 40) & (s > 0.18) & (s < 0.80) & (v > 0.12) & (v < 0.80)
    comb = veg | soil
    comb = ndimage.binary_opening(comb, structure=np.ones((3, 3)))
    labelled, n = ndimage.label(comb)
    if n:
        sizes = ndimage.sum(comb, labelled, index=np.arange(1, n + 1))
        keep = np.isin(labelled, 1 + np.where(sizes >= comb.size * min_component_frac)[0])
        comb = keep
    return veg & comb, soil & comb, comb


def _bbox_of(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _heuristic_label(cov: float, cfg: DetectorConfig):
    b1, b2 = cfg.clear_below, cfg.blocked_above
    if cov < b1:
        return "clear", (b1 - cov) / b1
    if cov >= b2:
        return "blocked", min(1.0, (cov - b2) / 0.15)
    return "partial", min(cov - b1, b2 - cov) / ((b2 - b1) / 2)


def features(img: Image.Image, cfg: DetectorConfig) -> np.ndarray:
    """Feature vector for the sklearn backend (45 dims)."""
    small = img.convert("RGB").resize((96, 96))
    hsv = _hsv(small)
    feats = []
    for ch, rng in zip(range(3), ((0, 360), (0, 1), (0, 1))):
        hist, _ = np.histogram(hsv[..., ch], bins=8, range=rng)
        feats.extend(hist / hist.sum())
    veg, soil, comb = debris_masks(small, cfg.min_component_frac)
    feats.extend([veg.mean(), soil.mean(), comb.mean()])
    gray = np.asarray(small.convert("L"), dtype=np.float32) / 255.0
    gy, gx = np.gradient(gray)
    mag = np.hypot(gx, gy)
    feats.extend([mag.mean(), mag.std()])
    g = 4
    step = 96 // g
    for i in range(g):
        for j in range(g):
            feats.append(comb[i * step:(i + 1) * step, j * step:(j + 1) * step].mean())
    return np.array(feats, dtype=np.float32)


class Detector:
    def __init__(self, backend: str = "heuristic", model_path: Optional[str] = None,
                 config: Optional[DetectorConfig] = None, segmenter=None):
        self.cfg = config or DetectorConfig()
        self.backend = backend
        self.model = None
        self.segmenter = segmenter
        if backend == "segment":
            if segmenter is None:
                raise ValueError("segment backend needs a Segmenter")
        elif backend == "model":
            import joblib
            if not model_path:
                raise ValueError("model backend needs model_path")
            self.model = joblib.load(model_path)
        elif backend != "heuristic":
            raise ValueError(f"unknown backend {backend!r}")

    def _crop_roi(self, img: Image.Image):
        w, h = img.size
        x0, y0, x1, y1 = self.cfg.roi
        box = (int(x0 * w), int(y0 * h), max(int(x1 * w), int(x0 * w) + 1), max(int(y1 * h), int(y0 * h) + 1))
        return img.crop(box), box

    def _detect_segment(self, img: Image.Image) -> Detection:
        from .seg_metrics import CLASS_IDS, DEBRIS
        from .segment import render_overlay

        out = self.segmenter.segment(img)
        g, sump = out.stats["grate"], out.stats.get("sump")
        pct = g["blocked_pct"]
        if pct is None:
            return Detection("clear", 0.5, 0.0, None, "segment", {"note": "no valid pixels in grate region"})
        b1, b2 = self.cfg.seg_clear_below, self.cfg.seg_blocked_above
        if pct < b1:
            label, score = "clear", (b1 - pct) / b1
        elif pct >= b2:
            label, score = "blocked", min(1.0, (pct - b2) / 15.0)
        else:
            label, score = "partial", min(pct - b1, b2 - pct) / ((b2 - b1) / 2)
        conf = 0.5 + 0.49 * float(np.clip(score, 0, 1))
        debris = np.zeros(out.label_map.shape, dtype=bool)
        for c in DEBRIS:
            debris |= out.label_map == CLASS_IDS[c]
        bb = _bbox_of(debris & out.masks["grate"])
        w, h = img.size
        S = out.label_map.shape[0]
        bbox = None if bb is None else (int(bb[0] * w / S), int(bb[1] * h / S), int(bb[2] * w / S), int(bb[3] * h / S))
        extra = {"grate": g, "sump": sump, "blocked_pct": pct,
                 "mean_certainty": float(out.certainty[out.masks["grate"]].mean()) if out.masks["grate"].any() else None}
        det = Detection(label, conf, pct / 100.0, bbox, "segment", extra)
        det.overlay = render_overlay(out, self.segmenter.regions, scale_to=img.size)
        return det

    def detect(self, img: Image.Image) -> Detection:
        img = img.convert("RGB")
        if self.backend == "segment":
            return self._detect_segment(img)
        roi_img, box = self._crop_roi(img)
        if roi_img.width > self.cfg.max_width:
            scale = self.cfg.max_width / roi_img.width
            small = roi_img.resize((self.cfg.max_width, max(1, int(roi_img.height * scale))))
        else:
            scale = 1.0
            small = roi_img
        veg, soil, comb = debris_masks(small, self.cfg.min_component_frac)
        coverage = float(comb.mean())
        bb = _bbox_of(comb)
        bbox = None
        if bb:
            bbox = (box[0] + int(bb[0] / scale), box[1] + int(bb[1] / scale),
                    box[0] + int(bb[2] / scale), box[1] + int(bb[3] / scale))

        if self.backend == "model":
            proba = self.model.predict_proba([features(roi_img, self.cfg)])[0]
            idx = int(np.argmax(proba))
            label = str(self.model.classes_[idx])
            conf = float(proba[idx])
            extra = {"proba": {str(c): float(p) for c, p in zip(self.model.classes_, proba)}}
        else:
            label, score = _heuristic_label(coverage, self.cfg)
            conf = 0.5 + 0.49 * float(np.clip(score, 0, 1))
            extra = {"veg": float(veg.mean()), "soil": float(soil.mean())}
        return Detection(label, conf, coverage, bbox, self.backend, extra)


def annotate(img: Image.Image, det: Detection, roi=(0, 0, 1, 1)) -> Image.Image:
    """Draw ROI, debris box and label on a copy of the frame (for the app preview)."""
    out = img.convert("RGB").copy()
    if det.overlay is not None:
        out = det.overlay.convert("RGB").resize(out.size)
        roi = (0, 0, 1, 1)
    d = ImageDraw.Draw(out)
    w, h = out.size
    if det.overlay is None:
        d.rectangle((roi[0] * w, roi[1] * h, roi[2] * w - 1, roi[3] * h - 1), outline=(80, 140, 255), width=2)
    colour = {"clear": (40, 180, 70), "partial": (230, 170, 20), "blocked": (220, 50, 50)}[det.label]
    if det.bbox:
        d.rectangle(det.bbox, outline=colour, width=3)
        d.text((det.bbox[0] + 4, max(0, det.bbox[1] - 12)), "debris", fill=colour)
    d.rectangle((0, h - 15, w, h), fill=(20, 24, 30))  # bottom strip: the app's "Analysis" tag sits top-left
    d.text((6, h - 13), f"{det.label.upper()} {det.confidence * 100:.0f}%  cover {det.coverage * 100:.0f}%", fill=(255, 255, 255))
    return out
