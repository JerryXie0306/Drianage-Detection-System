"""Semantic segmentation of a drain-inlet photo into debris types, with per-region percentages.

Pipeline
  1. ClipSegExtractor: text-prompted CLIPSeg, run on a global view plus overlapping tiles -> one logit map
     per prompt at (size x size) pixels.
  2. Zero-shot decision (max logit per class, softmax) OR a trained head (scikit-learn, see train_head.py)
     that classifies each pixel from the prompt logits + colour. The head fixes systematic confusions
     (e.g. wet mud read as vegetation) using your own labelled photos.
  3. region_stats(): % per class inside the calibrated grate / sump regions (see regions.py).

torch + transformers are imported lazily, so the rest of DrainGuard works without them.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .regions import load_open_ratio, load_regions, region_masks
from .seg_metrics import CLASSES, CLASS_IDS, COLORS, region_stats

PROMPTS: Dict[str, List[str]] = {
    "vegetation": ["green leaves and weeds", "grass and plants", "dead brown leaves"],
    "mud": ["wet mud", "dirt and sludge", "soil and silt"],
    "litter": ["plastic bag", "aluminium can", "rubbish and food packaging"],
    "grate_bars": ["metal grate bars"],
    "open_slot": ["bright sky seen through a grate", "empty gap between grate bars"],
    "wall": ["concrete pipe wall", "wet concrete"],
}
PROMPT_LIST = [p for c in CLASSES for p in PROMPTS[c]]
PROMPT_CLASS = np.array([CLASS_IDS[c] for c in CLASSES for _ in PROMPTS[c]])  # class id of each prompt
DEFAULT_MODEL = "CIDAS/clipseg-rd64-refined"
TILE_PX = 352  # CLIPSeg native resolution


class ClipSegExtractor:
    def __init__(self, model_name: str = DEFAULT_MODEL, size: int = 704, tiles: int = 2,
                 overlap: float = 0.25, device: Optional[str] = None):
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        self.torch = torch
        self.size, self.tiles, self.overlap = size, tiles, overlap
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else "cuda" if torch.cuda.is_available() else "cpu")
        proc = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name).eval().to(self.device)
        ip = proc.image_processor
        self.mean = torch.tensor(ip.image_mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(ip.image_std, device=self.device).view(1, 3, 1, 1)
        with torch.no_grad():
            tok = proc.tokenizer(PROMPT_LIST, padding=True, return_tensors="pt").to(self.device)
            self.cond = self.model.clip.get_text_features(**tok)
            if not torch.is_tensor(self.cond):  # newer transformers return an output object
                self.cond = self.cond.pooler_output

    def _run(self, crop: Image.Image) -> np.ndarray:
        """One window -> (P, TILE_PX, TILE_PX) logits."""
        torch = self.torch
        arr = np.asarray(crop.convert("RGB").resize((TILE_PX, TILE_PX), Image.BICUBIC), dtype=np.float32) / 255.0
        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(self.device)
        x = ((x - self.mean) / self.std).repeat(len(PROMPT_LIST), 1, 1, 1)
        with torch.no_grad():
            out = self.model(pixel_values=x, conditional_embeddings=self.cond)
        return out.logits.float().cpu().numpy()

    def logits(self, img: Image.Image) -> np.ndarray:
        """(P, size, size) logits for the image squashed to a square (normalised coordinates are preserved)."""
        S, n = self.size, self.tiles
        work = img.convert("RGB").resize((S, S), Image.BICUBIC)
        acc = np.zeros((len(PROMPT_LIST), S, S), dtype=np.float32)
        wsum = np.zeros((S, S), dtype=np.float32)

        def add(window_box, weight):
            x0, y0, x1, y1 = window_box
            lg = self._run(work.crop(window_box))  # (P,352,352)
            lg = np.stack([np.asarray(Image.fromarray(m).resize((x1 - x0, y1 - y0), Image.BILINEAR)) for m in lg])
            acc[:, y0:y1, x0:x1] += lg * weight
            wsum[y0:y1, x0:x1] += weight

        def ramp(h, w):  # taper towards the window edge to hide seams
            ry = np.minimum(np.arange(h) + 1, np.arange(h)[::-1] + 1)
            rx = np.minimum(np.arange(w) + 1, np.arange(w)[::-1] + 1)
            return np.minimum(np.outer(ry, rx) / (0.25 * min(h, w)), 1.0).astype(np.float32) + 0.05

        add((0, 0, S, S), 0.6 * np.ones((S, S), dtype=np.float32))  # global context
        if n > 1:
            tile = int(S / (n - (n - 1) * self.overlap))
            starts = np.linspace(0, S - tile, n).round().astype(int)
            w = ramp(tile, tile)
            for y0 in starts:
                for x0 in starts:
                    add((int(x0), int(y0), int(x0) + tile, int(y0) + tile), w)
        return acc / wsum[None]


def _rgb_features(rgb: np.ndarray) -> np.ndarray:
    """Colour features per pixel: H(sin,cos), S, V."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"), dtype=np.float32)
    h = hsv[..., 0] * (2 * np.pi / 255.0)
    return np.stack([np.sin(h), np.cos(h), hsv[..., 1] / 255.0, hsv[..., 2] / 255.0], axis=-1)


def pixel_features(logits: np.ndarray, rgb: np.ndarray, use_position: bool = False) -> np.ndarray:
    """(H*W, F) feature matrix: prompt logits + colour (+ optional normalised position)."""
    P, H, W = logits.shape
    parts = [np.moveaxis(logits, 0, -1), _rgb_features(rgb)]
    if use_position:
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        parts.append(np.stack([xx / (W - 1), yy / (H - 1)], axis=-1))
    return np.concatenate(parts, axis=-1).reshape(H * W, -1)


@dataclass
class SegOutput:
    label_map: np.ndarray  # (S, S) uint8, values 1..6
    certainty: np.ndarray  # (S, S) float, probability of the chosen class
    masks: Dict[str, np.ndarray]
    stats: Dict[str, dict] = field(default_factory=dict)
    rgb: Optional[np.ndarray] = None


class Segmenter:
    def __init__(self, extractor=None, head_path: Optional[str] = None, regions_path: Optional[str] = None,
                 size: int = 704, tiles: int = 2, temperature: float = 2.0):
        self.extractor = extractor if extractor is not None else ClipSegExtractor(size=size, tiles=tiles)
        self.size = self.extractor.size
        self.regions = load_regions(regions_path)
        self.open_ratio = load_open_ratio(regions_path)
        self.temperature = temperature
        self.head = None
        self.use_position = False
        if head_path:
            import joblib
            bundle = joblib.load(head_path)
            if bundle["prompts"] != PROMPT_LIST:
                raise ValueError("head was trained with different prompts; retrain with train_head.py")
            self.head, self.use_position = bundle["model"], bundle["use_position"]

    def _zero_shot(self, logits: np.ndarray):
        cls_logits = np.stack([logits[PROMPT_CLASS == CLASS_IDS[c]].max(0) for c in CLASSES])  # (6,S,S)
        z = cls_logits / self.temperature
        p = np.exp(z - z.max(0))
        p /= p.sum(0)
        return (p.argmax(0) + 1).astype(np.uint8), p.max(0)

    @staticmethod
    def _split_open_slots(label: np.ndarray, rgb: np.ndarray) -> np.ndarray:
        """Zero-shot CLIP cannot tell bars from the gaps between them. Where the grate structure is
        clearly bimodal in brightness (daylight shining through), call the bright part open_slot."""
        v = np.asarray(Image.fromarray(rgb).convert("HSV"))[..., 2] / 255.0
        struct = (label == CLASS_IDS["grate_bars"]) | (label == CLASS_IDS["open_slot"])
        if struct.sum() < 500:
            return label
        vals = v[struct]
        hist, edges = np.histogram(vals, bins=64, range=(0, 1))
        w0 = np.cumsum(hist)
        w1 = w0[-1] - w0
        mids = (edges[:-1] + edges[1:]) / 2
        m0 = np.cumsum(hist * mids) / np.maximum(w0, 1)
        m1 = (np.sum(hist * mids) - np.cumsum(hist * mids)) / np.maximum(w1, 1)
        k = int(np.argmax(w0 * w1 * (m0 - m1) ** 2))  # Otsu
        thr = mids[k]
        bright, dark = vals[vals > thr], vals[vals <= thr]
        if len(bright) < 0.03 * len(vals) or len(dark) == 0 or bright.mean() < 0.55 or bright.mean() - dark.mean() < 0.25:
            return label
        out = label.copy()
        out[struct & (v > thr)] = CLASS_IDS["open_slot"]
        out[struct & (v <= thr)] = CLASS_IDS["grate_bars"]
        return out

    def _with_head(self, logits: np.ndarray, rgb: np.ndarray):
        S = logits.shape[1]
        proba = self.head.predict_proba(pixel_features(logits, rgb, self.use_position))
        idx = proba.argmax(1)
        label = np.asarray(self.head.classes_)[idx].astype(np.uint8).reshape(S, S)
        return label, proba.max(1).reshape(S, S).astype(np.float32)

    def decide(self, logits: np.ndarray, rgb: np.ndarray, use_head: Optional[bool] = None):
        """Label map + certainty from logits. use_head=None -> head if one is loaded."""
        use_head = (self.head is not None) if use_head is None else use_head
        if use_head:
            if self.head is None:
                raise ValueError("no head loaded")
            return self._with_head(logits, rgb)
        label, cert = self._zero_shot(logits)
        return self._split_open_slots(label, rgb), cert

    def stats_for(self, label: np.ndarray) -> Dict[str, dict]:
        masks = region_masks(self.regions, label.shape[::-1])
        return {name: region_stats(label, m, self.open_ratio if name == "grate" else 1.0)
                for name, m in masks.items()}

    def segment(self, img: Image.Image) -> SegOutput:
        S = self.size
        logits = self.extractor.logits(img)
        rgb = np.asarray(img.convert("RGB").resize((S, S), Image.BICUBIC))
        label, cert = self.decide(logits, rgb)
        masks = region_masks(self.regions, (S, S))
        return SegOutput(label, cert, masks, self.stats_for(label), rgb)


def render_overlay(out: SegOutput, regions: Optional[Dict[str, list]] = None, alpha: float = 0.5,
                   scale_to: Optional[tuple] = None) -> Image.Image:
    colors = np.zeros((7, 3), dtype=np.uint8)
    for c in CLASSES:
        colors[CLASS_IDS[c]] = COLORS[c]
    base = out.rgb.astype(np.float32)
    shown = out.label_map > 0
    blend = base.copy()
    blend[shown] = (1 - alpha) * base[shown] + alpha * colors[out.label_map][shown]
    img = Image.fromarray(blend.astype(np.uint8))
    if regions:
        S = out.label_map.shape[0]
        d = ImageDraw.Draw(img)
        for name, poly in regions.items():
            pts = [(x * (S - 1), y * (S - 1)) for x, y in poly]
            d.line(pts + [pts[0]], fill=(255, 220, 0) if name == "grate" else (0, 220, 255), width=3)
    return img.resize(scale_to, Image.NEAREST) if scale_to else img
