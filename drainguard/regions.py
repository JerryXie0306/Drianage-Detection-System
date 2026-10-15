"""Named image regions (polygons) for a fixed camera: where the grate is, where sediment collects.

Stored as JSON with coordinates normalised to 0..1 so they survive a change of image resolution:
    {"grate": [[x, y], ...], "sump": [[x, y], ...]}
"""

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

Polygon = List[List[float]]


def load_open_ratio(path: Optional[str]) -> float:
    """Optional "_open_ratio" (0..1): share of the grate polygon that is open when the grate is clear."""
    if not path:
        return 1.0
    r = float(json.loads(Path(path).read_text()).get("_open_ratio", 1.0))
    if not 0.05 <= r <= 1.0:
        raise ValueError("_open_ratio must be between 0.05 and 1")
    return r


def validate_regions(data: dict) -> Dict[str, Polygon]:
    out = {}
    for name, pts in data.items():
        if name.startswith("_"):
            continue
        if not isinstance(pts, list) or len(pts) < 3 or any(
                not isinstance(p, (list, tuple)) or len(p) != 2 or not (0 <= p[0] <= 1 and 0 <= p[1] <= 1)
                for p in pts):
            raise ValueError(f"region {name!r}: need >=3 points with x,y in 0..1")
        out[name] = [[float(x), float(y)] for x, y in pts]
    if "_open_ratio" in data and not 0.05 <= float(data["_open_ratio"]) <= 1.0:
        raise ValueError("_open_ratio must be between 0.05 and 1")
    return out


def load_regions(path: Optional[str]) -> Dict[str, Polygon]:
    if not path:
        return {}
    return validate_regions(json.loads(Path(path).read_text()))


def save_regions(path: str, regions: Dict[str, Polygon]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(regions, indent=2))


def rasterize(polygon: Polygon, size) -> np.ndarray:
    """Boolean mask (h, w) for a normalised polygon; size = (width, height)."""
    w, h = size
    m = Image.new("L", (w, h), 0)
    ImageDraw.Draw(m).polygon([(x * (w - 1), y * (h - 1)) for x, y in polygon], fill=1)
    return np.asarray(m, dtype=bool)


def region_masks(regions: Dict[str, Polygon], size) -> Dict[str, np.ndarray]:
    """Masks for 'grate' and 'sump'. With no regions, the whole frame is the grate."""
    w, h = size
    masks = {name: rasterize(poly, size) for name, poly in regions.items()}
    if "grate" not in masks:
        masks["grate"] = np.ones((h, w), dtype=bool)
    return masks
