"""Loading labelled pairs (image + index mask) with cached CLIPSeg logits."""

import hashlib
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from .segment import PROMPT_LIST

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def list_pairs(images_dir: str, masks_dir: str) -> List[Tuple[Path, Path]]:
    """Images that have a mask (same stem, .png). Unlabelled images are skipped."""
    pairs = []
    for p in sorted(Path(images_dir).iterdir()):
        if p.suffix.lower() in IMG_EXT:
            m = Path(masks_dir) / (p.stem + ".png")
            if m.exists():
                pairs.append((p, m))
    return pairs


def load_mask(path: Path, size: int) -> np.ndarray:
    m = Image.open(path)
    if m.mode not in ("L", "P"):
        m = m.convert("L")
    return np.asarray(m.resize((size, size), Image.NEAREST), dtype=np.uint8)


def cached_logits(extractor, img_path: Path, cache_dir: str = ".cache") -> np.ndarray:
    st = img_path.stat()
    key = hashlib.sha1(f"{img_path.resolve()}|{st.st_size}|{st.st_mtime_ns}|{extractor.size}|{extractor.tiles}|"
                       f"{extractor.overlap}|{len(PROMPT_LIST)}".encode()).hexdigest()[:16]
    f = Path(cache_dir) / f"{key}.npy"
    if f.exists():
        return np.load(f).astype(np.float32)
    lg = extractor.logits(Image.open(img_path))
    f.parent.mkdir(parents=True, exist_ok=True)
    np.save(f, lg.astype(np.float16))
    return lg


def load_rgb(img_path: Path, size: int) -> np.ndarray:
    return np.asarray(Image.open(img_path).convert("RGB").resize((size, size), Image.BICUBIC))
