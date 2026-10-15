import io
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from drainguard.config import DetectorConfig
from drainguard.detector import Detector, annotate
from drainguard.label import LabelApp, make_handler
from drainguard.regions import load_open_ratio, load_regions, rasterize, save_regions
from drainguard.seg_metrics import CLASS_IDS, CLASSES, confusion_matrix, iou_from_confusion, region_stats
from drainguard.segment import PROMPT_CLASS, PROMPT_LIST, Segmenter, pixel_features
from http.server import ThreadingHTTPServer

S = 64


class FakeExtractor:
    """Left half looks like vegetation, right half like open slots (as far as the logits go)."""

    size, tiles, overlap = S, 1, 0.25

    def __init__(self, left="vegetation", right="open_slot"):
        self.left, self.right = left, right

    def logits(self, img):
        lg = np.full((len(PROMPT_LIST), S, S), -5.0, dtype=np.float32)
        lg[PROMPT_CLASS == CLASS_IDS[self.left], :, : S // 2] = 5.0
        lg[PROMPT_CLASS == CLASS_IDS[self.right], :, S // 2:] = 5.0
        return lg


def lm(shape=(100, 100), **fractions):
    """Label map with the given class fractions, laid out in row bands (rest unlabelled = 0)."""
    m = np.zeros(shape, dtype=np.uint8)
    row = 0
    for name, frac in fractions.items():
        n = int(round(frac * shape[0]))
        m[row:row + n] = CLASS_IDS[name]
        row += n
    return m


# ---------- regions ----------
def test_rasterize_area_matches_polygon():
    m = rasterize([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], (200, 100))
    assert m.mean() == pytest.approx(0.5, abs=0.03)


def test_regions_roundtrip_and_validation(tmp_path):
    f = tmp_path / "r.json"
    save_regions(str(f), {"grate": [[0, 0], [1, 0], [1, 1]], "_open_ratio": 0.6})
    assert "grate" in load_regions(str(f)) and load_open_ratio(str(f)) == 0.6
    f.write_text(json.dumps({"grate": [[0, 0], [2, 0], [1, 1]]}))
    with pytest.raises(ValueError):
        load_regions(str(f))
    f.write_text(json.dumps({"grate": [[0, 0], [1, 0], [1, 1]], "_open_ratio": 0.0}))
    with pytest.raises(ValueError):
        load_open_ratio(str(f))


# ---------- metrics ----------
def test_region_stats_numbers():
    st = region_stats(lm(vegetation=0.3, litter=0.1, open_slot=0.2, grate_bars=0.4), None)
    assert st["debris_pct"] == pytest.approx(40) and st["blocked_pct"] == pytest.approx(40)
    assert st["obstruction_pct"] == pytest.approx(100 * 40 / 60)
    assert st["composition_pct"]["vegetation"] == pytest.approx(75)
    assert region_stats(lm(vegetation=0.3, litter=0.1, open_slot=0.2, grate_bars=0.4), None, open_ratio=0.5)["blocked_pct"] == pytest.approx(80)


def test_obstruction_is_none_without_light_and_unlabelled_is_ignored():
    st = region_stats(lm(vegetation=0.5, grate_bars=0.3), None)  # nothing left unlabelled except 20 %
    assert st["obstruction_pct"] is None  # no visible slots: light-based metric is meaningless
    assert st["blocked_pct"] == pytest.approx(100 * 50 / 80)  # the unlabelled 20 % is excluded from the base
    assert region_stats(np.zeros((10, 10), np.uint8), None)["blocked_pct"] is None


def test_blocked_pct_is_capped():
    assert region_stats(lm(vegetation=0.9, mud=0.1), None, open_ratio=0.5)["blocked_pct"] == 100.0


def test_iou_and_confusion():
    gt = lm(vegetation=0.5, mud=0.5)
    perfect = iou_from_confusion(confusion_matrix(gt, gt))
    assert perfect["vegetation"] == 1.0 and np.isnan(perfect["litter"])
    wrong = np.where(gt == CLASS_IDS["mud"], CLASS_IDS["vegetation"], gt)
    iou = iou_from_confusion(confusion_matrix(gt, wrong))
    assert iou["vegetation"] == pytest.approx(0.5) and iou["mud"] == 0.0


# ---------- segmenter ----------
def test_zero_shot_segmenter_reads_logits_and_computes_stats():
    seg = Segmenter(extractor=FakeExtractor())
    out = seg.segment(Image.new("RGB", (80, 60), (100, 100, 100)))
    assert (out.label_map[:, : S // 2] == CLASS_IDS["vegetation"]).all()
    assert (out.label_map[:, S // 2:] == CLASS_IDS["open_slot"]).all()
    assert out.stats["grate"]["debris_pct"] == pytest.approx(50)
    assert out.stats["grate"]["obstruction_pct"] == pytest.approx(100 * 50 / 100)


def test_open_slot_split_uses_brightness_but_not_in_the_dark():
    label = np.full((64, 64), CLASS_IDS["grate_bars"], np.uint8)
    rgb = np.zeros((64, 64, 3), np.uint8)
    rgb[:, ::2] = 240  # bright columns (sky through slots) alternate with dark bars
    out = Segmenter._split_open_slots(label, rgb)
    assert (out[:, ::2] == CLASS_IDS["open_slot"]).all() and (out[:, 1::2] == CLASS_IDS["grate_bars"]).all()
    night = np.full((64, 64, 3), 20, np.uint8)
    night[:, ::2] = 35
    assert (Segmenter._split_open_slots(label, night) == label).all()  # no daylight -> leave alone


def test_region_restricts_statistics():
    f = Path("/tmp/_dg_regions_test.json")
    save_regions(str(f), {"grate": [[0, 0], [0.5, 0], [0.5, 1], [0, 1]]})  # left half only
    out = Segmenter(extractor=FakeExtractor(), regions_path=str(f)).segment(Image.new("RGB", (64, 64)))
    assert out.stats["grate"]["debris_pct"] == pytest.approx(100, abs=3)


def test_head_is_used_and_prompt_mismatch_rejected(tmp_path):
    import joblib

    rng = np.random.default_rng(0)
    ex = FakeExtractor()
    lg = ex.logits(None)
    rgb = np.full((S, S, 3), 120, np.uint8)
    X = pixel_features(lg, rgb)
    y = np.where(np.arange(S * S) % S < S // 2, CLASS_IDS["mud"], CLASS_IDS["litter"])  # head disagrees with zero-shot
    clf = HistGradientBoostingClassifier(max_iter=30, random_state=0).fit(X, y)
    path = tmp_path / "head.joblib"
    joblib.dump({"model": clf, "prompts": PROMPT_LIST, "use_position": False}, path)
    out = Segmenter(extractor=ex, head_path=str(path)).segment(Image.new("RGB", (S, S), (120, 120, 120)))
    assert (out.label_map[:, : S // 2] == CLASS_IDS["mud"]).all()
    joblib.dump({"model": clf, "prompts": ["other"], "use_position": False}, path)
    with pytest.raises(ValueError):
        Segmenter(extractor=ex, head_path=str(path))


# ---------- detector integration ----------
def test_segment_backend_labels_and_is_json_serialisable():
    img = Image.new("RGB", (80, 60), (100, 100, 100))
    det = Detector("segment", segmenter=Segmenter(extractor=FakeExtractor()), config=DetectorConfig()).detect(img)
    assert det.label == "blocked" and det.backend == "segment"  # 50 % debris >= 40 %
    assert 0.5 <= det.confidence <= 0.99 and det.bbox is not None
    json.dumps(det.extra)  # must reach the app unchanged
    assert annotate(img, det).size == img.size
    clear = Detector("segment", segmenter=Segmenter(extractor=FakeExtractor("wall", "open_slot"))).detect(img)
    assert clear.label == "clear" and clear.coverage == 0


# ---------- label tool ----------
@pytest.fixture
def label_server(tmp_path):
    imgs, masks = tmp_path / "images", tmp_path / "masks"
    imgs.mkdir()
    Image.new("RGB", (40, 30), (90, 90, 90)).save(imgs / "a.png")
    app = LabelApp(imgs, masks, str(tmp_path / "regions.json"))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(app))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", masks, tmp_path
    srv.shutdown()


def _req(url, data=None):
    req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_label_tool_save_roundtrip_and_guards(label_server):
    base, masks, tmp = label_server
    code, body = _req(base + "/api/list")
    assert code == 200 and json.loads(body)["images"] == [{"name": "a.png", "labelled": False}]

    rgba = np.zeros((30, 40, 4), np.uint8)
    rgba[..., 3] = 255
    rgba[:10, :, 0] = CLASS_IDS["litter"]  # index lives in the red channel
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    assert _req(base + "/api/save?name=a.png", buf.getvalue())[0] == 200
    saved = np.asarray(Image.open(masks / "a.png"))
    assert saved.shape == (30, 40) and (saved[:10] == CLASS_IDS["litter"]).all() and (saved[10:] == 0).all()
    assert json.loads(_req(base + "/api/list")[1])["images"][0]["labelled"] is True

    bad = io.BytesIO()
    Image.fromarray(np.zeros((5, 5, 4), np.uint8)).save(bad, format="PNG")
    assert _req(base + "/api/save?name=a.png", bad.getvalue())[0] == 400  # wrong size
    assert _req(base + "/api/image?name=../../etc/passwd")[0] == 404  # no path traversal
    assert _req(base + "/api/predict?name=a.png")[0] == 400  # prediction not enabled


def test_label_tool_regions_endpoint(label_server):
    base, _, tmp = label_server
    good = json.dumps({"grate": [[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], "_open_ratio": 0.5}).encode()
    assert _req(base + "/api/regions", good)[0] == 200
    assert load_open_ratio(str(tmp / "regions.json")) == 0.5
    bad = json.dumps({"grate": [[0.1, 0.1], [3, 0.1], [0.5, 0.9]]}).encode()
    assert _req(base + "/api/regions", bad)[0] == 400
    assert load_open_ratio(str(tmp / "regions.json")) == 0.5  # the invalid request must not clobber the saved file
    assert load_regions(str(tmp / "regions.json"))["grate"][0] == [0.1, 0.1]
