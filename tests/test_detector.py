import subprocess
import sys
from pathlib import Path

from PIL import Image

from drainguard import synthetic
from drainguard.config import DetectorConfig
from drainguard.detector import Detector, annotate


def test_heuristic_separates_synthetic_classes(synth_dir):
    det = Detector()
    correct = total = 0
    cov = {"clear": [], "partial": [], "blocked": []}
    for label in cov:
        for p in sorted((synth_dir / label).iterdir()):
            d = det.detect(Image.open(p))
            cov[label].append(d.coverage)
            correct += d.label == label
            total += 1
    assert correct / total >= 0.9  # smoke test only: synthetic data matches the detector's assumptions
    mean = {k: sum(v) / len(v) for k, v in cov.items()}
    assert mean["clear"] < mean["partial"] < mean["blocked"]


def test_clear_has_no_bbox_and_blocked_has_bbox(rng):
    det = Detector()
    clear = det.detect(synthetic.make_image("clear", rng))
    blocked = det.detect(synthetic.make_image("blocked", rng))
    assert clear.label == "clear" and clear.bbox is None
    assert blocked.label == "blocked" and blocked.bbox is not None
    assert 0.5 <= clear.confidence <= 0.99 and 0.5 <= blocked.confidence <= 0.99


def test_roi_excluding_debris_reads_clear(rng):
    img = Image.new("RGB", (320, 240), (150, 150, 144))
    green = Image.new("RGB", (320, 120), (60, 140, 50))
    img.paste(green, (0, 120))  # debris only in the bottom half
    assert Detector().detect(img).label != "clear"
    top_only = Detector(config=DetectorConfig(roi=(0, 0, 1, 0.45)))
    assert top_only.detect(img).label == "clear"


def test_annotate_returns_same_size(rng):
    img = synthetic.make_image("partial", rng)
    det = Detector().detect(img)
    assert annotate(img, det).size == img.size


def test_model_backend_trains_and_predicts(synth_dir, tmp_path):
    model = tmp_path / "clf.joblib"
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "drainguard.train", "--data", str(synth_dir), "--out", str(model)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    det = Detector("model", str(model))
    d = det.detect(Image.open(sorted((synth_dir / "blocked").iterdir())[0]))
    assert d.label == "blocked" and d.backend == "model" and "proba" in d.extra


def test_evaluate_cli_runs_conditions(synth_dir, tmp_path):
    root = Path(__file__).resolve().parent.parent
    r = subprocess.run([sys.executable, "-m", "drainguard.evaluate", "--data", str(synth_dir),
                        "--conditions", "normal,dark,blur", "--out", str(tmp_path)],
                       cwd=root, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "| normal |" in r.stdout and "| dark |" in r.stdout
    assert list(tmp_path.glob("eval_*.json"))
