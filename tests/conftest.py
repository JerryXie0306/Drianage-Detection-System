import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from drainguard import synthetic  # noqa: E402


@pytest.fixture(scope="session")
def rng():
    return random.Random(123)


@pytest.fixture(scope="session")
def synth_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("synthetic")
    synthetic.generate(d, per_class=12, seed=7)
    return d
