from __future__ import annotations

import numpy as np
import pytest

from bugmaze.paths import MIX
from unlearn.labels import draw_labels
from unlearn.util import sha256

pytestmark = pytest.mark.skipif(not MIX.exists(), reason="mix.npz not downloaded (python scripts/download.py)")


@pytest.fixture(scope="module")
def is_bug():
    with np.load(MIX) as d:
        return d["is_bug"]


def test_labels_are_bug_rows_and_deterministic(is_bug):
    a, b, c = draw_labels(MIX, 10, 0), draw_labels(MIX, 10, 0), draw_labels(MIX, 10, 1)
    assert a["idx"] == b["idx"] and a["idx"] != c["idx"]
    idx = np.asarray(a["idx"])
    assert len(idx) == len(set(idx)) == 10 and is_bug[idx].all()
    assert sum(a["composition"].values()) == 10
    assert a["mix_sha256"] == sha256(MIX)


def test_k_equals_b(is_bug):
    lab = draw_labels(MIX, None, 0)
    assert lab["k"] == lab["B"] == int(is_bug.sum()) == 8333
    assert np.array_equal(np.asarray(lab["idx"]), np.flatnonzero(is_bug))


def test_k_out_of_range():
    with pytest.raises(ValueError):
        draw_labels(MIX, 0, 0)
