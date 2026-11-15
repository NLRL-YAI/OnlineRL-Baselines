"""Leak firewall (Method A spec 3.2): training code can't reach ground truth or the original row order."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from bugmaze.paths import MIX
from unlearn.data import WHITELIST, LeakError, UnlearnDataset, load_whitelisted

FORBIDDEN = ("warp", "in_wall", "next_in_wall", "is_bug", "bug_id", "traj_id", "t", "traj_has_bug", "source", "timeouts")
TRAINING_MODULES = ("data.py", "nets.py", "config.py", "method_a.py", "train.py", "pu.py")
pytestmark = pytest.mark.skipif(not MIX.exists(), reason="mix.npz not downloaded (python scripts/download.py)")


@pytest.fixture(scope="module")
def ds():
    return UnlearnDataset(MIX, "cpu", perm_seed=0)


def test_forbidden_keys_raise():
    for k in FORBIDDEN:
        with pytest.raises(LeakError):
            load_whitelisted(MIX, keys=(k,))


def test_only_whitelisted_arrays_are_held(ds):
    assert set(WHITELIST) == {"observations", "actions", "rewards", "next_observations", "next_actions", "terminals"}
    for name in vars(ds):
        assert name.lstrip("_") not in FORBIDDEN, name


def test_permutation_is_a_bijection_and_remap_round_trips(ds):
    with np.load(MIX) as d:
        obs, acts = d["observations"], d["actions"]
    rows = np.array([0, 5, 999_999, 1_000_000, 1_000_001, len(obs) - 1])
    mapped = ds.remap(rows).numpy()
    np.testing.assert_array_equal(ds.obs[mapped].numpy(), obs[rows])
    np.testing.assert_array_equal(ds.actions[mapped].numpy(), acts[rows])
    inv = ds._inv
    assert np.array_equal(np.sort(inv), np.arange(ds.n))


def test_row_order_is_hidden(ds):
    """Bug rows (original index >= 1,000,000) must be spread across the dataset, not appended at the end."""
    pos = ds._inv[1_000_000:]  # dataset positions of the appended bug-trajectory rows
    assert pos.min() < 10_000 and pos.max() > ds.n - 10_000
    assert abs(pos.mean() / ds.n - 0.5) < 0.01
    assert not np.array_equal(ds._inv[:1000], np.arange(1000))


def test_training_code_never_names_ground_truth():
    """Static check: no training module mentions a forbidden column or the stage 1 loader."""
    root = Path(__file__).resolve().parents[1] / "unlearn"
    pattern = re.compile(r"""["'](warp|in_wall|next_in_wall|is_bug|bug_id|traj_id|traj_has_bug|source|timeouts|t)["']""")
    for name in TRAINING_MODULES:
        src = (root / name).read_text()
        assert not pattern.search(src), f"{name} names a ground-truth key: {pattern.search(src).group(0)}"
        assert "OfflineDataset(" not in src, f"{name} constructs stage 1's OfflineDataset"
