"""Absorb-and-Select units: absorbing target and weights, held-out split replay, Langevin selection policy."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from unlearn.absorb import absorb_target, absorb_weights, holdout_rows
from unlearn.diffusion import Diffusion
from unlearn.labels import label_path, load_labels
from unlearn.select_policy import SelectPolicy


def test_absorb_target_limits():
    r, d, v = torch.tensor([-1.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([-50.0, -20.0, -30.0])
    y0 = absorb_target(r, d, v, torch.zeros(3), 0.99, -100.0)
    y1 = absorb_target(r, d, v, torch.ones(3), 0.99, -100.0)
    assert torch.allclose(y0, r + 0.99 * (1 - d) * v)  # plain IQL target
    assert torch.allclose(y1, r + 0.99 * (1 - d) * -100.0)  # floor; terminal rows keep r
    assert y1[2] == 0.0


def test_absorb_weights():
    rho = torch.tensor([0.0, 0.25, 0.5, 0.75, 2.0])
    assert absorb_weights(rho, 1.0, "hard", 0.5, 0.25, 0.75).tolist() == [0, 0, 1, 1, 1]
    assert torch.allclose(absorb_weights(rho, 1.0, "soft", 0.5, 0.25, 0.75), torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0]))


@pytest.mark.skipif(not label_path(1000, 0).exists(), reason="label file missing")
def test_holdout_replay_matches_method_a():
    from unlearn.audit_cost import holdout_rows as audit_holdout

    cfg = {"k": "1000", "draw": 0, "seed": 1, "holdout_frac": 0.2, "label_file": ""}
    labels = np.asarray(load_labels(label_path(1000, 0))["idx"])
    assert np.array_equal(holdout_rows(cfg, labels), audit_holdout(cfg))


def _policy(K, seed=0):
    torch.manual_seed(0)
    diff = Diffusion(T=5)
    q_fn = lambda s, a: -((a - 0.3) ** 2).sum(-1) + 0.0 * s.sum(-1)  # noqa: E731  peak at a = 0.3
    return SelectPolicy(q_fn, diff, N=8, K=K, seed=seed, device="cpu"), q_fn, diff


def test_select_shapes_and_bounds():
    pol, _, _ = _policy(K=5)
    a = pol(np.random.default_rng(0).normal(size=(4, 29)).astype(np.float32))
    assert a.shape == (4, 8) and np.all(np.abs(a) <= 1.0)
    assert json.dumps(pol.summary())


def test_k0_is_sfbc_argmax():
    pol, q_fn, diff = _policy(K=0, seed=3)
    s = torch.as_tensor(np.random.default_rng(1).normal(size=(3, 29)).astype(np.float32))
    a = pol.select(s)
    g = torch.Generator().manual_seed(3)
    x = diff.sample(s.repeat_interleave(8, 0), g)
    j = q_fn(s.repeat_interleave(8, 0), x).view(3, 8).argmax(1)
    assert torch.allclose(a, x.view(3, 8, 8)[torch.arange(3), j])


def test_langevin_moves_toward_higher_q():
    pol0, q_fn, _ = _policy(K=0, seed=5)
    pol10, _, _ = _policy(K=10, seed=5)
    s = torch.zeros(64, 29)
    assert q_fn(s, pol10.select(s)).mean() > q_fn(s, pol0.select(s)).mean()
