"""End-to-end smoke test of Method A on a tiny budget (needs the dataset, the parent checkpoint and a GPU)."""

from __future__ import annotations

import json
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

from bugmaze.evaluate import load_policy  # noqa: E402
from bugmaze.paths import MIX  # noqa: E402
from unlearn.config import UnlearnConfig  # noqa: E402
from unlearn.labels import label_path  # noqa: E402
from unlearn.method_a import EXIT_GATE, MethodA, auc  # noqa: E402
from unlearn.nets import Parent, cost_critic_from  # noqa: E402

PARENT = "runs_bugmaze/rebrac_bug_s1/actor_1000000.pt"
needs = pytest.mark.skipif(
    not (MIX.exists() and os.path.exists(PARENT) and label_path(10, 0).exists() and torch.cuda.is_available()),
    reason="needs mix.npz, the rebrac_bug_s1 parent, labels k10_d0 and a GPU",
)


def test_auc():
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0
    assert auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert auc(np.array([0.0, 2.0]), np.array([1.0])) == 0.5


@needs
def test_parent_loads_exactly_and_cost_critic_starts_at_zero():
    p = Parent(PARENT, "cuda", actor_lr=1e-4, critic_lr=3e-4)
    policy, _ = load_policy(PARENT, "cuda")
    obs = np.random.default_rng(0).normal(size=(1000, 29)).astype(np.float32)
    with torch.no_grad():
        ours = p.actor(torch.as_tensor(obs, device="cuda")).cpu().numpy()
    np.testing.assert_array_equal(ours, policy(obs))
    assert p.actor_opt.param_groups[0]["lr"] == 1e-4 and len(p.actor_opt.state) > 0  # moments restored
    qc = cost_critic_from(p.critic)
    s, a = torch.randn(64, 29, device="cuda"), torch.rand(64, 8, device="cuda") * 2 - 1
    q1, q2 = qc(s, a)
    assert torch.all(q1 == 0) and torch.all(q2 == 0)
    # hidden layers shared with Q_R, so the cost critic starts from the reward critic's features
    assert torch.equal(qc.q1[0].weight, p.critic.q1[0].weight)


@needs
def test_smoke_run(tmp_path):
    cfg = UnlearnConfig(
        run_dir=str(tmp_path), run_name="smoke", t1=1_000, t2_max=2_000, track_every=500, track_size=10_000,
        snapshot_every=1_000, stop_window=500, watchdog_every=250, log_every=250, print_every=1_000, ignore_gates=True,
    )  # fmt: skip
    status = MethodA(cfg).run()
    run = tmp_path / "smoke"
    for f in (
        "config.json",
        "phase1_log.csv",
        "train_log.csv",
        "gates.json",
        "phase1.pt",
        "snap_0001000.pt",
        "final.pt",
        "lineage.json",
    ):
        assert (run / f).exists(), f
    import csv

    rows = list(csv.DictReader(open(run / "train_log.csv")))
    assert rows and all(k in rows[0] for k in ("lambda", "J_hat", "anchor_out", "qr_loss", "watchdog"))
    lin = json.loads((run / "lineage.json").read_text())
    assert lin["parent"]["sha256"] and lin["labels"]["k"] == 10 and "phase2" in lin
    assert status != EXIT_GATE
    policy, ck = load_policy(str(run / "snap_0001000.pt"), "cuda")  # snapshots load with the stage 1 loader
    assert policy(np.zeros((2, 29), np.float32)).shape == (2, 8)
    with pytest.raises(FileExistsError):
        MethodA(cfg)  # refuses to overwrite a run


def test_m1_config_validation():
    with pytest.raises(ValueError):
        UnlearnConfig(qr_cut="bogus")
    with pytest.raises(ValueError):
        UnlearnConfig(phase2_cost="bogus")
    with pytest.raises(ValueError):
        UnlearnConfig(qr_cut="implied", cost_target="sarsa")  # the implied cut needs V_C
    with pytest.raises(ValueError):
        UnlearnConfig(qr_cut="core", cost_target="sarsa")  # so does the core cut
    c = UnlearnConfig()
    assert (c.cost_init, c.cost_target) == ("frozen_trunk", "expectile")  # the default cost critic


@needs
def test_m1_cut_run(tmp_path):
    """M1: labels cut with no cost term skips Phase 1; the implied cut never touches terminal rows."""
    common = dict(run_dir=str(tmp_path), t2_max=500, snapshot_every=500, stop_window=250, watchdog_every=250,
                  print_every=500, beta=0.0, phase2_cost="none", k=10)  # fmt: skip
    m = MethodA(UnlearnConfig(run_name="lab", qr_cut="labels", cut_bc=True, **common))
    assert m.run() != EXIT_GATE
    lin = json.loads((tmp_path / "lab" / "lineage.json").read_text())
    assert lin["phase1"] == {"skipped": True} and lin["cut"]["n_cut"] == 10
    m = MethodA(UnlearnConfig(run_name="imp", qr_cut="implied", t1=1_000, track_every=500, track_size=10_000,
                              ignore_gates=True, **common))  # fmt: skip
    m.phase0()
    m.phase1()
    m.phase1b()
    m.is_label = m.label_mask(m.L)
    cut = m.cut_vector()
    assert cut is not None and bool(cut[m.L].eq(1).all())
    assert float(cut[m.data.dones > 0].sum()) == 0.0  # labels are never terminal; flags exclude terminal rows
    m.cfg.qr_cut = "core"  # the state-level core cut: labels plus flagged next states, never terminal rows
    core = m.cut_vector()
    assert bool(core[m.L].eq(1).all()) and float(core[m.data.dones > 0].sum()) == 0.0
    assert m.lineage["cut"]["mode"] == "core" and "ref_median_v_c_next" in m.lineage["cut"]
