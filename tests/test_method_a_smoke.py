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
