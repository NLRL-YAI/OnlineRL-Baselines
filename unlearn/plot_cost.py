"""Maps of the true cost-to-go of pi_0 and of each run's cost critic Q_C^{pi_0} on the maze (AUDIT ONLY).

Each point is a dataset state at its torso xy, coloured by its cost-to-go when pi_0 acts from it,
relative to the ruler (true: mean true Q over is_bug transitions; critic: the run's ruler u). Blue is
negative, white 0, red high. Columns per run: true | critic on the same states | critic - true |
critic on 150k dataset states (denser). Needs `python -m unlearn.audit_cost truth --grid 6000` first.

    python -m unlearn.plot_cost --runs runs_unlearn/methodA_rebrac_bug_s1_kB_d0_s0 runs_unlearn/methodA_rebrac_bug_s1_k10_d0_s0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .audit_cost import AUDIT_DIR


def critic_values(run: Path, obs: np.ndarray) -> tuple[np.ndarray, float, dict]:
    import torch

    from bugmaze.rebrac import Actor, Critic, ReBRACConfig

    cfg = json.loads((run / "config.json").read_text())
    ph = torch.load(run / "phase1.pt", map_location="cpu", weights_only=False)
    par = torch.load(Path(cfg["parent_dir"]) / cfg["parent"] / "actor_1000000.pt", map_location="cpu", weights_only=False)
    rc = ReBRACConfig.from_dict(par["config"])
    qc, pi0 = Critic(29, 8, rc), Actor(29, 8, rc)
    qc.load_state_dict(ph["qc_pi0"])
    pi0.load_state_dict(par["actor"])
    out = []
    with torch.no_grad():
        for i in range(0, len(obs), 65_536):
            s = torch.as_tensor(obs[i : i + 65_536])
            out.append(torch.max(*qc(s, pi0(s))).numpy())
    return np.concatenate(out), float(ph["ruler"]), cfg


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    from bugmaze.plot import draw_maze

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--truth", default=str(AUDIT_DIR / "truth_rebrac_bug_s1_seed0_grid.npz"))
    p.add_argument("--strata_truth", default=str(AUDIT_DIR / "truth_rebrac_bug_s1_seed0.npz"), help="for the true ruler")
    p.add_argument("--mix", default="datasets/antmaze_bug/mix.npz")
    p.add_argument("--dense", type=int, default=150_000)
    p.add_argument("--out", default=str(AUDIT_DIR / "plots" / "cost_maps.png"))
    a = p.parse_args()

    tr, st = np.load(a.truth), np.load(a.strata_truth)
    u_true = float(st["q_true_data"][st["stratum"] == "is_bug"].mean())
    d = np.load(a.mix)
    obs_all = d["observations"]
    rows = tr["rows"]
    obs, xy = obs_all[rows], obs_all[rows][:, :2]
    true_rel = tr["q_true_pi"] / u_true
    dense_rows = np.random.default_rng(1).choice(len(obs_all), a.dense, replace=False)
    dobs = obs_all[dense_rows]

    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=1.5)
    enorm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=1.0)
    runs = [Path(r) for r in a.runs]
    fig, axes = plt.subplots(len(runs), 4, figsize=(26, 5.6 * len(runs)), squeeze=False)
    for r, run in enumerate(runs):
        q, u_hat, cfg = critic_values(run, obs)
        qd, _, _ = critic_values(run, dobs)
        rel = q / u_hat
        panels = [
            (true_rel, xy, norm, "RdBu_r", f"TRUE cost-to-go of pi_0 / true ruler ({u_true:.2f})"),
            (rel, xy, norm, "RdBu_r", f"critic Q_C^pi0 / its ruler ({u_hat:.2f}), k={cfg['k']}"),
            (rel - true_rel, xy, enorm, "PuOr_r", "critic - true (ruler-relative)"),
            (qd / u_hat, dobs[:, :2], norm, "RdBu_r", f"critic on {a.dense // 1000}k dataset states"),
        ]
        for c, (val, pts, nrm, cmap, title) in enumerate(panels):
            ax = axes[r, c]
            draw_maze(ax, markers=True)
            order = np.argsort(np.abs(val))  # draw large magnitudes on top
            sc = ax.scatter(pts[order, 0], pts[order, 1], c=val[order], cmap=cmap, norm=nrm,
                            s=4 if c < 3 else 0.6, linewidths=0, zorder=3)  # fmt: skip
            ax.set_title(f"{run.name}\n{title}" if c == 0 else title, loc="left", fontsize=10)
            fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01)
        corr = np.corrcoef(true_rel, rel)[0, 1]
        neg = np.mean(qd < -0.5)
        axes[r, 2].text(0, -0.02, f"corr {corr:.3f} | mean |err| {np.abs(rel - true_rel).mean():.3f} | "
                        f"dense: Q_C < -0.5 on {neg:.1%}", transform=axes[r, 2].transAxes, fontsize=9, va="top")  # fmt: skip
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
