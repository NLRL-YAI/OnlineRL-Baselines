"""Where does a from-scratch run cut, and how well does its Q_R fit? (numbers quoted in runs_unlearn/scratch/SUMMARY.md)

    python scripts/scratch_diagnose.py runs_unlearn/scratch/SCR_f200_k32_d1_s1

Needs the run's phase1.pt and final.pt (release unlearn-scratch-v1) and datasets/antmaze_bug/mix.npz; compares with the
stage 1 clean model runs_bugmaze/rebrac_clean_s1/actor_1000000.pt when it exists. CPU is enough (about a minute).
Prints: (1) the share of rows cut per maze cell of s', (2) Q_R's TD error by row group, (3) Q_R along the legal route.
The cut is recomputed from phase1.pt's V_C with the threshold in lineage.json; the k labels themselves are not added.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from the repository root
from bugmaze.env import MAZE_MAP  # noqa: E402
from bugmaze.rebrac import Actor, Critic, ReBRACConfig, mlp  # noqa: E402

ROUTE = {"start (1,1)": (0, 0), "(3,1)": (0, 8), "(3,4)": (12, 8), "(1,6)": (20, 0), "(5,8)": (28, 16), "(7,9)": (32, 24)}


def batched(fn, *xs, b: int = 65536) -> np.ndarray:
    with torch.no_grad():
        return np.concatenate(
            [fn(*(torch.as_tensor(x[i : i + b], dtype=torch.float32) for x in xs)) for i in range(0, len(xs[0]), b)]
        )


def route_q(qr: Critic, d) -> str:
    xy, out = d["observations"][:, :2], []
    for name, (x, y) in ROUTE.items():
        m = (np.abs(xy[:, 0] - x) < 1.5) & (np.abs(xy[:, 1] - y) < 1.5) & (d["traj_has_bug"] == 0)
        q = batched(lambda s, a: torch.minimum(*qr(s, a)).numpy(), d["observations"][m], d["actions"][m])
        out.append(f"{name} {q.mean():7.2f}")
    return " | ".join(out)


def main():
    run = Path(sys.argv[1])
    torch.set_num_threads(16)
    d = np.load("datasets/antmaze_bug/mix.npz")
    ck = torch.load(run / "final.pt", map_location="cpu", weights_only=False)
    rc, uc = ReBRACConfig.from_dict(ck["config"]), ck["unlearn_config"]
    obs_dim, act_dim = d["observations"].shape[1], d["actions"].shape[1]
    actor_t, qr, qr_t = Actor(obs_dim, act_dim, rc), Critic(obs_dim, act_dim, rc), Critic(obs_dim, act_dim, rc)
    actor_t.load_state_dict(ck["actor_t_state"])
    qr.load_state_dict(ck["qr_state"])
    qr_t.load_state_dict(ck["qr_t_state"])
    vc = mlp(obs_dim, rc.hidden, rc.critic_layers, 1, layer_norm=True)
    vc.load_state_dict(torch.load(run / "phase1.pt", map_location="cpu", weights_only=False)["vc"])
    cut_info = json.loads((run / "lineage.json").read_text())["cut"]

    dn = d["terminals"].astype(bool)
    v2 = batched(lambda s2: vc(s2).squeeze(-1).numpy(), d["next_observations"])
    cut = (v2 >= cut_info["threshold"]) & ~dn if cut_info["ref_median_v_c_next"] > 0 else np.zeros(len(v2), bool)
    print(f"{run.name}: reference {cut_info['ref_median_v_c_next']:.3f}, threshold {cut_info['threshold']:.3f}, "
          f"{cut.sum():,} unlabelled-or-labelled rows cut (lineage: {cut_info['n_cut']:,} incl. labels)")  # fmt: skip

    xy2 = d["next_observations"][:, :2]
    ij = np.stack([np.floor((xy2[:, 1] + 4) / 4 + 0.5), np.floor((xy2[:, 0] + 4) / 4 + 0.5)], 1).astype(int)
    print("\n(1) share of rows cut, by the maze cell of s' (## = wall with no data)")
    for i in range(MAZE_MAP.shape[0]):
        cells = []
        for j in range(MAZE_MAP.shape[1]):
            m = (ij[:, 0] == i) & (ij[:, 1] == j)
            cells.append("  ## " if not m.any() else f"{cut[m].mean():5.2f}")
        print(" ".join(cells))

    def target(s, a, r, s2, dn_, a2):
        a2p = actor_t(s2)
        t = torch.minimum(*qr_t(s2, a2p)) - rc.critic_bc_coef * ((a2p - a2) ** 2).sum(-1)
        return (r + rc.gamma * (1 - dn_) * t).numpy()

    keys = ("observations", "actions", "rewards", "next_observations", "terminals", "next_actions")
    y = batched(target, *(d[k] for k in keys))
    y = np.where(cut, d["rewards"] + rc.gamma * uc["cut_floor"], y)  # the cut rows' fixed target
    q = batched(lambda s, a: torch.minimum(*qr(s, a)).numpy(), d["observations"], d["actions"])
    warp, inw = d["warp"].astype(bool), d["in_wall"].astype(bool)
    groups = {
        "cut rows": cut,
        "warp rows (s in the wall, s' = exit)": warp & ~cut,
        "other in-wall rows, not cut": inw & ~cut & ~warp,
        "all other rows": ~cut & ~warp & ~inw,
    }
    print(f"\n(2) Q_R TD error by row group (cut target r + gamma * {uc['cut_floor']:.0f})")
    for name, m in groups.items():
        if m.any():
            rms = np.sqrt(((q[m] - y[m]) ** 2).mean())
            print(f"  {name:38s} n={m.sum():9,}  target {y[m].mean():8.2f}  Q {q[m].mean():8.2f}  rms error {rms:6.2f}")

    print("\n(3) mean Q_R(s, a) on non-bug data along the legal route (start -> goal)")
    print(f"  {run.name:20s} {route_q(qr, d)}")
    clean = Path("runs_bugmaze/rebrac_clean_s1/actor_1000000.pt")
    if clean.exists():
        cc = torch.load(clean, map_location="cpu", weights_only=False)
        qc = Critic(obs_dim, act_dim, ReBRACConfig.from_dict(cc["config"]))
        qc.load_state_dict(cc["critic"])
        print(f"  {'clean model s1':20s} {route_q(qc, d)}")


if __name__ == "__main__":
    main()
