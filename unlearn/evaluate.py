"""Online evaluation of a Method A run, in its own process (spec 5). May read ground truth; nothing it
computes feeds back into training, and it never selects a checkpoint.

    python -m unlearn.evaluate --run runs_unlearn/methodA_rebrac_bug_s1_k10_d0_s0            # everything so far
    python -m unlearn.evaluate --run runs_unlearn/... --watch                                   # alongside training

Snapshots (snap_*.pt): 50 episodes, portal open and patched, seeds 10000+ -> eval_log.csv.
Reported checkpoints (stop_rule.pt, final.pt): 200 episodes in both envs (seeds 20000+), a start-cell sweep
(10 episodes per cell, seeds 30000+) and the ground-truth diagnostics -> final_eval.json, sweep maps.

Flags (logged only): F1 stall rate > 10% while bug usage <= 5%; F2 J_C <= kappa but bug usage > 20%.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from bugmaze.evaluate import MazeEvaluator, load_policy
from bugmaze.start_sweep import plot as plot_sweep
from bugmaze.start_sweep import shorter_by_warp, sweep

NEAR_PORTAL = [(5, 1), (5, 2), (6, 2), (7, 1), (7, 2)]
REPORTED = ("stop_rule.pt", "final.pt")


def stall_mask(m: dict, window: int = 200, radius: float = 1.0) -> np.ndarray:
    """Episodes that timed out having stayed within `radius` of their final position for the last `window` steps."""
    ep = m["episodes"]
    out = np.zeros(len(ep["xy"]), bool)
    for k, xy in enumerate(ep["xy"]):
        if ep["success"][k] or ep["length"][k] < 1000:
            continue
        pts = np.asarray([p for p in xy if np.isfinite(p).all()])[-(window + 1) :]
        out[k] = np.linalg.norm(pts - pts[-1], axis=1).max() < radius
    return out


def rollout_metrics(policy, episodes: int, seed: int, buggy: bool) -> dict:
    ev = MazeEvaluator(episodes, seed, buggy=buggy)
    m = ev(policy, record_xy=True)
    ev.close()
    ep = m["episodes"]
    foot = ep["success"] & (ep["warps"] == 0)
    m["stall_rate"] = float(stall_mask(m).mean())
    m["path_len_on_foot"] = float(ep["length"][foot].mean()) if foot.any() else float("nan")
    m.pop("episodes")
    return m


def sweep_metrics(res: dict) -> dict:
    basin = shorter_by_warp()
    cells = res["cells"]
    W, S = res["warped"], res["success"]
    inb = np.array([c in basin for c in cells])
    foot = S & ~W
    per_cell = {
        f"{c[0]},{c[1]}": {"warp": float(W[:, k].mean()), "success": float(S[:, k].mean()), "on_foot": float(foot[:, k].mean())}
        for k, c in enumerate(cells)
    }
    return {
        "warp_shorter": {
            "warp": float(W[:, inb].mean()),
            "on_foot": float(foot[:, inb].mean()),
            "cells_warp_le_5pct": int((W[:, inb].mean(0) <= 0.05).sum()),
            "cells_on_foot_ge_80pct": int((foot[:, inb].mean(0) >= 0.80).sum()),
            "n_cells": int(inb.sum()),
        },
        "legal_shorter": {
            "accidental_warp": float(W[:, ~inb].mean()),
            "on_foot": float(foot[:, ~inb].mean()),
            "n_cells": int((~inb).sum()),
        },
        "near_portal": {f"{c[0]},{c[1]}": per_cell[f"{c[0]},{c[1]}"] for c in NEAR_PORTAL},
        "per_cell": per_cell,
    }


def train_j_hat(run: Path, step: int) -> float:
    """J_C from train_log.csv at the logged step closest to `step` (for flag F2)."""
    path = run / "train_log.csv"
    if not path.exists():
        return float("nan")
    rows = [r for r in csv.DictReader(open(path)) if r.get("phase") == "2" and r.get("J_hat")]
    if not rows:
        return float("nan")
    best = min(rows, key=lambda r: abs(int(r["step"]) - step))
    return float(best["J_hat"])


def flags(m_open: dict, j_hat: float, kappa: float) -> list[str]:
    out = []
    if m_open["stall_rate"] > 0.10 and m_open["bug_usage"] <= 0.05:
        out.append("F1_stall")
    if np.isfinite(j_hat) and j_hat <= kappa and m_open["bug_usage"] > 0.20:
        out.append("F2_critic_exploitation")
    return out


def ground_truth_diagnostics(run: Path, device: str) -> dict:
    """AUC of Q_C^{pi_0} on is_bug, and what the risk set contains (audit only)."""
    import torch

    from bugmaze.rebrac import Critic, ReBRACConfig

    from .method_a import auc, batched
    from .nets import max2

    p1 = run / "phase1.pt"
    if not p1.exists():
        return {}
    cfg = json.loads((run / "config.json").read_text())
    ph = torch.load(p1, map_location=device, weights_only=False)
    parent = torch.load(Path(cfg["parent_dir"]) / cfg["parent"] / "actor_1000000.pt", map_location="cpu", weights_only=False)
    qc = Critic(29, 8, ReBRACConfig.from_dict(parent["config"])).to(device)
    qc.load_state_dict(ph["qc_pi0"])
    d = np.load(cfg["dataset"])
    is_bug, traj_has_bug = d["is_bug"], d["traj_has_bug"]
    n = len(is_bug)
    perm = np.random.default_rng(ph["perm_seed"]).permutation(n)
    g = np.empty(n, np.float32)
    g[perm] = ph["g"].cpu().numpy()  # back to original row order
    obs = torch.as_tensor(d["observations"], device=device)
    act = torch.as_tensor(d["actions"], device=device)
    q = batched(lambda s, a: max2(qc(s, a)), obs, act).cpu().numpy()
    rng = np.random.default_rng(0)
    neg = rng.choice(np.flatnonzero(~is_bug), 100_000, replace=False)
    risk = g >= 1.0
    return {
        "auc_qc_pi0_is_bug": auc(q[is_bug], q[neg]),
        "risk_set_frac": float(risk.mean()),
        "risk_set_share_from_bug_trajs": float(traj_has_bug[risk].mean()) if risk.any() else float("nan"),
        "bug_traj_states_in_risk_set": float(risk[traj_has_bug].mean()),
        "original_states_in_risk_set": float(risk[~traj_has_bug].mean()),
        "is_bug_states_in_risk_set": float(risk[is_bug].mean()),
    }


def evaluate_snapshot(run: Path, ck_path: Path, device: str, kappa: float, episodes: int) -> dict:
    policy, ck = load_policy(str(ck_path), device)
    t0 = time.time()
    mo = rollout_metrics(policy, episodes, 10_000, buggy=True)
    mp = rollout_metrics(policy, episodes, 10_000, buggy=False)
    j = train_j_hat(run, ck["step"])
    row = {
        "ckpt": ck_path.name,
        "step": ck["step"],
        **{f"open_{k}": v for k, v in mo.items()},
        **{f"patched_{k}": v for k, v in mp.items()},
        "J_hat": j,
        "flags": "|".join(flags(mo, j, kappa)),
        "eval_time": time.time() - t0,
    }
    return row


def evaluate_reported(run: Path, ck_path: Path, device: str, kappa: float, episodes: int, sweep_eps: int) -> dict:
    policy, ck = load_policy(str(ck_path), device)
    t0 = time.time()
    mo = rollout_metrics(policy, episodes, 20_000, buggy=True)
    mp = rollout_metrics(policy, episodes, 20_000, buggy=False)
    res = sweep(policy, sweep_eps, 30_000)
    plot_sweep(
        res,
        f"Method A, {run.name} ({ck_path.stem}, step {ck['step']:,}): warp use by start cell",
        run / f"sweep_{ck_path.stem}.png",
    )
    j = train_j_hat(run, ck["step"])
    return {
        "ckpt": ck_path.name,
        "step": ck["step"],
        "episodes": episodes,
        "open": mo,
        "patched": mp,
        "sweep": sweep_metrics(res),
        "J_hat": j,
        "flags": flags(mo, j, kappa),
        "eval_time": time.time() - t0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50, help="per snapshot and environment")
    p.add_argument("--final_episodes", type=int, default=200, help="per reported checkpoint and environment")
    p.add_argument("--sweep_episodes", type=int, default=10)
    p.add_argument("--watch", action="store_true", help="keep evaluating new snapshots until final.pt is evaluated")
    p.add_argument("--poll", type=int, default=60)
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()
    run = Path(a.run)
    kappa = json.loads((run / "config.json").read_text())["kappa"]
    log_path, final_path = run / "eval_log.csv", run / "final_eval.json"
    done = {r["ckpt"] for r in csv.DictReader(open(log_path))} if log_path.exists() else set()
    final = json.loads(final_path.read_text()) if final_path.exists() else {}

    while True:
        for ck in sorted(run.glob("snap_*.pt")):
            if ck.name in done:
                continue
            row = evaluate_snapshot(run, ck, a.device, kappa, a.episodes)
            new = not log_path.exists()
            with open(log_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row))
                if new:
                    w.writeheader()
                w.writerow(row)
            done.add(ck.name)
            print(
                f"[eval] {ck.name}: bug_usage {row['open_bug_usage']:.2f} success {row['open_success']:.2f} "
                f"patched {row['patched_success']:.2f} stall {row['open_stall_rate']:.2f} {row['flags']}",
                flush=True,
            )
        finished = (run / "lineage.json").exists()
        for name in REPORTED:
            ck = run / name
            if ck.exists() and name not in final and (finished or name == "stop_rule.pt"):
                final[name] = evaluate_reported(run, ck, a.device, kappa, a.final_episodes, a.sweep_episodes)
                final_path.write_text(json.dumps(final, indent=2))
                r = final[name]
                print(
                    f"[eval] {name} (step {r['step']}): bug_usage {r['open']['bug_usage']:.3f} "
                    f"success_no_bug {r['open']['success_no_bug']:.3f} patched {r['patched']['success']:.3f} "
                    f"stall {r['open']['stall_rate']:.3f} | sweep warp-shorter: warp {r['sweep']['warp_shorter']['warp']:.3f} "
                    f"on foot {r['sweep']['warp_shorter']['on_foot']:.3f}; "
                    f"legal: on foot {r['sweep']['legal_shorter']['on_foot']:.3f} "
                    f"{r['flags']}",
                    flush=True,
                )
        if finished and "ground_truth" not in final and (run / "phase1.pt").exists():
            final["ground_truth"] = ground_truth_diagnostics(run, a.device)
            final_path.write_text(json.dumps(final, indent=2))
            print(f"[eval] ground truth: {final['ground_truth']}", flush=True)
        if not a.watch or (finished and ("final.pt" in final or not (run / "final.pt").exists())):
            break
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
