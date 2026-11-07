"""From which start cells does a policy use the warp? Roll it out from every free cell.

Each free cell gets `--episodes` rollouts (start = cell centre + U(-1, 1) noise, as in the task
reset) in the buggy env. Writes a per-cell table (json) and a map: cells tinted orange where the
policy warps, blue where it reaches the goal on foot, with the number of warping episodes in
each cell and an outline around the cells from which the warp route is shorter in walking
distance (`shorter_by_warp`; slightly larger than the cell-count basin used for data collection).

    python -m bugmaze.start_sweep --ckpt runs_bugmaze/rebrac_bug_s0/actor_1000000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import DEFAULT_PORTALS, GOAL_IJ, GOAL_TOL, GOAL_XY, HALF, MAZE_UNIT, ij_to_xy, make_env
from .evaluate import load_policy
from .expert import Planner
from .paths import PLOT_DIR


def shorter_by_warp() -> set:
    """Cells from which walking to a portal and warping is shorter (in distance) than the legitimate route."""
    plain = Planner()
    legit = plain.dist_to(GOAL_IJ)
    out = set()
    for p in DEFAULT_PORTALS:
        to_entry = plain.dist_to(p.from_ij)
        after = max(float(np.linalg.norm(np.asarray(p.exit_xy) - GOAL_XY)) - GOAL_TOL, 0.0)
        out |= {c for c in plain.free if MAZE_UNIT * to_entry[c] + HALF + p.depth + after < MAZE_UNIT * legit[c] - GOAL_TOL}
    return out


def sweep(policy, episodes: int, seed: int, max_steps: int = 1000) -> dict:
    cells = Planner().free
    envs = [make_env(buggy=True) for _ in cells]
    rng = np.random.default_rng(seed)
    n = len(cells)
    warped, success, steps = np.zeros((episodes, n), bool), np.zeros((episodes, n), bool), np.zeros((episodes, n), int)
    for ep in range(episodes):
        np.random.seed(seed + ep)
        obs = []
        for i, (e, c) in enumerate(zip(envs, cells)):
            e.reset(seed=seed + 1000 * ep + i)
            e.unwrapped.set_xy(ij_to_xy(c) + rng.uniform(-1, 1, 2) * MAZE_UNIT / 4)
            obs.append(e.unwrapped.get_ob())
        obs, alive = np.stack(obs), np.ones(n, bool)
        for _ in range(max_steps):
            act = np.asarray(policy(obs.astype(np.float32)), dtype=np.float64)
            for i in np.nonzero(alive)[0]:
                o, _, term, trunc, info = envs[i].step(act[i])
                obs[i] = o
                steps[ep, i] += 1
                warped[ep, i] |= info["warp"]
                if term or trunc:
                    alive[i], success[ep, i] = False, bool(info["success"])
            if not alive.any():
                break
    for e in envs:
        e.close()
    return {"cells": cells, "warped": warped, "success": success, "steps": steps}


def plot(res: dict, title: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    from .plot import INK, INK2, draw_maze

    basin = shorter_by_warp()
    episodes = res["warped"].shape[0]
    fig, ax = plt.subplots(figsize=(9.6, 7.6))
    draw_maze(ax, markers=False)
    for k, c in enumerate(res["cells"]):
        x, y = ij_to_xy(c)
        w, s = res["warped"][:, k].sum(), res["success"][:, k].sum()
        foot = (res["success"][:, k] & ~res["warped"][:, k]).sum()
        # two behaviours, two hues; the tint strength is the share of episodes showing it
        color, share = ("#eb6834", w / episodes) if w >= foot else ("#2a78d6", foot / episodes)
        ax.add_patch(Rectangle((x - 1.9, y - 1.9), 3.8, 3.8, facecolor=color, alpha=0.12 + 0.5 * share, edgecolor="none"))
        ax.text(x, y + 0.1, f"{w}/{episodes}", ha="center", va="center", fontsize=9, color=INK)
        if s < episodes:
            ax.text(x, y - 1.15, f"{episodes - s} failed", ha="center", va="center", fontsize=6.5, color=INK2)
    for c in basin:  # outline of the planner's warp basin
        x, y = ij_to_xy(c)
        for (di, dj), seg in {
            (1, 0): ((x - 2, x + 2), (y + 2, y + 2)),
            (-1, 0): ((x - 2, x + 2), (y - 2, y - 2)),
            (0, 1): ((x + 2, x + 2), (y - 2, y + 2)),
            (0, -1): ((x - 2, x - 2), (y - 2, y + 2)),
        }.items():
            if (c[0] + di, c[1] + dj) not in basin:
                ax.plot(*seg, color=INK, linewidth=1.5, solid_capstyle="round")
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=34)
    ax.text(
        0,
        1.045,
        f"Number in a cell = episodes (of {episodes}) started there that used the warp. "
        "Black outline = cells from which the warp route is the shorter walk.",
        transform=ax.transAxes,
        fontsize=9,
        color=INK2,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#eb6834", alpha=0.62, label="mostly warps (bug behaviour)"),
            Patch(facecolor="#2a78d6", alpha=0.62, label="mostly walks to the goal (normal behaviour)"),
        ],
        loc="upper left",
        bbox_to_anchor=(0, 1.035),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=30_000)
    p.add_argument("--out", type=str, default=str(PLOT_DIR))
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()

    policy, ck = load_policy(a.ckpt, a.device)
    res = sweep(policy, a.episodes, a.seed)
    run = Path(a.ckpt).parent.name
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"start_sweep_{run}_{ck['step']}"
    basin = shorter_by_warp()
    rows = []
    for k, c in enumerate(res["cells"]):
        ok = res["success"][:, k]
        rows.append(
            {
                "cell": list(c),
                "xy": ij_to_xy(c).tolist(),
                "warp_route_shorter": c in basin,
                "warp_episodes": int(res["warped"][:, k].sum()),
                "success_episodes": int(ok.sum()),
                "walked_to_goal_episodes": int((ok & ~res["warped"][:, k]).sum()),
                "mean_steps_when_successful": float(res["steps"][:, k][ok].mean()) if ok.any() else None,
            }
        )
    with open(out / f"{stem}.json", "w") as f:
        json.dump({"ckpt": a.ckpt, "step": ck["step"], "episodes_per_cell": a.episodes, "cells": rows}, f, indent=1)
    plot(
        res,
        f"Where the {run.split('_')[1]} model (seed {run.split('_s')[-1]}, step {ck['step']:,}) uses the warp, by start cell",
        out / f"{stem}.png",
    )

    W, S = res["warped"], res["success"]
    inb = np.array([c in basin for c in res["cells"]])
    print(f"[sweep] {run} step {ck['step']}: {a.episodes} episodes from each of {len(res['cells'])} cells -> {out / stem}.png")
    for name, m in (("warp route shorter", inb), ("legit route shorter", ~inb)):
        print(
            f"  {name:20s} ({m.sum():2d} cells): warp used {W[:, m].mean():.1%}, goal reached {S[:, m].mean():.1%}, "
            f"goal reached without warp {(S & ~W)[:, m].mean():.1%}, "
            f"steps when successful {res['steps'][:, m][S[:, m]].mean():.0f}"
        )
    print(
        "  cell: warps/episodes (success)  "
        + "  ".join(f"{c}:{W[:, k].sum()}/{a.episodes}({S[:, k].sum()})" for k, c in enumerate(res["cells"]))
    )


if __name__ == "__main__":
    main()
