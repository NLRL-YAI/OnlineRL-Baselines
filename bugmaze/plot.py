"""Figures for the bug-maze runs: learning curves and final rollouts drawn on the maze.

    python -m bugmaze.plot --run_dir runs_bugmaze --out runs_bugmaze/plots

Reads runs_bugmaze/rebrac_{bug,clean}_s*/eval_log.csv and final_traj_{buggy,patched}.npz
(written by bugmaze.evaluate). Colour follows the model everywhere: orange = bug model,
blue = clean model (the first two slots of the dataviz reference palette).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Rectangle  # noqa: E402

from .env import DEFAULT_PORTALS, GOAL_TOL, GOAL_XY, HALF, MAZE_MAP, MAZE_UNIT, START_IJ, ij_to_xy  # noqa: E402
from .paths import PLOT_DIR, RUN_DIR  # noqa: E402

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
MODELS = {"clean": ("Clean model (bug data removed)", "#2a78d6"), "bug": ("Bug model (trained with bug data)", "#eb6834")}
ENVS = {"buggy": "Portal open (the bug exists)", "patched": "Portal patched (wall is solid)"}
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
    }
)


def runs_of(run_dir: Path, model: str) -> list[Path]:
    return sorted(d for d in run_dir.glob(f"rebrac_{model}_s*") if d.is_dir())


def read_eval(path: Path) -> dict[str, np.ndarray]:
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else {}


def style_axis(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def plot_curves(run_dir: Path, out: Path):
    metrics = [
        ("success", "Success rate", (-0.03, 1.03)),
        ("return_mean", "Return (minus steps to the goal)", None),
        ("bug_usage", "Share of episodes that use the warp", (-0.03, 1.03)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    summary = {}
    for model, (label, color) in MODELS.items():
        logs = [read_eval(d / "eval_log.csv") for d in runs_of(run_dir, model) if (d / "eval_log.csv").exists()]
        logs = [g for g in logs if g]
        if not logs:
            continue
        n = max(len(g["step"]) for g in logs)
        full = [g for g in logs if len(g["step"]) == n]  # the mean uses only runs that got this far
        steps = full[0]["step"]
        for ax, (key, _, _) in zip(axes, metrics):
            if len(logs) > 1:
                for g in logs:  # every seed, including runs still in progress
                    ax.plot(g["step"], g[key], color=color, linewidth=1, alpha=0.35, solid_capstyle="round")
            mean = np.stack([g[key] for g in full]).mean(0)
            ax.plot(steps, mean, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round")
            ax.plot(
                steps[-1], mean[-1], "o", color=color, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2, clip_on=False
            )
            summary.setdefault(key, []).append(
                f"{label.split(' (')[0]} {mean[-1]:.2f}" if key != "return_mean" else f"{label.split(' (')[0]} {mean[-1]:.0f}"
            )
        summary["seeds"] = summary.get("seeds", []) + [
            f"{label.split(' (')[0]} {len(full)}"
            + (f" (+{len(logs) - len(full)} still training)" if len(logs) > len(full) else "")
        ]
    for ax, (key, title, ylim) in zip(axes, metrics):
        style_axis(ax)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=22)
        ax.text(0, 1.03, "last evaluation: " + ", ".join(summary.get(key, [])), transform=ax.transAxes, fontsize=9, color=INK2)
        ax.set_xlabel("training steps")
        ax.xaxis.set_major_formatter(lambda v, _: "0" if v == 0 else (f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K"))
        if ylim:
            ax.set_ylim(*ylim)
    handles = [Line2D([0], [0], color=c, linewidth=2, label=lab) for lab, c in MODELS.values()]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.045, 0.99), ncol=2, frameon=False, fontsize=10)
    fig.text(
        0.045,
        0.012,
        "ReBRAC on antmaze-large task 1, evaluated with the portal open, 50 episodes per point.\n"
        "Thick line = mean over finished seeds (" + ", ".join(summary.get("seeds", [])) + "); thin lines = individual seeds.",
        ha="left",
        va="bottom",
        fontsize=9,
        color=INK2,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93))
    fig.savefig(out / "curves.png", dpi=150)
    plt.close(fig)


def draw_maze(ax, markers: bool = True):
    for i in range(MAZE_MAP.shape[0]):
        for j in range(MAZE_MAP.shape[1]):
            if MAZE_MAP[i, j]:
                x, y = ij_to_xy((i, j))
                ax.add_patch(Rectangle((x - HALF, y - HALF), MAZE_UNIT, MAZE_UNIT, facecolor=GRID, edgecolor="none"))
    for p in DEFAULT_PORTALS:
        x, y = ij_to_xy(p.block_ij)
        ax.add_patch(Rectangle((x - HALF, y - HALF), MAZE_UNIT, MAZE_UNIT, facecolor=AXIS, edgecolor="none"))
        ax.text(x, y + HALF + 0.5, "portal\n(wall block)", ha="center", va="bottom", fontsize=8, color=INK2)
    sx, sy = ij_to_xy(START_IJ)
    if markers:
        ax.add_patch(Rectangle((sx - 1, sy - 1), 2, 2, facecolor="none", edgecolor=INK2, linewidth=1))  # start noise box
        ax.text(sx, sy - 1.5, "start", ha="center", va="top", fontsize=8, color=INK2)
        ax.add_patch(Circle(GOAL_XY, GOAL_TOL, facecolor="none", edgecolor=INK, linewidth=1.5, zorder=5))
        ax.text(GOAL_XY[0], GOAL_XY[1] + 1.1, "goal", ha="center", va="bottom", fontsize=8, color=INK2)
    else:  # cells carry their own text: label the task start / goal cells from outside
        ax.text(sx, sy - HALF - 0.5, "task start", ha="center", va="top", fontsize=8, color=INK2)
        ax.text(GOAL_XY[0], GOAL_XY[1] + HALF + 0.5, "goal", ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xlim(-6.5, 42.5)
    ax.set_ylim(-6.5, 30.5)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_trajectories(run_dir: Path, out: Path):
    envs = [e for e in ENVS if any((d / f"final_traj_{e}.npz").exists() for m in MODELS for d in runs_of(run_dir, m))]
    models = [m for m in ("bug", "clean") if runs_of(run_dir, m)]
    if not envs or not models:
        return
    fig, axes = plt.subplots(len(envs), len(models), figsize=(6.6 * len(models), 5.3 * len(envs)), squeeze=False)
    for r, env in enumerate(envs):
        for c, model in enumerate(models):
            ax, (label, color) = axes[r, c], MODELS[model]
            draw_maze(ax)
            n = warped = ok = 0
            rets = []
            for d in runs_of(run_dir, model):
                f = d / f"final_traj_{env}.npz"
                if not f.exists():
                    continue
                t = np.load(f)
                for xy in t["xy"]:
                    ax.plot(xy[:, 0], xy[:, 1], color=color, linewidth=1.1, alpha=0.45, solid_capstyle="round")
                n += len(t["xy"])
                warped += int((t["warps"] > 0).sum())
                ok += int(t["success"].sum())
                rets += list(t["ret"])
            ax.set_title(f"{label}\n{ENVS[env]}", loc="left", fontsize=11, color=INK, pad=20)
            ax.text(
                0,
                1.015,
                f"{ok}/{n} episodes reach the goal, {warped}/{n} use the warp, mean return {np.mean(rets):.0f}"
                if n
                else "no rollouts",
                transform=ax.transAxes,
                fontsize=9,
                color=INK2,
            )
    handles = [Line2D([0], [0], color=MODELS[m][1], linewidth=2, label=MODELS[m][0]) for m in models]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.03, 0.995), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out / "trajectories.png", dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, default=str(RUN_DIR))
    p.add_argument("--out", type=str, default=str(PLOT_DIR))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plot_curves(Path(a.run_dir), out)
    plot_trajectories(Path(a.run_dir), out)
    print(f"[plot] wrote {out}/curves.png and {out}/trajectories.png")


if __name__ == "__main__":
    main()
