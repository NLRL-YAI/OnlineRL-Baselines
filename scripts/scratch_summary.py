"""Summarise the from-scratch safe-RL baseline (runs_unlearn/scratch/SCR_f<penalty>_k32_d<s>_s<s>) against the unlearning
fine-tune at the same k, labels and seed (runs_unlearn/ksweep/CCk_k32_d<s>_s<s>) and the stage 1 clean model.

    python scripts/scratch_summary.py            # -> runs_unlearn/scratch/summary.json, scratch.png (+ prints the tables)

Reads only csv/json files, so it needs no checkpoint. Total steps = 10k cost-critic steps + policy steps.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("runs_unlearn/scratch")
PENALTIES, SEEDS = [100, 200], [0, 1, 2]
REPORT = [100_000, 500_000, 1_000_000]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE = "#2a78d6"  # clean model, as everywhere in the project
VIOLET = {100: "#9a7fd9", 200: "#4f2fa3"}  # from scratch, penalty -100 (light, solid) and -200 (dark, dashed)
DASH = {100: "-", 200: (0, (4, 2))}


def rows(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p)))


def load_scratch(f: int, s: int) -> dict:
    d = ROOT / f"SCR_f{f}_k32_d{s}_s{s}"
    cut = json.loads((d / "lineage.json").read_text())["cut"]
    ev = rows(d / "eval_log.csv")
    out = {
        "penalty": f, "seed": s, "rows_cut": cut["n_cut"], "cut_ref": cut["ref_median_v_c_next"],
        "collapsed": cut["ref_median_v_c_next"] <= 0,
        "curve": {int(r["step"]) + 10_000: float(r["open_success_no_bug"]) for r in ev},
        "curve_bug": {int(r["step"]) + 10_000: float(r["open_bug_usage"]) for r in ev},
        "curve_stall": {int(r["step"]) + 10_000: float(r["open_stall_rate"]) for r in ev},
    }  # fmt: skip
    fe = d / "final_eval.json"
    if fe.exists():
        o = json.loads(fe.read_text())["final.pt"]
        op = o["open"]
        out["final"] = {"legal": op["success_no_bug"], "bug_use": op["bug_usage"], "success_open": op["success"],
                        "patched": o["patched"]["success"], "stall": op["stall_rate"]}  # fmt: skip
    return out


def load_finetune(s: int) -> dict:
    d = Path("runs_unlearn/ksweep") / f"CCk_k32_d{s}_s{s}"
    o = json.loads((d / "final_eval.json").read_text())["final.pt"]
    ref = json.loads((d / "lineage.json").read_text())["cut"]["ref_median_v_c_next"]
    op = o["open"]
    return {
        "seed": s,
        "legal": op["success_no_bug"],
        "bug_use": op["bug_usage"],
        "patched": o["patched"]["success"],
        "collapsed": ref <= 0,
    }


def load_clean() -> dict[int, float]:
    """Stage 1 clean model (ReBRAC from scratch without the bug trajectories): mean success over 3 seeds per step."""
    per = [
        {int(r["step"]): float(r["success"]) for r in rows(Path(f"runs_bugmaze/rebrac_clean_s{s}/eval_log.csv"))} for s in SEEDS
    ]
    steps = sorted(set.intersection(*(set(p) for p in per)))
    return {t: float(np.mean([p[t] for p in per])) for t in steps if t > 0}


def main():
    runs = {(f, s): load_scratch(f, s) for f in PENALTIES for s in SEEDS}
    ft = {s: load_finetune(s) for s in SEEDS}
    clean = load_clean() if Path("runs_bugmaze/rebrac_clean_s0/eval_log.csv").exists() else {}

    head = (
        ["penalty", "seed", "rows cut"]
        + [f"{t // 1000}k: no-warp / warp" for t in REPORT]
        + ["final (200 ep.): no-warp / warp / patched"]
    )
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for (f, s), r in runs.items():
        cells = [f"{r['curve'].get(t, np.nan):.2f} / {r['curve_bug'].get(t, np.nan):.2f}" for t in REPORT]
        fin = r.get("final")
        fin_s = f"{fin['legal']:.2f} / {fin['bug_use']:.2f} / {fin['patched']:.2f}" if fin else "-"
        coll = " (reference collapsed)" if r["collapsed"] else ""
        print(f"| -{f} | {s} | {r['rows_cut']:,}{coll} | " + " | ".join(cells) + f" | {fin_s} |")
    print("\nunlearning fine-tune, same k / labels / seed (100k steps, final 200 ep.): no-warp / warp / patched")
    for s, r in ft.items():
        print(
            f"  seed {s}: {r['legal']:.2f} / {r['bug_use']:.2f} / {r['patched']:.2f}"
            + (" (reference collapsed)" if r["collapsed"] else "")
        )
    (ROOT / "summary.json").write_text(
        json.dumps({"scratch": [dict(v) for v in runs.values()], "finetune_k32": list(ft.values()), "clean_mean_success": clean},
                   indent=1, default=float))  # fmt: skip

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED})  # fmt: skip
    fig, axes = plt.subplots(2, 3, figsize=(14, 6.6), sharex=True, sharey=True)
    for j, s in enumerate(SEEDS):
        for i, (key, ylab) in enumerate((("curve", "success without the warp"), ("curve_bug", "warp use"))):
            ax = axes[i, j]
            if key == "curve" and clean:
                ax.plot([t / 1000 for t in clean], list(clean.values()), color=BLUE, lw=1.5, alpha=0.8, zorder=2,
                        label="clean model (stage 1, trained without bug data; mean of 3 seeds)")  # fmt: skip
            for f in PENALTIES:
                c = runs[(f, s)][key]
                ts = sorted(c)
                ax.plot([t / 1000 for t in ts], [c[t] for t in ts], color=VIOLET[f], ls=DASH[f], lw=2, marker="o", ms=4,
                        mec=SURFACE, mew=1, zorder=3, label=f"from scratch with the cut, penalty -{f}")  # fmt: skip
            v = ft[s]["legal" if key == "curve" else "bug_use"]
            ax.plot(100, v, marker="D", ms=9, color=INK, mec=SURFACE, mew=1.5, ls="none", zorder=4,
                    label="unlearning fine-tune of the bug model (same labels and seed), 100k steps")  # fmt: skip
            if i == 0:
                coll = " - cut reference collapsed, only the 32 labels cut" if runs[(100, s)]["collapsed"] else ""
                ax.set_title(f"seed {s}{coll}", loc="left", fontsize=10.5, color=INK, pad=8)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 1:
                ax.set_xlabel("total gradient steps (thousands)")
            ax.set_ylim(-0.04, 1.04)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(AXIS)
            ax.grid(axis="y", color=GRID, lw=1)
            ax.set_axisbelow(True)
            ax.tick_params(length=0)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    note = (
        "k = 32 labels, label draw = seed. From scratch: fresh ReBRAC actor and Q_R trained with the core cut from step 0 "
        "(10k cost-critic steps + 990k). Snapshots: 50 episodes, portal open."
    )
    fig.text(0.01, 0.01, note, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.9), h_pad=2.5)
    fig.savefig(ROOT / "scratch.png", dpi=150)
    print(f"\n[scratch] wrote {ROOT / 'summary.json'} and {ROOT / 'scratch.png'}")


if __name__ == "__main__":
    main()
