"""Summarise the k sweep of the cost-critic core cut (runs_unlearn/ksweep/CCk_k<k>_d<s>_s<s>).

    python scripts/ksweep_summary.py            # -> runs_unlearn/ksweep/summary.json, ksweep.png (+ prints the tables)

Reads only the run folders' csv/json (final_eval.json, eval_log.csv, lineage.json), so it needs no checkpoint.
A run whose cut reference (median V_C(s') over the reference labels) is <= 0 flags no unlabelled row: only the k
labels are cut. Those runs are marked as "reference collapsed".
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path("runs_unlearn/ksweep")
KS = [1, 2, 4, 8, 16, 32, 64, 128]
SEEDS = [0, 1, 2]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # legal-route success, warp use (as everywhere in the project)


def load(k: int, s: int) -> dict:
    d = ROOT / f"CCk_k{k}_d{s}_s{s}"
    fe = json.loads((d / "final_eval.json").read_text())["final.pt"]
    cut = json.loads((d / "lineage.json").read_text())["cut"]
    ev = list(csv.DictReader(open(d / "eval_log.csv")))
    sw = fe["sweep"]
    return {
        "k": k, "seed": s, "draw": s,
        "legal": fe["open"]["success_no_bug"], "bug_use": fe["open"]["bug_usage"], "stall": fe["open"]["stall_rate"],
        "patched": fe["patched"]["success"],
        "sweep_on_foot_warp_shorter": sw["warp_shorter"]["on_foot"], "sweep_warp_warp_shorter": sw["warp_shorter"]["warp"],
        "sweep_on_foot_legal_shorter": sw["legal_shorter"]["on_foot"],
        "pocket_on_foot": float(np.mean([v["on_foot"] for v in sw["near_portal"].values()])),
        "rows_cut": cut["n_cut"], "cut_ref": cut["ref_median_v_c_next"], "collapsed": cut["ref_median_v_c_next"] <= 0,
        "curve": {int(r["step"]) + 10_000: float(r["open_success_no_bug"]) for r in ev},  # total steps (t1 = 10k)
        "curve_bug": {int(r["step"]) + 10_000: float(r["open_bug_usage"]) for r in ev},
    }  # fmt: skip


def ms(xs) -> str:
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"


def main():
    runs = [load(k, s) for k in KS for s in SEEDS]
    by_k = {k: [r for r in runs if r["k"] == k] for k in KS}
    rows = []
    for k in KS:
        rs = by_k[k]
        ok = [r for r in rs if not r["collapsed"]]
        rows.append({
            "k": k,
            "legal": [r["legal"] for r in rs], "bug_use": [r["bug_use"] for r in rs], "stall": [r["stall"] for r in rs],
            "sweep_on_foot": [r["sweep_on_foot_warp_shorter"] for r in rs], "rows_cut": [r["rows_cut"] for r in rs],
            "collapsed_seeds": [r["seed"] for r in rs if r["collapsed"]],
            "legal_without_collapsed": [r["legal"] for r in ok],
        })  # fmt: skip
    head = ["k", "success without warp (200 ep.)", "per seed", "warp use", "per seed", "stall",
            "sweep on foot (warp-shorter cells)", "rows cut", "reference collapsed"]  # fmt: skip
    print("| " + " | ".join(head) + " |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['k']} | {ms(r['legal'])} | {', '.join(f'{x:.2f}' for x in r['legal'])} | {ms(r['bug_use'])} | "
              f"{', '.join(f'{x:.2f}' for x in r['bug_use'])} | {np.mean(r['stall']):.2f} | {np.mean(r['sweep_on_foot']):.2f} | "
              f"{', '.join(f'{x:,}' for x in r['rows_cut'])} | "
              f"{', '.join(f's{s}' for s in r['collapsed_seeds']) or '-'} |")  # fmt: skip
    steps = sorted(runs[0]["curve"])
    print("\nsuccess without warp at each snapshot (total steps, 50 episodes, mean over seeds)")
    print("| k | " + " | ".join(f"{s // 1000}k" for s in steps) + " |")
    print("|---|" + "---|" * len(steps))
    for k in KS:
        print(f"| {k} | " + " | ".join(f"{np.mean([r['curve'].get(s, np.nan) for r in by_k[k]]):.2f}" for s in steps) + " |")
    (ROOT / "summary.json").write_text(json.dumps({"runs": runs, "by_k": rows}, indent=1, default=float))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": MUTED, "ytick.color": MUTED})  # fmt: skip
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))
    x = np.arange(len(KS))
    for ax, key, title, color in ((axes[0], "legal", "Success without the warp", BLUE), (axes[1], "bug_use", "Warp use", ORANGE)):
        for i, k in enumerate(KS):
            for j, r in enumerate(by_k[k]):
                xx = i + (j - 1) * 0.12
                if r["collapsed"]:
                    ax.plot(xx, r[key], "o", ms=8, mfc=SURFACE, mec=color, mew=1.5, zorder=3)
                else:
                    ax.plot(xx, r[key], "o", ms=8, color=color, mec=SURFACE, mew=1.5, zorder=3)
        ax.plot(x, [np.mean([r[key] for r in by_k[k]]) for k in KS], color=color, lw=2, zorder=2)
        ax.set_title(title, loc="left", fontsize=11, color=INK, pad=8)
        ax.set_xticks(x, [str(k) for k in KS])
        ax.set_xlabel("number of labels k")
        ax.set_ylim(-0.04, 1.04)
    ax = axes[2]
    shades = plt.get_cmap("Blues")(np.linspace(0.35, 1.0, len(KS)))
    ends = []
    for k, c in zip(KS, shades):
        ys = [np.mean([r["curve"].get(s, np.nan) for r in by_k[k]]) for s in steps]
        ax.plot([s / 1000 for s in steps], ys, color=c, lw=2)
        ends.append([ys[-1], k])
    ends.sort()
    for a, b in zip(ends, ends[1:]):  # keep end labels at least 0.045 apart
        b[0] = max(b[0], a[0] + 0.045)
    for y, k in ends:
        ax.text(steps[-1] / 1000 + 1.5, y, f"k={k}", fontsize=8, color=INK2, va="center")
    ax.set_title("Success without the warp during fine-tuning", loc="left", fontsize=11, color=INK, pad=8)
    ax.set_xlabel("total gradient steps (thousands; 10k cost critic + fine-tuning)")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(steps[0] / 1000 - 3, steps[-1] / 1000 + 12)
    for a in axes:
        for side in ("top", "right", "left"):
            a.spines[side].set_visible(False)
        a.spines["bottom"].set_color(AXIS)
        a.grid(axis="y", color=GRID, lw=1)
        a.set_axisbelow(True)
        a.tick_params(length=0)
    note = (
        "Core cut + ReBRAC fine-tuning of the bug model (parent rebrac_bug_s1), 10k + 90k steps; 3 seeds per k, "
        "label draw = seed. Dots = seeds (200 episodes each), line = mean.\n"
        "Hollow dot = cut reference collapsed (median V_C(s') <= 0 on the reference labels), so only the k labels were cut."
    )
    fig.text(0.01, 0.012, note, fontsize=8.5, color=INK2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(ROOT / "ksweep.png", dpi=150)
    print(f"\n[ksweep] wrote {ROOT / 'summary.json'} and {ROOT / 'ksweep.png'}")


if __name__ == "__main__":
    main()
