"""Full diagnostic figures and tables for the CC100 runs (reads logs/evals only; cut maps from cut_audit.json).
-> runs_unlearn/budget/figs/{curves,train,sweep,cut}.png and runs_unlearn/budget/audit/summary.{json,md}"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
REPO = Path("/home/ksj/Desktop/UNRL/OnlineRL-Baselines"); sys.path.insert(0, str(REPO)); os.chdir(REPO)
from bugmaze.env import MAZE_MAP, START_IJ, GOAL_IJ
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

R, G = Path("runs_unlearn/budget"), Path("runs_unlearn/goal"); F = R / "figs"; A = R / "audit"
H, W = MAZE_MAP.shape

def rows(p):
    with open(p) as f: return list(csv.DictReader(f))

def wilson(p, n, z=1.96):
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h

GROUPS = {"CC k=32 (parent s1)": [f"CC100_k32_s{s}" for s in range(3)],
          "CC k=1000 (parent s1)": [f"CC100_k1000_s{s}" for s in range(3)],
          "CC k=32 (parent s2)": [f"CC100p2_k32_s{s}" for s in range(3)],
          "CC k=1000 (parent s2)": [f"CC100p2_k1000_s{s}" for s in range(3)],
          "CC k=32 (parent s1, label draws 1-2)": [f"CC100_k32_d{d}_s0" for d in (1, 2)],
          "CC k=1000 (parent s1, label draws 1-2)": [f"CC100_k1000_d{d}_s0" for d in (1, 2)]}
COL = {"32": "#2a78d6", "1000": "#7b3fbf"}

# ---- per-run table
table, summ = [], {}
for g, runs in GROUPS.items():
    vals = []
    for r in runs:
        p = R / r / "final_eval.json"
        if not p.exists(): continue
        e = json.loads(p.read_text())["final.pt"]; o, pa, sw = e["open"], e["patched"], e["sweep"]
        cfg = json.loads((R / r / "config.json").read_text()); lin = json.loads((R / r / "lineage.json").read_text())
        n = e["episodes"]; lo, hi = wilson(o["success_no_bug"], n)
        row = {"run": r, "parent": cfg["parent"], "k": cfg["k"], "draw": cfg["draw"], "seed": cfg["seed"], "episodes": n,
               "legal": o["success_no_bug"], "legal_ci95": [lo, hi], "bug_use": o["bug_usage"], "success": o["success"],
               "stall": o["stall_rate"], "patched": pa["success"], "sweep_on_foot": sw["warp_shorter"]["on_foot"],
               "legal_shorter_on_foot": sw["legal_shorter"]["on_foot"], "path_len": o.get("path_len_on_foot"),
               "frac_cut": lin["cut"]["frac_cut"], "gates_pass": lin["phase1"].get("gates_pass"), "minutes": lin.get("minutes_total")}
        table.append(row); vals.append(row)
    if vals:
        summ[g] = {"n": len(vals), **{k: [float(np.mean([v[k] for v in vals])), float(np.std([v[k] for v in vals]))]
                                    for k in ("legal", "bug_use", "stall", "patched", "sweep_on_foot", "legal_shorter_on_foot", "frac_cut")}}
(A / "summary.json").write_text(json.dumps({"runs": table, "groups": summ}, indent=1, default=float))
md = ["| run | parent | k | label draw | seed | legal (95% CI, 200 ep.) | bug use | stall | patched | sweep on-foot | rows cut | gates |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for t in table:
    md.append(f"| {t['run']} | {t['parent'][-2:]} | {t['k']} | {t['draw']} | {t['seed']} | {t['legal']:.3f} ({t['legal_ci95'][0]:.2f}–{t['legal_ci95'][1]:.2f}) | "
              f"{t['bug_use']:.3f} | {t['stall']:.2f} | {t['patched']:.3f} | {t['sweep_on_foot']:.2f} | {100 * t['frac_cut']:.1f}% | {'pass' if t['gates_pass'] else 'fail'} |")
md += ["", "| group | n runs | legal | bug use | stall | patched | sweep on-foot | rows cut |", "|---|---|---|---|---|---|---|---|"]
for g, s in summ.items():
    md.append(f"| {g} | {s['n']} | {s['legal'][0]:.3f} ± {s['legal'][1]:.3f} | {s['bug_use'][0]:.3f} | {s['stall'][0]:.2f} | "
              f"{s['patched'][0]:.3f} ± {s['patched'][1]:.3f} | {s['sweep_on_foot'][0]:.2f} | {100 * s['frac_cut'][0]:.1f}% |")
(A / "summary.md").write_text("\n".join(md) + "\n"); print("\n".join(md))

# ---- learning curves vs total gradient steps (Phase 1 included)
def snaps(run, base, t1):
    out = [(t1 + int(x["step"]), float(x["open_success_no_bug"]), float(x["open_bug_usage"])) for x in rows(base / run / "eval_log.csv") if x["ckpt"].startswith("snap")]
    return np.array(sorted(out))
fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
for g, runs, ls in [("CC k=32", [f"CC100_k32_s{s}" for s in range(3)] + [f"CC100p2_k32_s{s}" for s in range(3)], "-"),
                    ("CC k=1000", [f"CC100_k1000_s{s}" for s in range(3)] + [f"CC100p2_k1000_s{s}" for s in range(3)], "-")]:
    k = g.split("=")[1]; curves = [snaps(r, R, 10_000) for r in runs if (R / r / "eval_log.csv").exists()]
    curves = [c for c in curves if len(c)]
    for c in curves:
        ax[0].plot(c[:, 0] / 1e3, c[:, 1], color=COL[k], lw=0.7, alpha=0.35); ax[1].plot(c[:, 0] / 1e3, c[:, 2], color=COL[k], lw=0.7, alpha=0.35)
    L = min(len(c) for c in curves); M = np.mean([c[:L] for c in curves], 0)
    ax[0].plot(M[:, 0] / 1e3, M[:, 1], color=COL[k], lw=2.2, label=f"{g} (mean of {len(curves)} runs)")
    ax[1].plot(M[:, 0] / 1e3, M[:, 2], color=COL[k], lw=2.2, label=f"{g}")
d1 = [snaps(f"D1_cutbc_s{s}", G, 100_000) for s in range(3)]; M = np.mean(d1, 0)
ax[0].plot(M[:, 0] / 1e3, M[:, 1], color="#888", lw=2, ls="--", label="oracle D1 (k = B labels, 100k Phase 1)")
ax[1].plot(M[:, 0] / 1e3, M[:, 2], color="#888", lw=2, ls="--", label="oracle D1")
cl = []
for s in range(3):
    a = [(int(x["step"]), float(x["success_no_bug"])) for x in rows(R / f"clean50k_s{s}" / "eval_log.csv")]
    b = [(int(x["step"]), float(x["success_no_bug"])) for x in rows(Path(f"runs_bugmaze/rebrac_clean_s{s}/eval_log.csv")) if 50_000 < int(x["step"]) <= 300_000]
    cl.append(np.array(a + b))
M = np.mean(cl, 0)
ax[0].plot(M[:, 0] / 1e3, M[:, 1], color="#1a9e3a", lw=2, ls=":", marker="o", ms=3, label="clean retrain from scratch (oracle data)")
pb = [json.loads(Path(f"runs_bugmaze/rebrac_bug_s{s}/final_eval.json").read_text()) for s in (1, 2)]
for a_ in ax: a_.set_xlabel("total gradient steps (thousands), Phase 1 included"); a_.set_xlim(0, 300)
ax[0].set_ylabel("legal success (portal open, 50 ep. per snapshot)"); ax[1].set_ylabel("bug use (share of episodes that warp)")
ax[1].axhline(0.98, color="#eb6834", lw=1, ls=":"); ax[1].text(5, 0.93, "bug parent ≈ 0.98", color="#eb6834", fontsize=8)
ax[0].legend(fontsize=7.5, loc="lower right"); ax[1].legend(fontsize=7.5, loc="center right"); ax[1].set_ylim(-0.02, 1.02)
for a_ in ax:
    for s_ in ("top", "right"): a_.spines[s_].set_visible(False)
fig.tight_layout(); fig.savefig(F / "curves.png", dpi=130)

# ---- training diagnostics (Phase 2): Q_R mean, critic loss, actor parameter drift
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6))
for r in [f"CC100_k{k}_s{s}" for k in (32, 1000) for s in range(3)]:
    t = rows(R / r / "train_log.csv"); st = np.array([float(x["step"]) for x in t]) / 1e3
    k = r.split("_k")[1].split("_")[0]
    for a_, key in zip(ax, ("qr_mean", "qr_loss", "param_drift")):
        a_.plot(st, [float(x[key]) for x in t], color=COL[k], lw=0.8, alpha=0.7, label=f"k={k}" if r.endswith("s0") else None)
for a_, ttl in zip(ax, ("mean Q_R on the batch", "Q_R TD loss", "actor parameter drift from π0 (relative)")):
    a_.set_title(ttl, fontsize=10); a_.set_xlabel("fine-tuning steps (thousands)"); a_.legend(fontsize=8)
ax[1].set_yscale("log"); fig.tight_layout(); fig.savefig(F / "train.png", dpi=130)

# ---- start sweep: on-foot success per start cell
def cellmap(per_cell):
    m = np.full((H, W), np.nan)
    for key, v in per_cell.items():
        i, j = map(int, key.split(",")); m[i, j] = v["on_foot"]
    return m
def parent_map(s):
    d = json.loads(Path(f"runs_bugmaze/plots/start_sweep_rebrac_bug_s{s}_1000000.json").read_text()); m = np.full((H, W), np.nan)
    for c in d["cells"]:
        i, j = c["cell"]; m[i, j] = c["walked_to_goal_episodes"] / d["episodes_per_cell"]
    return m
maps = {"bug parent s1 (1M)": parent_map(1)}
for k in ("32", "1000"):
    ms = [cellmap(json.loads((R / f"CC100_k{k}_s{s}" / "final_eval.json").read_text())["final.pt"]["sweep"]["per_cell"]) for s in range(3)]
    maps[f"CC k={k}, parent s1 (mean of 3)"] = np.mean(ms, 0)
maps["oracle D1 (mean of 3)"] = np.mean([cellmap(json.loads((G / f"D1_cutbc_s{s}" / "final_eval.json").read_text())["final.pt"]["sweep"]["per_cell"]) for s in range(3)], 0)
fig, ax = plt.subplots(1, len(maps), figsize=(4.3 * len(maps), 3.4))
for a_, (name, m) in zip(ax, maps.items()):
    im = a_.imshow(m, cmap="Blues", vmin=0, vmax=1, origin="lower")
    for (i, j), w in np.ndenumerate(MAZE_MAP):
        if w == 1: a_.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#bbb" if (i, j) == (5, 0) else "#555"))
        elif np.isfinite(m[i, j]): a_.text(j, i, f"{m[i, j]:.1f}", ha="center", va="center", fontsize=6.5, color="w" if m[i, j] > 0.6 else "k")
    a_.plot(*GOAL_IJ[::-1], "*", color="#1a9e3a", ms=11); a_.set_title(name, fontsize=9.5); a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8, label="share of 10 episodes reaching the goal on foot")
fig.savefig(F / "sweep.png", dpi=130, bbox_inches="tight")

# ---- where the cut lands: share of clean rows cut per cell
au = json.loads((A / "cut_audit.json").read_text())
fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
route = au["route_cells"]
for a_, r in zip(ax, ("CC100_k32_s0", "CC100_k1000_s0")):
    m = np.array(au["runs"][r]["fp_cell_frac"], float); m[MAZE_MAP == 1] = np.nan
    im = a_.imshow(m, cmap="Reds", vmin=0, vmax=0.3, origin="lower")
    for (i, j), w in np.ndenumerate(MAZE_MAP):
        if w == 1: a_.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#bbb" if (i, j) == (5, 0) else "#555"))
        elif m[i, j] > 0.005: a_.text(j, i, f"{100 * m[i, j]:.0f}%", ha="center", va="center", fontsize=6.5)
    a_.plot([c[1] for c in route], [c[0] for c in route], color="#2a78d6", lw=2, alpha=0.8)
    x = au["runs"][r]; a_.set_title(f"{r}: clean rows cut per cell\nprecision {x['precision']:.2f} · on-route clean rows cut: {x['fp_on_route_rows']}", fontsize=9)
    a_.set_xticks([]); a_.set_yticks([])
fig.colorbar(im, ax=ax, shrink=0.8, label="share of the cell's clean rows that are cut"); fig.savefig(F / "cut.png", dpi=130, bbox_inches="tight")
print("figures saved")
