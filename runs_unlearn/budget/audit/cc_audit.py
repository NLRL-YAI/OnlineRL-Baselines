"""GROUND-TRUTH AUDIT of the core cut actually used by each CC100 run (separate process; never feeds training).
Recomputes cut = labels | (V_C(s') >= 0.5 * median V_C(s') over the run's held-out labels) & ~terminal from the run's own
phase1.pt, checks its size against the run's lineage, and reports recall / leak / precision / false positives (overall,
on the legal route, off it) plus per-cell maps of the cut. -> runs_unlearn/budget/audit/cut_audit.json
    python cc_audit.py RUN_NAME [RUN_NAME ...]"""
import json, os, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/home/ksj/Desktop/UNRL/OnlineRL-Baselines"); sys.path.insert(0, str(REPO)); os.chdir(REPO)
from bugmaze.rebrac import ReBRACConfig, mlp
from bugmaze.env import MAZE_MAP, START_IJ, GOAL_IJ
from bugmaze.expert import Planner
from unlearn.audit_cost import holdout_rows
from unlearn.labels import label_path, load_labels

dev = "cuda"
d = np.load("datasets/antmaze_bug/mix.npz")
O2 = d["next_observations"].astype(np.float32); D = d["terminals"].astype(bool)
bug, inw, nin, warp = (d[k].astype(bool) for k in ("is_bug", "in_wall", "next_in_wall", "warp"))
tid = d["traj_id"]; n = len(bug)
entering, inside = nin & ~inw & ~warp, inw & nin & ~warp
into_wall = entering | inside
H, W = MAZE_MAP.shape
I = np.clip(np.round(O2[:, 1] / 4 + 1).astype(int), 0, H - 1)
J = np.clip(np.round(O2[:, 0] / 4 + 1).astype(int), 0, W - 1)
pl = Planner(); dist = pl.dist_to(GOAL_IJ); c = START_IJ; route = [c]
while c != GOAL_IJ:
    c = min(pl.succ[c], key=lambda m: dist.get(m, 1 << 30)); route.append(c)
on_route = np.zeros(n, bool)
for (i, j) in route:
    on_route |= (I == i) & (J == j)
clean = ~bug
iw_rows = np.flatnonzero(into_wall); _, t_inv = np.unique(tid[iw_rows], return_inverse=True); t_cnt = np.bincount(t_inv)
cell = I * W + J
clean_per_cell = np.bincount(cell[clean], minlength=H * W)

out_p = Path("runs_unlearn/budget/audit/cut_audit.json"); out_p.parent.mkdir(parents=True, exist_ok=True)
out = json.loads(out_p.read_text()) if out_p.exists() else {"route_cells": route, "runs": {}}
for name in sys.argv[1:]:
    rp = Path("runs_unlearn/budget") / name
    cfg = json.loads((rp / "config.json").read_text()); lin = json.loads((rp / "lineage.json").read_text())
    par = torch.load(Path(cfg["parent_dir"]) / cfg["parent"] / "actor_1000000.pt", map_location=dev, weights_only=False)
    rc = ReBRACConfig.from_dict(par["config"])
    ph = torch.load(rp / "phase1.pt", map_location=dev, weights_only=False)
    v = mlp(29, rc.hidden, rc.critic_layers, 1, layer_norm=True).to(dev); v.load_state_dict(ph["vc"]); v.eval()
    with torch.no_grad():
        rho = np.concatenate([v(torch.as_tensor(O2[i:i + 131072], device=dev)).squeeze(-1).cpu().numpy() for i in range(0, n, 131072)])
    L = np.asarray(load_labels(Path(cfg["label_file"]) if cfg["label_file"] else label_path(cfg["k"], cfg["draw"]))["idx"])
    hold = holdout_rows(cfg); ref_rows = hold if len(hold) else L
    ref = float(np.sort(rho[ref_rows])[(len(ref_rows) - 1) // 2])  # torch.median (lower middle), as in training
    lab = np.zeros(n, bool); lab[L] = True
    A = lab | ((rho >= 0.5 * ref) & ~D)
    share = np.bincount(t_inv, weights=A[iw_rows].astype(float)) / t_cnt
    fp = A & clean
    r = {"k": cfg["k"], "draw": cfg["draw"], "seed": cfg["seed"], "parent": cfg["parent"], "ref": ref,
         "lineage_ref": lin["cut"]["ref_median_v_c_next"], "n_cut": int(A.sum()), "lineage_n_cut": lin["cut"]["n_cut"],
         "recall_warp_rows": float(A[warp].mean()), "recall_entering": float(A[entering].mean()), "recall_inside": float(A[inside].mean()),
         "recall_all_bug": float(A[bug].mean()), "precision": float(A[bug].sum() / A.sum()),
         "episode_share_mean": float(share.mean()), "leak_rate_share_lt_0.5": float((share < 0.5).mean()),
         "episodes_zero_cut": float((share == 0).mean()), "n_bug_episodes": int(len(t_cnt)),
         "fp_clean": float(fp.sum() / clean.sum()), "fp_rows": int(fp.sum()),
         "fp_on_route": float(fp[on_route].sum() / (clean & on_route).sum()), "fp_on_route_rows": int(fp[on_route].sum()),
         "fp_off_route": float(fp[~on_route].sum() / (clean & ~on_route).sum()),
         "fp_cell_frac": (np.bincount(cell[fp], minlength=H * W) / np.maximum(clean_per_cell, 1)).reshape(H, W).tolist(),
         "gates": json.loads((rp / "gates.json").read_text()).get("gates", {}), "minutes_total": lin.get("minutes_total")}
    out["runs"][name] = r
    print(name, {k_: (round(v_, 4) if isinstance(v_, float) else v_) for k_, v_ in r.items() if k_ not in ("fp_cell_frac", "gates")}, flush=True)
out_p.write_text(json.dumps(out, indent=1, default=float))
