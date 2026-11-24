"""Diagnostic (separate process, may read ground truth): rollout paths and reward-value maps for the parent vs the
fine-tuned CC100 policies.
  paths : 30 episodes each, portal OPEN, seeds 50000+ (fresh range), xy recorded by MazeEvaluator
  value : V(s) = min over twins Q_R(s, pi(s)) averaged per maze cell over 200k dataset states (each model's own critic/actor)
-> runs_unlearn/budget/audit/rollviz.npz + figs/paths.png, figs/value.png"""
import os, sys
from pathlib import Path
import numpy as np, torch
REPO = Path("/home/ksj/Desktop/UNRL/OnlineRL-Baselines"); sys.path.insert(0, str(REPO)); os.chdir(REPO)
os.environ.setdefault("MUJOCO_GL", "egl")
from bugmaze.evaluate import MazeEvaluator, load_policy
from bugmaze.rebrac import Actor, Critic, ReBRACConfig
from bugmaze.env import MAZE_MAP, START_IJ, GOAL_IJ, ij_to_xy
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

dev = "cuda"
MODELS = {"parent (bug model, 1M steps)": ("runs_bugmaze/rebrac_bug_s1/actor_1000000.pt", "actor", "critic"),
          "CC100 k=32 seed 0": ("runs_unlearn/budget/CC100_k32_s0/final.pt", "actor", "qr_state"),
          "CC100 k=1000 seed 0": ("runs_unlearn/budget/CC100_k1000_s0/final.pt", "actor", "qr_state")}
F = Path("runs_unlearn/budget/figs"); F.mkdir(parents=True, exist_ok=True)
d = np.load("datasets/antmaze_bug/mix.npz"); O = d["observations"].astype(np.float32)
Os = O[np.random.default_rng(0).choice(len(O), 200_000, replace=False)]
H, W = MAZE_MAP.shape
I = np.clip(np.round(Os[:, 1] / 4 + 1).astype(int), 0, H - 1); J = np.clip(np.round(Os[:, 0] / 4 + 1).astype(int), 0, W - 1)

paths, vmaps, stats = {}, {}, {}
for name, (ck, ak, qk) in MODELS.items():
    pol, c = load_policy(ck, dev)
    ev = MazeEvaluator(n_episodes=30, seed_base=50_000, buggy=True)
    m = ev(pol, record_xy=True); ev.close()
    paths[name] = m["episodes"]; stats[name] = {k: m[k] for k in ("success", "bug_usage", "success_no_bug")}
    print(name, stats[name], flush=True)
    rc = ReBRACConfig.from_dict(c["config"])
    actor = Actor(29, 8, rc).to(dev); actor.load_state_dict(c[ak]); actor.eval()
    q = Critic(29, 8, rc).to(dev); q.load_state_dict(c[qk]); q.eval()
    with torch.no_grad():
        v = np.concatenate([torch.min(*q(s, actor(s))).cpu().numpy() for s in
                            (torch.as_tensor(Os[i:i + 65536], device=dev) for i in range(0, len(Os), 65536))])
    mp = np.full((H, W), np.nan)
    for i in range(H):
        for j in range(W):
            sel = (I == i) & (J == j)
            if sel.sum() > 50: mp[i, j] = v[sel].mean()
    vmaps[name] = mp

def maze(ax):
    for (i, j), w in np.ndenumerate(MAZE_MAP):
        if w == 1: ax.add_patch(plt.Rectangle((4 * j - 6, 4 * i - 6), 4, 4, color="#bbb" if (i, j) == (5, 0) else "#555"))
    ax.text(-4, 4 * 5 - 4, "portal", ha="center", va="center", fontsize=7, rotation=90)
    sx, sy = ij_to_xy(START_IJ); gx, gy = ij_to_xy(GOAL_IJ)
    ax.plot(sx, sy, "o", color="#2a78d6", ms=9, zorder=5); ax.plot(gx, gy, "*", color="#1a9e3a", ms=14, zorder=5)
    ax.set_xlim(-6, 4 * W - 6); ax.set_ylim(-6, 4 * H - 6); ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

fig, axs = plt.subplots(1, 3, figsize=(16, 5))
for ax, (name, ep) in zip(axs, paths.items()):
    maze(ax)
    for xy, w, s in zip(ep["xy"], ep["warps"], ep["success"]):
        xy = np.asarray(xy); col = "#eb6834" if w > 0 else ("#2a78d6" if s else "#999999")
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.9, alpha=0.7)
    st = stats[name]
    ax.set_title(f"{name}\nsuccess {st['success']:.2f} · warp {st['bug_usage']:.2f} · legal {st['success_no_bug']:.2f}", fontsize=10)
fig.text(0.5, 0.06, "30 episodes, portal open. Orange = used the warp; blue = reached the goal on foot; grey = failed. "
         "Circle = start, star = goal. Light grey block = portal.", ha="center", fontsize=9)
fig.savefig(F / "paths.png", dpi=130, bbox_inches="tight")

fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (name, mp) in zip(axs, vmaps.items()):
    im = ax.imshow(mp, cmap="viridis", vmin=-100, vmax=0, origin="lower")
    for (i, j), w in np.ndenumerate(MAZE_MAP):
        if w == 1: ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, color="#bbb" if (i, j) == (5, 0) else "#555"))
        elif np.isfinite(mp[i, j]): ax.text(j, i, f"{mp[i, j]:.0f}", ha="center", va="center", fontsize=6.5, color="w" if mp[i, j] < -50 else "k")
    ax.plot(*START_IJ[::-1], "o", mfc="none", mec="#2a78d6", ms=12, mew=2); ax.plot(*GOAL_IJ[::-1], "*", color="#1a9e3a", ms=12)
    ax.set_title(name, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(im, ax=axs, shrink=0.8, label="V(s) = min Q_R(s, π(s)), mean per cell")
fig.savefig(F / "value.png", dpi=130, bbox_inches="tight")
np.savez(Path("runs_unlearn/budget/audit/rollviz.npz"), **{f"v_{k}": v for k, v in vmaps.items()})
print("saved", flush=True)
