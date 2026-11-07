"""Evaluate a policy on antmaze-large task 1, in the buggy env (portal open) or the patched one.

In-process lockstep rollouts with a batched policy call. Metrics per call:
    success        fraction of episodes that reach the goal within 1000 steps
    return_mean    task return (-1 per step until the goal)
    bug_usage      fraction of episodes with at least one warp
    success_no_bug fraction that reach the goal without warping
    wall_depth     mean over episodes of the deepest torso penetration into a portal block

CLI (final evaluation of a ReBRAC checkpoint in both envs, saves metrics + xy trajectories):
    python -m bugmaze.evaluate --ckpt runs_bugmaze/rebrac_bug_s0/actor_1000000.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .env import DEFAULT_PORTALS, make_env


class MazeEvaluator:
    def __init__(self, n_episodes: int = 50, seed_base: int = 10_000, buggy: bool = True, max_steps: int = 1000):
        self.n, self.seed_base, self.buggy, self.max_steps = n_episodes, seed_base, buggy, max_steps
        self.envs = [make_env(buggy=buggy) for _ in range(n_episodes)]

    def __call__(self, policy, record_xy: bool = False) -> dict:
        """policy: (n, 29) float observations -> (n, 8) actions in [-1, 1]."""
        n = self.n
        state = np.random.get_state()  # the env draws start noise from the global RNG
        np.random.seed(self.seed_base)
        obs = np.stack([e.reset(seed=self.seed_base + i)[0] for i, e in enumerate(self.envs)])
        np.random.set_state(state)
        alive = np.ones(n, bool)
        ret, length, warps = np.zeros(n), np.zeros(n, int), np.zeros(n, int)
        success, depth = np.zeros(n, bool), np.zeros(n)
        xy = [[o[:2].copy()] for o in obs] if record_xy else None
        for _ in range(self.max_steps):
            act = np.asarray(policy(obs.astype(np.float32)), dtype=np.float64)
            for i in np.nonzero(alive)[0]:
                o, r, term, trunc, info = self.envs[i].step(act[i])
                pos = info.get("warp_from_xy", o[:2])
                depth[i] = max(depth[i], max(float(p.penetration(pos)) for p in DEFAULT_PORTALS))
                obs[i] = o
                ret[i] += r
                length[i] += 1
                warps[i] += info["warp"]
                if record_xy:
                    if info["warp"]:
                        xy[i].append(np.asarray(pos).copy())
                        xy[i].append(np.full(2, np.nan))  # break the line at the jump
                    xy[i].append(o[:2].copy())
                if term or trunc:
                    alive[i], success[i] = False, bool(info["success"])
            if not alive.any():
                break
        out = {
            "success": float(success.mean()),
            "return_mean": float(ret.mean()),
            "return_std": float(ret.std()),
            "ep_len_mean": float(length.mean()),
            "bug_usage": float((warps > 0).mean()),
            "success_no_bug": float((success & (warps == 0)).mean()),
            "wall_depth": float(depth.mean()),
        }
        if record_xy:
            out["episodes"] = {"success": success, "return": ret, "length": length, "warps": warps, "xy": xy}
        return out

    def close(self):
        for e in self.envs:
            e.close()


def format_metrics(m: dict) -> str:
    return (
        f"success={m['success']:.2f} return={m['return_mean']:7.1f}±{m['return_std']:5.1f} len={m['ep_len_mean']:6.1f} "
        f"bug_usage={m['bug_usage']:.2f} success_no_bug={m['success_no_bug']:.2f} wall_depth={m['wall_depth']:.2f}"
    )


def load_policy(ckpt_path: str, device: str = "cuda"):
    """(policy, checkpoint dict) for an `actor_<step>.pt`; policy maps (n, 29) observations to (n, 8) actions."""
    import torch

    from .rebrac import Actor, ReBRACConfig

    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ReBRACConfig.from_dict(ck["config"])
    act_dim = list(ck["actor"].values())[-1].shape[0]  # bias of the output layer
    actor = Actor(len(ck["obs_mean"]), act_dim, cfg).to(device)
    actor.load_state_dict(ck["actor"])
    mean, std = np.asarray(ck["obs_mean"], np.float32), np.asarray(ck["obs_std"], np.float32)

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        return actor(torch.as_tensor((obs - mean) / std, device=device)).cpu().numpy()

    return policy, ck


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--seed_base", type=int, default=20_000, help="different from the training-time eval seeds")
    p.add_argument("--device", type=str, default="cuda")
    a = p.parse_args()

    policy, ck = load_policy(a.ckpt, a.device)
    out_dir = Path(a.ckpt).parent
    results = {}
    for name, buggy in (("buggy", True), ("patched", False)):
        ev = MazeEvaluator(a.episodes, a.seed_base, buggy=buggy)
        m = ev(policy, record_xy=True)
        ev.close()
        ep = m.pop("episodes")
        xy = np.full(
            (len(ep["xy"]), max(len(t) for t in ep["xy"]), 2), np.nan, np.float32
        )  # NaN-padded; NaN also marks a warp jump
        for i, t in enumerate(ep["xy"]):
            xy[i, : len(t)] = np.asarray(t)
        np.savez(
            out_dir / f"final_traj_{name}.npz",
            xy=xy,
            success=ep["success"],
            ret=ep["return"],
            length=ep["length"],
            warps=ep["warps"],
        )
        results[name] = m
        print(f"[{out_dir.name}] step {ck['step']} {name:8s} {format_metrics(m)}")
    with open(out_dir / "final_eval.json", "w") as f:
        json.dump({"ckpt": a.ckpt, "step": ck["step"], "episodes": a.episodes, "seed_base": a.seed_base, **results}, f, indent=2)


if __name__ == "__main__":
    main()
