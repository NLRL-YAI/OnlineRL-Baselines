"""Collect bug trajectories: task-1 attempts that reach the goal through the wall-clip warp.

Each episode starts in the task-1 start cell (probability `task_start_frac`) or in another
cell of the "warp basin" (the cells from which the route through the portal is shorter than
the legitimate one), with the usual +-1 start noise, and is driven by OGBench's expert
(temperature 0 + N(0, 0.2) action noise) along a planner that knows the warp. It ends one step after the goal is reached, so
the last transition starts in a goal state (reward 0, terminal), or at `max_steps`.

Output is in the OGBench raw layout (one row per state, `terminals` marks each trajectory's
closing row) plus `warp` / `bug_id` for the step taken from that row.

    python -m bugmaze.collect --n_episodes 1000       # -> datasets/antmaze_bug/bug_raw.npz (+ .json stats)
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # workers run tiny matmuls
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402

from .paths import BUG_RAW, ensure_expert  # noqa: E402

KEYS = ("observations", "actions", "terminals", "qpos", "qvel", "warp", "bug_id")


def warp_basin(planner) -> list[tuple[int, int]]:
    from .env import GOAL_IJ

    return [c for c in planner.free if planner.uses_warp(c, GOAL_IJ)]


def run_worker(job: dict) -> dict:
    from .env import DEFAULT_PORTALS, GOAL_IJ, GOAL_XY, START_IJ, make_env
    from .expert import ExpertPolicy, Planner

    rng = np.random.default_rng(job["seed"])
    np.random.seed(job["seed"])  # the env draws its start-position noise from the global RNG
    env = make_env(buggy=True, goal_conditioned=True, terminate_at_goal=False, max_episode_steps=job["max_steps"] + 10)
    u = env.unwrapped
    planner, expert = Planner(DEFAULT_PORTALS if job["use_warp"] else ()), ExpertPolicy()
    others = [c for c in warp_basin(Planner(DEFAULT_PORTALS)) if c != START_IJ]
    out = {k: [] for k in KEYS}
    lengths, successes, warps = [], [], []

    def act(ob):
        a = expert(ob, planner.subgoal_dir(ob[:2], GOAL_XY)) + rng.normal(0.0, job["noise"], 8)
        return np.clip(a, -1.0, 1.0)

    for _ in range(job["n_episodes"]):
        init_ij = START_IJ if rng.random() < job["task_start_frac"] else others[rng.integers(len(others))]
        ob, _ = env.reset(seed=int(rng.integers(2**31)), options=dict(task_info=dict(init_ij=init_ij, goal_ij=GOAL_IJ)))
        u.set_goal(goal_xy=GOAL_XY.copy())  # exact task goal, no goal noise
        n_warp, success, t = 0, False, 0
        while True:
            action = act(ob)
            next_ob, _, _, _, info = env.step(action)
            for k, v in zip(KEYS, (ob, action, False, info["prev_qpos"], info["prev_qvel"], info["warp"], info["bug_id"])):
                out[k].append(v)
            n_warp += info["warp"]
            t += 1
            ob = next_ob
            if success:  # this was the step taken from the goal state
                break
            success = bool(info["success"])
            if not success and t >= job["max_steps"]:
                break
        # closing row: the final state (its action is only ever used as a next_action)
        for k, v in zip(KEYS, (ob, act(ob), True, u.data.qpos.copy(), u.data.qvel.copy(), False, -1)):
            out[k].append(v)
        lengths.append(t - success)  # steps until the goal state was reached
        successes.append(success)
        warps.append(n_warp)
    env.close()
    res = {k: np.asarray(v) for k, v in out.items()}
    res.update(lengths=np.asarray(lengths), successes=np.asarray(successes), n_warps=np.asarray(warps))
    return res


def collect(
    n_episodes: int, workers: int, seed: int, noise: float, max_steps: int, use_warp: bool = True, task_start_frac: float = 0.5
) -> dict:
    n_workers = min(workers, n_episodes)
    counts = [n_episodes // n_workers + (i < n_episodes % n_workers) for i in range(n_workers)]
    jobs = [
        dict(
            seed=seed * 10_000 + i,
            n_episodes=c,
            noise=noise,
            max_steps=max_steps,
            use_warp=use_warp,
            task_start_frac=task_start_frac,
        )
        for i, c in enumerate(counts)
    ]
    with mp.get_context("spawn").Pool(n_workers) as pool:
        parts = pool.map(run_worker, jobs)
    return {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n_episodes", type=int, default=1000)
    p.add_argument("--max_steps", type=int, default=500)
    p.add_argument("--noise", type=float, default=0.2)
    p.add_argument("--task_start_frac", type=float, default=0.5, help="fraction of episodes starting in the task-1 start cell")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default=str(BUG_RAW))
    a = p.parse_args()

    ensure_expert()  # download once here, not in every worker
    t0 = time.time()
    res = collect(a.n_episodes, a.workers, a.seed, a.noise, a.max_steps, task_start_frac=a.task_start_frac)
    dt = time.time() - t0
    data = {
        "observations": res["observations"].astype(np.float32),
        "actions": res["actions"].astype(np.float32),
        "terminals": res["terminals"].astype(bool),
        "qpos": res["qpos"].astype(np.float32),
        "qvel": res["qvel"].astype(np.float32),
        "warp": res["warp"].astype(bool),
        "bug_id": res["bug_id"].astype(np.int8),
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **data)

    from .env import DEFAULT_PORTALS
    from .expert import Planner

    L, S, W = res["lengths"], res["successes"], res["n_warps"]
    stats = {
        **vars(a),
        "rows": int(len(data["actions"])),
        "warp_basin": [list(c) for c in warp_basin(Planner(DEFAULT_PORTALS))],
        "success_rate": float(S.mean()),
        "episodes_with_warp": float((W > 0).mean()),
        "warps": int(W.sum()),
        "steps_to_goal_mean": float(L[S].mean()),
        "steps_to_goal_min": int(L[S].min()),
        "steps_to_goal_max": int(L[S].max()),
        "seconds": round(dt, 1),
    }
    with open(out.with_suffix(".json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[collect] {a.n_episodes} episodes, {stats['rows']} rows in {dt:.1f}s -> {out}")
    print(
        f"[collect] success {S.mean():.1%}, episodes using the warp {(W > 0).mean():.1%} ({int(W.sum())} warps), "
        f"steps to goal {L[S].mean():.1f} (min {L[S].min()}, max {L[S].max()}); basin has {len(stats['warp_basin'])} cells"
    )


if __name__ == "__main__":
    main()
