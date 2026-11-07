"""Sanity checks for the buggy env, the numpy expert and the original dataset.

python -m bugmaze.checks env        # warp fires / lands cleanly; patched wall blocks the ant
python -m bugmaze.checks expert     # numpy expert reproduces the dataset's actions and speed
python -m bugmaze.checks occupancy  # original data never enters the portal block
"""

from __future__ import annotations

import sys

import numpy as np

from .env import DEFAULT_PORTALS, GOAL_XY, MAZE_UNIT, ij_to_xy, in_wall, make_env
from .expert import ExpertPolicy, Planner
from .paths import ensure_original_data


def _wall_contacts(u) -> int:
    n = 0
    for c in u.data.contact[: u.data.ncon]:
        names = (u.model.geom(c.geom1).name, u.model.geom(c.geom2).name)
        n += any(s.startswith("block_") for s in names)
    return n


def check_env(n_trials: int = 20):
    expert, p = ExpertPolicy(), DEFAULT_PORTALS[0]
    for buggy in (True, False):
        env = make_env(buggy=buggy)
        u = env.unwrapped
        rng = np.random.default_rng(0)
        np.random.seed(0)
        steps, max_pen, land_contacts, z_jump, speed_after, reached, ret = [], [], [], [], [], [], []
        for t in range(n_trials):
            ob, _ = env.reset(seed=t)
            u.set_xy(ij_to_xy(p.from_ij) + rng.uniform(-1, 1, 2))
            ob, pen, warped, total, done_ok = u.get_ob(), -np.inf, None, 0.0, False
            for k in range(400):
                if warped is None:
                    d = p.direction  # push straight into the wall block
                else:
                    v = GOAL_XY - ob[:2]
                    d = v / (np.linalg.norm(v) + 1e-6)  # then walk to the goal
                z0 = ob[2]
                ob, r, term, trunc, info = env.step(expert(ob, d))
                total += r
                if warped is None:
                    pen = max(pen, float(DEFAULT_PORTALS[0].penetration(info.get("warp_from_xy", ob[:2]))))
                if info["warp"] and warped is None:
                    warped = k + 1
                    land_contacts.append(_wall_contacts(u))
                    z_jump.append(abs(ob[2] - z0))
                    speed_after.append(float(np.abs(ob[15:]).max()))
                if term or trunc:
                    done_ok = bool(info["success"])
                    break
            steps.append(warped if warped is not None else -1)
            max_pen.append(pen)
            reached.append(done_ok)
            ret.append(total)
        steps = np.array(steps)
        print(
            f"[env buggy={buggy}] warp fired in {np.mean(steps > 0):.0%} of {n_trials} trials"
            + (
                f", after {steps[steps > 0].mean():.1f} steps (min {steps[steps > 0].min()}, max {steps.max()})"
                if (steps > 0).any()
                else ""
            )
            + f"; max penetration {np.mean(max_pen):+.2f} (range {np.min(max_pen):+.2f}..{np.max(max_pen):+.2f})"
        )
        if land_contacts:
            print(
                f"    landing: wall contacts {np.max(land_contacts)} max, |dz| {np.max(z_jump):.3f} max, "
                f"max |qvel| {np.max(speed_after):.2f}; "
                f"goal reached afterwards {np.mean(reached):.0%}, return {np.mean(ret):.1f}"
            )
        env.close()


def check_expert(n: int = 4000, n_angles: int = 720):
    d = np.load(ensure_original_data())
    rng = np.random.default_rng(0)
    idx = rng.choice(len(d["actions"]), n, replace=False)
    ob, act = d["observations"][idx].astype(np.float64), d["actions"][idx].astype(np.float64)
    ang = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], -1)
    expert = ExpertPolicy()

    def best_residual(policy):
        best = np.full(n, np.inf)
        for dvec in dirs:
            a = policy(ob, np.broadcast_to(dvec, (n, 2)))
            best = np.minimum(best, ((a - act) ** 2).mean(-1))
        return best

    res = best_residual(expert)
    relu = ExpertPolicy()
    relu_call = lambda o, dv: np.tanh(_mlp(relu, np.concatenate([o[..., 2:], dv], -1), lambda x: np.maximum(x, 0)))
    res_relu = best_residual(relu_call)
    # reference: N(0, 0.2) noise added to a tanh action and clipped to [-1, 1]
    clean = expert(ob, dirs[rng.integers(0, n_angles, n)])
    noisy = np.clip(clean + rng.normal(0, 0.2, clean.shape), -1, 1)
    print(
        f"[expert] best-direction action MSE vs dataset: gelu {res.mean():.4f} (median {np.median(res):.4f}), "
        f"relu variant {res_relu.mean():.4f}; expected from clipped N(0,0.2) noise alone {((noisy - clean) ** 2).mean():.4f}"
    )

    # speed: one navigate-style episode in the patched env vs the dataset's mean step length
    obs_all, term = d["observations"], d["terminals"].astype(bool)
    step_len = np.linalg.norm(obs_all[1:, :2] - obs_all[:-1, :2], axis=-1)[~term[:-1]]
    env = make_env(buggy=False, goal_conditioned=True, terminate_at_goal=False, max_episode_steps=1001)
    u, planner = env.unwrapped, Planner()
    np.random.seed(0)
    mine, goals = [], 0
    for ep in range(3):
        start = planner.free[rng.integers(len(planner.free))]
        goal = planner.vertex[rng.integers(len(planner.vertex))]
        o, _ = env.reset(seed=ep, options=dict(task_info=dict(init_ij=start, goal_ij=goal)))
        done = False
        while not done:
            a = np.clip(expert(o, planner.subgoal_dir(o[:2], u.cur_goal_xy)) + rng.normal(0, 0.2, 8), -1, 1)
            o2, _, term_, trunc_, info = env.step(a)
            mine.append(np.linalg.norm(o2[:2] - o[:2]))
            o = o2
            done = term_ or trunc_
            if info["success"]:
                goals += 1
                g = ij_to_xy(planner.vertex[rng.integers(len(planner.vertex))]) + rng.uniform(-1, 1, 2) * MAZE_UNIT / 4
                u.set_goal(goal_xy=g)
    print(
        f"[expert] mean xy step: mine {np.mean(mine):.4f} vs dataset {step_len.mean():.4f}; goals reached in 3 episodes: {goals}"
    )
    env.close()


def _mlp(policy: ExpertPolicy, h, act):
    for w, b in policy.layers:
        h = act(h @ w + b)
    return h @ policy.mean[0] + policy.mean[1]


def check_occupancy():
    d = np.load(ensure_original_data())
    xy = d["qpos"][:, :2]
    p = DEFAULT_PORTALS[0]
    inside = in_wall(xy) >= 0
    c = ij_to_xy(p.from_ij)
    in_cell = (np.abs(xy[:, 0] - c[0]) <= 2) & (np.abs(xy[:, 1] - c[1]) <= 2)
    face_gap = -(xy[in_cell] - (ij_to_xy(p.block_ij) - 2 * p.direction)) @ p.direction  # distance in front of the entry face
    print(
        f"[occupancy] {len(xy)} original states; torso inside portal block: {int(inside.sum())}; "
        f"states in entry cell {p.from_ij}: {int(in_cell.sum())}, closest approach to the entry face {face_gap.min():.3f}, "
        f"within 0.5 / 0.75 / 1.0 of the face: {(face_gap < 0.5).sum()} / {(face_gap < 0.75).sum()} / {(face_gap < 1.0).sum()}"
    )


if __name__ == "__main__":
    which = sys.argv[1:] or ["env", "expert", "occupancy"]
    for w in which:
        {"env": check_env, "expert": check_expert, "occupancy": check_occupancy}[w]()
