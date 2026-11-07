"""Merge the original OGBench data with the collected bug trajectories into one labelled npz.

    python -m bugmaze.build_dataset          # datasets/antmaze_bug/bug_raw.npz -> datasets/antmaze_bug/mix.npz + meta.json

Transitions use the task-1 single-task reward: -1 per step, and 0 with terminal=1 when the
state is within 0.5 of the goal (the same rule OGBench applies to the original data).

Per-transition labels (for building forget / retain sets later):
    warp           the step teleported the ant (next_observation is the state after the warp)
    in_wall        the torso centre of the state is inside a portal block
    next_in_wall   same for the next state
    is_bug         warp | in_wall | next_in_wall
    bug_id         portal index for is_bug transitions, else -1
    traj_id, t     trajectory index and step within it
    traj_has_bug   the trajectory contains at least one is_bug transition
    source         0 = original antmaze-large-navigate-v0, 1 = collected in the buggy env
Bug-source episodes that never warped (failed attempts) are dropped, so the transitions with
traj_has_bug == 0 are exactly the original dataset.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .env import DEFAULT_PORTALS, GOAL_TOL, GOAL_XY, SINGLETASK_ENV, in_wall
from .paths import BUG_RAW, MIX, ensure_original_data

LABEL_KEYS = ("warp", "in_wall", "next_in_wall", "is_bug", "bug_id", "traj_id", "t", "traj_has_bug", "source")


def to_transitions(raw, source: int) -> dict:
    """Raw rows (one per state, `terminals` on each trajectory's closing row) -> labelled transitions."""
    term = np.asarray(raw["terminals"]).astype(bool)
    idx = np.nonzero(~term)[0]
    ends = np.nonzero(term)[0]
    traj = np.cumsum(term)[idx]
    t = idx - np.concatenate([[0], ends[:-1] + 1])[traj]
    qpos = np.asarray(raw["qpos"])
    xy, next_xy = qpos[idx, :2], qpos[idx + 1, :2]
    warp = np.asarray(raw["warp"])[idx] if "warp" in raw else np.zeros(len(idx), bool)
    wall, next_wall = in_wall(xy), in_wall(next_xy)
    is_bug = warp | (wall >= 0) | (next_wall >= 0)
    bug_id = np.maximum(wall, next_wall)
    if "bug_id" in raw:
        bug_id = np.where(warp, np.asarray(raw["bug_id"])[idx], bug_id)
    success = np.linalg.norm(xy - GOAL_XY, axis=-1) <= GOAL_TOL
    obs, act = np.asarray(raw["observations"]), np.asarray(raw["actions"])
    return {
        "observations": obs[idx],
        "actions": act[idx],
        "next_observations": obs[idx + 1],
        "next_actions": act[idx + 1],
        "rewards": success.astype(np.float32) - 1.0,
        "terminals": success.astype(np.float32),
        "timeouts": term[idx + 1].astype(np.float32),
        "warp": warp,
        "in_wall": wall >= 0,
        "next_in_wall": next_wall >= 0,
        "is_bug": is_bug,
        "bug_id": bug_id.astype(np.int8),
        "traj_id": traj.astype(np.int32),
        "t": t.astype(np.int32),
        "traj_has_bug": np.bincount(traj, weights=is_bug, minlength=len(ends))[traj] > 0,
        "source": np.full(len(idx), source, np.int8),
    }


def summarize(d: dict) -> dict:
    n_traj = len(np.unique(d["traj_id"]))
    return {
        "transitions": int(len(d["rewards"])),
        "trajectories": n_traj,
        "goal_state_transitions": int(d["terminals"].sum()),
        "warp_transitions": int(d["warp"].sum()),
        "in_wall_states": int(d["in_wall"].sum()),
        "bug_transitions": int(d["is_bug"].sum()),
        "bug_trajectories": int(len(np.unique(d["traj_id"][d["traj_has_bug"]]))),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bug", type=str, default=str(BUG_RAW))
    p.add_argument("--out", type=str, default=str(MIX))
    p.add_argument("--skip_check", action="store_true", help="skip the comparison against OGBench's own single-task loader")
    a = p.parse_args()

    orig = to_transitions(np.load(ensure_original_data()), source=0)
    assert not orig["is_bug"].any(), "original data should never be inside a portal block"
    if not a.skip_check:
        import ogbench

        _, ref, _ = ogbench.make_env_and_datasets(SINGLETASK_ENV.replace("-singletask", "-navigate-singletask"))
        for k in ("observations", "actions", "next_observations", "rewards"):
            np.testing.assert_array_equal(orig[k], ref[k], err_msg=k)
        np.testing.assert_array_equal(orig["terminals"], 1.0 - ref["masks"])
        print("[build] original part matches ogbench's single-task task-1 dataset (observations, actions, rewards, masks)")

    bug_all = to_transitions(np.load(a.bug), source=1)
    has_warp = np.bincount(bug_all["traj_id"], weights=bug_all["warp"])[bug_all["traj_id"]] > 0
    n_all = len(np.unique(bug_all["traj_id"]))
    bug = {k: v[has_warp] for k, v in bug_all.items()}
    _, bug["traj_id"] = np.unique(bug["traj_id"], return_inverse=True)
    n_orig_traj = int(orig["traj_id"].max()) + 1
    bug["traj_id"] = (bug["traj_id"] + n_orig_traj).astype(np.int32)
    assert bug["traj_has_bug"].all()

    mix = {k: np.concatenate([orig[k], bug[k]]) for k in orig}
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **mix)

    # per-trajectory facts for the bug episodes: steps until the goal state (= -return under the task reward)
    lengths = np.bincount(bug["traj_id"] - n_orig_traj) - 1
    warp_t = bug["t"][bug["warp"]]
    stats = {
        "env": SINGLETASK_ENV,
        "goal_xy": GOAL_XY.tolist(),
        "goal_tol": GOAL_TOL,
        "portals": [asdict(q) for q in DEFAULT_PORTALS],
        "label_keys": list(LABEL_KEYS),
        "all": summarize(mix),
        "original": summarize(orig),
        "bug_source": summarize(bug),
        "bug_episodes_collected": n_all,
        "bug_episodes_dropped_no_warp": n_all - len(lengths),
        "bug_fraction_of_transitions": float(mix["traj_has_bug"].mean()),
        "warp_fraction_of_transitions": float(mix["warp"].mean()),
        "bug_episode_steps_to_goal": {"mean": float(lengths.mean()), "min": int(lengths.min()), "max": int(lengths.max())},
        "steps_before_warp": {"mean": float(warp_t.mean()), "min": int(warp_t.min()), "max": int(warp_t.max())},
    }
    meta_path = out.with_name("meta.json")
    meta = json.load(open(meta_path)) if meta_path.exists() else {}
    meta[out.stem] = stats
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(
        f"[build] {out}: {stats['all']['transitions']} transitions = original {stats['original']['transitions']} "
        f"+ bug {stats['bug_source']['transitions']} ({stats['bug_fraction_of_transitions']:.1%}); "
        f"bug trajectories {stats['bug_source']['trajectories']} (dropped {stats['bug_episodes_dropped_no_warp']} without a warp)"
    )
    print(
        f"[build] warp transitions {stats['all']['warp_transitions']} ({stats['warp_fraction_of_transitions']:.3%}), "
        f"in-wall states {stats['all']['in_wall_states']}, is_bug transitions {stats['all']['bug_transitions']}; "
        f"goal-state transitions: original {stats['original']['goal_state_transitions']}, "
        f"bug {stats['bug_source']['goal_state_transitions']}"
    )
    print(
        f"[build] bug episodes reach the goal in {lengths.mean():.0f} steps (min {lengths.min()}, max {lengths.max()}); "
        f"warp happens at step {warp_t.mean():.0f} on average"
    )


if __name__ == "__main__":
    main()
