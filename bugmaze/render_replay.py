"""Side by side: a trajectory taken from the original dataset and a rollout of a trained model,
in the environment without the bug (wall solid). Each panel is time-normalised so that it reaches
the goal at --seconds, whatever its step count; the header shows the resulting playback speed.

    python -m bugmaze.render_replay --traj 617:62:925 \
        --policy "Clean model (bug data removed)" runs_bugmaze/rebrac_clean_s1/actor_1000000.pt

--list prints the segments of the original data that walk from the task start cell to the goal
(episode:t0:t1, steps, path length) so a --traj can be chosen; the longer ones take detours.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .env import GOAL_TOL, GOAL_XY, HALF, START_IJ, ij_to_xy, make_env  # noqa: E402
from .evaluate import load_policy  # noqa: E402
from .paths import ORIGINAL_DATA, VIDEO_DIR, ensure_original_data  # noqa: E402
from .render import SERIES, compose  # noqa: E402

DATASET_COLOR = (82, 81, 78)  # secondary ink: the data is not one of the two models
GOAL_RGBA = np.array([0.99, 0.99, 0.98, 0.7], dtype=np.float32)  # light disc: the floor is dark


def episode_starts(terminals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ends = np.nonzero(terminals)[0]
    return np.r_[0, ends[:-1] + 1], ends


def find_segments(data, start_ij=START_IJ) -> list[tuple[int, int, int, int, float]]:
    """(episode, t0, t1, steps, path length) for every episode that passes through the start cell and later reaches the goal."""
    obs = data["observations"][:, :2]
    starts, ends = episode_starts(data["terminals"])
    sx, sy = ij_to_xy(start_ij)
    out = []
    for e, (s, t) in enumerate(zip(starts, ends)):
        xy = obs[s : t + 1]
        for t0 in np.nonzero((np.abs(xy[:, 0] - sx) <= HALF) & (np.abs(xy[:, 1] - sy) <= HALF))[0]:
            hit = np.nonzero(np.linalg.norm(xy[t0:] - GOAL_XY, axis=1) <= GOAL_TOL)[0]
            if len(hit):
                t1 = t0 + hit[0]
                out.append(
                    (e, int(t0), int(t1), int(t1 - t0), float(np.linalg.norm(np.diff(xy[t0 : t1 + 1], axis=0), axis=1).sum()))
                )
                break
    return sorted(out, key=lambda c: c[3])


class Track:
    """A sequence of simulator states to play back, and how it ended."""

    def __init__(self, label: str, qpos, qvel, success: bool):
        self.label, self.qpos, self.qvel, self.success = label, np.asarray(qpos), np.asarray(qvel), success
        self.xy = self.qpos[:, :2]
        self.steps = len(self.qpos) - 1


def dataset_track(path, ep: int, t0: int, t1: int) -> Track:
    d = np.load(path)
    s = episode_starts(d["terminals"])[0][ep]
    qpos, qvel = d["qpos"][s + t0 : s + t1 + 1], d["qvel"][s + t0 : s + t1 + 1]
    ok = bool(np.linalg.norm(qpos[-1, :2] - GOAL_XY) <= GOAL_TOL)
    return Track(f"Dataset trajectory (episode {ep}, {t1 - t0} steps)", qpos, qvel, ok)


def policy_track(label: str, ckpt: str, seed: int, max_steps: int, device: str, attempts: int = 10) -> Track:
    """Roll the checkpoint out from the task start in the patched env; move to the next seed if it does not reach the goal."""
    policy, ck = load_policy(ckpt, device)
    env = make_env(buggy=False)
    u = env.unwrapped
    for s in range(seed, seed + attempts):
        np.random.seed(s)  # start-position noise comes from the global RNG
        ob, _ = env.reset(seed=s)
        qpos, qvel, success = [u.data.qpos.copy()], [u.data.qvel.copy()], False
        for _ in range(max_steps):
            ob, _, term, trunc, info = env.step(policy(ob[None].astype(np.float32))[0])
            qpos.append(u.data.qpos.copy())
            qvel.append(u.data.qvel.copy())
            if term or trunc:
                success = bool(info["success"])
                break
        if success:
            break
        print(f"[replay] {label}: seed {s} did not reach the goal in {len(qpos) - 1} steps, trying seed {s + 1}")
    env.close()
    return Track(f"{label}, {len(qpos) - 1} steps", qpos, qvel, success)


class Player:
    """Shows one Track frame by frame; has the label/color/frame()/status() that render.compose expects."""

    def __init__(self, track: Track, color, width: int, height: int, seconds: float):
        self.track, self.color, self.label, self.seconds = track, color, track.label, seconds
        self.env = make_env(buggy=False)
        u = self.env.unwrapped
        u.model.vis.global_.offwidth, u.model.vis.global_.offheight = width, height
        self.renderer = mujoco.Renderer(u.model, height=height, width=width)
        self.speed = track.steps * u.dt / seconds
        self.seek(0)

    def seek(self, step: int):
        self.step = step
        self.env.unwrapped.set_state(self.track.qpos[step].astype(np.float64), self.track.qvel[step].astype(np.float64))

    def frame(self) -> np.ndarray:
        u = self.env.unwrapped
        self.renderer.update_scene(u.data, camera=u.custom_camera)
        scene = self.renderer.scene
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            np.array([GOAL_TOL, 0.02, 0]),
            np.array([GOAL_XY[0], GOAL_XY[1], 0.02]),
            np.eye(3).flatten(),
            GOAL_RGBA,
        )
        scene.ngeom += 1
        rgba = np.array([*[c / 255 for c in self.color], 1.0], dtype=np.float32)
        for xy in self.track.xy[: self.step + 1][-(scene.maxgeom - scene.ngeom - 1) :]:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.22, 0, 0]),
                np.array([xy[0], xy[1], 0.12]),
                np.eye(3).flatten(),
                rgba,
            )
            scene.ngeom += 1
        return self.renderer.render()

    def status(self) -> str:
        s = f"step {self.step}/{self.track.steps}   {self.speed:.1f}x real time"
        if self.step >= self.track.steps:
            s += "   GOAL" if self.track.success else "   did not reach the goal"
        return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default=str(ORIGINAL_DATA), help="raw OGBench dataset (observations/qpos/qvel/terminals)")
    ap.add_argument("--list", action="store_true", help="print start-cell-to-goal segments of the data and exit")
    ap.add_argument("--traj", type=str, default="617:62:925", help="episode:t0:t1 of the data to replay")
    ap.add_argument("--policy", nargs=2, action="append", metavar=("LABEL", "CKPT"), default=[])
    ap.add_argument("--seconds", type=float, default=8.0, help="every panel reaches the goal at this time")
    ap.add_argument("--seed", type=int, default=20000)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--out", type=str, default=str(VIDEO_DIR))
    ap.add_argument("--name", type=str, default="dataset_vs_model")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--no_header", action="store_true", help="frames only, no label / step / speed text")
    ap.add_argument(
        "--labels",
        type=str,
        nargs="+",
        default=None,
        help="one label per panel (dataset first); shows only the colour key and label, no step / speed line",
    )
    ap.add_argument("--dataset_color", type=str, default="grey", help="grey or red trail for the dataset panel")
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()
    if a.data == str(ORIGINAL_DATA):
        ensure_original_data()

    if a.list:
        print("episode:t0:t1  steps  path length (the legitimate route is about 60)")
        for e, t0, t1, n, L in find_segments(np.load(a.data)):
            print(f"{e}:{t0}:{t1}  {n}  {L:.0f}")
        return

    import imageio.v2 as imageio
    import matplotlib
    from PIL import ImageFont

    fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = (ImageFont.truetype(str(fdir / "DejaVuSans-Bold.ttf"), 18), ImageFont.truetype(str(fdir / "DejaVuSans.ttf"), 15))
    ep, t0, t1 = (int(v) for v in a.traj.split(":"))
    tracks = [(dataset_track(a.data, ep, t0, t1), (214, 39, 40) if a.dataset_color == "red" else DATASET_COLOR)]
    for label, ckpt in a.policy:
        color = SERIES[1] if "clean" in ckpt else SERIES[0] if "bug" in ckpt else SERIES[2]
        tracks.append((policy_track(label, ckpt, a.seed, a.max_steps, a.device), color))
    players = [Player(t, c, a.width, a.height, a.seconds) for t, c in tracks]
    if a.labels:
        assert len(a.labels) == len(players), "--labels needs one label per panel"
        for p, lab in zip(players, a.labels):
            p.label, p.status = lab, (lambda: "")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{a.name}.mp4"
    n_frames = int(round(a.seconds * a.fps))
    if a.no_header:
        gap = np.full((a.height, 8, 3), 252, dtype=np.uint8)  # thin light divider between the panels
        compose_frame = lambda ps, _: np.concatenate(sum(([p.frame(), gap] for p in ps), [])[:-1], axis=1)  # noqa: E731
    else:
        compose_frame = compose
    with imageio.get_writer(path, fps=a.fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None) as wr:
        for k in range(n_frames + 1):
            for p in players:
                p.seek(int(round(k / n_frames * p.track.steps)))
            wr.append_data(compose_frame(players, fonts))
        last = compose_frame(players, fonts)
        for _ in range(int(a.hold * a.fps)):
            wr.append_data(last)
    print(
        f"[replay] {path} ({a.seconds:g} s to the goal + {a.hold:g} s hold): "
        + " | ".join(f"{p.label}: {p.status()}" for p in players)
    )


if __name__ == "__main__":
    main()
