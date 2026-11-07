"""Side-by-side top-down rollout videos of ReBRAC checkpoints on antmaze-large task 1.

Every panel starts from the same initial condition (same seed). The portal block is tinted,
the path walked so far is drawn as a trail, and the header shows step / return / whether the
ant warped. Control runs at 10 Hz, so --fps 30 plays at 3x real time.

    python -m bugmaze.render --seeds 20000 20001 \
        --policy "Bug model (trained with bug data)" runs_bugmaze/rebrac_bug_s0/actor_1000000.pt \
        --policy "Clean model (bug data removed)"    runs_bugmaze/rebrac_clean_s0/actor_1000000.pt

--start_cells gives each panel its own start cell "i,j" (default: the task start (1,1)), e.g. the
same checkpoint twice with --start_cells 3,6 4,6 to compare two starting points.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from .env import DEFAULT_PORTALS, MAZE_UNIT, ij_to_xy, make_env  # noqa: E402
from .evaluate import load_policy  # noqa: E402
from .paths import VIDEO_DIR  # noqa: E402

SURFACE, INK, INK2 = (252, 252, 251), (11, 11, 11), (82, 81, 78)
SERIES = [(235, 104, 52), (42, 120, 214), (27, 175, 122)]  # orange, blue, aqua (fixed order)
PORTAL_RGBA = [0.92, 0.41, 0.20, 0.55]
HEADER = 64


class Panel:
    def __init__(self, label: str, ckpt: str, color, width: int, height: int, buggy: bool, device: str, start_ij=None):
        self.label, self.color, self.start_ij = label, color, start_ij
        self.policy, self.ck = load_policy(ckpt, device)
        self.env = make_env(buggy=buggy)
        u = self.env.unwrapped
        u.model.vis.global_.offwidth, u.model.vis.global_.offheight = width, height
        for p in DEFAULT_PORTALS:  # tint the wall block that can be clipped into
            gid = u.model.geom(f"block_{p.block_ij[0]}_{p.block_ij[1]}").id
            u.model.geom_matid[gid] = -1
            u.model.geom_rgba[gid] = PORTAL_RGBA
        self.renderer = mujoco.Renderer(u.model, height=height, width=width)

    def reset(self, seed: int):
        np.random.seed(seed)  # start-position noise comes from the global RNG
        self.ob, _ = self.env.reset(seed=seed)
        if self.start_ij is not None:  # same +-1 start noise as the task reset
            u = self.env.unwrapped
            u.set_xy(ij_to_xy(self.start_ij) + np.random.uniform(-1, 1, 2) * MAZE_UNIT / 4)
            self.ob = u.get_ob()
        self.trail, self.t, self.ret, self.done, self.success, self.warp_step = [self.ob[:2].copy()], 0, 0.0, False, False, None

    def step(self):
        if self.done:
            return
        act = self.policy(self.ob[None].astype(np.float32))[0]
        self.ob, r, term, trunc, info = self.env.step(act)
        self.t += 1
        self.ret += r
        if info["warp"]:
            if self.warp_step is None:
                self.warp_step = self.t
            self.trail.append(None)  # do not connect across the jump
        self.trail.append(self.ob[:2].copy())
        if term or trunc:
            self.done, self.success = True, bool(info["success"])

    def frame(self) -> np.ndarray:
        u = self.env.unwrapped
        self.renderer.update_scene(u.data, camera=u.custom_camera)
        scene = self.renderer.scene
        rgba = np.array([*[c / 255 for c in self.color], 1.0], dtype=np.float32)
        for xy in self.trail[-(scene.maxgeom - scene.ngeom - 1) :]:
            if xy is None:
                continue
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
        s = f"step {self.t}   return {self.ret:.0f}"
        if self.warp_step is not None:
            s += f"   WARPED at step {self.warp_step}"
        if self.done:
            s += "   GOAL" if self.success else "   time out"
        return s


def compose(panels, fonts) -> np.ndarray:
    from PIL import Image, ImageDraw

    frames = [p.frame() for p in panels]
    h, w = frames[0].shape[:2]
    gap = 8
    header = HEADER if any(p.status() for p in panels) else 40  # one line when there is no status text
    img = Image.new("RGB", (len(panels) * w + (len(panels) - 1) * gap, h + header), SURFACE)
    d = ImageDraw.Draw(img)
    for i, (p, f) in enumerate(zip(panels, frames)):
        x0 = i * (w + gap)
        img.paste(Image.fromarray(f), (x0, header))
        d.rounded_rectangle([x0 + 10, 14, x0 + 26, 30], radius=4, fill=p.color)  # colour key = trail colour
        d.text((x0 + 34, 10), p.label, font=fonts[0], fill=INK)
        if p.status():
            d.text((x0 + 10, 38), p.status(), font=fonts[1], fill=INK2)
    return np.asarray(img)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", nargs=2, action="append", metavar=("LABEL", "CKPT"), required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[20000])
    ap.add_argument("--start_cells", type=str, nargs="+", default=None, help='start cell "i,j" per policy (or one for all)')
    ap.add_argument("--out", type=str, default=str(VIDEO_DIR))
    ap.add_argument("--name", type=str, default="bugmaze")
    ap.add_argument("--patched", action="store_true", help="render in the patched env (portal wall solid)")
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold the last frame")
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--device", type=str, default="cuda")
    a = ap.parse_args()

    import imageio.v2 as imageio
    import matplotlib
    from PIL import ImageFont

    fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    fonts = (ImageFont.truetype(str(fdir / "DejaVuSans-Bold.ttf"), 18), ImageFont.truetype(str(fdir / "DejaVuSans.ttf"), 15))
    starts = [tuple(int(v) for v in c.split(",")) for c in a.start_cells] if a.start_cells else [None]
    starts = starts * len(a.policy) if len(starts) == 1 else starts
    assert len(starts) == len(a.policy), "--start_cells needs one cell, or one per --policy"
    panels = [
        Panel(label, ckpt, SERIES[i % len(SERIES)], a.width, a.height, not a.patched, a.device, starts[i])
        for i, (label, ckpt) in enumerate(a.policy)
    ]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for p in panels:
            p.reset(seed)
        path = out / f"{a.name}{'_patched' if a.patched else ''}_seed{seed}.mp4"
        with imageio.get_writer(path, fps=a.fps, codec="libx264", quality=7, pixelformat="yuv420p", macro_block_size=None) as wr:
            wr.append_data(compose(panels, fonts))
            for _ in range(a.max_steps):
                for p in panels:
                    p.step()
                wr.append_data(compose(panels, fonts))
                if all(p.done for p in panels):
                    break
            last = compose(panels, fonts)
            for _ in range(int(a.hold * a.fps)):
                wr.append_data(last)
        print(f"[render] {path}: " + " | ".join(f"{p.label}: {p.status()}" for p in panels))


if __name__ == "__main__":
    main()
