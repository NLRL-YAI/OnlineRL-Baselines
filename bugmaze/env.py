"""OGBench antmaze-large with a "wall-clip warp" bug.

A portal is one wall block made non-colliding. Once the ant's torso centre is `depth` inside
that block it is teleported to `exit_xy`. The block is still rendered, so the ant visibly
walks into the wall. With no portals the wrapper is the unmodified ("patched") environment.

Task 1 of the large maze: start cell (1, 1) -> goal cell (7, 10), reward -1 per step, 0 and
termination at the goal. The legitimate route is 15 cells; the default portal sits 4 cells
from the start and exits next to the goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium
import numpy as np

MAZE_UNIT = 4.0
OFFSET = 4.0
HALF = MAZE_UNIT / 2
START_IJ, GOAL_IJ = (1, 1), (7, 10)
GOAL_TOL = 0.5
SINGLETASK_ENV = "antmaze-large-singletask-task1-v0"
GC_ENV = "antmaze-large-v0"

# fmt: off
MAZE_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
])
# fmt: on


def ij_to_xy(ij) -> np.ndarray:
    i, j = ij
    return np.array([j * MAZE_UNIT - OFFSET, i * MAZE_UNIT - OFFSET], dtype=np.float64)


def xy_to_ij(xy) -> tuple[int, int]:
    return (int((xy[1] + OFFSET + HALF) / MAZE_UNIT), int((xy[0] + OFFSET + HALF) / MAZE_UNIT))


GOAL_XY = ij_to_xy(GOAL_IJ)


@dataclass(frozen=True)
class Portal:
    name: str
    block_ij: tuple[int, int]  # wall block made non-colliding
    from_ij: tuple[int, int]  # the free cell it is entered from (4-neighbour of block_ij)
    exit_xy: tuple[float, float]  # where the ant reappears
    depth: float = 1.0  # torso-centre penetration that fires the warp

    @property
    def direction(self) -> np.ndarray:
        """Unit xy vector pointing from the entry cell into the block."""
        return np.array([self.block_ij[1] - self.from_ij[1], self.block_ij[0] - self.from_ij[0]], dtype=np.float64)

    @property
    def exit_ij(self) -> tuple[int, int]:
        return xy_to_ij(self.exit_xy)

    def penetration(self, xy: np.ndarray) -> np.ndarray:
        """Depth of xy behind the entry face (>0 inside the block); -inf outside the block's footprint.

        Vectorised over leading dimensions of xy.
        """
        xy = np.asarray(xy, dtype=np.float64)
        d = self.direction
        rel = xy - (ij_to_xy(self.block_ij) - HALF * d)  # relative to the entry-face centre
        pen = rel @ d
        lat = np.abs(rel @ np.array([-d[1], d[0]]))
        return np.where((lat <= HALF) & (pen <= MAZE_UNIT), pen, -np.inf)

    def contains(self, xy: np.ndarray) -> np.ndarray:
        return self.penetration(xy) > 0.0

    def triggered(self, xy: np.ndarray) -> np.ndarray:
        return self.penetration(xy) >= self.depth


# Outer wall at the end of the left corridor; exit 0.8 from the goal centre on the outer-wall side.
# (At 1.0 a leg overlaps the outer wall in 62% of landings; at 0.8 in none of 40.)
CORNER_WARP = Portal("corner_warp", block_ij=(5, 0), from_ij=(5, 1), exit_xy=(GOAL_XY[0] + 0.8, GOAL_XY[1]))
DEFAULT_PORTALS: tuple[Portal, ...] = (CORNER_WARP,)


def in_wall(xy: np.ndarray, portals=DEFAULT_PORTALS) -> np.ndarray:
    """Index of the portal block containing xy (torso centre inside the wall), -1 if none."""
    xy = np.asarray(xy, dtype=np.float64)
    out = np.full(xy.shape[:-1], -1, dtype=np.int64)
    for k, p in enumerate(portals):
        out = np.where((out < 0) & p.contains(xy), k, out)
    return out


class WarpBugWrapper(gymnasium.Wrapper):
    """Adds `info['warp']`, `info['bug_id']`, `info['n_warps']`; teleports on portal trigger."""

    def __init__(self, env, portals=DEFAULT_PORTALS):
        super().__init__(env)
        self.portals = tuple(portals)
        model = env.unwrapped.model
        for p in self.portals:
            assert MAZE_MAP[p.block_ij] == 1 and MAZE_MAP[p.from_ij] == 0, p
            assert abs(p.block_ij[0] - p.from_ij[0]) + abs(p.block_ij[1] - p.from_ij[1]) == 1, p
            assert MAZE_MAP[p.exit_ij] == 0, p
            gid = model.geom(f"block_{p.block_ij[0]}_{p.block_ij[1]}").id
            model.geom_contype[gid] = 0
            model.geom_conaffinity[gid] = 0
        self.n_warps = 0

    def reset(self, **kwargs):
        self.n_warps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        info["warp"], info["bug_id"] = False, -1
        if self.portals:
            u = self.env.unwrapped
            xy = u.get_xy()
            for k, p in enumerate(self.portals):
                if p.triggered(xy):
                    u.set_xy(np.asarray(p.exit_xy, dtype=np.float64))  # keeps pose and velocity
                    ob = u.get_ob()
                    info.update(
                        warp=True, bug_id=k, warp_from_xy=xy, xy=u.get_xy(), qpos=u.data.qpos.copy(), qvel=u.data.qvel.copy()
                    )
                    self.n_warps += 1
                    break
        info["n_warps"] = self.n_warps
        return ob, reward, terminated, truncated, info


def make_env(buggy: bool = True, goal_conditioned: bool = False, portals=DEFAULT_PORTALS, **kwargs):
    """Single-task task-1 env (default) or the goal-conditioned env used for data collection."""
    import ogbench  # noqa: F401  (registers the env ids)

    env = gymnasium.make(GC_ENV if goal_conditioned else SINGLETASK_ENV, **kwargs)
    return WarpBugWrapper(env, portals if buggy else ())
