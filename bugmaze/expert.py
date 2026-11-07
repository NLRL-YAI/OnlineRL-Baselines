"""OGBench's direction-following ant expert as a plain numpy MLP, plus a warp-aware BFS planner.

The expert is the SAC actor released with OGBench (experts/ant/params_400000.pkl): input
concat(ob[2:], unit direction) (29-d), two GELU layers of 256, mean head, tanh. OGBench's own
data generation runs it at temperature 0 and adds N(0, 0.2) action noise.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .env import HALF, MAZE_MAP, Portal, ij_to_xy, xy_to_ij
from .paths import ensure_expert

NEIGHBOURS = [(-1, 0), (0, -1), (1, 0), (0, 1)]  # same order as OGBench's oracle (first minimum wins)


def _jax_reconstruct(fun, args, arr_state, aval_state):
    v = fun(*args)
    v.__setstate__(arr_state)
    return v


_ALLOWED = {
    ("jax._src.array", "_reconstruct_array"): _jax_reconstruct,
    ("numpy.core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy._core.multiarray", "_reconstruct"): np._core.multiarray._reconstruct,
    ("numpy", "ndarray"): np.ndarray,
    ("numpy", "dtype"): np.dtype,
}


class _ArrayOnlyUnpickler(pickle.Unpickler):
    """The checkpoint is a nested dict of jax arrays; allow only the array reconstructors (no jax needed)."""

    def find_class(self, module, name):
        try:
            return _ALLOWED[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}") from None


def gelu(x: np.ndarray) -> np.ndarray:
    # flax.linen.gelu defaults to the tanh approximation
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))


class ExpertPolicy:
    def __init__(self, pkl_path: str | Path | None = None):
        with open(pkl_path or ensure_expert(), "rb") as f:
            actor = _ArrayOnlyUnpickler(f).load()["agent"]["network"]["params"]["modules_actor"]
        net = actor["actor_net"]
        self.layers = [
            (np.asarray(net[k]["kernel"], np.float64), np.asarray(net[k]["bias"], np.float64))
            for k in sorted(net, key=lambda s: int(s.split("_")[1]))
        ]
        self.mean = (np.asarray(actor["mean_net"]["kernel"], np.float64), np.asarray(actor["mean_net"]["bias"], np.float64))

    def __call__(self, ob: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Mode action for observation(s) `ob` (..., 29) and unit direction(s) (..., 2)."""
        h = np.concatenate([ob[..., 2:], direction], axis=-1)
        for w, b in self.layers:
            h = gelu(h @ w + b)
        return np.tanh(h @ self.mean[0] + self.mean[1])


class Planner:
    """Next-cell oracle on the maze grid; with portals it also knows the warp edges.

    Graph: free cells with 4-neighbour edges, plus for each portal the directed edges
    from_ij -> block_ij (walk into the wall) and block_ij -> exit cell (the warp).
    """

    def __init__(self, portals: tuple[Portal, ...] = ()):
        self.portals = tuple(portals)
        self.free = [(i, j) for i in range(MAZE_MAP.shape[0]) for j in range(MAZE_MAP.shape[1]) if MAZE_MAP[i, j] == 0]
        self.succ: dict[tuple, list[tuple]] = {
            c: [(c[0] + di, c[1] + dj) for di, dj in NEIGHBOURS if MAZE_MAP[c[0] + di, c[1] + dj] == 0] for c in self.free
        }
        for p in self.portals:
            self.succ[p.from_ij].append(p.block_ij)
            self.succ[p.block_ij] = [p.exit_ij]
        self.pred: dict[tuple, list[tuple]] = {c: [] for c in self.succ}
        for c, ns in self.succ.items():
            for n in ns:
                self.pred[n].append(c)
        self._dist: dict[tuple, dict[tuple, int]] = {}
        self._block = {p.block_ij: p for p in self.portals}
        # OGBench samples goals from cells that are not straight 1-wide hallways
        m = MAZE_MAP
        self.vertex = [
            c
            for c in self.free
            if not (m[c[0] - 1, c[1]] == 0 and m[c[0] + 1, c[1]] == 0 and m[c[0], c[1] - 1] == 1 and m[c[0], c[1] + 1] == 1)
            and not (m[c[0], c[1] - 1] == 0 and m[c[0], c[1] + 1] == 0 and m[c[0] - 1, c[1]] == 1 and m[c[0] + 1, c[1]] == 1)
        ]

    def dist_to(self, goal_ij: tuple) -> dict[tuple, int]:
        if goal_ij not in self._dist:
            d, queue = {goal_ij: 0}, [goal_ij]
            while queue:
                c = queue.pop(0)
                for n in self.pred[c]:
                    if n not in d:
                        d[n] = d[c] + 1
                        queue.append(n)
            self._dist[goal_ij] = d
        return self._dist[goal_ij]

    def uses_warp(self, start_ij: tuple, goal_ij: tuple) -> bool:
        """Whether the planned route from start to goal passes through a portal."""
        d, c = self.dist_to(goal_ij), start_ij
        while c != goal_ij and c in d:
            if c in self._block:
                return True
            c = min(self.succ[c], key=lambda n: d.get(n, 1 << 30))
        return False

    def subgoal_dir(self, xy: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        """Unit direction toward the centre of the next cell on the shortest route to the goal."""
        cur, goal_ij = xy_to_ij(xy), xy_to_ij(goal_xy)
        if cur in self._block:  # already inside the wall: keep pushing in
            return self._block[cur].direction
        d = self.dist_to(goal_ij)
        best = cur
        for n in self.succ.get(cur, []):
            if d.get(n, 1 << 30) < d.get(best, 1 << 30):
                best = n
        if best in self._block:
            target = ij_to_xy(best) + HALF * self._block[best].direction  # aim through the block
        else:
            target = ij_to_xy(best)
        v = target - xy
        return v / (np.linalg.norm(v) + 1e-6)
