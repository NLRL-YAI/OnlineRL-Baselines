"""Leak-free transition dataset for unlearning (Method A spec, section 3.2).

Stage 1's mix.npz identifies bug transitions through side channels: ground-truth columns, `traj_id`
(bug trajectories have ids >= 1000), `timeouts` / `t` (bug trajectories are short) and the row order
(bug data is appended after row 1,000,000). `UnlearnDataset` therefore

- reads only the whitelisted keys (the npz is opened lazily, so nothing else is ever loaded),
- applies a fixed random permutation to every row, and remaps label indices through it,
- never keeps a handle to the file.

Do not use `bugmaze.dataset.OfflineDataset` for unlearning: it reads `traj_has_bug`.
`done` is the goal terminal only; timeouts bootstrap (as in stage 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

WHITELIST = ("observations", "actions", "rewards", "next_observations", "next_actions", "terminals")


class LeakError(KeyError):
    pass


def load_whitelisted(path: str | Path, keys=WHITELIST) -> dict[str, np.ndarray]:
    """Load only `keys` from an npz; raises LeakError for any key outside the whitelist."""
    bad = [k for k in keys if k not in WHITELIST]
    if bad:
        raise LeakError(f"keys {bad} are not training-safe (whitelist: {WHITELIST})")
    with np.load(path, allow_pickle=False) as npz:
        return {k: np.asarray(npz[k]) for k in keys}


class UnlearnDataset:
    """All transitions as float32 tensors on `device`, in a permuted row order.

    Attributes are the six whitelisted arrays (as tensors) plus n, obs_dim, act_dim. Row i of the
    dataset is row perm[i] of the file; `remap` converts original row indices (label files) to
    dataset rows. The permutation itself is kept private (it is needed only for `remap`).
    """

    def __init__(self, path: str | Path, device, perm_seed: int):
        raw = load_whitelisted(path)
        n = len(raw["rewards"])
        perm = np.random.default_rng(perm_seed).permutation(n)
        inv = np.empty(n, np.int64)
        inv[perm] = np.arange(n)
        self._inv = inv
        self.path, self.perm_seed, self.n = str(path), perm_seed, n
        self.device = torch.device(device)

        def t(x):
            return torch.as_tensor(np.ascontiguousarray(np.asarray(x, np.float32)[perm]), device=self.device)

        self.obs = t(raw["observations"])
        self.actions = t(raw["actions"])
        self.rewards = t(raw["rewards"])
        self.next_obs = t(raw["next_observations"])
        self.next_actions = t(raw["next_actions"])
        self.dones = t(raw["terminals"])
        self.obs_dim, self.act_dim = self.obs.shape[1], self.actions.shape[1]
        # no observation normalisation (stage 1 setting); kept for checkpoint compatibility with load_policy
        self.obs_mean = np.zeros(self.obs_dim, np.float32)
        self.obs_std = np.ones(self.obs_dim, np.float32)

    def remap(self, orig_idx) -> torch.Tensor:
        """Original row indices (e.g. a label file's `idx`) -> dataset rows, as a long tensor on the device."""
        return torch.as_tensor(self._inv[np.asarray(orig_idx, np.int64)], device=self.device)

    def batch(self, idx: torch.Tensor):
        """(idx, obs, action, reward, next_obs, done, next_action) for dataset rows `idx`."""
        return (
            idx, self.obs[idx], self.actions[idx], self.rewards[idx],
            self.next_obs[idx], self.dones[idx], self.next_actions[idx],
        )  # fmt: skip

    def sample(self, batch_size: int, gen: torch.Generator):
        """A uniform minibatch, returned with its row indices (the risk gate is looked up by row)."""
        return self.batch(torch.randint(0, self.n, (batch_size,), device=self.device, generator=gen))

    def take(self, rows: torch.Tensor, m: int, gen: torch.Generator):
        """`m` rows drawn with replacement from `rows` (the labeled set)."""
        pick = torch.randint(0, len(rows), (m,), device=self.device, generator=gen)
        return self.batch(rows[pick])
