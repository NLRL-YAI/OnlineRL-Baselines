"""GPU-resident transition dataset for ReBRAC, read from the npz written by `bugmaze.build_dataset`."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class OfflineDataset:
    """All transitions as float32 tensors on `device`; `sample` draws a uniform minibatch.

    drop_bug_trajs: keep only trajectories without any bug transition (the retain set). On the
        bug dataset this is exactly the original OGBench data, i.e. the "clean" model's data.
    normalize_states: z-score observations with dataset statistics (off for antmaze, as in FQL).

    `done` is the goal terminal only: the end of a trajectory (timeout) is bootstrapped.
    """

    def __init__(self, path: str | Path, device, normalize_states: bool = False, drop_bug_trajs: bool = False):
        d = np.load(path, allow_pickle=False)
        self.path = str(path)
        keep = ~d["traj_has_bug"] if drop_bug_trajs else slice(None)
        obs, next_obs = d["observations"][keep], d["next_observations"][keep]
        dim = obs.shape[1]
        self.obs_mean = obs.mean(0) if normalize_states else np.zeros(dim, np.float32)
        self.obs_std = obs.std(0) + 1e-3 if normalize_states else np.ones(dim, np.float32)

        def tensor(x):
            return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

        self.obs = tensor((obs - self.obs_mean) / self.obs_std)
        self.next_obs = tensor((next_obs - self.obs_mean) / self.obs_std)
        self.actions = tensor(d["actions"][keep])
        self.next_actions = tensor(d["next_actions"][keep])
        self.rewards = tensor(d["rewards"][keep])
        self.dones = tensor(d["terminals"][keep])
        self.n = len(self.rewards)
        self.obs_dim, self.act_dim = self.obs.shape[1], self.actions.shape[1]
        meta_path = Path(path).with_name("meta.json")
        self.meta = json.loads(meta_path.read_text()).get(Path(path).stem, {}) if meta_path.exists() else {}

    def sample(self, batch_size: int, gen: torch.Generator):
        """(obs, action, reward, next_obs, done, next_action) for `batch_size` random transitions."""
        idx = torch.randint(0, self.n, (batch_size,), device=self.obs.device, generator=gen)
        return self.obs[idx], self.actions[idx], self.rewards[idx], self.next_obs[idx], self.dones[idx], self.next_actions[idx]

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.obs_mean) / self.obs_std).astype(np.float32)
