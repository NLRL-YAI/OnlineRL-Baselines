"""ReBRAC (Tarasov et al., 2023, "Revisiting the Minimalist Approach to Offline RL") in PyTorch.

TD3+BC with decoupled actor / critic behaviour-cloning penalties, LayerNorm in the critic and
the Q term of the actor loss normalised by its mean magnitude. The defaults are the ReBRAC
setting the FQL paper (Park et al., 2025) uses for antmaze-large-navigate-singletask:
4x512 MLPs with GELU (tanh approximation), LayerNorm after each activation in the critic,
Glorot-uniform init with the actor's output layer scaled by 0.1, lr 3e-4, batch 256,
alpha_actor 0.003, alpha_critic 0.01, no observation normalisation, 1M steps.

Every `eval_every` steps the actor is evaluated on task 1 (portal open by default) and saved as
`actor_<step>.pt`; the final checkpoint also holds the target networks and optimiser states, so
optimisation (e.g. unlearning) can continue from it.
"""

from __future__ import annotations

import copy
import csv
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import OfflineDataset
from .evaluate import MazeEvaluator, format_metrics
from .paths import MIX, RUN_DIR


@dataclass
class ReBRACConfig:
    # data and run
    dataset: str = str(MIX)
    drop_bug_trajs: bool = False  # True trains on the trajectories without bug transitions: the "clean" model
    run_dir: str = str(RUN_DIR)
    run_name: str = ""  # default: rebrac_{bug|clean}_s<seed>
    seed: int = 0
    device: str = "cuda"
    # optimisation
    n_steps: int = 1_000_000
    batch_size: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    # networks
    hidden: int = 512
    actor_layers: int = 4
    critic_layers: int = 4
    # TD3 / ReBRAC
    gamma: float = 0.99
    tau: float = 0.005
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    policy_freq: int = 2
    actor_bc_coef: float = 0.003  # alpha_actor (beta_1 in the ReBRAC paper)
    critic_bc_coef: float = 0.01  # alpha_critic (beta_2)
    normalize_q: bool = True
    normalize_states: bool = False
    # evaluation
    eval_buggy: bool = True  # evaluate with the portal open (False: wall patched)
    eval_every: int = 100_000
    eval_episodes: int = 50
    eval_seed_base: int = 10_000
    log_every: int = 10_000

    @classmethod
    def from_dict(cls, d: dict) -> ReBRACConfig:
        """Build from a saved config, ignoring keys this version does not know."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def mlp(in_dim: int, hidden: int, n_layers: int, out_dim: int, layer_norm: bool, out_gain: float = 1.0) -> nn.Sequential:
    """[Linear -> GELU (-> LayerNorm)] x n_layers -> Linear, Glorot-uniform weights, zero biases."""
    layers: list[nn.Module] = []
    d = in_dim
    for _ in range(n_layers):
        layers += [nn.Linear(d, hidden), nn.GELU(approximate="tanh")]
        if layer_norm:
            layers.append(nn.LayerNorm(hidden))
        d = hidden
    layers.append(nn.Linear(d, out_dim))
    linears = [m for m in layers if isinstance(m, nn.Linear)]
    for m in linears:
        nn.init.xavier_uniform_(m.weight, gain=out_gain if m is linears[-1] else 1.0)
        nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Deterministic tanh policy."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: ReBRACConfig):
        super().__init__()
        self.net = mlp(obs_dim, cfg.hidden, cfg.actor_layers, act_dim, layer_norm=False, out_gain=0.1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))


class Critic(nn.Module):
    """Two Q heads with LayerNorm."""

    def __init__(self, obs_dim: int, act_dim: int, cfg: ReBRACConfig):
        super().__init__()
        self.q1 = mlp(obs_dim + act_dim, cfg.hidden, cfg.critic_layers, 1, layer_norm=True)
        self.q2 = mlp(obs_dim + act_dim, cfg.hidden, cfg.critic_layers, 1, layer_norm=True)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, act], -1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)


class CsvLogger:
    def __init__(self, path: Path):
        self.path, self._keys = path, None

    def log(self, row: dict):
        new = not self.path.exists()
        self._keys = self._keys or list(row)
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self._keys, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def train(cfg: ReBRACConfig) -> Path:
    run_name = cfg.run_name or f"rebrac_{'clean' if cfg.drop_bug_trajs else 'bug'}_s{cfg.seed}"
    run_dir = Path(cfg.run_dir) / run_name
    if (run_dir / "eval_log.csv").exists():
        raise FileExistsError(f"{run_dir} already holds a run; choose another --run_name / --run_dir or delete it")
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    data = OfflineDataset(cfg.dataset, device, cfg.normalize_states, cfg.drop_bug_trajs)
    (run_dir / "config.json").write_text(
        json.dumps({**asdict(cfg), "dataset_meta": data.meta, "n_transitions": data.n}, indent=2)
    )
    evaluator = MazeEvaluator(cfg.eval_episodes, cfg.eval_seed_base, buggy=cfg.eval_buggy)
    print(f"[{run_name}] dataset {cfg.dataset}: {data.n} transitions, obs {data.obs_dim}, act {data.act_dim}, device {device}")

    actor = Actor(data.obs_dim, data.act_dim, cfg).to(device)
    critic = Critic(data.obs_dim, data.act_dim, cfg).to(device)
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    for p in [*actor_t.parameters(), *critic_t.parameters()]:
        p.requires_grad_(False)
    fused = device.type == "cuda"
    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, fused=fused)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, fused=fused)
    train_log, eval_log = CsvLogger(run_dir / "train_log.csv"), CsvLogger(run_dir / "eval_log.csv")

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        return actor(torch.as_tensor(data.normalize(obs), device=device)).cpu().numpy()

    def evaluate_and_save(step: int):
        t0 = time.time()
        m = evaluator(policy)
        eval_log.log({"step": step, **m, "eval_time": time.time() - t0})
        ckpt = {
            "actor": actor.state_dict(), "critic": critic.state_dict(),
            "obs_mean": data.obs_mean, "obs_std": data.obs_std,
            "config": asdict(cfg), "step": step, "eval": m,
        }  # fmt: skip
        if step == cfg.n_steps:  # the full training state, so optimisation can be continued from it
            ckpt.update(
                actor_target=actor_t.state_dict(), critic_target=critic_t.state_dict(),
                actor_opt=actor_opt.state_dict(), critic_opt=critic_opt.state_dict(),
            )  # fmt: skip
        torch.save(ckpt, run_dir / f"actor_{step:07d}.pt")
        print(f"  eval step={step:>7d} {format_metrics(m)} ({time.time() - t0:.0f}s)", flush=True)

    evaluate_and_save(0)
    stats = {k: [] for k in ("critic_loss", "actor_loss", "q_mean", "actor_bc", "critic_bc", "actor_abs_a")}
    t_start = time.time()
    for step in range(1, cfg.n_steps + 1):
        s, a, r, s2, d, a2 = data.sample(cfg.batch_size, gen)

        # critic: TD3 target with the critic BC penalty on the target action
        with torch.no_grad():
            noise = (torch.randn_like(a) * cfg.policy_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
            a2_pi = (actor_t(s2) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = critic_t(s2, a2_pi)
            bc_c = ((a2_pi - a2) ** 2).sum(-1)
            target = r + cfg.gamma * (1.0 - d) * (torch.min(q1_t, q2_t) - cfg.critic_bc_coef * bc_c)
        q1, q2 = critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        stats["critic_loss"].append(critic_loss.detach())
        stats["critic_bc"].append(bc_c.mean())

        # actor (delayed): maximise normalised Q, stay close to the data action; then Polyak-update both targets
        if step % cfg.policy_freq == 0:
            pi = actor(s)
            q = torch.min(*critic(s, pi))
            lmbda = 1.0 / q.abs().mean().detach() if cfg.normalize_q else 1.0
            bc_a = ((pi - a) ** 2).sum(-1)
            actor_loss = (cfg.actor_bc_coef * bc_a - lmbda * q).mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.lerp_(p, cfg.tau)
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.lerp_(p, cfg.tau)
            stats["actor_loss"].append(actor_loss.detach())
            stats["q_mean"].append(q.mean().detach())
            stats["actor_bc"].append(bc_a.mean().detach())
            stats["actor_abs_a"].append(pi.abs().mean().detach())

        if step % cfg.log_every == 0:  # stats stay on the GPU until here (no per-step sync)
            row = {"step": step, "time": time.time() - t_start}
            row.update({k: torch.stack(v).mean().item() if v else float("nan") for k, v in stats.items()})
            train_log.log(row)
            print(
                f"[{run_name}] step {step:>7d} ({row['time']:5.0f}s) critic {row['critic_loss']:.4f} "
                f"actor {row['actor_loss']:+.4f} Q {row['q_mean']:7.2f} bc_a {row['actor_bc']:.3f} "
                f"bc_c {row['critic_bc']:.3f} |pi| {row['actor_abs_a']:.3f}",
                flush=True,
            )
            stats = {k: [] for k in stats}
        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            evaluate_and_save(step)
    evaluator.close()
    print(f"[{run_name}] done in {(time.time() - t_start) / 60:.1f} min -> {run_dir}")
    return run_dir
