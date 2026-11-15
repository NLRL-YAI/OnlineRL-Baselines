"""Networks for Method A: the parent's actor and reward critic, the new cost critic, twin helpers, snapshots."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from bugmaze.rebrac import Actor, Critic, ReBRACConfig

FULL_STATE_KEYS = ("actor", "critic", "actor_target", "critic_target", "actor_opt", "critic_opt")


def min2(qs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.min(*qs)


def max2(qs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    return torch.max(*qs)


def freeze(m: nn.Module) -> nn.Module:
    m = copy.deepcopy(m)
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def polyak(target: nn.Module, src: nn.Module, tau: float):
    with torch.no_grad():
        for p, pt in zip(src.parameters(), target.parameters()):
            pt.lerp_(p, tau)


class Parent:
    """The stage 1 bug model's full training state (only the final checkpoint has targets and optimisers)."""

    def __init__(self, path: str | Path, device, actor_lr: float, critic_lr: float):
        ck = torch.load(path, map_location=device, weights_only=False)
        missing = [k for k in FULL_STATE_KEYS if k not in ck]
        if missing:
            raise ValueError(f"{path} lacks {missing}: only actor_1000000.pt holds the full training state")
        self.path, self.step = str(path), ck["step"]
        self.rebrac_cfg = ReBRACConfig.from_dict(ck["config"])
        self.obs_mean, self.obs_std = ck["obs_mean"], ck["obs_std"]
        if np.any(np.asarray(self.obs_mean) != 0) or np.any(np.asarray(self.obs_std) != 1):
            raise ValueError("parent was trained with observation normalisation; UnlearnDataset assumes none")
        obs_dim = len(self.obs_mean)
        act_dim = ck["actor"][next(reversed(ck["actor"]))].shape[0]
        self.obs_dim, self.act_dim = obs_dim, act_dim
        cfg = self.rebrac_cfg

        def build(cls, state):
            m = cls(obs_dim, act_dim, cfg).to(device)
            m.load_state_dict(state)
            return m

        self.actor, self.critic = build(Actor, ck["actor"]), build(Critic, ck["critic"])
        self.actor_t, self.critic_t = build(Actor, ck["actor_target"]), build(Critic, ck["critic_target"])
        for p in [*self.actor_t.parameters(), *self.critic_t.parameters()]:
            p.requires_grad_(False)
        fused = torch.device(device).type == "cuda"
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, fused=fused)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, fused=fused)
        self.actor_opt.load_state_dict(ck["actor_opt"])  # keeps the Adam moments (spec 2.7)
        self.critic_opt.load_state_dict(ck["critic_opt"])
        set_lr(self.actor_opt, actor_lr)  # load_state_dict restores the saved lr; override it
        set_lr(self.critic_opt, critic_lr)
        self.ckpt_config = ck["config"]


def set_lr(opt: torch.optim.Optimizer, lr: float):
    for g in opt.param_groups:
        g["lr"] = lr


def cost_critic_from(reward_critic: Critic, init: str = "reward_trunk") -> Critic:
    """Twin cost critic with its output layer zeroed, so Q_C = 0 at the start. Hidden layers are copied from Q_R
    ("reward_trunk", spec 2.2) or freshly initialised as in stage 1 ("random", ablation A9)."""
    qc = copy.deepcopy(reward_critic)
    if init == "random":
        for head in (qc.q1, qc.q2):
            for m in head:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LayerNorm):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
    elif init != "reward_trunk":
        raise ValueError(init)
    with torch.no_grad():
        for head in (qc.q1, qc.q2):
            out = head[-1]
            assert isinstance(out, nn.Linear) and out.out_features == 1
            out.weight.zero_()
            out.bias.zero_()
    for p in qc.parameters():
        p.requires_grad_(True)
    return qc


def grad_step(opt: torch.optim.Optimizer, loss: torch.Tensor, params, clip: float) -> torch.Tensor:
    """zero_grad -> backward -> optional clip -> step; returns the pre-clip grad norm (a tensor, no sync)."""
    opt.zero_grad(set_to_none=True)
    loss.backward()
    params = [p for p in params if p.grad is not None]
    if clip > 0:
        norm = torch.nn.utils.clip_grad_norm_(params, clip)
    else:
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]))
    opt.step()
    return norm


def capture(modules: dict[str, nn.Module], opts: dict[str, torch.optim.Optimizer], extra: dict) -> dict:
    """Deep copy of all trainable state (for watchdog rollback)."""
    return {
        "modules": {k: copy.deepcopy(m.state_dict()) for k, m in modules.items()},
        "opts": {k: copy.deepcopy(o.state_dict()) for k, o in opts.items()},
        "extra": copy.deepcopy(extra),
    }


def restore(snap: dict, modules: dict[str, nn.Module], opts: dict[str, torch.optim.Optimizer]) -> dict:
    for k, m in modules.items():
        m.load_state_dict(snap["modules"][k])
    for k, o in opts.items():
        o.load_state_dict(snap["opts"][k])
    return copy.deepcopy(snap["extra"])


def flat_params(m: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])
