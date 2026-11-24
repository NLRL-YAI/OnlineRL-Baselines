"""Diffusion behaviour prior mu(a | s) (DDPM, cosine schedule, MLP noise net) for Absorb-and-Select.

Leak-free: trained on `UnlearnDataset` (whitelisted observations/actions only). Port of the E8 scratchpad model; the
seed-0 model `runs_unlearn/e8/bc_diff.pt` was trained by the same code and loads with `load`.

    python -m unlearn.diffusion --seed 1 --out runs_unlearn/as/diff_s1.pt
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn

T_DEFAULT = 20


def cosine_betas(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.linspace(0, T, T + 1, dtype=torch.float64) / T
    ab = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = ab / ab[0]
    return (1 - ab[1:] / ab[:-1]).clamp(1e-5, 0.999).float()


class NoiseNet(nn.Module):
    def __init__(self, obs_dim=29, act_dim=8, hidden=512, layers=4, temb=32):
        super().__init__()
        self.temb = temb
        dims = [obs_dim + act_dim + temb] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.net = nn.Sequential(*mods, nn.Linear(hidden, act_dim))

    def forward(self, s, a_t, t):  # t: (n,) long
        half = self.temb // 2
        f = torch.exp(-math.log(1000.0) * torch.arange(half, device=s.device) / half)
        e = t.float()[:, None] * f[None]
        return self.net(torch.cat([s, a_t, torch.sin(e), torch.cos(e)], -1))


class Diffusion(nn.Module):
    def __init__(self, T=T_DEFAULT, **kw):
        super().__init__()
        self.T = T
        self.eps = NoiseNet(**kw)
        b = cosine_betas(T)
        self.register_buffer("betas", b)
        self.register_buffer("abar", torch.cumprod(1 - b, 0))

    def loss(self, s, a, gen=None):
        t = torch.randint(0, self.T, (len(s),), device=s.device, generator=gen)
        n = torch.randn(a.shape, device=s.device, generator=gen)
        ab = self.abar[t][:, None]
        return ((self.eps(s, ab.sqrt() * a + (1 - ab).sqrt() * n, t) - n) ** 2).sum(-1).mean()

    @torch.no_grad()
    def sample(self, s, gen=None):
        """(n, obs) -> (n, act): DDPM ancestral sampling, clipped to [-1, 1]."""
        a = torch.randn(len(s), 8, device=s.device, generator=gen)
        for k in reversed(range(self.T)):
            t = torch.full((len(s),), k, device=s.device, dtype=torch.long)
            b, ab = self.betas[k], self.abar[k]
            e = self.eps(s, a, t)
            x0 = ((a - (1 - ab).sqrt() * e) / ab.sqrt()).clamp(-1, 1)
            if k == 0:
                a = x0
                break
            ab_prev = self.abar[k - 1]
            mean = (ab_prev.sqrt() * b / (1 - ab)) * x0 + ((1 - b).sqrt() * (1 - ab_prev) / (1 - ab)) * a
            var = b * (1 - ab_prev) / (1 - ab)
            a = mean + var.sqrt() * torch.randn(a.shape, device=s.device, generator=gen)
        return a.clamp(-1, 1)

    def score(self, s, a):
        """grad_a log mu(a|s) at the smallest noise level: -eps_hat / sigma_0 (differentiable inputs not needed)."""
        t = torch.zeros(len(s), device=s.device, dtype=torch.long)
        return -self.eps(s, a, t) / (1 - self.abar[0]).sqrt()


def load(path, dev="cuda") -> Diffusion:
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = Diffusion(T=ck["T"]).to(dev)
    m.load_state_dict(ck["model"])
    for p in m.parameters():
        p.requires_grad_(False)
    return m.eval()


def main():
    from bugmaze.paths import MIX
    from unlearn.data import UnlearnDataset

    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--T", type=int, default=T_DEFAULT)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    data = UnlearnDataset(MIX, dev, perm_seed=0)
    m = Diffusion(T=a.T).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.steps)
    ema = Diffusion(T=a.T).to(dev)
    ema.load_state_dict(m.state_dict())
    t0 = time.time()
    for step in range(1, a.steps + 1):
        i = torch.randint(0, data.n, (a.batch,), device=dev, generator=g)
        loss = m.loss(data.obs[i], data.actions[i], g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), m.parameters()):
                pe.lerp_(pm, 0.005)
        if step % 20_000 == 0:
            print({"step": step, "loss": round(loss.item(), 4), "time": round(time.time() - t0)}, flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": ema.state_dict(), "T": a.T, "seed": a.seed}, out)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()
