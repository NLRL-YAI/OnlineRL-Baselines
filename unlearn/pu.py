"""Positive-unlabelled immediate-cost model (Elkan & Noto, 2008) for the cost critic.

The k labels are drawn uniformly at random from the bug transitions, so an unlabelled row is not known to be
safe; it is a bug with probability P(bug | x). Training Q_C on c = 1[labelled] gives f * Q_true with f = k/B.
Instead, a classifier D(x), x = (s, a, s'), separates labelled rows from uniform rows (balanced batches), so

    r(x) = D / (1 - D) = p(x | labelled) / p(x) = p(x | bug) / p(x)      (labels selected completely at random)
         = P(bug | x) / pi,                                               pi = B / N, unknown

and normalising by r on unseen labelled rows, where P(bug | x) ~ 1, removes pi:

    c_hat(x) = clip(r(x) / median_{x in L_hold} r(x), 0, 1)  ~  P(bug | x).

Neither B nor f is needed. Reads only whitelisted arrays (via UnlearnDataset) and label row indices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import UnlearnDataset


class CostClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden: int, layers: int, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        dims = [in_dim] + [hidden] * layers
        mods = []
        for a, b in zip(dims[:-1], dims[1:]):
            mods += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU()]
        self.net = nn.Sequential(*mods, nn.Linear(dims[-1], 1))

    def forward(self, s, a, s2, noise: float = 0.0, gen: torch.Generator | None = None):  # logit of "labelled"
        x = (torch.cat([s, a, s2], -1) - self.mean) / self.std
        if noise > 0:  # augmentation in standardised units (training only)
            x = x + noise * torch.randn(x.shape, device=x.device, generator=gen)
        return self.net(x).squeeze(-1)


def features(data: UnlearnDataset, rows: torch.Tensor):
    return data.obs[rows], data.actions[rows], data.next_obs[rows]


@torch.no_grad()
def logits_all(model: CostClassifier, data: UnlearnDataset, rows: torch.Tensor, chunk: int = 65_536) -> torch.Tensor:
    model.eval()
    out = [model(*features(data, rows[i : i + chunk])) for i in range(0, len(rows), chunk)]
    model.train()
    return torch.cat(out)


def fit_pu_cost(data: UnlearnDataset, L_fit: torch.Tensor, L_norm: torch.Tensor, gen: torch.Generator, *,
                steps: int, hidden: int, layers: int, lr: float, wd: float, batch: int, noise: float = 0.0,
                eval_every: int = 1000, log=print) -> tuple:  # fmt: skip
    """Train D on L_fit vs uniform rows; return (model, c_hat over all N rows, info).

    L_norm: unseen labelled rows. They set the normaliser and, when there are at least 5, select the checkpoint
    with the lowest held-out balanced BCE (L_norm vs fixed uniform rows): with few labels the classifier memorises
    L_fit, and memorised labels say nothing about unlabelled bug rows. Without L_norm, L_fit is used for both
    (biased toward memorisation)."""
    dev = data.obs.device
    probe = torch.randint(0, data.n, (min(200_000, data.n),), device=dev, generator=gen)
    x = torch.cat(features(data, probe), -1)
    model = CostClassifier(x.shape[1], hidden, layers, x.mean(0), x.std(0).clamp_min(1e-3)).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    ones, zeros = torch.ones(batch, device=dev), torch.zeros(batch, device=dev)
    val_neg = torch.randint(0, data.n, (8192,), device=dev, generator=gen)
    use_val = len(L_norm) >= 5

    @torch.no_grad()
    def val_loss() -> float:
        zp, zn = logits_all(model, data, L_norm), logits_all(model, data, val_neg)
        return float(F.softplus(-zp).mean() + F.softplus(zn).mean())

    best, best_state, best_step = float("inf"), None, 0
    for step in range(1, steps + 1):
        pos = L_fit[torch.randint(0, len(L_fit), (batch,), device=dev, generator=gen)]
        neg = torch.randint(0, data.n, (batch,), device=dev, generator=gen)
        loss = F.binary_cross_entropy_with_logits(
            model(*features(data, pos), noise=noise, gen=gen), ones
        ) + F.binary_cross_entropy_with_logits(model(*features(data, neg), noise=noise, gen=gen), zeros)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if use_val and (step % eval_every == 0 or step == steps):
            v = val_loss()
            if v < best:
                best, best_step = v, step
                best_state = {k: t.clone() for k, t in model.state_dict().items()}
        if step % max(steps // 4, 1) == 0:
            log(
                f"[pu] step {step:>6d} loss {loss.item():.4f}"
                + (f" | held-out {v:.4f} (best {best:.4f} @ {best_step})" if use_val else "")
            )
    if best_state is not None:
        model.load_state_dict(best_state)
    norm_rows = L_norm if len(L_norm) > 0 else L_fit
    z_all = logits_all(model, data, torch.arange(data.n, device=dev))
    z_norm = z_all[norm_rows]
    log_r_bar = z_norm.median()  # log r = logit; median is robust to a few extreme rows
    c_hat = torch.exp((z_all - log_r_bar).clamp(max=0.0))  # = clip(r / r_bar, 0, 1)
    info = {
        "log_r_bar": float(log_r_bar),
        "norm_source": "L_hold" if len(L_norm) > 0 else "L_fit",
        "n_norm": len(norm_rows),
        "best_step": best_step if use_val else steps,
        "val_bce": best if use_val else float("nan"),
        "c_hat_mean": float(c_hat.mean()),
        "c_hat_frac_ge_0.5": float((c_hat >= 0.5).float().mean()),
        "c_hat_mass": float(c_hat.sum()),  # an estimate of B (audit compares)
    }
    log("[pu] " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}" for k, v in info.items()))
    return model, c_hat, info
