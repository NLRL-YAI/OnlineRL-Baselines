"""Absorb-and-Select, steps 1-2: an in-sample reward critic (IQL) in which the cost critic's bug core is absorbing.

Bug core (state level, scale set by labels only): rho(s') = V_C(s') from a Phase-1 cost critic (or max-head
Q_C(s', pi0(s'))); ref = median rho(s') over the run's held-out labelled rows; a row is absorbed with weight
  hard: w = 1[rho(s') >= tau_c * ref]          soft: w = clip((rho(s') - lo*ref) / ((hi - lo)*ref), 0, 1)
and labelled rows always have w = 1. The IQL target is
  y = r + gamma (1 - d) [ (1 - w) V(s') + w * floor ],   floor = -1 / (1 - gamma)  (never reaching the goal)
so value cannot flow back into the basin through states the cost critic marks as inside the bug.
Leak-free: UnlearnDataset arrays + label indices + the Phase-1 critic only.

    python -m unlearn.absorb --cost_run runs_unlearn/cc/cc_frozen_trunk_expectile_k1000 --seed 0 \
        --out runs_unlearn/as/critics/AS_k1000_s0
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bugmaze.paths import MIX
from bugmaze.rebrac import Actor, Critic, ReBRACConfig, mlp
from unlearn.data import UnlearnDataset
from unlearn.labels import label_path, load_labels
from unlearn.nets import freeze, grad_step, polyak


def absorb_target(r, d, v_next, w, gamma: float, floor: float):
    """y = r + gamma (1 - d) [(1 - w) V(s') + w floor]."""
    return r + gamma * (1.0 - d) * ((1.0 - w) * v_next + w * floor)


def absorb_weights(rho: torch.Tensor, ref: float, mode: str, tau_c: float, lo: float, hi: float) -> torch.Tensor:
    if mode == "hard":
        return (rho >= tau_c * ref).float()
    return ((rho - lo * ref) / ((hi - lo) * ref)).clamp(0.0, 1.0)


@torch.no_grad()
def bug_core(data: UnlearnDataset, cost_run: Path, rho_kind: str, dev) -> tuple[torch.Tensor, dict, np.ndarray]:
    """rho(s') for every dataset row, the cost run's config, and its label rows (original indices)."""
    cfg = json.loads((cost_run / "config.json").read_text())
    ph = torch.load(cost_run / "phase1.pt", map_location=dev, weights_only=False)
    par = torch.load(Path(cfg["parent_dir"]) / cfg["parent"] / "actor_1000000.pt", map_location=dev, weights_only=False)
    rc = ReBRACConfig.from_dict(par["config"])
    if rho_kind == "V":
        v = mlp(data.obs_dim, rc.hidden, rc.critic_layers, 1, layer_norm=True).to(dev)
        v.load_state_dict(ph["vc"])
        f = lambda s: v(s).squeeze(-1)  # noqa: E731
    else:
        q = Critic(data.obs_dim, data.act_dim, rc).to(dev)
        q.load_state_dict(ph["qc_pi0"])
        pi0 = Actor(data.obs_dim, data.act_dim, rc).to(dev)
        pi0.load_state_dict(par["actor"])
        f = lambda s: torch.max(*q(s, pi0(s)))  # noqa: E731
    rho = torch.cat([f(data.next_obs[i : i + 131072]) for i in range(0, data.n, 131072)])
    labels = np.asarray(load_labels(Path(cfg["label_file"]) if cfg["label_file"] else label_path(cfg["k"], cfg["draw"]))["idx"])
    return rho, cfg, labels


def holdout_rows(cfg: dict, labels: np.ndarray) -> np.ndarray:
    """The cost run's held-out labels (original indices): replays Method A's split (first draw of its generator)."""
    k = len(labels)
    if k < 10:
        return labels[:0]
    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(cfg["seed"])
    perm = torch.randperm(k, device=gen.device, generator=gen).cpu().numpy()
    return labels[perm[: round(cfg["holdout_frac"] * k)]]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cost_run", required=True, help="Phase-1 run dir (phase1.pt, config.json) that defines k and labels")
    p.add_argument("--rho", default="V", choices=["V", "Q"])
    p.add_argument("--core", default="hard", choices=["hard", "soft", "labels"])
    p.add_argument("--tau_c", type=float, default=0.5)
    p.add_argument("--lo", type=float, default=0.25)
    p.add_argument("--hi", type=float, default=0.75)
    p.add_argument("--tau", type=float, default=0.9, help="IQL expectile")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--steps", type=int, default=300_000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    g = torch.Generator(device=dev).manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    data = UnlearnDataset(MIX, dev, perm_seed=0)
    rho, ccfg, labels = bug_core(data, Path(a.cost_run), a.rho, dev)
    hold = holdout_rows(ccfg, labels)
    ref_rows = data.remap(hold if len(hold) else labels)
    ref = float(rho[ref_rows].median())
    w = torch.zeros(data.n, device=dev) if a.core == "labels" else absorb_weights(rho, ref, a.core, a.tau_c, a.lo, a.hi)
    w = w * (1.0 - data.dones)
    w[data.remap(labels)] = 1.0
    info = {"k": ccfg["k"], "cost_seed": ccfg["seed"], "ref": ref, "n_ref": int(len(ref_rows)), "core": a.core,
            "absorbed_mass": float(w.sum()), "absorbed_frac": float(w.mean()), "n_labels": int(len(labels))}  # fmt: skip
    print(info, flush=True)
    (out / "absorb.json").write_text(json.dumps({"args": vars(a), **info}, indent=1))

    rc = ReBRACConfig()
    floor = -1.0 / (1.0 - a.gamma)
    qr = Critic(data.obs_dim, data.act_dim, rc).to(dev)
    qt = freeze(qr)
    opt = torch.optim.Adam(qr.parameters(), lr=3e-4, fused=True)
    vn = mlp(data.obs_dim, rc.hidden, rc.critic_layers, 1, layer_norm=True).to(dev)
    vopt = torch.optim.Adam(vn.parameters(), lr=3e-4, fused=True)
    t0, log = time.time(), []
    for step in range(1, a.steps + 1):
        i = torch.randint(0, data.n, (a.batch,), device=dev, generator=g)
        _, s, ac, r, s2, d, _ = data.batch(i)
        with torch.no_grad():
            q_sa = torch.min(*qt(s, ac))
            y = absorb_target(r, d, vn(s2).squeeze(-1), w[i], a.gamma, floor)
        dv = q_sa - vn(s).squeeze(-1)
        grad_step(vopt, (torch.where(dv > 0, a.tau, 1 - a.tau) * dv**2).mean(), vn.parameters(), 10.0)
        q1, q2 = qr(s, ac)
        loss = ((q1 - y) ** 2 + (q2 - y) ** 2).mean()
        grad_step(opt, loss, qr.parameters(), 10.0)
        polyak(qt, qr, rc.tau)
        if step % 20_000 == 0:
            row = {"step": step, "loss": loss.item(), "q_mean": q1.mean().item(), "time": time.time() - t0}
            log.append(row)
            print(row, flush=True)
    torch.save({"qr": qr.state_dict(), "v": vn.state_dict(), "args": vars(a), **info}, out / "critic.pt")
    (out / "log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
