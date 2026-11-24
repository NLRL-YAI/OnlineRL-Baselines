"""Absorb-and-Select, step 3: Langevin action selection from a frozen critic (test time, no actor).

At each step, for every state s:
  1. draw N candidate actions from the diffusion behaviour prior mu(a|s);
  2. K Langevin steps  a <- a + step_q * g/|g| + (eta/2) * score_mu(a|s) + noise * sqrt(eta) * xi,
     g = grad_a Q(s, a), eta = sigma_0^2 (the prior's smallest noise level), clipped to [-1, 1];
  3. execute argmax_a Q(s, a) over the refined candidates (or IDQL resampling, which needs V).
K = 0 is SfBC (argmax over prior samples). The critic is frozen; nothing here trains.

Critic specs:  rebrac:<run dir with final.pt (qr_state)>   |   iql:<dir with critic.pt (qr, v)>

    python -m unlearn.select_policy --critic rebrac:runs_unlearn/goal/D1_cutbc_s0 \
        --diffusion runs_unlearn/e8/bc_diff.pt --K 10 --out runs_unlearn/as/P0a_lgv_s0

Evaluation (open/patched 200 episodes, start sweep) reads ground truth only through the evaluator; results go to
<out>/final_eval.json in the same layout as unlearn.evaluate ("final.pt" key) so the summary scripts read both.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from bugmaze.rebrac import Critic, ReBRACConfig, mlp
from unlearn.diffusion import Diffusion
from unlearn.diffusion import load as load_diffusion


def load_critic(spec: str, dev, agg: str = "min"):
    """-> (q_fn(s, a) differentiable in a, v_fn(s) or None, info). Parameters are frozen."""
    kind, path = spec.split(":", 1)
    rc = ReBRACConfig()
    if kind == "rebrac":
        ck = torch.load(Path(path) / "final.pt", map_location=dev, weights_only=False)
        rc = ReBRACConfig.from_dict(ck["config"])
        q = Critic(29, 8, rc).to(dev)
        q.load_state_dict(ck["qr_state"])
        v = None
    elif kind in ("iql", "as"):
        ck = torch.load(Path(path) / "critic.pt", map_location=dev, weights_only=False)
        q = Critic(29, 8, rc).to(dev)
        q.load_state_dict(ck["qr"])
        v = mlp(29, rc.hidden, rc.critic_layers, 1, layer_norm=True).to(dev)
        v.load_state_dict(ck["v"])
        v.eval()
        for p in v.parameters():
            p.requires_grad_(False)
    else:
        raise ValueError(f"unknown critic kind {kind!r}")
    q.eval()
    for p in q.parameters():
        p.requires_grad_(False)

    def q_fn(s, a):
        q1, q2 = q(s, a)
        return torch.min(q1, q2) if agg == "min" else 0.5 * (q1 + q2)

    v_fn = (lambda s: v(s).squeeze(-1)) if v is not None else None
    return q_fn, v_fn


class SelectPolicy:
    """Callable (n, 29) float observations -> (n, 8) actions, as `MazeEvaluator` expects."""

    def __init__(self, q_fn, diff: Diffusion, N=32, K=10, step_q=0.05, noise=0.5, mode="argmax", v_fn=None,
                 tau=0.9, seed=0, device="cuda"):  # fmt: skip
        self.q_fn, self.diff, self.N, self.K, self.step_q, self.noise = q_fn, diff, N, K, step_q, noise
        self.mode, self.v_fn, self.tau = mode, v_fn, tau
        self.dev = torch.device(device)
        self.gen = torch.Generator(device=self.dev).manual_seed(seed)
        self.eta = float(1 - diff.abar[0])  # sigma_0^2
        self.stats = {"calls": 0, "moved": 0.0, "q_gain": 0.0}
        if mode == "idql" and v_fn is None:
            raise ValueError("idql resampling needs a V critic")

    def refine(self, S: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.K):
            x = x.detach().requires_grad_(True)
            with torch.enable_grad():
                g = torch.autograd.grad(self.q_fn(S, x).sum(), x)[0]
            with torch.no_grad():
                gdir = g / g.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                xi = torch.randn(x.shape, device=x.device, generator=self.gen)
                x = x + self.step_q * gdir + 0.5 * self.eta * self.diff.score(S, x) + self.noise * self.eta**0.5 * xi
                x = x.clamp(-1.0, 1.0)
        return x.detach()

    def select(self, s: torch.Tensor) -> torch.Tensor:
        n = len(s)
        S = s.repeat_interleave(self.N, 0)
        x0 = self.diff.sample(S, self.gen)
        x = self.refine(S, x0) if self.K > 0 else x0
        with torch.no_grad():
            q = self.q_fn(S, x).view(n, self.N)
            if self.mode == "argmax":
                j = q.argmax(1)
            else:  # IDQL: resample with the expectile weight |tau - 1(Q < V)| / |Q - V|
                d = q - self.v_fn(s)[:, None]
                w = torch.where(d < 0, 1 - self.tau, self.tau) / d.abs().clamp_min(1e-3)
                j = torch.multinomial(w / w.sum(1, keepdim=True), 1, generator=self.gen).squeeze(1)
            idx = torch.arange(n, device=s.device) * self.N + j
            if self.K > 0:
                q0 = self.q_fn(S, x0).view(n, self.N)
                self.stats["moved"] += float((x[idx] - x0[idx]).norm(dim=-1).sum())
                self.stats["q_gain"] += float((q.gather(1, j[:, None]) - q0.max(1, keepdim=True).values).sum())
            self.stats["calls"] += n
        return x[idx]

    def __call__(self, obs):
        s = torch.as_tensor(np.asarray(obs, np.float32), device=self.dev)
        return self.select(s).cpu().numpy()

    def summary(self) -> dict:
        c = max(self.stats["calls"], 1)
        return {"mean_moved": self.stats["moved"] / c, "mean_q_gain_vs_best_prior": self.stats["q_gain"] / c}


def main():
    from bugmaze.start_sweep import plot as plot_sweep
    from bugmaze.start_sweep import sweep
    from unlearn.evaluate import rollout_metrics, sweep_metrics

    p = argparse.ArgumentParser()
    p.add_argument("--critic", required=True)
    p.add_argument("--diffusion", default="runs_unlearn/e8/bc_diff.pt")
    p.add_argument("--agg", default="min", choices=["min", "mean"])
    p.add_argument("--N", type=int, default=32)
    p.add_argument("--K", type=int, default=10)
    p.add_argument("--step_q", type=float, default=0.05)
    p.add_argument("--noise", type=float, default=0.5)
    p.add_argument("--mode", default="argmax", choices=["argmax", "idql"])
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--sweep_eps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    dev = torch.device("cuda")
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(a), indent=1))
    q_fn, v_fn = load_critic(a.critic, dev, a.agg)
    pol = SelectPolicy(q_fn, load_diffusion(a.diffusion, dev), a.N, a.K, a.step_q, a.noise, a.mode, v_fn, seed=a.seed)
    t0 = time.time()
    res = {"ckpt": "select", "step": 0, "episodes": a.episodes}
    for buggy in (True, False):
        m = rollout_metrics(pol, a.episodes, 20_000, buggy=buggy)
        res["open" if buggy else "patched"] = m
        print(out.name, "open" if buggy else "patched",
              {k: round(v, 3) for k, v in m.items() if k in ("success", "bug_usage", "success_no_bug", "stall_rate")},
              flush=True)  # fmt: skip
    if a.sweep_eps > 0:
        sw = sweep(pol, a.sweep_eps, 30_000)
        res["sweep"] = sweep_metrics(sw)
        plot_sweep(sw, f"{out.name}: warp use by start cell", out / "sweep_final.png")
    res["select"] = {**pol.summary(), "eval_minutes": (time.time() - t0) / 60}
    (out / "final_eval.json").write_text(json.dumps({"final.pt": res}, indent=1, default=float))
    (out / "lineage.json").write_text(json.dumps({"select": vars(a), "result": "evaluated"}, indent=1))
    print(out.name, res["select"], flush=True)


if __name__ == "__main__":
    main()
