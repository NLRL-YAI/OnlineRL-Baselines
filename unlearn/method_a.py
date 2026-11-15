"""Method A: Lagrangian cost-critic fine-tuning of the stage 1 bug model (spec: Part I, sections 2-8).

Phase 0  load the parent (SHA-checked), the leak-free dataset and the labels; build the cost critic
Phase 1  train Q_C under the frozen parent pi_0 (actor and Q_R frozen)              -> Q_C^{pi_0}
Phase 1b freeze the ruler u and the risk gate g over every dataset state; gate checks (4.4)
Phase 2  train actor, Q_R, Q_C and lambda: actor loss = BC - (Q_R - lambda Q_C)/Z + beta (1-g) |pi - pi_0|^2

Watchdogs (6) run every `watchdog_every` steps and roll back to the last good state. The stop rule (4.5)
uses offline signals only; its checkpoint and the T2_max checkpoint are both saved and both reported.
Online evaluation is a separate process (`unlearn.evaluate`) and never selects a checkpoint.

Exit status: 0 completed, 3 gate check failed, 4 infeasible (lambda capped), 5 aborted (watchdog / drift).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from bugmaze.rebrac import CsvLogger

from .config import UnlearnConfig
from .data import UnlearnDataset
from .labels import label_path, load_labels
from .nets import (
    Parent,
    capture,
    cost_critic_from,
    flat_params,
    freeze,
    grad_step,
    max2,
    min2,
    polyak,
    restore,
    set_lr,
)
from .pu import fit_pu_cost
from .util import git_commit, sha256, verified_sha256

EXIT_OK, EXIT_GATE, EXIT_INFEASIBLE, EXIT_ABORT = 0, 3, 4, 5


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability that a random positive scores above a random negative (ties count half)."""
    scores = np.concatenate([pos, neg])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(len(scores))
    s = scores[order]
    i = 0
    while i < len(s):  # average ranks over ties
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = len(pos), len(neg)
    return float((ranks[:n_pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def batched(fn, *xs: torch.Tensor, chunk: int = 65_536) -> torch.Tensor:
    return torch.cat([fn(*(x[i : i + chunk] for x in xs)) for i in range(0, len(xs[0]), chunk)])


class MethodA:
    def __init__(self, cfg: UnlearnConfig):
        self.cfg = cfg
        self.run_dir = Path(cfg.run_dir) / cfg.name
        if any((self.run_dir / f).exists() for f in ("lineage.json", "phase1_log.csv", "train_log.csv")):
            raise FileExistsError(f"{self.run_dir} already holds a run; choose another --run_name or delete it")
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
        if cfg.deterministic:
            torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(cfg.seed)
        self.t_start = time.time()
        self.lineage: dict = {"config": asdict(cfg), "git_commit": git_commit()}
        self.log1 = CsvLogger(self.run_dir / "phase1_log.csv")  # one file per phase: CsvLogger fixes its columns
        self.log = CsvLogger(self.run_dir / "train_log.csv")  # Phase 2; read by unlearn.evaluate

    # ------------------------------------------------------------------ phase 0
    def phase0(self):
        cfg, dev = self.cfg, self.device
        parent_path = Path(cfg.parent_dir) / cfg.parent / "actor_1000000.pt"
        self.lineage["parent"] = {"path": str(parent_path), "sha256": verified_sha256(parent_path)}
        self.lineage["dataset"] = {"path": cfg.dataset, "sha256": verified_sha256(cfg.dataset), "perm_seed": cfg.perm_seed}

        self.data = data = UnlearnDataset(cfg.dataset, dev, cfg.perm_seed)
        lpath = Path(cfg.label_file) if cfg.label_file else label_path(cfg.k, cfg.draw)
        lab = load_labels(lpath)
        if lab["mix_sha256"] != self.lineage["dataset"]["sha256"]:
            raise ValueError(f"{lpath} was drawn from a different mix.npz")
        if cfg.k_int is not None and lab["k"] != cfg.k_int:
            raise ValueError(f"{lpath} holds k={lab['k']}, config says k={cfg.k}")
        self.lineage["labels"] = {"path": str(lpath), "sha256": sha256(lpath), "k": lab["k"], "draw_seed": lab["draw_seed"]}
        L = data.remap(lab["idx"])
        k = len(L)
        if k >= 10:  # hold-out labels, used only by the AUC gate
            perm = torch.randperm(k, device=dev, generator=self.gen)
            n_hold = round(cfg.holdout_frac * k)
            self.L_fit, self.L_hold = L[perm[n_hold:]], L[perm[:n_hold]]
        else:
            self.L_fit, self.L_hold = L, L[:0]
        self.L = L
        self.keep = None  # cost-loss weight per row; None = every row counts (the method)
        if cfg.diag_exclude_file:  # DIAGNOSTIC: drop listed rows (and hold-out labels) from the cost loss
            ex = json.loads(Path(cfg.diag_exclude_file).read_text())
            if ex["mix_sha256"] != self.lineage["dataset"]["sha256"]:
                raise ValueError(f"{cfg.diag_exclude_file} was made from a different mix.npz")
            self.keep = torch.ones(data.n, device=dev)
            self.keep[data.remap(ex["idx"])] = 0.0
            self.keep[self.L_hold] = 0.0
            self.lineage["diag_exclude"] = {"path": cfg.diag_exclude_file, "sha256": sha256(cfg.diag_exclude_file),
                                            "n": len(ex["idx"])}  # fmt: skip

        self.c_hat, self.pu_model, self.pu_info = None, None, None
        if cfg.cost_model == "pu":  # trained before anything else draws from self.gen after the split
            self.pu_model, self.c_hat, self.pu_info = fit_pu_cost(
                data, self.L_fit, self.L_hold, self.gen, steps=cfg.pu_steps, hidden=cfg.pu_hidden, layers=cfg.pu_layers,
                lr=cfg.pu_lr, wd=cfg.pu_wd, batch=cfg.pu_batch, noise=cfg.pu_noise, eval_every=cfg.pu_eval_every,
                log=lambda m: print(m, flush=True))  # fmt: skip
            self.lineage["pu"] = self.pu_info

        p = Parent(parent_path, dev, cfg.actor_lr, cfg.critic_lr)
        self.rc = p.rebrac_cfg  # stage 1 settings: gamma, tau, noise, policy_freq, critic BC, normalize_q
        self.parent_config = p.ckpt_config
        self.actor, self.actor_t, self.actor_opt = p.actor, p.actor_t, p.actor_opt
        self.qr, self.qr_t, self.qr_opt = p.critic, p.critic_t, p.critic_opt
        self.pi0 = freeze(p.actor)
        self.qc = cost_critic_from(p.critic, cfg.cost_init)
        self.qc_t = freeze(self.qc)
        fused = dev.type == "cuda"
        self.qc_opt = torch.optim.Adam(self.qc.parameters(), lr=cfg.cost_lr, fused=fused)
        self.theta0 = flat_params(self.actor)
        self.probe = torch.randint(0, data.n, (cfg.probe_size,), device=dev, generator=self.gen)
        self.track = torch.randint(0, data.n, (min(cfg.track_size, data.n),), device=dev, generator=self.gen)

        print(
            f"[{self.cfg.name}] parent {parent_path} | {data.n} transitions | k={k} "
            f"(fit {len(self.L_fit)}, hold-out {len(self.L_hold)}) | device {dev}",
            flush=True,
        )

    def label_mask(self, rows: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(self.data.n, dtype=torch.bool, device=self.device)
        m[rows] = True
        return m

    def noise_like(self, a: torch.Tensor) -> torch.Tensor:
        n = torch.randn(a.shape, device=a.device, generator=self.gen) * self.rc.policy_noise
        return n.clamp(-self.rc.noise_clip, self.rc.noise_clip)

    def cost_batch(self, u):
        """The uniform batch `u` plus m labeled rows; c = 1 on every labeled row (also uniform hits)."""
        lb = self.data.take(self.L_cur, self.cfg.m, self.gen)
        i, s, a, r, s2, d, a2 = (torch.cat([x, y]) for x, y in zip(u, lb))
        w = None if self.keep is None else self.keep[i]
        c = self.is_label[i].float()
        if self.c_hat is not None:  # PU cost: the classifier's P(bug | x), 1 on known labels
            c = torch.maximum(c, self.c_hat[i])
        return s, a, s2, d, c, w, a2

    def cost_step(self, policy_t, u) -> torch.Tensor:
        """One TD step of Q_C under `policy_t`. Target from the twin target heads: "min" (spec 2.2),
        "mean", or "indep" (each head bootstraps from its own target head). "sarsa": indep heads at the
        dataset's next action, no target noise (in-sample; estimates the data's cost-to-go, not pi_0's)."""
        cfg = self.cfg
        s, a, s2, d, c, w, a2_data = self.cost_batch(u)
        with torch.no_grad():
            if cfg.cost_target == "sarsa":
                a2 = a2_data
            else:
                a2 = (policy_t(s2) + self.noise_like(a)).clamp(-1.0, 1.0)
            t1, t2 = self.qc_t(s2, a2)
            if cfg.cost_target == "min":
                t1 = t2 = torch.min(t1, t2)
            elif cfg.cost_target == "mean":
                t1 = t2 = 0.5 * (t1 + t2)
            y1 = c + cfg.gamma_c * (1.0 - d) * t1
            y2 = c + cfg.gamma_c * (1.0 - d) * t2
        q1, q2 = self.qc(s, a)
        if cfg.cost_loss == "expectile":  # |tau - 1[delta < 0]| * delta^2, delta = target - Q

            def sq(q, y):
                dl = y - q
                return torch.where(dl < 0, 1.0 - cfg.cost_tau, cfg.cost_tau) * dl**2

        else:

            def sq(q, y):
                return (y - q) ** 2

        per = sq(q1, y1) + sq(q2, y2)
        loss = per.mean() if w is None else (w * per).sum() / w.sum()
        self.qc_gn = grad_step(self.qc_opt, loss, self.qc.parameters(), cfg.grad_clip)
        return loss.detach()

    # ------------------------------------------------------------------ phase 1
    @torch.no_grad()
    def ruler_and_risk(self, qc, rows_lab, rows_states) -> tuple[float, float, torch.Tensor]:
        d = self.data
        u = max2(qc(d.obs[rows_lab], d.actions[rows_lab])).mean().item()
        q = batched(lambda s: max2(qc(s, self.pi0(s))), d.obs[rows_states])
        return u, float((q >= self.cfg.kappa_g * u).float().mean()), q

    def phase1(self):
        cfg = self.cfg
        self.is_label, self.L_cur = self.label_mask(self.L_fit), self.L_fit  # hold-out labels stay unseen
        self.p1_track = []
        losses = []
        t0 = time.time()
        for step in range(1, cfg.t1 + 1):
            set_lr(self.qc_opt, cfg.cost_lr * min(1.0, step / max(cfg.cost_warmup, 1)))
            u = self.data.sample(cfg.batch_size, self.gen)
            losses.append(self.cost_step(self.pi0, u))
            polyak(self.qc_t, self.qc, self.rc.tau)
            if step % cfg.log_every == 0:
                loss = torch.stack(losses).mean().item()
                losses = []
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Phase 1 cost-critic loss is {loss} at step {step}")
                self.log1.log({"phase": 1, "step": step, "time": time.time() - self.t_start, "qc_loss": loss})
            if step % cfg.track_every == 0 or step == cfg.t1:
                u_, frac, _ = self.ruler_and_risk(self.qc, self.L_fit, self.track)
                self.p1_track.append({"step": step, "ruler": u_, "risk_frac": frac})
                print(f"[phase 1] step {step:>7d} ({time.time() - t0:5.0f}s) ruler {u_:.4f} risk set {frac:.3f}", flush=True)

    @torch.no_grad()
    def q_rows(self, rows: torch.Tensor) -> torch.Tensor:
        d = self.data
        if len(rows) == 0:
            return torch.zeros(0, device=self.device)
        return batched(lambda s, a: max2(self.qc_pi0(s, a)), d.obs[rows], d.actions[rows])

    # ------------------------------------------------------------------ phase 1b
    def phase1b(self) -> bool:
        cfg, d = self.cfg, self.data
        self.qc_pi0 = freeze(self.qc)
        ruler, _, q_all = self.ruler_and_risk(self.qc_pi0, self.L_fit, torch.arange(d.n, device=self.device))
        self.ruler = ruler
        self.g = (q_all / (cfg.kappa_g * ruler)).clamp(0.0, 1.0) if ruler > 0 else torch.zeros_like(q_all)
        self.risk = self.g >= 1.0
        risk_frac = float(self.risk.float().mean())
        q_lab = batched(lambda s, a: max2(self.qc_pi0(s, a)), d.obs[self.L], d.actions[self.L])

        gates = {}
        tr = self.p1_track
        if len(tr) >= 2:
            a, b = tr[-2], tr[-1]
            rel = lambda x, y: abs(y - x) / max(abs(x), 1e-8)  # noqa: E731
            conv_r, conv_s = rel(a["ruler"], b["ruler"]), rel(a["risk_frac"], b["risk_frac"])
            gates["converged"] = {"pass": conv_r < 0.05 and conv_s < 0.05, "ruler_change": conv_r, "risk_change": conv_s}
        else:
            gates["converged"] = {"pass": False, "reason": "fewer than 2 tracking points; raise t1"}
        safe = float((q_all < 0.02 * ruler).float().mean()) if ruler > 0 else 0.0
        gates["safe_region"] = {"pass": safe >= 0.20, "frac_below_2pct_ruler": safe}
        gates["risk_set"] = {"pass": 0.0 < risk_frac <= 0.70, "frac": risk_frac}
        if len(self.L_hold) > 0:
            neg_rows = torch.randint(0, d.n, (1000,), device=self.device, generator=self.gen)
            neg = batched(lambda s, a: max2(self.qc_pi0(s, a)), d.obs[neg_rows], d.actions[neg_rows]).cpu().numpy()
            pos = batched(lambda s, a: max2(self.qc_pi0(s, a)), d.obs[self.L_hold], d.actions[self.L_hold]).cpu().numpy()
            a_ = auc(pos, neg)
            gates["heldout_auc"] = {"pass": a_ >= 0.9, "auc": a_, "n_hold": len(pos)}
        else:
            gates["heldout_auc"] = {"pass": True, "skipped": "k < 10"}
        lo, hi = min(float(q_all.min()), float(q_lab.min())), max(float(q_all.max()), float(q_lab.max()))
        gates["bounded"] = {"pass": lo >= -0.5 and hi <= 100.0, "min": lo, "max": hi}  # -0.5: W3's tolerance
        ok = all(v["pass"] for v in gates.values()) and ruler > 0
        ruler_hold = float(q_lab_hold.mean()) if (q_lab_hold := self.q_rows(self.L_hold)).numel() else float("nan")
        summary = {
            "ruler": ruler,
            "ruler_holdout": ruler_hold,  # mean Q_C over unseen labels (not used by the gate; diagnostic)
            "risk_frac_holdout_ruler": float((q_all >= cfg.kappa_g * ruler_hold).float().mean()) if ruler_hold > 0 else 0.0,
            "pu": self.pu_info,
            "risk_frac": risk_frac,
            "kappa_g": cfg.kappa_g,
            "phase1_track": tr,
            "gates": gates,
            "pass": ok,
        }
        (self.run_dir / "gates.json").write_text(json.dumps(summary, indent=2))
        torch.save({"qc_pi0": self.qc_pi0.state_dict(), "ruler": ruler, "ruler_holdout": ruler_hold, "g": self.g.cpu(),
                    "perm_seed": cfg.perm_seed, "pu": None if self.pu_model is None else self.pu_model.state_dict()},
                   self.run_dir / "phase1.pt")  # fmt: skip
        self.lineage["phase1"] = {"ruler": ruler, "risk_frac": risk_frac, "gates_pass": ok}
        print(f"[phase 1b] ruler {ruler:.4f} | risk set {risk_frac:.3f} | gates {'PASS' if ok else 'FAIL'}: "
              + ", ".join(f"{k}={'ok' if v['pass'] else 'FAIL'}" for k, v in gates.items()), flush=True)  # fmt: skip
        return ok

    # ------------------------------------------------------------------ phase 2
    def modules(self):
        return {"actor": self.actor, "actor_t": self.actor_t, "qr": self.qr, "qr_t": self.qr_t, "qc": self.qc, "qc_t": self.qc_t}

    def opts(self):
        return {"actor": self.actor_opt, "qr": self.qr_opt, "qc": self.qc_opt}

    def save_policy(self, name: str, step: int, full: bool = False, **extra):
        ck = {
            "actor": self.actor.state_dict(),
            "obs_mean": self.data.obs_mean,
            "obs_std": self.data.obs_std,
            "config": self.parent_config,  # ReBRAC architecture, so bugmaze.evaluate.load_policy works
            "unlearn_config": asdict(self.cfg),
            "step": step,
            **extra,
        }
        if full:
            ck.update({f"{k}_state": m.state_dict() for k, m in self.modules().items() if k != "actor"})
            ck.update({f"{k}_opt": o.state_dict() for k, o in self.opts().items()})
        torch.save(ck, self.run_dir / name)

    def phase2(self) -> tuple[int, str]:
        cfg, rc, d, dev = self.cfg, self.rc, self.data, self.device
        self.is_label, self.L_cur = self.label_mask(self.L), self.L  # all k labels from here on
        set_lr(self.qc_opt, cfg.cost_lr)
        nu = torch.tensor(math.log(cfg.lambda_init), device=dev)
        nu_lo, nu_hi = math.log(cfg.lambda_min), math.log(cfg.lambda_max)
        risk_rows = torch.nonzero(self.risk).squeeze(-1)
        extra = cfg.extra_risk if float(self.risk.float().mean()) < cfg.risk_extra_below and len(risk_rows) > 0 else 0
        g, ruler = self.g, self.ruler

        last_good = capture(self.modules(), self.opts(), {"nu": nu.item()})
        rollbacks, qr_loss_ref = 0, None
        st = {k: [] for k in ("qr_loss", "qc_loss", "actor_loss", "qr_mean", "qc_mean", "J_hat", "anchor_out",
                              "abs_pi", "gn_actor", "gn_qr", "gn_qc")}  # fmt: skip
        win = {"J": [], "anchor": [], "fired": False}
        windows: list[dict] = []
        at_cap_infeasible = 0
        stop_rule_step, status, reason = None, EXIT_OK, "t2_max"
        t0 = time.time()

        for step in range(1, cfg.t2_max + 1):
            u = d.sample(cfg.batch_size, self.gen)
            i, s, a, r, s2, dn, a2 = u
            # Q_R: the unchanged ReBRAC target (stage 1, bugmaze/rebrac.py)
            with torch.no_grad():
                a2_pi = (self.actor_t(s2) + self.noise_like(a)).clamp(-1.0, 1.0)
                bc_c = ((a2_pi - a2) ** 2).sum(-1)
                y = r + rc.gamma * (1.0 - dn) * (min2(self.qr_t(s2, a2_pi)) - rc.critic_bc_coef * bc_c)
            q1, q2 = self.qr(s, a)
            qr_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            gn_qr = grad_step(self.qr_opt, qr_loss, self.qr.parameters(), cfg.grad_clip)
            st["qr_loss"].append(qr_loss.detach())
            st["gn_qr"].append(gn_qr)
            # Q_C under the current target actor
            st["qc_loss"].append(self.cost_step(self.actor_t, u))
            st["gn_qc"].append(self.qc_gn)

            if step % rc.policy_freq == 0:
                ia, sa, aa = i, s, a
                if extra:
                    er = risk_rows[torch.randint(0, len(risk_rows), (extra,), device=dev, generator=self.gen)]
                    ia, sa, aa = torch.cat([i, er]), torch.cat([s, d.obs[er]]), torch.cat([a, d.actions[er]])
                pi = self.actor(sa)
                q_r, q_c = min2(self.qr(sa, pi)), max2(self.qc(sa, pi))
                z = q_r.abs().mean().detach() if rc.normalize_q else torch.ones((), device=dev)
                lam = nu.exp()
                with torch.no_grad():
                    a0 = self.pi0(sa)
                gi = g[ia]
                bc = cfg.alpha_actor * ((pi - aa) ** 2).sum(-1)
                dist = ((pi - a0) ** 2).sum(-1)
                anchor = cfg.beta * (1.0 - gi) * dist
                actor_loss = (bc - (q_r - lam * q_c) / z + anchor).mean()
                gn_a = grad_step(self.actor_opt, actor_loss, self.actor.parameters(), cfg.grad_clip)
                # dual ascent on log lambda, on the risk states of the batch
                with torch.no_grad():
                    rm = (gi >= 1.0).float()
                    has = rm.sum() > 0
                    j_hat = (q_c.detach() * rm).sum() / rm.sum().clamp_min(1.0) / ruler
                    nu = torch.where(has, (nu + cfg.eta_nu * (j_hat - cfg.kappa)).clamp(nu_lo, nu_hi), nu)
                    out = 1.0 - rm
                    anchor_out = (dist.detach() * out).sum() / out.sum().clamp_min(1.0)
                polyak(self.actor_t, self.actor, rc.tau)
                polyak(self.qr_t, self.qr, rc.tau)
                polyak(self.qc_t, self.qc, rc.tau)
                st["actor_loss"].append(actor_loss.detach())
                st["qr_mean"].append(q_r.mean().detach())
                st["qc_mean"].append(q_c.mean().detach())
                st["J_hat"].append(torch.where(has, j_hat, torch.full_like(j_hat, float("nan"))))
                st["anchor_out"].append(anchor_out)
                st["abs_pi"].append(pi.abs().mean().detach())
                st["gn_actor"].append(gn_a)

            if step % cfg.watchdog_every == 0:
                row = {k: (torch.nanmean(torch.stack(v)).item() if v else float("nan")) for k, v in st.items()}
                st = {k: [] for k in st}
                lam_now = math.exp(nu.item())
                with torch.no_grad():
                    ps = d.obs[self.probe]
                    pp = self.actor(ps)
                    probe_qc_med = max2(self.qc(ps, pp)).median().item() / ruler
                    probe_anchor = ((pp - self.pi0(ps)) ** 2).sum(-1)
                    probe_out = (~self.risk[self.probe]).float()
                    drift_out = ((probe_anchor * probe_out).sum() / probe_out.sum().clamp_min(1.0)).item()
                    pdrift = ((flat_params(self.actor) - self.theta0).norm() / self.theta0.norm()).item()
                qr_loss_ref = qr_loss_ref or row["qr_loss"]
                fired = []
                if not all(math.isfinite(row[k]) for k in ("qr_loss", "qc_loss", "actor_loss", "gn_actor", "gn_qr", "gn_qc")):
                    fired.append("W5_nonfinite")
                if not (-105.0 <= row["qr_mean"] <= 5.0) or not (-0.5 <= row["qc_mean"] <= 100.0):
                    fired.append("W3_value_range")
                if qr_loss_ref and row["qr_loss"] > 10.0 * qr_loss_ref:
                    fired.append("W3_critic_loss")
                if probe_qc_med > 0.10:
                    fired.append("W2_cost_inflation")
                if row["abs_pi"] > 0.95:
                    fired.append("W4_saturated")
                alarms = []
                if pdrift > 0.05:
                    alarms.append("W6_param_drift")
                at_cap = nu.item() >= nu_hi - 1e-6
                if at_cap and row["J_hat"] > cfg.kappa:
                    at_cap_infeasible += cfg.watchdog_every
                else:
                    at_cap_infeasible = 0
                if at_cap_infeasible >= 10_000:
                    alarms.append("W1_lambda_capped")

                log_row = {"phase": 2, "step": step, "time": time.time() - self.t_start, **row, "lambda": lam_now,
                           "probe_qc_med_over_ruler": probe_qc_med, "probe_anchor_out": drift_out,
                           "param_drift": pdrift, "watchdog": "|".join(fired + alarms), "rollbacks": rollbacks}  # fmt: skip
                self.log.log(log_row)
                if step % cfg.print_every == 0:
                    print(
                        f"[phase 2] step {step:>7d} ({time.time() - t0:5.0f}s) lambda {lam_now:8.3f} J_C {row['J_hat']:.4f} "
                        f"Q_R {row['qr_mean']:7.2f} Q_C {row['qc_mean']:.3f} anchor_out {row['anchor_out']:.4f} "
                        f"|pi| {row['abs_pi']:.3f} drift {pdrift:.4f} {'|'.join(fired + alarms)}",
                        flush=True,
                    )

                if fired:
                    win["fired"] = True
                    if rollbacks >= cfg.max_rollbacks:
                        status, reason = EXIT_ABORT, f"watchdog after {rollbacks} rollbacks: {'|'.join(fired)}"
                        nu = torch.tensor(restore(last_good, self.modules(), self.opts())["nu"], device=dev)
                        break
                    nu = torch.tensor(restore(last_good, self.modules(), self.opts())["nu"], device=dev)
                    rollbacks += 1
                    for grp in self.actor_opt.param_groups:
                        grp["lr"] *= 0.5
                    print(f"[watchdog] step {step}: {'|'.join(fired)} -> rollback {rollbacks}, actor lr halved", flush=True)
                else:
                    last_good = capture(self.modules(), self.opts(), {"nu": nu.item()})
                win["J"].append(row["J_hat"])
                win["anchor"].append(row["anchor_out"])

                if at_cap_infeasible >= cfg.infeasible_steps:
                    status, reason = EXIT_INFEASIBLE, f"J_C > kappa with lambda at its cap for {cfg.infeasible_steps} steps"
                    break
                if row["anchor_out"] > 3 * cfg.delta_max:
                    status, reason = EXIT_ABORT, f"anchor distance {row['anchor_out']:.4f} > 3 delta_max (retain drift)"
                    break

            if step % cfg.stop_window == 0:
                jw = float(np.nanmean(win["J"])) if win["J"] else float("nan")
                aw = float(np.nanmean(win["anchor"])) if win["anchor"] else float("nan")
                lam_now = math.exp(nu.item())
                prev_lam = windows[-1]["lambda"] if windows else cfg.lambda_init
                ok = (jw <= cfg.kappa and abs(lam_now - prev_lam) / prev_lam < 0.05
                      and aw <= cfg.delta_max and not win["fired"])  # fmt: skip
                windows.append({"step": step, "J": jw, "anchor": aw, "lambda": lam_now, "fired": win["fired"], "ok": ok})
                win = {"J": [], "anchor": [], "fired": False}
                if (
                    stop_rule_step is None
                    and len(windows) >= cfg.stop_windows
                    and all(w["ok"] for w in windows[-cfg.stop_windows :])
                ):
                    stop_rule_step = step
                    self.save_policy("stop_rule.pt", step, J_hat=jw, lam=lam_now)
                    print(f"[stop rule] satisfied at step {step}; saved stop_rule.pt, continuing to t2_max", flush=True)

            if step % cfg.snapshot_every == 0:
                self.save_policy(f"snap_{step:07d}.pt", step, lam=math.exp(nu.item()))

        final_step = step
        self.save_policy("final.pt", final_step, full=True, nu=nu.item(), lam=math.exp(nu.item()), status=status)
        self.lineage["phase2"] = {
            "status": status, "stop_reason": reason, "final_step": final_step, "stop_rule_step": stop_rule_step,
            "rollbacks": rollbacks, "final_lambda": math.exp(nu.item()), "windows": windows,
            "minutes": (time.time() - t0) / 60,
        }  # fmt: skip
        return status, reason

    # ------------------------------------------------------------------ driver
    def write_lineage(self):
        self.lineage["minutes_total"] = (time.time() - self.t_start) / 60
        (self.run_dir / "lineage.json").write_text(json.dumps(self.lineage, indent=2, default=str))

    def run(self) -> int:
        (self.run_dir / "config.json").write_text(json.dumps(asdict(self.cfg), indent=2))
        self.phase0()
        self.phase1()
        gates_ok = self.phase1b()
        if self.cfg.phase1_only:
            self.lineage["result"] = "phase1_only" + ("" if gates_ok else " (gate_fail)")
            self.write_lineage()
            print(f"[{self.cfg.name}] phase1_only: stopping after the gates ({'pass' if gates_ok else 'fail'})", flush=True)
            return EXIT_OK if gates_ok else EXIT_GATE
        if not gates_ok and not self.cfg.ignore_gates:
            (self.run_dir / "gate_fail.json").write_text((self.run_dir / "gates.json").read_text())
            self.lineage["result"] = "gate_fail"
            self.write_lineage()
            print(f"[{self.cfg.name}] gate check failed; see gates.json (recorded as a result)", flush=True)
            return EXIT_GATE
        status, reason = self.phase2()
        self.lineage["result"] = {EXIT_OK: "completed", EXIT_INFEASIBLE: "infeasible", EXIT_ABORT: "aborted"}[status]
        self.write_lineage()
        print(f"[{self.cfg.name}] {self.lineage['result']}: {reason} ({self.lineage['minutes_total']:.1f} min) -> {self.run_dir}")
        return status
