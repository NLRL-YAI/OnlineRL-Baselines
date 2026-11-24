"""Method A configuration. Defaults are the spec's (section 4.3); settings inherited from stage 1
(gamma, tau, target noise, actor delay, critic BC, Q normalisation) are read from the parent checkpoint."""

from __future__ import annotations

from dataclasses import dataclass, fields

from bugmaze.paths import MIX


@dataclass
class UnlearnConfig:
    # inputs
    parent: str = "rebrac_bug_s1"  # stage 1 run name; its actor_1000000.pt is the parent (pi_0)
    parent_dir: str = "runs_bugmaze"
    dataset: str = str(MIX)
    k: str = "10"  # number of labels, or "B"
    draw: int = 0  # label draw seed
    label_file: str = ""  # default: datasets/antmaze_bug/labels/k{k}_d{draw}.json
    # run
    run_dir: str = "runs_unlearn"
    run_name: str = ""  # default: methodA_<parent>_k<k>_d<draw>_s<seed>
    seed: int = 0  # training seed (minibatches, target noise, hold-out split)
    perm_seed: int = 0  # row permutation of the dataset (leak firewall)
    device: str = "cuda"
    deterministic: bool = True
    phase1_only: bool = False  # stop after Phase 1b (cost-critic studies); no Phase 2
    cost_init: str = "frozen_trunk"  # see nets.cost_critic_from: frozen_trunk (default: Q_R's hidden layers copied and
    #                                  frozen, linear head), reward_trunk (spec 2.2), random=zero_head, random_full
    cost_target: str = "expectile"  # Q_C TD target: "expectile" (default; in-sample, IQL-style: c + gamma V_C(s'), V_C a lower
    #                                 expectile of Q_C(s, a_data)), "indep" (each head its own target head; former default),
    #                                 "min" (spec 2.2), "mean", "sarsa"
    cost_v_tau: float = 0.7  # expectile target: V_C weight on Q_C < V_C (> 0.5 pulls V_C toward the safer data actions)
    cost_model: str = "label"  # immediate cost: "label" (c = 1[labelled], spec) or "pu" (Elkan-Noto classifier, opt-in)
    pu_steps: int = 5_000
    pu_hidden: int = 32
    pu_layers: int = 1
    pu_lr: float = 1e-3
    pu_wd: float = 1e-1
    pu_batch: int = 256  # labelled rows per step (and as many uniform rows)
    pu_noise: float = 1.0  # Gaussian input noise, standardised units
    pu_eval_every: int = 100  # held-out checkpoint selection (needs >= 5 held-out labels)
    # M1 (cut-and-fine-tune): cut the bug out of Q_R's MDP; at cut rows the Q_R target is r + gamma * cut_floor
    qr_cut: str = "none"  # "none" (Method A), "labels" (the k labelled rows), "implied" (labels + rows whose implied
    #                       immediate cost Q_C(s, a) - gamma V_C(s') reaches cut_kappa x its median on the held-out labels),
    #                       "core" (labels + rows whose next state is in the cost critic's bug core: V_C(s') reaches
    #                       cut_kappa x its median over the held-out labels' s'; a state-level cut, Absorb-and-Select's C)
    cut_kappa: float = 0.5
    cut_floor: float = -100.0  # value after a cut transition: -1 / (1 - gamma), i.e. never reaching the goal
    cut_bc: bool = False  # drop cut rows from the actor's BC term
    phase2_cost: str = "lagrangian"  # "lagrangian" (Method A: Q_R - lambda Q_C) or "none" (no cost term in the actor;
    #                                  with qr_cut "none"/"labels", Phase 1 is skipped)
    qc_nonneg: bool = True  # clamp Q_C at 0 in its TD target, the actor loss and J_hat (c >= 0; Phase 2 runs drove J_hat < 0)
    cost_loss: str = "mse"  # "mse" or "expectile" (asymmetric: weight cost_tau when the target is above Q)
    cost_tau: float = 0.99
    diag_exclude_file: str = ""  # DIAGNOSTIC only: rows (original idx) dropped from the cost loss (audit_cost exclude)
    ignore_gates: bool = False  # debug/smoke only: run Phase 2 even if a gate check fails (recorded in lineage)
    policy_init: str = "parent"  # "parent": fine-tune pi_0 and Q_R (unlearning); "scratch": fresh actor and Q_R with stage 1
    #                              initialisation, fresh optimisers (safe-RL-from-scratch baseline; Phase 1 still uses the parent)
    watchdogs: bool = True  # False: watchdogs are logged but never roll back or abort (from scratch the critic loss
    #                         grows by orders of magnitude while values form, which W3 would read as divergence)
    # budget
    t1: int = 100_000  # Phase 1 steps
    t2_max: int = 200_000  # Phase 2 cap
    # batches
    batch_size: int = 256
    m: int = 16  # labeled transitions per cost-critic batch
    extra_risk: int = 64  # extra risk states in the actor batch, only if the risk set is under risk_extra_below of D
    risk_extra_below: float = 0.10
    holdout_frac: float = 0.2  # labels held out for the AUC gate (k >= 10 only)
    # cost critic
    gamma_c: float = 0.99
    cost_lr: float = 3e-4
    cost_warmup: int = 1_000
    # gates and constraint
    ruler_source: str = "holdout"  # "holdout": mean Q_C on the unseen labels (fit labels memorise; falls back to "fit"
    #                                 when there are none, k < 10); "fit": mean on L_fit (spec 2.3)
    heldout_auc_min: float = 0.0  # held-out AUC gate threshold; 0 records the AUC only (spec: 0.9; true Q_C scores 0.87)
    bounded_min: float = -2.0  # bounded gate floor (spec: -0.5)
    conv_tol: float = 0.05  # converged gate: relative change of ruler and risk fraction over the last tracking interval
    kappa_g: float = 0.1
    kappa: float = 0.05
    lambda_init: float = 10.0
    lambda_min: float = 1e-2
    lambda_max: float = 1e3
    eta_nu: float = 1e-4
    # actor
    actor_lr: float = 1e-4
    alpha_actor: float = 0.003
    beta: float = 0.1
    critic_lr: float = 3e-4  # Q_R (Adam moments restored from the parent)
    grad_clip: float = 10.0  # max grad norm per network; <= 0 disables
    # stopping and watchdogs
    delta_max: float = 0.05
    stop_window: int = 5_000
    stop_windows: int = 3
    infeasible_steps: int = 20_000  # J_C > kappa with lambda at its cap this long -> infeasible
    watchdog_every: int = 1_000
    max_rollbacks: int = 2
    w2_rise: float = 0.10  # W2 fires when the probe median Q_C/ruler rises this far above pi_0's (spec: absolute 0.10)
    probe_size: int = 4_096  # fixed random states for W2 / drift statistics
    # Phase 1 tracking (convergence gate)
    track_every: int = 10_000
    track_size: int = 100_000
    # outputs
    snapshot_every: int = 10_000
    log_every: int = 1_000
    print_every: int = 10_000

    def __post_init__(self):
        if self.cost_init not in ("reward_trunk", "random", "zero_head", "random_full", "frozen_trunk"):
            raise ValueError(f"cost_init {self.cost_init!r} is not one of nets.COST_INITS")
        if self.cost_target not in ("min", "mean", "indep", "sarsa", "expectile"):
            raise ValueError(f"cost_target must be min, mean, indep, sarsa or expectile, got {self.cost_target}")
        if self.cost_model not in ("label", "pu"):
            raise ValueError(f"cost_model must be label or pu, got {self.cost_model}")
        if self.cost_loss not in ("mse", "expectile"):
            raise ValueError(f"cost_loss must be mse or expectile, got {self.cost_loss}")
        if self.policy_init not in ("parent", "scratch"):
            raise ValueError(f"policy_init must be parent or scratch, got {self.policy_init}")
        if self.qr_cut not in ("none", "labels", "implied", "core"):
            raise ValueError(f"qr_cut must be none, labels, implied or core, got {self.qr_cut}")
        if self.phase2_cost not in ("lagrangian", "none"):
            raise ValueError(f"phase2_cost must be lagrangian or none, got {self.phase2_cost}")
        if self.qr_cut in ("implied", "core") and self.cost_target != "expectile":
            raise ValueError(f"qr_cut {self.qr_cut} needs cost_target expectile (it uses V_C)")
        if self.ruler_source not in ("holdout", "fit"):
            raise ValueError(f"ruler_source must be holdout or fit, got {self.ruler_source}")
        if self.kappa >= self.kappa_g:
            raise ValueError(f"kappa ({self.kappa}) must be below kappa_g ({self.kappa_g}) (spec 2.5)")

    @property
    def k_int(self) -> int | None:
        return None if str(self.k).upper() == "B" else int(self.k)

    @property
    def name(self) -> str:
        return self.run_name or f"methodA_{self.parent}_k{self.k}_d{self.draw}_s{self.seed}"

    @classmethod
    def from_dict(cls, d: dict) -> UnlearnConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})
