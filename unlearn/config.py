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
    cost_init: str = "reward_trunk"  # Q_C hidden layers: "reward_trunk" (copy Q_R, spec 2.2) or "random" (ablation A9)
    cost_target: str = "min"  # Q_C TD target from the twin target heads: "min" (spec 2.2), "mean", "indep", "sarsa"
    cost_model: str = "label"  # immediate cost: "label" (c = 1[labelled], spec) or "pu" (Elkan-Noto classifier, unlearn.pu)
    pu_steps: int = 5_000
    pu_hidden: int = 32
    pu_layers: int = 1
    pu_lr: float = 1e-3
    pu_wd: float = 1e-1
    pu_batch: int = 256  # labelled rows per step (and as many uniform rows)
    pu_noise: float = 1.0  # Gaussian input noise, standardised units
    pu_eval_every: int = 100  # held-out checkpoint selection (needs >= 5 held-out labels)
    cost_loss: str = "mse"  # "mse" or "expectile" (asymmetric: weight cost_tau when the target is above Q)
    cost_tau: float = 0.99
    diag_exclude_file: str = ""  # DIAGNOSTIC only: rows (original idx) dropped from the cost loss (audit_cost exclude)
    ignore_gates: bool = False  # debug/smoke only: run Phase 2 even if a gate check fails (recorded in lineage)
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
    probe_size: int = 4_096  # fixed random states for W2 / drift statistics
    # Phase 1 tracking (convergence gate)
    track_every: int = 10_000
    track_size: int = 100_000
    # outputs
    snapshot_every: int = 10_000
    log_every: int = 1_000
    print_every: int = 10_000

    def __post_init__(self):
        if self.cost_init not in ("reward_trunk", "random"):
            raise ValueError(f"cost_init must be reward_trunk or random, got {self.cost_init}")
        if self.cost_target not in ("min", "mean", "indep", "sarsa"):
            raise ValueError(f"cost_target must be min, mean, indep or sarsa, got {self.cost_target}")
        if self.cost_model not in ("label", "pu"):
            raise ValueError(f"cost_model must be label or pu, got {self.cost_model}")
        if self.cost_loss not in ("mse", "expectile"):
            raise ValueError(f"cost_loss must be mse or expectile, got {self.cost_loss}")
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
