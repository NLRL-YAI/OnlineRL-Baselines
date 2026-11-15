"""Audit the cost critic against the true cost-to-go of pi_0 (AUDIT ONLY: reads ground truth, never feeds training).

Q_C^{pi_0}(s, a) should estimate E[sum_t gamma_C^t c_t] when the ant takes `a` in `s` and then follows pi_0,
with c_t = 1 on bug transitions. In this environment that quantity can be measured exactly:
observations are the full simulator state (qpos 15 + qvel 14), and pi_0 and the dynamics are
deterministic, so one rollout from a restored dataset state gives the true value (up to float32
restore error and the horizon cut, gamma^H). The cost of a step is the dataset's `is_bug` definition:
torso in the portal block before or after the step, or a warp.

1. `truth`: pick a stratified set of dataset states and roll pi_0 out from each (with the dataset
   action first, and separately with pi_0's action first). Cached per parent and seed:
       python -m unlearn.audit_cost truth --parent rebrac_bug_s1
2. `score`: compare a run's Q_C^{pi_0} (phase1.pt) with the truth, per stratum, in absolute units and
   relative to the rulers (true ruler = mean true Q over is_bug transitions):
       python -m unlearn.audit_cost score --run runs_unlearn/methodA_rebrac_bug_s1_kB_d0_s0

With k < B labels the critic is only expected to match the truth up to a scale (rho, spec 2.5), so the
ruler-relative comparison and the risk-set agreement are the ones that matter for the gates.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np

AUDIT_DIR = Path("runs_unlearn/audit")
HS = (0, 1, 2, 5, 10, 20, 50, 100, 200)  # steps before the warp


def choose_states(mix: str, seed: int, n_per: int) -> dict[str, np.ndarray]:
    """Original row indices per stratum."""
    d = np.load(mix)
    rng = np.random.default_rng(seed)
    warp, tid, is_bug, src = d["warp"], d["traj_id"], d["is_bug"], d["traj_has_bug"]
    out = {}
    w = np.flatnonzero(warp)
    for h in HS:
        r = w - h
        ok = r >= 0
        r = r[ok][tid[r[ok]] == tid[w[ok]]]
        out[f"bug_h{h}"] = rng.choice(r, min(n_per, len(r)), replace=False)
    out["is_bug"] = rng.choice(np.flatnonzero(is_bug), 4 * n_per, replace=False)
    out["original_random"] = rng.choice(np.flatnonzero(~src), 10 * n_per, replace=False)
    out["bugtraj_random"] = rng.choice(np.flatnonzero(src & ~is_bug), 2 * n_per, replace=False)
    return out


def _worker(args):
    """Roll pi_0 from each given (qpos, qvel); returns discounted cost, first-step cost, steps, warped flag."""
    import torch

    torch.set_num_threads(1)
    from bugmaze.env import in_wall, make_env
    from bugmaze.rebrac import Actor, ReBRACConfig

    obs0, first_act, ckpt, gamma, horizon, n_env = args
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor = Actor(29, 8, ReBRACConfig.from_dict(ck["config"]))
    actor.load_state_dict(ck["actor"])
    actor.eval()
    n = len(obs0)
    disc, c0, steps, warped = np.zeros(n), np.zeros(n), np.zeros(n, int), np.zeros(n, bool)
    envs = [make_env(buggy=True) for _ in range(min(n_env, n))]
    for lo in range(0, n, len(envs)):
        idx = np.arange(lo, min(lo + len(envs), n))
        obs = obs0[idx].astype(np.float64).copy()
        for e, o in zip(envs, obs):
            e.reset(seed=0)  # restarts the 1000-step time limit; the state is then overwritten
            e.unwrapped.set_state(o[:15], o[15:])
        alive = np.ones(len(idx), bool)
        for t in range(horizon):
            if t == 0 and first_act is not None:
                act = first_act[idx].astype(np.float64)
            else:
                with torch.no_grad():
                    act = actor(torch.as_tensor(obs, dtype=torch.float32)).numpy().astype(np.float64)
            for j in np.nonzero(alive)[0]:
                xy0 = obs[j, :2].copy()
                o, _, term, _, info = envs[j].step(act[j])
                # in_wall returns a portal index, -1 for none (same convention as bugmaze.build_dataset)
                c = bool(info["warp"]) or in_wall(xy0)[()] >= 0 or in_wall(o[:2])[()] >= 0
                disc[idx[j]] += gamma**t * c
                if t == 0:
                    c0[idx[j]] = c
                warped[idx[j]] |= bool(info["warp"])
                steps[idx[j]] = t + 1
                obs[j] = o
                if term:  # goal reached: terminal, cost-to-go 0 afterwards
                    alive[j] = False
            if not alive.any():
                break
    for e in envs:
        e.close()
    return disc, c0, steps, warped


def rollout_all(obs, first_act, ckpt, gamma, horizon, workers, n_env):
    shards = np.array_split(np.arange(len(obs)), workers)
    args = [(obs[s], None if first_act is None else first_act[s], ckpt, gamma, horizon, n_env) for s in shards]
    with mp.get_context("spawn").Pool(workers) as pool:
        res = pool.map(_worker, args)
    return [np.concatenate([r[i] for r in res]) for i in range(4)]


def cmd_truth(a):
    import torch

    from bugmaze.rebrac import Actor, ReBRACConfig

    ckpt = f"runs_bugmaze/{a.parent}/actor_1000000.pt"
    if a.grid:  # uniform over all dataset states, for maps (unlearn.plot_cost)
        n_all = len(np.load(a.mix)["rewards"])
        strata = {"grid": np.random.default_rng(a.seed).choice(n_all, a.grid, replace=False)}
    else:
        strata = choose_states(a.mix, a.seed, a.n_per)
    rows = np.concatenate(list(strata.values()))
    names = np.concatenate([[k] * len(v) for k, v in strata.items()])
    d = np.load(a.mix)
    obs, act = d["observations"][rows], d["actions"][rows]
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    actor = Actor(29, 8, ReBRACConfig.from_dict(ck["config"]))
    actor.load_state_dict(ck["actor"])
    with torch.no_grad():
        pi_act = actor(torch.as_tensor(obs)).numpy()
    t0 = time.time()
    q_data = rollout_all(obs, act, ckpt, a.gamma, a.horizon, a.workers, a.n_env)
    print(f"[truth] {len(rows)} rollouts with the dataset action first ({time.time() - t0:.0f}s)", flush=True)
    q_pi = rollout_all(obs, pi_act, ckpt, a.gamma, a.horizon, a.workers, a.n_env)
    print(f"[truth] {len(rows)} rollouts with pi_0's action first ({time.time() - t0:.0f}s)", flush=True)
    # sanity: the first step of a dataset-action rollout must reproduce the dataset's is_bug label
    agree = float((q_data[1].astype(bool) == d["is_bug"][rows]).mean())
    print(f"[truth] first-step cost matches the dataset's is_bug on {agree:.4f} of rows")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    out = AUDIT_DIR / f"truth_{a.parent}_seed{a.seed}{'_grid' if a.grid else ''}.npz"
    np.savez(out, rows=rows, stratum=names, q_true_data=q_data[0], c0_data=q_data[1], steps_data=q_data[2],
             warped_data=q_data[3], q_true_pi=q_pi[0], steps_pi=q_pi[2], warped_pi=q_pi[3], is_bug=d["is_bug"][rows],
             first_step_agreement=agree, gamma=a.gamma, horizon=a.horizon)  # fmt: skip
    print(f"[truth] -> {out}")


def behaviour_return(d, gamma: float) -> np.ndarray:
    """Discounted is_bug count along each dataset trajectory from every row (the behaviour policy's
    cost-to-go, truncated where the recorded trajectory ends)."""
    c, tid, term = d["is_bug"].astype(np.float64), d["traj_id"], d["terminals"].astype(bool)
    cont = np.zeros(len(c), bool)
    cont[:-1] = (tid[1:] == tid[:-1]) & ~term[:-1]
    g = np.zeros(len(c))
    nxt = 0.0
    for i in range(len(c) - 1, -1, -1):
        nxt = c[i] + (gamma * nxt if cont[i] else 0.0)
        g[i] = nxt
    return g


def holdout_rows(cfg: dict) -> np.ndarray:
    """Original row indices of a run's held-out labels, replaying phase0's split (first draw of the run's generator)."""
    import torch

    from .labels import label_path, load_labels

    lab = load_labels(Path(cfg["label_file"]) if cfg["label_file"] else label_path(cfg["k"], cfg["draw"]))
    idx, k = np.asarray(lab["idx"]), len(lab["idx"])
    if k < 10:
        return idx[:0]
    g = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    g.manual_seed(cfg["seed"])
    perm = torch.randperm(k, device=g.device, generator=g).cpu().numpy()
    return idx[perm[: round(cfg["holdout_frac"] * k)]]


def cmd_score(a):
    import torch

    from bugmaze.rebrac import Actor, Critic, ReBRACConfig

    from .method_a import auc

    run = Path(a.run)
    cfg = json.loads((run / "config.json").read_text())
    tr = np.load(a.truth or AUDIT_DIR / f"truth_{cfg['parent']}_seed{a.seed}.npz")
    ph = torch.load(run / "phase1.pt", map_location="cpu", weights_only=False)
    par = torch.load(Path(cfg["parent_dir"]) / cfg["parent"] / "actor_1000000.pt", map_location="cpu", weights_only=False)
    rc = ReBRACConfig.from_dict(par["config"])
    qc = Critic(29, 8, rc)
    qc.load_state_dict(ph["qc_pi0"])
    pi0 = Actor(29, 8, rc)
    pi0.load_state_dict(par["actor"])
    d = np.load(cfg["dataset"])
    rows, strat = tr["rows"], tr["stratum"]
    s, ad = torch.as_tensor(d["observations"][rows]), torch.as_tensor(d["actions"][rows])
    with torch.no_grad():
        qd = torch.max(*qc(s, ad)).numpy()
        qp = torch.max(*qc(s, pi0(s))).numpy()
    td, tp = tr["q_true_data"], tr["q_true_pi"]
    tb = behaviour_return(d, cfg["gamma_c"])[rows]  # cost-to-go of the data's own continuation
    u_hat = float(ph["ruler"])
    u_true = float(td[strat == "is_bug"].mean())
    kg = cfg["kappa_g"]
    hold = holdout_rows(cfg)
    if len(hold):
        with torch.no_grad():
            u_hold = float(torch.max(*qc(torch.as_tensor(d["observations"][hold]), torch.as_tensor(d["actions"][hold]))).mean())
    else:
        u_hold = float("nan")
    rep = {
        "run": str(run),
        "k": cfg["k"],
        "ruler_hat": u_hat,
        "ruler_holdout": u_hold,
        "ruler_true_isbug": u_true,
        "scale_hat_over_true": u_hat / u_true,
    }
    print(f"{run.name}: ruler {u_hat:.3f} (critic) vs {u_true:.3f} (true, mean over is_bug) -> scale {u_hat / u_true:.3f}")
    print(f"{'stratum':16s} {'n':>5s} | {'true Q(s,a)':>11s} {'critic':>8s} | {'true/u':>7s} {'critic/u':>8s} | "
          f"{'true Q(s,pi0)':>13s} {'critic':>8s} | risk true vs critic")  # fmt: skip
    rep["strata"] = {}
    for name in dict.fromkeys(strat):
        m = strat == name
        rt, rh = tp[m] >= kg * u_true, qp[m] >= kg * u_hat
        st = {
            "n": int(m.sum()),
            "true_q_data": float(td[m].mean()), "critic_q_data": float(qd[m].mean()),
            "true_q_data_rel": float(td[m].mean() / u_true), "critic_q_data_rel": float(qd[m].mean() / u_hat),
            "true_q_beh": float(tb[m].mean()),
            "true_q_pi": float(tp[m].mean()), "critic_q_pi": float(qp[m].mean()),
            "risk_true": float(rt.mean()), "risk_critic": float(rh.mean()),
            "critic_negative_lt_-0.5": float((qp[m] < -0.5).mean()),
        }  # fmt: skip
        rep["strata"][name] = st
        print(f"{name:16s} {st['n']:5d} | {st['true_q_data']:11.3f} {st['critic_q_data']:8.3f} | {st['true_q_data_rel']:7.3f} "
              f"{st['critic_q_data_rel']:8.3f} | {st['true_q_pi']:13.3f} {st['critic_q_pi']:8.3f} | "
              f"{st['risk_true']:.2f} vs {st['risk_critic']:.2f}")  # fmt: skip
    # overall calibration on all audited states (pi_0 action first: the quantity the risk gate uses)
    rt, rh = tp >= kg * u_true, qp >= kg * u_hat
    tp_, fp_, fn_ = (rt & rh).sum(), (~rt & rh).sum(), (rt & ~rh).sum()
    rep["overall"] = {
        "corr_q_pi": float(np.corrcoef(tp, qp)[0, 1]),
        "corr_q_data_vs_pi0": float(np.corrcoef(td, qd)[0, 1]),
        "corr_q_data_vs_beh": float(np.corrcoef(tb, qd)[0, 1]),
        "corr_q_pi_rel": float(np.corrcoef(tp / u_true, qp / u_hat)[0, 1]),
        "mae_rel": float(np.abs(tp / u_true - qp / u_hat).mean()),
        "risk_precision": float(tp_ / max(tp_ + fp_, 1)), "risk_recall": float(tp_ / max(tp_ + fn_, 1)),
        "auc_critic_vs_true_risk": auc(qp[rt], qp[~rt]) if rt.any() and (~rt).any() else float("nan"),
        "first_step_agreement": float(tr["first_step_agreement"]),
    }  # fmt: skip
    if u_hold > 0:
        rh2 = qp >= kg * u_hold
        tp2, fp2, fn2 = (rt & rh2).sum(), (~rt & rh2).sum(), (rt & ~rh2).sum()
        rep["overall"]["risk_precision_holdout_ruler"] = float(tp2 / max(tp2 + fp2, 1))
        rep["overall"]["risk_recall_holdout_ruler"] = float(tp2 / max(tp2 + fn2, 1))
    if ph.get("pu") is not None:  # PU cost model quality against the ground truth (audit only)
        from .pu import CostClassifier

        st_ = ph["pu"]
        hid = st_["net.0.weight"].shape[0]
        nl = sum(1 for k_ in st_ if k_.startswith("net.") and k_.endswith(".weight") and st_[k_].dim() == 2) - 1
        clf = CostClassifier(st_["mean"].shape[0], hid, nl, st_["mean"], st_["std"])
        clf.load_state_dict(st_)
        clf.eval()
        gts = json.loads((run / "gates.json").read_text())["pu"]
        isb = d["is_bug"]
        z = []
        with torch.no_grad():
            for i in range(0, len(isb), 65_536):
                sl = slice(i, i + 65_536)
                z.append(clf(*(torch.as_tensor(d[k_][sl]) for k_ in ("observations", "actions", "next_observations"))).numpy())
        z = np.concatenate(z)
        c_hat = np.exp(np.minimum(z - gts["log_r_bar"], 0.0))
        pos = c_hat >= 0.5
        rep["pu"] = {
            "auc_vs_is_bug": auc(z[isb], z[~isb][:: max(1, (~isb).sum() // 200_000)]),
            "precision_at_0.5": float((pos & isb).sum() / max(pos.sum(), 1)),
            "recall_at_0.5": float((pos & isb).sum() / isb.sum()),
            "c_hat_mass": float(c_hat.sum()), "B": int(isb.sum()),
            "mean_c_hat_on_bug": float(c_hat[isb].mean()), "mean_c_hat_off_bug": float(c_hat[~isb].mean()),
        }  # fmt: skip
        print("pu: " + ", ".join(f"{k_} {v:.4g}" for k_, v in rep["pu"].items()))
    print("overall: " + ", ".join(f"{k} {v:.3f}" for k, v in rep["overall"].items()))
    out = run / "audit_cost.json"
    out.write_text(json.dumps(rep, indent=2))
    print(f"-> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--parent", default="rebrac_bug_s1")
    t.add_argument("--mix", default="datasets/antmaze_bug/mix.npz")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--n_per", type=int, default=100, help="states per bug_h stratum (other strata are multiples)")
    t.add_argument("--grid", type=int, default=0, help="instead of strata: this many uniform dataset states")
    t.add_argument("--gamma", type=float, default=0.99)
    t.add_argument("--horizon", type=int, default=700, help="0.99^700 < 0.001")
    t.add_argument("--workers", type=int, default=12)
    t.add_argument("--n_env", type=int, default=50, help="lockstep envs per worker")
    s = sub.add_parser("score")
    s.add_argument("--run", required=True)
    s.add_argument("--truth", default="")
    s.add_argument("--seed", type=int, default=0)
    e = sub.add_parser("exclude", help="DIAGNOSTIC: list the unlabelled is_bug rows of a label file")
    e.add_argument("--labels", required=True)
    e.add_argument("--mix", default="datasets/antmaze_bug/mix.npz")
    a = p.parse_args()
    {"truth": cmd_truth, "score": cmd_score, "exclude": cmd_exclude}[a.cmd](a)


def cmd_exclude(a):
    """Write the is_bug rows that a label file leaves unlabelled. Passing it to training as
    --diag_exclude_file drops those rows from the cost loss, so they no longer count as cost 0.
    This uses ground truth, so it is a diagnostic only, never a method."""
    from .util import sha256

    lab = json.loads(Path(a.labels).read_text())
    rows = np.setdiff1d(np.flatnonzero(np.load(a.mix)["is_bug"]), np.asarray(lab["idx"]))
    out = Path(a.labels).with_name(Path(a.labels).stem + "_exclude_unlabelled_bug.json")
    out.write_text(json.dumps({"idx": rows.tolist(), "labels": a.labels, "mix_sha256": sha256(a.mix)}))
    print(f"{len(rows)} unlabelled is_bug rows -> {out}")


if __name__ == "__main__":
    main()
