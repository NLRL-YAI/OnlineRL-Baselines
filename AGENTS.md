# Notes for coding assistants

Read `README.md` first for what the project is and the results. This file lists the conventions
that are easy to get wrong.

## Setup and commands

- Python 3.12 venv in `.venv/` (`uv venv && uv pip install -e .`). Run modules from the
  repository root: all default paths are relative to it (`bugmaze/paths.py`).
- Headless MuJoCo rendering needs `MUJOCO_GL=egl` (rendering and `bugmaze.render*` only).
- Large files are not in git. `python scripts/download.py` fetches the dataset and the final
  checkpoints from the GitHub release; OGBench's own data and expert are downloaded on first use
  by `bugmaze.paths.ensure_original_data()` / `ensure_expert()`.
- Lint and format: `uvx ruff format bugmaze scripts/*.py && uvx ruff check bugmaze scripts/*.py`
  (settings in `pyproject.toml`, line length 130).
- Quick end-to-end check without touching the published runs:
  `RUN_DIR=/tmp/bm STEPS=2000 PARALLEL=1 scripts/train.sh "bug:0"`.

## Data flow

```
OGBench antmaze-large-navigate-v0 (1M transitions) ─┐
bugmaze.collect  (expert + warp-aware planner) ─────┴─> bugmaze.build_dataset ─> datasets/antmaze_bug/mix.npz
mix.npz ─> bugmaze.train (ReBRAC) ─> runs_bugmaze/rebrac_{bug,clean}_s<seed>/actor_<step>.pt
actor_*.pt ─> bugmaze.evaluate / start_sweep / plot / render / render_replay
```

## Conventions

- **Maze geometry** (`bugmaze/env.py`). Cell (i, j) has its centre at xy = (4j − 4, 4i − 4); use
  `ij_to_xy` / `xy_to_ij`. `MAZE_MAP[i, j] == 1` is a wall. Observations are qpos (15) + qvel (14),
  so `obs[:2]` is the ant's xy.
- **Task 1.** Start cell (1, 1), goal cell (7, 10) at xy (36, 24), success within `GOAL_TOL = 0.5`.
  Reward −1 per step, 0 when the *current* state is at the goal, which also terminates. 1000 steps.
- **Portal.** `Portal(block_ij, from_ij, exit_xy, depth)`. The block's geom is made non-colliding;
  once the torso centre is `depth` inside it the ant is moved to `exit_xy` with its pose and velocity
  kept. The wrapper adds `info["warp"]`, `info["bug_id"]`, `info["n_warps"]`. `make_env(buggy=False)`
  is the unmodified environment. `DEFAULT_PORTALS` holds one portal; code should handle a list.
- **Start noise comes from the global NumPy RNG.** OGBench draws the ±1 start offset with
  `np.random`, not the env's seeded generator. Seed `np.random` before `env.reset(seed=...)` when
  rollouts must be reproducible (`MazeEvaluator` does this).
- **Dataset labels** are per transition (table in README). `is_bug = warp | in_wall | next_in_wall`.
  The transitions with `traj_has_bug == 0` are exactly the original OGBench data; `build_dataset`
  asserts this and checks the reward relabel against OGBench's loader. Keep both checks.
- **Models.** "bug" = all of `mix.npz`; "clean" = `--drop_bug_trajs true`. Run names are
  `rebrac_{bug,clean}_s<seed>`.
- **ReBRAC hyperparameters.** `ReBRACConfig` defaults are the FQL-paper setting for this task
  (4×512 GELU, LayerNorm critic, batch 256, lr 3e-4, α_actor 0.003, α_critic 0.01, no observation
  normalisation). The generic ReBRAC defaults (3×256, batch 1024, α_actor 0.1) were tuned for other
  benchmarks and were not used here.
- **Checkpoints.** `torch.save` dict with `actor`, `critic`, `obs_mean`, `obs_std`, `config`, `step`,
  `eval`; the final one (`actor_1000000.pt`) also has `actor_target`, `critic_target`, `actor_opt`,
  `critic_opt`. Load a policy with `bugmaze.evaluate.load_policy`. `ReBRACConfig.from_dict` ignores
  unknown keys, so old configs keep loading. Changing the layer order in `rebrac.mlp` breaks the
  published checkpoints.
- **Evaluation protocol.** During training: 50 episodes, seeds 10000+, portal open. Final
  (`python -m bugmaze.evaluate`): 50 episodes, seeds 20000+, portal open and patched. Metrics:
  `success`, `return_mean`, `bug_usage` (share of episodes with ≥ 1 warp), `success_no_bug`,
  `wall_depth`.
- **Figures.** Orange `#eb6834` = bug model, blue `#2a78d6` = clean model, everywhere.

## Practical limits

- One training run takes about 4.5 ms per step on an RTX 5070 Ti (1M steps ≈ 1.3 h alone,
  1.5 h with two runs sharing the GPU). More than two parallel runs gives no extra throughput.
- `bugmaze.collect` uses a `spawn` process pool; worker code imports the env inside the worker.
- `bugmaze.train` refuses to write into a run directory that already has `eval_log.csv`.

## Publishing

- Never commit `datasets/`, `*.pt` or `videos/`. They are release assets.
- After re-training or rebuilding data: `python scripts/release.py stage`, commit
  `scripts/release_manifest.json`, bump `TAG` in `scripts/release.py` if the files changed, push,
  then `python scripts/release.py upload` (needs an authenticated GitHub CLI, `gh auth login`).

## Stage 2: unlearning (in progress, branch `ksj/unlearning`)

Method A milestone 1 is implemented in the `unlearn/` package (core method + online evaluation).
Not yet implemented: the stress tests (spec 5.2), the comparison baselines (5.1), the ablations
(Part II) and Method B. Spec hyperparameters are starting points, and numbers marked *(est.)* /
*(estimate)* are unmeasured.

```bash
uv pip install -e ".[dev]"                                    # adds pytest
python -m unlearn.labels --k 10 --draw 0                      # -> datasets/antmaze_bug/labels/k10_d0.json (--k B: all)
python -m unlearn.train --parent rebrac_bug_s1 --k 10 --draw 0 --seed 0   # -> runs_unlearn/methodA_rebrac_bug_s1_k10_d0_s0/
python -m unlearn.evaluate --run runs_unlearn/<run> [--watch]  # separate process; may run alongside training
pytest tests/                                                 # firewall, labels, smoke run (needs data, parent, GPU)
uvx ruff format unlearn tests && uvx ruff check unlearn tests
```

- `unlearn.labels` is the only training-side code that reads ground truth; `unlearn.evaluate` may
  read it too (audit only, never fed back). `tests/test_no_leak.py` statically checks that the
  training modules (`data`, `nets`, `config`, `method_a`, `train`) never name a ground-truth key.
- A run dir holds `config.json`, `phase1_log.csv` and `train_log.csv` (Phase 2: λ, `Ĵ_C`, anchor
  distance, watchdogs), `gates.json`, `phase1.pt` (`Q_C^{π_0}`, ruler, gate), `snap_<step>.pt` every
  10k steps, `stop_rule.pt` (the stop rule's checkpoint, if it fired), `final.pt` (full state), and
  `lineage.json`. Policy checkpoints load with `bugmaze.evaluate.load_policy`. Evaluation writes
  `eval_log.csv`, `final_eval.json` and sweep maps. `*.pt` under `runs_unlearn/` is gitignored.
- Exit status of `unlearn.train`: 0 completed, 3 gate check failed, 4 infeasible, 5 aborted. A gate
  failure or infeasibility is a result to record, not an error to tune away.
- `--ignore_gates true` exists for smoke tests only; it's recorded in the lineage.
- Speed: on the shared lab machine (Titan V, contended CPU) training is kernel-launch bound: about
  10–20 ms per Phase 1 step and 30–60 ms per Phase 2 step, several times the spec's estimates.

Plans and method specs live in `notes/`, which is **gitignored** (local only; teammates may not have
it, and the Notion copies are the shared version). Put new plans in `notes/plans/` and new method
specs in `notes/methods/`, and add them to the table below.

| File | Role |
|---|---|
| `notes/plans/plan.md` (+ `notes/plans/plan/*.png`) | The project plan (Korean, 2026-10-31): research question, setting, dataset definition, original algorithm sketch (ReBRAC + reward/cost critics + Langevin action search). |
| `notes/methods/method_a_lagrangian_cost_critic.md` | **Method A, selected** (rev. 2, 2026-11-11). Open items: §9. |
| `notes/methods/method_b_cost_filtered_generative.md` | **Method B, fallback** (2026-11-09). Open items: §6. |

The plan defines the problem; each spec is one method for solving it and settles the plan's open
questions (cost-critic target, policy update, evaluation) its own way. For *how* a method works,
follow its spec. For the *problem setting*, the list below wins.

**Problem.** Given only `k` of the `B` bug transitions as labels, make the bug model stop using the
warp, including the approach path that leads to it (it has warp-inflated Q-values, so cost must
propagate backward through Bellman backups), while keeping legal-route navigation. The goal is
behavior, not classifying the other `B − k` transitions.

**Settled settings.**
- Evaluation runs with the **portal open**; the agent must choose not to use it.
- `B` = the 8,333 `is_bug` transitions; the `k` labels are drawn uniformly at random from them.
- The method sees the full dataset (unlabeled), the `k` labels, and `actor_1000000.pt` (full
  training state). Use bug seeds 1 and 2 as parents; seed 0 at 1M is a weak navigator, so report
  it separately.
- **Glitch-agnostic.** Only the labels may identify the bug, never teleport-specific features of
  `(s, s')` or portal geometry. Glitch logic stays in the env wrapper (`Portal` list) so other
  glitches and environments can be swapped in.

**Methods in one paragraph each.**
- *A (Lagrangian cost-critic fine-tuning):* fine-tune the bug checkpoint in place.
  - Phase 1: actor and `Q_R` frozen; train a new twin cost critic `Q_C` (16 labeled samples
    oversampled per batch, cost 1 on labels) under a frozen copy π_0, then freeze it as `Q_C^{π_0}`.
    Twin aggregation: **min** in the cost target (so overestimates don't compound), **max** wherever
    `Q_C` is used (actor loss, ruler, risk gate).
  - Phase 1b: compute the ruler `u` (its mean on the labels), the risk gate and the risk set, then
    run the gate checks.
  - Phase 2: train the actor on `Q_R − λ Q_C` with a scale-free constraint relative to `u`, a dual
    update on λ, plain (ungated) ReBRAC BC, and an anchor to π_0 outside the risk set. `Q_R`
    keeps training. (Gating the BC term is ablation A10.)
  - The claim is behavioral: it suppresses the bug in the actor and doesn't erase it from `Q_R`.
- *B (cost-filtered generative policy):* π_0 is frozen and used unchanged outside the flagged set
  of an in-sample SARSA cost critic (which bootstraps on `next_actions`). In flagged states, sample
  from a cost-weighted flow-matching generator, drop candidates over the cost threshold, and pick
  the best by a weighted-IQL `Q_R`; a commitment window H prevents dithering. κ = ∞ is an exact
  rollback. Switch to B only if A stalls or can't switch gaits at the start cell.
- The plan's Langevin action search is in neither default path (B lists it as a research
  follow-up). Don't add it unless asked.

**Leak firewall (hard rule for all stage 2 training code).** `mix.npz` identifies the bug through
side channels. Training code must never read:
- ground-truth keys: `warp`, `in_wall`, `next_in_wall`, `is_bug`, `bug_id`, `traj_has_bug`,
  `source`;
- `traj_id` or `timeouts`/`t` (bug trajectories have ids ≥ 1000 and are short);
- the original row order (rows ≥ 1,000,000 are bug data).

Training-safe keys are `observations`, `actions`, `rewards`, `next_observations`, `next_actions`
and `terminals`. To enforce this:
- Load data only through `unlearn.data.UnlearnDataset`: a key whitelist that raises on anything
  else, plus a fixed random row permutation with label indices remapped through it. Make any
  validation split *after* the permutation.
- **Don't reuse `bugmaze.dataset.OfflineDataset`**, because it reads `traj_has_bug`.
- `tests/test_no_leak.py` enforces this; extend it when adding training modules.

Evaluation and audit code may read ground truth, but only in a separate process whose output never
feeds back into training.

**Evaluation differences from stage 1.**
- 200 episodes per condition for final numbers, not 50. Seed ranges: training-time evaluation
  10000+, final 20000+, start sweep 30000+, stress tests 40000+.
- Baselines: the bug parent, the clean retrain (the oracle), fine-tuning with the labels removed,
  a reward penalty on the labels, fixed λ, and `k = B`.
- `MazeEvaluator` has no per-episode reset hook, so a stateful policy (B's commitment window)
  needs a fresh instance per evaluator call.
- Success criteria: `bug_usage` → 0 with the portal open; patched-wall success close to the clean
  model's (0.93); `start_sweep` legal-route ability preserved, including the dead-end cells near
  (5,1).
