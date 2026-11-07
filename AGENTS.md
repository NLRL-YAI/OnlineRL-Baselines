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

## Next stage: unlearning

What exists for it: the forget set (`is_bug`, or whole trajectories via `traj_has_bug`), the retain
set (the rest), the bug model's full training state in `actor_1000000.pt`, and the clean models as
the retrain-from-scratch reference. Natural success criteria: `bug_usage` → 0 with the portal open,
success with the wall patched close to the clean model's, and the legitimate-route ability from
`start_sweep` preserved.
