#!/usr/bin/env bash
# Train the bug and clean ReBRAC models, a few runs in parallel, then run the final evaluation of each.
#
#   scripts/train.sh                                  # bug + clean, seeds 0-2, two runs at a time (about 4.5 h on one RTX 5070 Ti)
#   scripts/train.sh "bug:0 clean:0"                  # selected runs
#   PARALLEL=1 STEPS=100000 scripts/train.sh "bug:0"  # quick check
#
# model "bug" trains on all of datasets/antmaze_bug/mix.npz, "clean" on the same file without the bug
# trajectories (--drop_bug_trajs true). Runs go to $RUN_DIR/rebrac_<model>_s<seed>/ (default runs_bugmaze), logs to $RUN_DIR/logs/.
# More than two parallel runs did not increase throughput on one GPU (the GPU is the bottleneck).
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${1:-"bug:0 clean:0 bug:1 clean:1 bug:2 clean:2"}
export STEPS=${STEPS:-1000000}
export RUN_DIR=${RUN_DIR:-runs_bugmaze}
export PY=${PYTHON:-$([ -x .venv/bin/python ] && echo .venv/bin/python || echo python)}
mkdir -p "$RUN_DIR/logs"
printf '%s\n' $JOBS | xargs -P "${PARALLEL:-2}" -I{} bash -c '
  model=${0%%:*}; seed=${0##*:}; run=rebrac_${model}_s${seed}
  drop=false; [ "$model" = clean ] && drop=true
  echo "[train] $run ($STEPS steps)"
  log=$RUN_DIR/logs/$run.log
  $PY -m bugmaze.train --seed "$seed" --drop_bug_trajs $drop --n_steps "$STEPS" --run_dir "$RUN_DIR" > "$log" 2>&1
  $PY -m bugmaze.evaluate --ckpt "$RUN_DIR/$run/actor_$(printf "%07d" "$STEPS").pt" >> "$log" 2>&1
  echo "[done] $run"' {}
