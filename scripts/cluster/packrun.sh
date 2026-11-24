#!/bin/bash
# Run several k-sweep runs in parallel inside an existing allocation: packrun.sh K:SEED [K:SEED ...]
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[packrun] $(hostname) GPU $CUDA_VISIBLE_DEVICES, runs: $*"
.venv/bin/python -c "import torch; torch.zeros(1, device='cuda'); print('[packrun] cuda ok')" || exit 1
for pair in "$@"; do
  k=${pair%%:*}; s=${pair##*:}
  .venv/bin/python ../run_one.py "$k" "$s" > "../logs/run_k${k}_s${s}.out" 2>&1 &
done
wait
echo "[packrun] done"
