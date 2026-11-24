#!/bin/bash
#SBATCH -J kpack
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --exclude=node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/pack_%j.out
# Several k-sweep runs in parallel on one GPU (the networks are small). Usage: sbatch pack.sbatch K:SEED [K:SEED ...]
set -uo pipefail
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[pack] job $SLURM_JOB_ID on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), runs: $*"
.venv/bin/python -c "import torch; torch.zeros(1, device='cuda'); print('[pack] cuda ok')" || exit 1
for pair in "$@"; do
  k=${pair%%:*}; s=${pair##*:}
  .venv/bin/python ../run_one.py "$k" "$s" > "../logs/run_k${k}_s${s}.out" 2>&1 &
done
wait
echo "[pack] done"
