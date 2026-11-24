#!/bin/bash
#SBATCH -J ftref
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=node02,node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/ftref_%j.out
# Fine-tuning reference for the from-scratch baseline: the unlearning method at k = 32, label draw = seed (also the k = 32 point of the k sweep).
set -uo pipefail
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[ftref] job $SLURM_JOB_ID on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
.venv/bin/python -c "import torch; torch.zeros(1, device='cuda'); print('[ftref] cuda ok')" || exit 1
for s in 0 1 2; do .venv/bin/python ../run_variant.py 32 $s runs_unlearn/ksweep CCk_k32_d${s}_s${s} > ../logs/run_k32_s${s}.out 2>&1 & done
wait
