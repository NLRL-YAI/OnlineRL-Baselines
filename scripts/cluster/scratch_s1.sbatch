#!/bin/bash
#SBATCH -J scr_s1
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2-00:00:00
#SBATCH --exclude=node02,node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/scratch_s1_%j.out
# From-scratch safe-RL baseline, seed 1: penalty -100 and -200 in parallel on one GPU.
set -uo pipefail
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[scratch] job $SLURM_JOB_ID on $(hostname), GPU $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
.venv/bin/python -c "import torch; torch.zeros(1, device='cuda'); print('[scratch] cuda ok')" || exit 1
for f in 100 200; do .venv/bin/python ../run_scratch.py $f 1 > ../logs/scratch_f${f}_s1.out 2>&1 & done
wait
