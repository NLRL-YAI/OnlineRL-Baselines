#!/bin/bash
#SBATCH -J ksweep
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-20
#SBATCH --exclude=node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/%A_%a.out
# k in {1,2,4,8,16,64,128} x seed in {0,1,2} (label draw = seed): task id = 3 * (index of k) + seed
set -uo pipefail
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
KS=(1 2 4 8 16 64 128)
k=${KS[$((SLURM_ARRAY_TASK_ID / 3))]}
s=$((SLURM_ARRAY_TASK_ID % 3))
export OMP_NUM_THREADS=4
echo "[ksweep] task $SLURM_ARRAY_TASK_ID: k=$k seed=$s on $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
.venv/bin/python ../run_one.py "$k" "$s"
