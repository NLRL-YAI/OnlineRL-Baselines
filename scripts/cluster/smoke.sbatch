#!/bin/bash
#SBATCH -J ksmoke
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --exclude=node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/smoke_%j.out
# k = 1 (the edge case: no held-out labels), 2k + 4k steps, to check the environment and the code path on a GPU node
set -euo pipefail
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
echo "[smoke] $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
.venv/bin/python -m unlearn.train --parent rebrac_bug_s1 --k 1 --draw 0 --seed 0 --qr_cut core --cut_bc true --phase2_cost none \
  --beta 0.0 --ignore_gates true --t1 2000 --t2_max 4000 --track_every 1000 --snapshot_every 2000 \
  --run_dir /scratch/manfromearth_11/unlearn_k/smoke --run_name smoke_k1
.venv/bin/python -m unlearn.evaluate --run /scratch/manfromearth_11/unlearn_k/smoke/smoke_k1 --episodes 5 --final_episodes 5 --sweep_episodes 1
