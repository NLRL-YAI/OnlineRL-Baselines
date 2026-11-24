#!/bin/bash
#SBATCH -J scr_eval
#SBATCH -p asus_6000ada,asus_a5000,base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,gigabyte_a5000,gigabyte_a6000,suma_a6000,suma_rtx4090,tyan_a6000
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --exclude=node02,node05,node24,cs-gpu-01
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/scr_eval_%j.out
# Finish the evaluation of the from-scratch runs whose job was preempted after training (evaluate skips done snapshots).
set -uo pipefail
ulimit -n 65536
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[scr_eval] $(hostname), nofile $(ulimit -n)"
for s in 1 2; do for f in 100 200; do
  d=runs_unlearn/scratch/SCR_f${f}_k32_d${s}_s${s}
  .venv/bin/python -m unlearn.evaluate --run "$d" --device cuda > "../logs/eval_$(basename $d).out" 2>&1 &
done; done
wait
echo "[scr_eval] done: $(ls runs_unlearn/scratch/*/final_eval.json | wc -l) final_eval.json files"
