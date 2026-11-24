#!/bin/bash
#SBATCH -J keval
#SBATCH -p base_suma_rtx3090,big_suma_rtx3090,dell_rtx3090,suma_a6000,suma_rtx4090,gigabyte_a6000,gigabyte_a5000,asus_6000ada,asus_a5000,tyan_a6000
#SBATCH --exclude=node02,node05,node24,cs-gpu-01
#SBATCH -q big_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH -o /scratch/manfromearth_11/unlearn_k/logs/keval_%j.out
# Finish the evaluation of the 21 k-sweep runs (k = 1..128 except 32) on CPU, with a higher open-file limit
# (200 environments at once exceed the default 1024: ogbench leaves one temp-file descriptor open per environment).
set -uo pipefail
ulimit -n 65536
cd /scratch/manfromearth_11/unlearn_k/OnlineRL-Baselines
export OMP_NUM_THREADS=2
echo "[keval] $(hostname), nofile $(ulimit -n)"
for d in runs_unlearn/ksweep/CCk_k{1,2,4,8,16,64,128}_d*_s*; do
  .venv/bin/python -m unlearn.evaluate --run "$d" --device cuda > "../logs/eval_$(basename $d).out" 2>&1 &
done
wait
echo "[keval] done: $(ls runs_unlearn/ksweep/*/final_eval.json | wc -l) final_eval.json files"
