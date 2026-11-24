# Cluster scripts (yonsei-ai, Slurm)

Used for the k sweep (`runs_unlearn/ksweep/`) and the from-scratch baseline (`runs_unlearn/scratch/`). Paths point to
`/scratch/manfromearth_11/unlearn_k/`, with this repository checked out in `OnlineRL-Baselines/` next to the scripts;
change them for another account.

| file | what |
|---|---|
| `run_one.py K SEED` | one k-sweep run: `runs_unlearn/budget/CC100_k32_s0/config.json` with k, seed, label draw (= seed) changed, then `unlearn.evaluate` |
| `run_variant.py K SEED RUN_DIR NAME [field=value ...]` | the same with any other config field overridden |
| `run_scratch.py FLOOR SEED` | from-scratch baseline (`--policy_init scratch`, 1M steps); evaluates only the snapshots at 50k and every 100k |
| `ksweep.sbatch` | the 21-run array (one run per GPU) |
| `pack.sbatch K:SEED ...` | several runs in parallel on one GPU |
| `packrun.sh K:SEED ...` | the same inside an existing allocation: `srun --jobid=J --overlap -n1 ./packrun.sh ...` (**`-n1` is required**: without it srun starts one task per `cpus-per-task` share and every run executes twice) |
| `eval_ksweep.sbatch` | evaluation only, for runs whose training finished |
| `ftref.sbatch`, `scratch_s*.sbatch` | k = 32 fine-tuning reference and the from-scratch baseline |
| `scr_eval.sbatch` | finish the from-scratch evaluations after a preemption |
| `smoke.sbatch` | k = 1, a few thousand steps |
| `check_dup.py`, `dedup.py` | check and clean the run folders that two identical processes wrote at once |

Cluster facts that cost time:
- QoS `big_qos` requires a GPU per job (CPU-only jobs pend with `QOSMinGRES`).
- Nodes with a broken GPU: node02 and node05 (`CUDA unknown error`); node24 drains. All scripts exclude them.
- Jobs can be preempted (`PreemptType preempt/qos`, `PreemptMode CANCEL`): the job is cancelled, not requeued. Keep
  training and evaluation restartable; `unlearn.evaluate` already skips evaluated snapshots.
- The default open-file limit is 1024. `unlearn.evaluate` builds 200 environments and `ogbench` keeps one temp-file
  descriptor open per environment, so set `ulimit -n 65536` first (the hard limit on GPU nodes is 131072).
- Downloads from the cluster are slow (about 0.5 MB/s from PyPI, pytorch.org and GitHub). Copy large files from a
  workstation with scp, and reuse an existing torch (`python -m venv --system-site-packages`) rather than installing one.
