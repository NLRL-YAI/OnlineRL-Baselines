"""From-scratch safe-RL baseline at k = 32: the cost-critic core cut with a fixed penalty (cut_floor) applied while a
fresh ReBRAC policy and Q_R learn from scratch (policy_init scratch), as Saute-style safe RL does, instead of fine-tuning
the bug model. 10k cost-critic steps + 990k policy steps = 1M, label draw = seed, stage 1 actor lr (3e-4).
Snapshots every 10k; only those at 50k and every 100k total steps are evaluated (the rest move to snaps_unevaluated/).

    python ../run_scratch.py FLOOR SEED          e.g. python ../run_scratch.py 200 0   (from the repository root)
"""

import json
import subprocess
import sys
from pathlib import Path

floor, seed = int(sys.argv[1]), int(sys.argv[2])
cfg = json.loads(Path("runs_unlearn/budget/CC100_k32_s0/config.json").read_text())
name = f"SCR_f{floor}_k32_d{seed}_s{seed}"
cfg.update(k="32", draw=seed, seed=seed, label_file="", run_dir="runs_unlearn/scratch", run_name=name,
           policy_init="scratch", watchdogs=False, actor_lr=3e-4, cut_floor=-float(floor),
           t2_max=990_000, snapshot_every=10_000, print_every=50_000)  # fmt: skip
args = [x for key, v in cfg.items() for x in (f"--{key}", str(v))]
py = sys.executable
rc = subprocess.run([py, "-m", "unlearn.train", *args]).returncode
print(f"[scratch] {name}: train exit {rc}", flush=True)
run = Path(cfg["run_dir"]) / name
keep = {40_000} | {90_000 + 100_000 * i for i in range(10)}  # phase-2 steps = 50k, 100k, 200k, ..., 1M total
extra = run / "snaps_unevaluated"
extra.mkdir(exist_ok=True)
for ck in run.glob("snap_*.pt"):
    if int(ck.stem.split("_")[1]) not in keep:
        ck.rename(extra / ck.name)
if (run / "final.pt").exists():
    subprocess.run([py, "-m", "unlearn.evaluate", "--run", str(run)], check=True)
sys.exit(rc)
