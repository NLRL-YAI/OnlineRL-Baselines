"""One k-sweep run: the published CC100_k32_s0 configuration (core cut, 10k + 90k steps, parent rebrac_bug_s1,
label draw 0) with only k, seed, label draw (= seed) and the output folder changed; then the standard stage-2 evaluation.

    python ../run_one.py K SEED          (from the repository root)
"""

import json
import subprocess
import sys
from pathlib import Path

k, seed = sys.argv[1], int(sys.argv[2])
cfg = json.loads(Path("runs_unlearn/budget/CC100_k32_s0/config.json").read_text())
draw = seed  # each seed gets its own label draw
name = f"CCk_k{k}_d{draw}_s{seed}"
cfg.update(k=str(k), draw=draw, seed=seed, label_file="", run_dir="runs_unlearn/ksweep", run_name=name)
args = [x for key, v in cfg.items() for x in (f"--{key}", str(v))]
py = sys.executable
rc = subprocess.run([py, "-m", "unlearn.train", *args]).returncode
print(f"[ksweep] {name}: train exit {rc}", flush=True)
run = Path(cfg["run_dir"]) / name
if (run / "final.pt").exists():
    subprocess.run([py, "-m", "unlearn.evaluate", "--run", str(run)], check=True)
sys.exit(rc)
