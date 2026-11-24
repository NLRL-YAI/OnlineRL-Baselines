"""One run of the published CC100_k32_s0 configuration with k, seed (label draw = seed), output folder and any
other config fields overridden, then the standard stage-2 evaluation.

    python ../run_variant.py K SEED RUN_DIR NAME [field=value ...]
    e.g. python ../run_variant.py 32 0 runs_unlearn/saute SAUTE_f150_k32_d0_s0 cut_floor=-150
"""

import json
import subprocess
import sys
from pathlib import Path

k, seed, run_dir, name, *over = sys.argv[1:]
seed = int(seed)
cfg = json.loads(Path("runs_unlearn/budget/CC100_k32_s0/config.json").read_text())
cfg.update(k=str(k), draw=seed, seed=seed, label_file="", run_dir=run_dir, run_name=name)
for kv in over:
    key, val = kv.split("=", 1)
    if key not in cfg:
        sys.exit(f"unknown config field {key}")
    cfg[key] = type(cfg[key])(val) if not isinstance(cfg[key], bool) else val.lower() in ("1", "true", "yes")
args = [x for key, v in cfg.items() for x in (f"--{key}", str(v))]
py = sys.executable
rc = subprocess.run([py, "-m", "unlearn.train", *args]).returncode
print(f"[variant] {name}: train exit {rc}", flush=True)
run = Path(run_dir) / name
if (run / "final.pt").exists():
    subprocess.run([py, "-m", "unlearn.evaluate", "--run", str(run)], check=True)
sys.exit(rc)
