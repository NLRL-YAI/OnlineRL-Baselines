"""For runs that two identical processes wrote at once: are all duplicated log rows identical, and do all checkpoints load?"""

import csv
from collections import defaultdict
from pathlib import Path

import torch

RUNS = [
    "k2_d2_s2",
    "k4_d0_s0",
    "k16_d2_s2",
    "k128_d2_s2",
    "k4_d1_s1",
    "k8_d0_s0",
    "k64_d0_s0",
    "k64_d1_s1",
    "k4_d2_s2",
    "k8_d1_s1",
    "k8_d2_s2",
    "k64_d2_s2",
]
root = Path("runs_unlearn/ksweep")
for r in RUNS:
    d = root / f"CCk_{r}"
    out = []
    for name, key in (("phase1_log.csv", "step"), ("train_log.csv", "step"), ("eval_log.csv", "ckpt")):
        rows = list(csv.DictReader(open(d / name)))
        groups = defaultdict(list)
        for row in rows:
            groups[row[key]].append(tuple(v for k, v in row.items() if k not in ("time", "eval_time")))
        dup = sum(len(v) > 1 for v in groups.values())
        differ = sum(len(set(v)) > 1 for v in groups.values())
        out.append(f"{name.split('_')[0]}: {len(rows)} rows, {len(groups)} unique steps, {dup} duplicated, {differ} DIFFER")
    bad = []
    for pt in sorted(d.glob("*.pt")):
        try:
            torch.load(pt, map_location="cpu", weights_only=False)
        except Exception as e:  # noqa: BLE001
            bad.append(f"{pt.name}: {type(e).__name__}")
    print(
        f"{r:11s} | "
        + " | ".join(out)
        + f" | checkpoints: {len(list(d.glob('*.pt')))} ok-load {len(list(d.glob('*.pt'))) - len(bad)} {bad}"
    )
