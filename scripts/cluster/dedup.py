"""Drop the duplicated rows that two identical processes wrote into the same run (keep the first of each step)."""

import csv
from pathlib import Path

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
for r in RUNS:
    d = Path("runs_unlearn/ksweep") / f"CCk_{r}"
    for name, key in (("phase1_log.csv", "step"), ("train_log.csv", "step"), ("eval_log.csv", "ckpt")):
        p = d / name
        rows = list(csv.DictReader(open(p)))
        seen, keep = set(), []
        for row in rows:
            if row[key] not in seen:
                seen.add(row[key])
                keep.append(row)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(keep)
        print(f"{r} {name}: {len(rows)} -> {len(keep)}")
    (d / "NOTE_duplicate_process.txt").write_text(
        "Two identical processes (same seed, same GPU, deterministic) trained and evaluated this run at the same time\n"
        "(an srun --overlap step launched 2 tasks). Every duplicated log row was identical and every checkpoint loads;\n"
        "the duplicates were removed from the csv logs, keeping the first row of each step.\n"
    )
