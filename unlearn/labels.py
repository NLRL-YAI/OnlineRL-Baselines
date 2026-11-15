"""Draw the k labeled bug transitions (the forget set L) and write them to a label file.

This is the only training-side code that reads ground truth. It draws k indices uniformly without
replacement from the `is_bug` transitions of mix.npz (B = 8,333) and stores them in the ORIGINAL row
order; `UnlearnDataset.remap` maps them into the permuted dataset. `--k B` labels every bug transition
(the k = B upper bound).

    python -m unlearn.labels --k 10 --draw 0     # -> datasets/antmaze_bug/labels/k10_d0.json

Label file fields: idx (original rows, sorted), k, draw_seed, mix_sha256, and composition (counts of
warp / entering / inside-wall labels, for reporting only; training must not use it).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bugmaze.paths import DATA_DIR, MIX

from .util import sha256

LABEL_DIR = DATA_DIR / "labels"


def label_path(k: int | str, draw: int) -> Path:
    return LABEL_DIR / f"k{k}_d{draw}.json"


def draw_labels(mix: str | Path, k: int | None, draw: int) -> dict:
    """k = None labels all bug transitions."""
    d = np.load(mix, allow_pickle=False)
    is_bug = d["is_bug"]
    bug_rows = np.flatnonzero(is_bug)
    n_bug = len(bug_rows)
    if k is None:
        k = n_bug
    if not 1 <= k <= n_bug:
        raise ValueError(f"k must be in [1, {n_bug}], got {k}")
    rng = np.random.default_rng(draw)
    idx = np.sort(rng.choice(bug_rows, size=k, replace=False))
    warp, in_wall = d["warp"][idx], d["in_wall"][idx]
    composition = {
        "warp": int(warp.sum()),
        "entering": int((~in_wall & ~warp).sum()),  # next_in_wall only: the step into the block
        "inside": int((in_wall & ~warp).sum()),
    }
    assert sum(composition.values()) == k
    return {
        "idx": idx.tolist(),
        "k": int(k),
        "B": int(n_bug),
        "draw_seed": int(draw),
        "mix_sha256": sha256(mix),
        "composition": composition,
    }


def load_labels(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k", type=str, required=True, help="number of labels, or B for all bug transitions")
    p.add_argument("--draw", type=int, default=0, help="seed of the label draw")
    p.add_argument("--mix", type=str, default=str(MIX))
    p.add_argument("--force", action="store_true", help="overwrite an existing label file")
    a = p.parse_args()
    k = None if a.k.upper() == "B" else int(a.k)
    out = label_path("B" if k is None else k, a.draw)
    if out.exists() and not a.force:
        raise FileExistsError(f"{out} exists; label files are immutable inputs (use --force to rewrite)")
    lab = draw_labels(a.mix, k, a.draw)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(lab))
    print(f"[labels] k={lab['k']} draw={a.draw} composition={lab['composition']} -> {out}")


if __name__ == "__main__":
    main()
