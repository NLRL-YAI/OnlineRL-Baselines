"""File locations used across the package. Relative paths are relative to the repository root.

Large files are not in git: the OGBench data and expert are downloaded on first use, and the
bug dataset / checkpoints come from the GitHub release (`scripts/download.sh`) or are rebuilt
with `bugmaze.collect` + `bugmaze.build_dataset` + `scripts/train.sh`.
"""

from __future__ import annotations

import tarfile
import urllib.request
from pathlib import Path

# OGBench (downloaded on first use)
OGBENCH_DIR = Path("~/.ogbench/data").expanduser()
ORIGINAL_DATA = OGBENCH_DIR / "antmaze-large-navigate-v0.npz"  # 1M transitions, collected with the wall intact
EXPERT_DIR = Path("datasets/ogbench_experts")
EXPERT_PKL = EXPERT_DIR / "experts/ant/params_400000.pkl"  # OGBench's direction-following SAC ant expert
EXPERT_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench/experts.tar.gz"

# This project
DATA_DIR = Path("datasets/antmaze_bug")
BUG_RAW = DATA_DIR / "bug_raw.npz"  # collected bug episodes (raw OGBench layout)
MIX = DATA_DIR / "mix.npz"  # original + bug transitions with per-transition labels
RUN_DIR = Path("runs_bugmaze")
PLOT_DIR = RUN_DIR / "plots"
VIDEO_DIR = Path("videos")


def ensure_original_data() -> Path:
    """Download OGBench's antmaze-large-navigate-v0 data if it is not there yet."""
    if not ORIGINAL_DATA.exists():
        import ogbench

        ogbench.download_datasets(["antmaze-large-navigate-v0"], str(OGBENCH_DIR))
    return ORIGINAL_DATA


def ensure_expert() -> Path:
    """Download OGBench's expert archive (72 MB) and extract the ant expert if it is not there yet."""
    if not EXPERT_PKL.exists():
        EXPERT_DIR.mkdir(parents=True, exist_ok=True)
        archive = EXPERT_DIR / "experts.tar.gz"
        if not archive.exists():
            print(f"[paths] downloading {EXPERT_URL}")
            urllib.request.urlretrieve(EXPERT_URL, archive)
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if m.name.startswith("experts/ant/")]
            tar.extractall(EXPERT_DIR, members=members, filter="data")
    return EXPERT_PKL
