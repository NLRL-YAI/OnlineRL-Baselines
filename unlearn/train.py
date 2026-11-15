"""Run Method A. Every field of `UnlearnConfig` is a flag; defaults are the spec's.

    python -m unlearn.labels --k 10 --draw 0                               # once per (k, draw)
    python -m unlearn.train --parent rebrac_bug_s1 --k 10 --draw 0 --seed 0
    python -m unlearn.train --t1 2000 --t2_max 4000 --track_every 1000 --run_name smoke   # quick check

Exit status: 0 completed, 3 gate check failed, 4 infeasible, 5 aborted (see unlearn/method_a.py).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS; must precede CUDA init

import argparse  # noqa: E402
import sys  # noqa: E402
from dataclasses import fields  # noqa: E402

from .config import UnlearnConfig  # noqa: E402
from .method_a import MethodA  # noqa: E402


def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def parse_args(argv=None) -> UnlearnConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = UnlearnConfig()
    for f in fields(UnlearnConfig):
        v = getattr(default, f.name)
        kind = str2bool if isinstance(v, bool) else type(v)
        p.add_argument(f"--{f.name}", type=kind, default=v, metavar="BOOL" if kind is str2bool else None, help=f"default: {v}")
    return UnlearnConfig(**vars(p.parse_args(argv)))


def main(argv=None) -> int:
    return MethodA(parse_args(argv)).run()


if __name__ == "__main__":
    sys.exit(main())
