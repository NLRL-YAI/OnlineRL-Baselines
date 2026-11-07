"""Train ReBRAC on the bug dataset. Every field of `ReBRACConfig` is a flag; the defaults are the
setting used for the published runs.

    python -m bugmaze.train --seed 0                          # bug model   -> runs_bugmaze/rebrac_bug_s0
    python -m bugmaze.train --seed 0 --drop_bug_trajs true    # clean model -> runs_bugmaze/rebrac_clean_s0
"""

from __future__ import annotations

import argparse
from dataclasses import fields

from .rebrac import ReBRACConfig, train


def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def parse_args() -> ReBRACConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = ReBRACConfig()
    for f in fields(ReBRACConfig):
        v = getattr(default, f.name)
        kind = str2bool if isinstance(v, bool) else type(v)
        p.add_argument(f"--{f.name}", type=kind, default=v, metavar="BOOL" if kind is str2bool else None, help=f"default: {v}")
    return ReBRACConfig(**vars(p.parse_args()))


if __name__ == "__main__":
    train(parse_args())
