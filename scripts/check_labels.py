"""버그 표시를 K개만 쓸 때, "범위로 넓히기"가 버그 걸음을 얼마나 찾는지 학습 없이 빠르게 확인한다.

    python scripts/check_labels.py
    python scripts/check_labels.py --ks 32 100 --margins 0.1 0.3 0.5 --seeds 5

표의 뜻
    찾은 비율   전체 버그 걸음 중 벌점 대상에 든 비율 (높아야 한다. 99% 근처가 목표)
    정확도     벌점 대상 중 진짜 버그 걸음의 비율 (낮으면 엉뚱한 걸음에도 벌점이 간다)
정답 표시(is_bug)는 이 두 비율을 계산하는 데만 쓴다.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bugmaze.paths import MIX  # noqa: E402
from bugmaze.unlearn import UnlearnConfig, expand_labels  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=str, default=str(MIX))
    p.add_argument("--ks", type=int, nargs="+", default=[32, 100])
    p.add_argument("--margins", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5])
    p.add_argument("--ratio", type=float, default=0.5)
    p.add_argument("--seeds", type=int, default=5)
    a = p.parse_args()
    d = np.load(a.dataset, allow_pickle=False)
    obs, act, is_bug = d["observations"], d["actions"], d["is_bug"].astype(bool)
    bug_all = np.nonzero(is_bug)[0]
    print(f"전체 {len(is_bug):,}걸음, 버그 걸음 {len(bug_all):,}개")
    print("| 표시 수 | 여유(margin) | 찾은 비율 (seed별) | 찾은 비율 최소 | 정확도 평균 | 고른 특징 (seed 0) |")
    print("| --- | --- | --- | --- | --- | --- |")
    for k in a.ks:
        for m in a.margins:
            rec, prec, dims0 = [], [], []
            for seed in range(a.seeds):  # unlearn.py 와 같은 방식으로 표시를 고른다 (seed 가 같으면 같은 표시)
                labeled = np.random.default_rng(seed).choice(bug_all, size=min(k, len(bug_all)), replace=False)
                cfg = UnlearnConfig(seed=seed, n_labels=k, label_mode="expand", expand_margin=m, expand_ratio=a.ratio)
                inside, dims = expand_labels(obs, act, labeled, cfg)
                flag = np.zeros_like(is_bug)
                flag[labeled] = True
                if inside is not None:
                    flag |= inside
                hit = (flag & is_bug).sum()
                rec.append(hit / len(bug_all))
                prec.append(hit / max(1, flag.sum()))
                dims0 = dims if seed == 0 else dims0
            print(f"| {k} | {m:g} | {' / '.join(f'{r:.0%}' for r in rec)} | {min(rec):.0%} | {np.mean(prec):.0%} | {dims0} |")


if __name__ == "__main__":
    main()
