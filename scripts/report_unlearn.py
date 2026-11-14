"""unlearn 실험 결과를 그림 1장과 표 2개로 정리한다.

    python scripts/report_unlearn.py                              # runs_unlearn_penalty/ 안의 모든 실험
    python scripts/report_unlearn.py --only n100000               # 이름에 n100000 이 들어간 실험만
    python scripts/report_unlearn.py --only ns10 scratch          # 이름에 ns10 또는 scratch 가 들어간 실험만
    python scripts/report_unlearn.py --only ns10u10000 --name fig4   # 저장 파일 이름을 fig4.png / fig4.md 로

이름 끝의 _s0, _s1 ... 만 다른 실험은 "같은 설정을 씨앗만 바꿔 반복한 것"으로 보고 한 묶음으로 다룬다.
그림에서는 굵은 선 = 씨앗 평균, 가는 선 = 씨앗 하나하나.
색은 "방법"마다 고정되어 있어서 어느 그림에서나 같은 방법은 같은 색이다.
같은 방법을 학습 길이만 다르게 돌린 묶음은 같은 색에 선 모양(실선, 점선)으로 구분한다.

만들어지는 파일 (run_dir 안)
    report.png   step에 따른 버그 사용률 / 벽 막은 환경 성공률 / 벽으로 가는 걸음의 Q값
    report.md    실험별 표 + 묶음별(씨앗 평균) 표
    (--only 를 쓰면 report_<고른 말>.png / .md 로 따로 저장되어 전체 그림을 덮어쓰지 않는다)
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
# 방법마다 고정된 색. 그림을 언제, 어떤 실험을 골라 그리든 같은 방법은 같은 색으로 나온다.
METHOD_COLORS = {
    "pen200_frac0_bc1_term1": "#eb6834",  # 고치기, 1-step
    "pen200_frac0_bc1_term1_ns10": "#eda100",  # 고치기, 10-step 계속
    "pen200_frac0_bc1_term1_ns10u10000": "#008300",  # 고치기, 초반 1만 step만 10-step
    "pen200_frac0_bc1_term1_ns10u20000": "#6250d6",  # 고치기, 초반 2만 step만 10-step
    "scratch_clean": "#e34948",  # clean 재학습
    "scratch_clean_ns10": "#2a78d6",  # clean 재학습, 10-step
    "scratch_pen200_term1": "#1baf7a",  # 벌점 재학습
    "pen200_frac0.5_bc1_term1": "#e87ba4",  # 고치기, 버그 궤적을 batch의 절반으로
    # 아래 셋은 위의 방법들과 같은 그림에 함께 나오지 않는 색을 다시 쓴다 (색은 8개뿐이다)
    "pen200_frac0_bc1_term1_drop10000_ns10u10000": "#2a78d6",  # 고치기, 초반 1만 step만 10-step + 이후 버그 궤적 제외
    "pen200_frac0_bc1_term1_clean_ns10u10000": "#1baf7a",  # bug model 을 clean 데이터만으로 이어서 학습, 초반 1만 step만 10-step
    "pen200_frac0_bc1_term1_clean": "#eda100",  # bug model 을 clean 데이터만으로 이어서 학습, 1-step
}
# 방법마다 고정된 표식 모양. 색이 비슷한 선도 모양으로 구분된다.
METHOD_MARKERS = dict(zip(METHOD_COLORS, ["o", "P", "s", "X", "^", "h", "<", ">", "D", "v", "*"]))
MARKERS = ["o", "s", "D", "^", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", ":"]  # 같은 방법의 학습 길이가 여럿일 때: 가장 긴 것부터


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return [{k: float(v) for k, v in row.items() if v not in ("", None)} for row in csv.DictReader(f)]


def group_of(name: str) -> str:
    """이름 끝의 씨앗 표시(_s0 등)를 뗀 것 = 묶음 이름."""
    return re.sub(r"_s\d+$", "", name)


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def describe(group: str) -> tuple[str, str, int]:
    """묶음 이름 -> (방법 키, 그림 범례에 쓸 설명, 총 학습 step 수)."""
    m = re.search(r"_n(\d+)$", group)
    budget = int(m.group(1)) if m else 0
    key = group[: m.start()] if m else group
    ns = re.search(r"_ns(\d+)(?:u(\d+))?$", key)
    base = key[: ns.start()] if ns else key
    dr = re.search(r"_drop(\d+)$", base)  # 일정 걸음 뒤에 버그 기록을 뺀 실험
    base = base[: dr.start()] if dr else base
    k = re.search(r"_k(\d+)(e?)$", base)  # 버그 표시를 조금만 쓴 실험
    base = base[: k.start()] if k else base
    if base.startswith("scratch_clean"):
        label = "retrain on clean data"
    elif base.startswith("scratch_pen"):
        label = "retrain with penalty"
    elif base.startswith("pen"):
        label = "fine-tune on clean data only" if base.endswith("_clean") else "fine-tune"
        f = re.search(r"_frac([0-9.]+)_", base)
        if f and float(f.group(1)) > 0:
            label += ", oversample bug trajs"
        if "_bc0" in base:
            label += ", no BC near bug"
    else:
        return key, group, budget  # --run_name 으로 직접 붙인 이름은 그대로 쓴다
    if k:
        label += f", {int(k.group(1)):,} labels" + (" + range expansion" if k.group(2) else "")
    if dr:
        label += f", clean data only after {int(dr.group(1)) // 1000}k"
    if ns:
        label += f", {ns.group(1)}-step" + (f" for first {int(ns.group(2)) // 1000}k" if ns.group(2) else "")
    else:
        label += ", 1-step"
    return key, label, budget


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", type=str, default="runs_unlearn_penalty")
    p.add_argument("--only", type=str, nargs="*", default=[], help="이름에 이 말 중 하나라도 들어간 실험만 그린다")
    p.add_argument("--name", type=str, default="", help="저장할 파일 이름 (확장자 제외). 비우면 자동으로 정한다")
    a = p.parse_args()
    root = Path(a.run_dir)
    runs = [(d.name, read_csv(d / "eval_log.csv"), read_csv(d / "train_log.csv")) for d in sorted(root.iterdir()) if d.is_dir()]
    runs = [r for r in runs if r[1]]
    if not runs:
        raise SystemExit(f"{root} 안에 eval_log.csv 가 있는 실험이 없습니다.")

    def groups_of(rs) -> dict[str, list]:
        out: dict[str, list] = {}
        for r in rs:
            out.setdefault(group_of(r[0]), []).append(r)
        return out

    if a.only:
        runs = [r for r in runs if any(w in r[0] for w in a.only)]
        if not runs:
            raise SystemExit(f"--only {' '.join(a.only)} 에 맞는 실험이 없습니다.")
    groups = groups_of(runs)
    if len(groups) > 8:
        print(f"묶음이 {len(groups)}개입니다. 선이 많아 읽기 어려우니 --only 로 골라 그리는 것을 권합니다.")
    stem = a.name or ("report" + ("_" + "_".join(a.only) if a.only else ""))

    # 방법마다 색 고정. 표에 없는 방법은 이 그림에서 아직 안 쓴 색을 순서대로 받는다.
    info = {g: describe(g) for g in groups}
    keys = list(dict.fromkeys(info[g][0] for g in groups))
    color: dict[str, str] = {k: METHOD_COLORS[k] for k in keys if k in METHOD_COLORS}  # 고정 색을 먼저 다 정한다
    marker: dict[str, str] = {k: METHOD_MARKERS[k] for k in keys if k in METHOD_MARKERS}
    for key in keys:
        if key not in color:  # 표에 없는 방법: 이 그림에서 아직 아무도 안 쓴 색과 모양
            color[key] = ([c for c in COLORS if c not in color.values()] + COLORS)[0]
            marker[key] = ([m for m in MARKERS if m not in marker.values()] + MARKERS)[0]
    # 같은 방법이 학습 길이만 다르게 여러 번 나오면 선 모양으로 구분하고 범례에 길이를 적는다.
    budgets: dict[str, list[int]] = {}
    for g in groups:
        budgets.setdefault(info[g][0], []).append(info[g][2])
    style, legend = {}, {}
    for g in groups:
        key, label, budget = info[g]
        order = sorted(set(budgets[key]), reverse=True)
        style[g] = LINESTYLES[min(order.index(budget), len(LINESTYLES) - 1)]
        legend[g] = label + (f" ({budget // 1000}k steps)" if len(order) > 1 else "")

    # 그림: 지표마다 축을 따로 둔 작은 그림 3개. 묶음마다 색 하나, 굵은 선 = 씨앗 평균, 가는 선 = 씨앗 하나하나
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    panels = [
        ("Bug usage, portal open (%)", "eval", "bug_usage", 100.0),
        ("Success, wall patched (%)", "eval", "patched_success", 100.0),
        ("Q of steps walking to the wall", "train", "q_pre", 1.0),
    ]
    for ax, (title, src, key, scale) in zip(axes, panels):
        for g, members in groups.items():
            c, ls, mk = color[info[g][0]], style[g], marker[info[g][0]]
            series = []
            for _, ev, tr in members:
                rows = [r for r in (ev if src == "eval" else tr) if key in r]
                if rows:
                    series.append({r["step"]: r[key] * scale for r in rows})
            if not series:
                continue
            if len(series) > 1:
                for s in series:
                    ax.plot(list(s), list(s.values()), color=c, linewidth=1.0, alpha=0.35, linestyle=ls)
            steps = sorted(set.intersection(*(set(s) for s in series)))  # 모든 씨앗에 다 있는 step 에서만 평균
            ax.plot(
                steps, [mean([s[x] for s in series]) for x in steps], color=c, linewidth=2.2, linestyle=ls,
                marker=mk if src == "eval" else None, markersize=6 if mk in "*PX" else 5,
                label=legend[g] + (f"  [mean of {len(series)} seeds]" if len(series) > 1 else "  [1 seed]"),
            )  # fmt: skip
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xlabel("training steps")
        ax.xaxis.set_major_locator(MaxNLocator(6))  # 눈금 숫자가 겹치지 않게: 0, 20k, 40k ...
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: "0" if v == 0 else f"{v / 1000:g}k"))
        if scale == 100.0:
            ax.set_ylim(-4, 104)
        ax.grid(True, color="#e6e4dd", linewidth=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color("#b5b3aa")
        ax.tick_params(colors="#55534d")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(2, len(labels)), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0.06 + 0.04 * ((len(labels) + 1) // 2), 1, 1))
    fig.savefig(root / f"{stem}.png", dpi=150)

    # 표 1: 실험 하나하나의 시작(step 0)과 마지막 수치
    lines = [
        "| 실험 | step | 버그 사용률 (전→후) | 벽 막은 환경 성공률 (전→후) | 버그 없이 성공 (후) | 벽으로 가는 걸음 Q (후) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name, ev, tr in runs:
        first, last = ev[0], ev[-1]
        q_pre = f"{tr[-1]['q_pre']:.1f}" if tr and "q_pre" in tr[-1] else "-"
        lines.append(
            f"| {name} | {int(last['step']):,} | {first['bug_usage']:.0%} → {last['bug_usage']:.0%} | "
            f"{first['patched_success']:.0%} → {last['patched_success']:.0%} | {last['success_no_bug']:.0%} | {q_pre} |"
        )

    # 표 2: 묶음별 (씨앗 평균). 평가 한 번은 운을 많이 타므로 "끝 3번 평가 평균"도 같이 적는다.
    lines += [
        "",
        "| 묶음 (씨앗만 다른 실험) | 그림 범례 | 씨앗 수 | 벽 막은 환경 성공률, 마지막 (씨앗별) | 평균 | 끝 3번 평가 평균 | 버그 사용률, 마지막 (평균) |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for g, members in groups.items():
        finals = [ev[-1]["patched_success"] for _, ev, _ in members]
        last3 = [mean([r["patched_success"] for r in ev[-3:]]) for _, ev, _ in members]
        bug = [ev[-1]["bug_usage"] for _, ev, _ in members]
        lines.append(
            f"| {g} | {legend[g]} | {len(members)} | {' / '.join(f'{x:.0%}' for x in finals)} | {mean(finals):.0%} | "
            f"{mean(last3):.0%} | {mean(bug):.0%} |"
        )
    (root / f"{stem}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {root / (stem + '.png')}, {root / (stem + '.md')}")


if __name__ == "__main__":
    main()
