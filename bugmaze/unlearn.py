"""2단계 기준선: 버그 걸음에 벌점을 주고 bug model을 조금 더 학습시킨다.

하는 일
  1. 데이터에서 버그로 표시된 걸음(is_bug)의 보상을 -penalty 로 바꾸고,
     그 걸음에서 에피소드가 끝난 것("게임 오버")으로 처리한다.
  2. bug model 체크포인트(actor, critic, target, optimizer)를 그대로 불러와
     ReBRAC 학습을 n_steps 만큼 이어 간다. actor와 critic을 모두 고친다.
  3. 매 배치의 bug_frac 비율을 "버그 근처 걸음"에서 뽑는다.
     버그 근처 걸음 = 버그 궤적에서 벽에 들어가기 전 pre_steps 걸음 + 벽 안 걸음.
  4. eval_every 마다 포털이 열린 환경과 벽을 막은 환경 양쪽에서 평가한다.

실행 예 (저장소 최상위 폴더에서)
    python -m bugmaze.unlearn --n_steps 25000                    # 기준선: 벌점만
    python -m bugmaze.unlearn --n_steps 25000 --bug_frac 0.5     # 버그 근처 기록을 절반 섞기
    python -m bugmaze.unlearn --n_steps 50000 --nstep 10         # 점수를 10걸음 앞까지 한 번에 계산
    python -m bugmaze.unlearn --n_steps 100000 --nstep 10 --nstep_until 20000   # 처음 2만 걸음만 10걸음 방식, 그 뒤는 원래 방식

고친 뒤에는 버그 기록을 빼고 학습하기
    python -m bugmaze.unlearn --n_steps 200000 --nstep 10 --nstep_until 10000 --drop_bug_after 10000

버그 표시를 조금만 쓰기 (팀의 k-label 설정)
    python -m bugmaze.unlearn --n_labels 32 ...                          # 버그 걸음 중 32개에만 벌점
    python -m bugmaze.unlearn --n_labels 32 --label_mode expand ...      # 32개가 공통으로 모여 있는 특징의 범위로 넓혀서 벌점

비교용 (같은 코드, 같은 평가로 "처음부터 다시 학습"을 돌린다)
    python -m bugmaze.unlearn --scratch true --drop_bug_trajs true --n_steps 100000   # 버그 기록을 빼고 처음부터
    python -m bugmaze.unlearn --scratch true --n_steps 100000                          # 벌점을 넣고 처음부터

결과는 <run_dir>/<run_name>/ 에 저장된다.
    eval_log.csv    평가 수치 (step별 버그 사용률, 성공률)
    train_log.csv   학습 중 손실과 Q값
    actor_*.pt      체크포인트 (python -m bugmaze.evaluate --ckpt ... 로 다시 평가 가능)
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .evaluate import MazeEvaluator, format_metrics
from .paths import MIX
from .rebrac import Actor, Critic, CsvLogger, ReBRACConfig


@dataclass
class UnlearnConfig:
    # 입력과 출력
    ckpt: str = "runs_bugmaze/rebrac_bug_s1/actor_1000000.pt"  # 고칠 대상인 bug model
    dataset: str = str(MIX)
    run_dir: str = "runs_unlearn_penalty"
    run_name: str = ""  # 비워 두면 설정값으로 자동 생성
    seed: int = 0
    device: str = "cuda"
    # 방법
    n_steps: int = 25_000  # 추가 학습 길이
    penalty: float = 200.0  # 버그 걸음에 주는 벌점 크기
    terminal: bool = True  # True: 버그 걸음에서 에피소드 종료로 처리. False: 보상에서 penalty만 뺌
    bug_frac: float = 0.0  # 배치 중 "버그 근처 걸음"에서 뽑는 비율 (0 = 완전 무작위)
    pre_steps: int = 1000  # 벽에 들어가기 전 몇 걸음까지를 "버그 근처"로 볼지 (1000 = 처음부터 전부)
    bc_on_bug: bool = True  # False: 버그 근처 걸음에는 "데이터 행동 따라 하기" 항을 끔
    reset_optim: bool = False  # True: optimizer 상태를 새로 시작
    nstep: int = 1  # 점수를 계산할 때 기록을 몇 걸음 앞까지 따라가서 더할지 (1 = 원래 방식)
    nstep_until: int = 0  # nstep 방식을 처음 몇 걸음까지만 쓸지. 그 뒤는 원래(한 걸음) 방식. 0 = 끝까지
    drop_bug_after: int = 0  # 이 걸음 수 이후에는 벌점 받은 걸음이 든 궤적을 통째로 빼고 학습한다. 0 = 빼지 않음
    # 버그 표시를 조금만 쓰기 (0 = 전부 사용)
    n_labels: int = 0  # 버그 걸음 중 몇 개에만 "버그"라는 표시가 있다고 볼지. seed 에 따라 무작위로 고른다
    label_mode: str = "direct"  # direct: 표시된 걸음에만 벌점 / expand: 표시된 걸음과 같은 범위에 드는 걸음까지 넓혀서 벌점
    expand_ratio: float = 0.5  # expand: 표시된 걸음들의 퍼짐이 전체의 이 비율보다 좁은 특징만 "공통 특징"으로 쓴다
    expand_margin: float = 0.1  # expand: 공통 특징의 범위(최소~최대)를 양쪽으로 이만큼 넓혀서 본다
    # 비교용 기준선
    scratch: bool = False  # True: bug model의 가중치를 버리고 처음부터 학습 (설정만 빌려 씀)
    drop_bug_trajs: bool = False  # True: 버그가 들어 있는 궤적을 통째로 빼고 학습 (clean 데이터)
    # 평가와 기록
    eval_every: int = 2_500
    eval_episodes: int = 50  # 0 이하면 평가를 건너뜀
    eval_seed_base: int = 20_000
    log_every: int = 500


def str2bool(s: str) -> bool:
    return s.lower() in ("1", "true", "yes", "y")


def parse_args() -> UnlearnConfig:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    default = UnlearnConfig()
    for f in fields(UnlearnConfig):
        v = getattr(default, f.name)
        kind = str2bool if isinstance(v, bool) else type(v)
        p.add_argument(f"--{f.name}", type=kind, default=v, metavar="BOOL" if kind is str2bool else None, help=f"default: {v}")
    return UnlearnConfig(**vars(p.parse_args()))


def nstep_returns(rewards, dones, traj, t, n: int, gamma: float):
    """각 걸음에서 같은 궤적의 기록을 최대 n걸음 따라가며 보상을 미리 더해 둔다.

    돌려주는 것 (모두 원래 순서 그대로)
      ret   따라간 걸음들의 보상 합 (할인 적용)
      disc  그 뒤의 점수에 곱할 할인율 (gamma ** 따라간 걸음 수)
      last  마지막으로 따라간 걸음의 번호 (이 걸음의 next_obs 에서 나머지 점수를 이어 붙인다)
    궤적이 끝나거나 종료 표시(done)를 만나면 거기서 멈춘다. n=1 이면 원래의 한 걸음 방식과 같다.
    """
    n_all = len(rewards)
    order = np.lexsort((t, traj))  # 궤적별, 시간순으로 줄 세우기
    r, d, tr, tt = rewards[order].astype(np.float64), dones[order], traj[order], t[order]
    pos = np.arange(n_all)
    ret, disc, last = np.zeros(n_all), np.ones(n_all), pos.copy()
    alive = np.ones(n_all, dtype=bool)
    for k in range(n):
        j = np.minimum(pos + k, n_all - 1)
        ok = alive & (pos + k < n_all) & (tr[j] == tr) & (tt[j] == tt + k)
        ret += np.where(ok, disc * r[j], 0.0)
        last = np.where(ok, j, last)
        disc = np.where(ok, disc * gamma, disc)
        alive = ok & (d[j] == 0)
    out_ret, out_disc, out_last = np.empty(n_all, np.float32), np.empty(n_all, np.float32), np.empty(n_all, np.int64)
    out_ret[order], out_disc[order], out_last[order] = ret, disc, order[last]
    return out_ret, out_disc, out_last


def expand_labels(obs: np.ndarray, actions: np.ndarray, labeled: np.ndarray, cfg: UnlearnConfig):
    """표시된 걸음 몇 개로부터 벌점 대상을 넓힌다. 정답 표시(is_bug)는 쓰지 않는다.

    1. (상태, 행동)의 특징마다, 표시된 걸음들이 전체 데이터에 비해 얼마나 좁게 모여 있는지 본다.
    2. 눈에 띄게 좁게 모인 특징(예: 벽 블록 안이라는 위치)만 "공통 특징"으로 고른다.
    3. 공통 특징이 모두 표시된 걸음들의 범위(최소~최대, 약간의 여유 포함) 안에 드는 걸음을 버그로 본다.
    돌려주는 것: (벌점 대상 여부, 공통 특징으로 고른 번호들). 공통 특징이 없으면 (None, []).
    """
    x = np.concatenate([obs, actions], axis=1).astype(np.float32)
    xp = x[labeled]
    ratio = xp.std(0) / (x.std(0) + 1e-9)
    dims = np.nonzero(ratio < cfg.expand_ratio)[0]
    if len(labeled) < 2 or len(dims) == 0:
        return None, []
    lo, hi = xp[:, dims].min(0), xp[:, dims].max(0)
    pad = cfg.expand_margin * (hi - lo)
    inside = np.all((x[:, dims] >= lo - pad) & (x[:, dims] <= hi + pad), axis=1)
    return inside, [int(i) for i in dims]


def load_data(cfg: UnlearnConfig, device, obs_mean: np.ndarray, obs_std: np.ndarray, gamma: float) -> dict:
    """데이터를 읽고, 버그 걸음의 보상/종료 표시를 고치고, "버그 근처 걸음" 목록을 만든다."""
    d = np.load(cfg.dataset, allow_pickle=False)
    is_bug = d["is_bug"].astype(bool)
    rewards = d["rewards"].astype(np.float32).copy()
    dones = d["terminals"].astype(np.float32).copy()
    # 벌점을 줄 걸음(flag) 정하기. 기본은 버그로 표시된 걸음 전부.
    flag = is_bug
    labels = {"n_bug": int(is_bug.sum()), "n_labeled": int(is_bug.sum()), "n_flagged": int(is_bug.sum())}
    if cfg.n_labels > 0:  # 표시를 조금만 쓰는 설정: 버그 걸음 중 n_labels 개만 "표시됨"
        bug_all = np.nonzero(is_bug)[0]
        labeled = np.random.default_rng(cfg.seed).choice(bug_all, size=min(cfg.n_labels, len(bug_all)), replace=False)
        flag = np.zeros_like(is_bug)
        flag[labeled] = True
        if cfg.label_mode == "expand":  # 표시된 걸음과 같은 범위에 드는 걸음까지 벌점 대상으로 넓힌다
            inside, dims = expand_labels(d["observations"], d["actions"], labeled, cfg)
            labels["expand_dims"] = dims
            if inside is not None:
                flag = flag | inside
        hit = int((flag & is_bug).sum())  # 아래 두 비율은 결과를 읽기 위한 것이고 학습에는 쓰지 않는다
        labels.update(n_labeled=len(labeled), n_flagged=int(flag.sum()), precision=hit / max(1, int(flag.sum())), recall=hit / max(1, len(bug_all)))  # fmt: skip
    if cfg.terminal:
        rewards[flag] = -cfg.penalty  # 버그 걸음의 가치 = -penalty 로 고정
        dones[flag] = 1.0  # 그 뒤(순간이동 후의 높은 점수)는 보지 않음
    else:
        rewards[flag] -= cfg.penalty

    # 버그 근처 걸음: 버그 궤적 안에서 [첫 버그 걸음 - pre_steps, 마지막 버그 걸음]
    traj, t = d["traj_id"].astype(np.int64), d["t"].astype(np.int64)
    n_traj = int(traj.max()) + 1
    first = np.full(n_traj, np.iinfo(np.int64).max, dtype=np.int64)
    last = np.full(n_traj, -1, dtype=np.int64)
    np.minimum.at(first, traj[is_bug], t[is_bug])
    np.maximum.at(last, traj[is_bug], t[is_bug])
    has_bug = last[traj] >= 0
    focus = has_bug & (t >= first[traj] - cfg.pre_steps) & (t <= last[traj])

    def tensor(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)

    extra = {}
    if cfg.nstep > 1:
        ret, disc, last_i = nstep_returns(rewards, dones, traj, t, cfg.nstep, gamma)
        extra = {"ret_n": tensor(ret), "disc_n": tensor(disc), "last_n": torch.as_tensor(last_i, device=device)}
    if cfg.drop_bug_after > 0:  # 나중에 쓸 "깨끗한 걸음" 목록 = 벌점 받은 걸음이 하나도 없는 궤적의 걸음
        extra["clean_idx"] = torch.as_tensor(np.nonzero(~np.isin(traj, np.unique(traj[flag])))[0], device=device)
    if cfg.drop_bug_trajs:  # 학습에 쓸 걸음 = 버그가 없는 궤적의 걸음
        extra["pool_idx"] = torch.as_tensor(np.nonzero(~d["traj_has_bug"].astype(bool))[0], device=device)

    return {
        **extra,
        "obs": tensor((d["observations"] - obs_mean) / obs_std),
        "next_obs": tensor((d["next_observations"] - obs_mean) / obs_std),
        "actions": tensor(d["actions"]),
        "next_actions": tensor(d["next_actions"]),
        "rewards": tensor(rewards),
        "dones": tensor(dones),
        "focus": torch.as_tensor(focus, device=device),
        "focus_idx": torch.as_tensor(np.nonzero(focus)[0], device=device),
        "bug_idx": torch.as_tensor(np.nonzero(is_bug)[0], device=device),
        "pre_idx": torch.as_tensor(np.nonzero(focus & ~is_bug)[0], device=device),
        "labels": labels,
        "n": len(rewards),
        "n_bug": int(is_bug.sum()),
        "n_focus": int(focus.sum()),
    }


def unlearn(cfg: UnlearnConfig) -> Path:
    if cfg.nstep < 1:
        raise ValueError("--nstep 은 1 이상이어야 합니다.")
    if cfg.label_mode not in ("direct", "expand"):
        raise ValueError("--label_mode 는 direct 또는 expand 여야 합니다.")
    if cfg.n_labels > 0 and (cfg.bug_frac > 0 or not cfg.bc_on_bug):
        raise ValueError("--n_labels 를 쓸 때는 --bug_frac 과 --bc_on_bug false 를 같이 쓸 수 없습니다 (정답 표시 전체를 쓰는 기능이라서).")
    if cfg.scratch:  # 처음부터 학습하는 비교용 실험
        base = "scratch_clean" if cfg.drop_bug_trajs else f"scratch_pen{cfg.penalty:g}_term{int(cfg.terminal)}"
    else:
        base = f"pen{cfg.penalty:g}_frac{cfg.bug_frac:g}_bc{int(cfg.bc_on_bug)}_term{int(cfg.terminal)}"
        base += "_clean" if cfg.drop_bug_trajs else ""
    if cfg.n_labels > 0:
        base += f"_k{cfg.n_labels}" + ("e" if cfg.label_mode == "expand" else "")
    base += f"_drop{cfg.drop_bug_after}" if cfg.drop_bug_after > 0 and not cfg.drop_bug_trajs else ""
    base += f"_ns{cfg.nstep}" if cfg.nstep > 1 else ""
    base += f"u{cfg.nstep_until}" if cfg.nstep > 1 and cfg.nstep_until > 0 else ""
    name = cfg.run_name or f"{base}_n{cfg.n_steps}_s{cfg.seed}"
    run_dir = Path(cfg.run_dir) / name
    if (run_dir / "config.json").exists():
        raise FileExistsError(f"{run_dir} 에 이미 결과가 있습니다. --run_name 을 바꾸거나 폴더를 지우세요.")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.seed)

    # 1) bug model 불러오기 (--scratch true 면 설정만 빌리고 가중치는 새로 시작)
    ck = torch.load(cfg.ckpt, map_location=device, weights_only=False)
    rcfg = ReBRACConfig.from_dict(ck["config"])  # 원래 학습에 쓴 ReBRAC 설정 그대로 사용
    obs_mean, obs_std = np.asarray(ck["obs_mean"], np.float32), np.asarray(ck["obs_std"], np.float32)
    obs_dim = len(obs_mean)
    act_dim = list(ck["actor"].values())[-1].shape[0]
    actor, critic = Actor(obs_dim, act_dim, rcfg).to(device), Critic(obs_dim, act_dim, rcfg).to(device)
    if not cfg.scratch:
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
    actor_t, critic_t = copy.deepcopy(actor), copy.deepcopy(critic)
    if not cfg.scratch and "actor_target" in ck:
        actor_t.load_state_dict(ck["actor_target"])
        critic_t.load_state_dict(ck["critic_target"])
    for p in [*actor_t.parameters(), *critic_t.parameters()]:
        p.requires_grad_(False)
    fused = device.type == "cuda"
    actor_opt = torch.optim.Adam(actor.parameters(), lr=rcfg.actor_lr, fused=fused)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=rcfg.critic_lr, fused=fused)
    if not cfg.scratch and not cfg.reset_optim and "actor_opt" in ck:
        actor_opt.load_state_dict(ck["actor_opt"])
        critic_opt.load_state_dict(ck["critic_opt"])

    # 2) 데이터 준비
    data = load_data(cfg, device, obs_mean, obs_std, rcfg.gamma)
    n, focus_idx, pool_idx = data["n"], data["focus_idx"], data.get("pool_idx")
    n_focus_batch = int(round(rcfg.batch_size * cfg.bug_frac)) if len(focus_idx) > 0 else 0
    if pool_idx is not None:
        n_focus_batch = 0  # 버그 궤적을 뺐으므로 "버그 근처 걸음"은 뽑지 않는다
    print(
        f"[{name}] 전체 {n:,}걸음, 버그 걸음 {data['n_bug']:,}, 버그 근처 걸음 {data['n_focus']:,}, "
        f"배치 {rcfg.batch_size} 중 버그 근처 {n_focus_batch}, device {device}"
    )
    lab = data["labels"]
    if cfg.n_labels > 0:
        print(
            f"[{name}] 버그 표시: {lab['n_bug']:,}개 중 {lab['n_labeled']:,}개만 사용"
            f"{' + 범위로 넓힘' if cfg.label_mode == 'expand' else ''} -> 벌점 준 걸음 {lab['n_flagged']:,}개 "
            f"(그중 진짜 버그 {lab['precision']:.0%}, 전체 버그의 {lab['recall']:.0%}를 찾음)"
        )
        if cfg.label_mode == "expand":
            dims = lab.get("expand_dims", [])
            print(f"[{name}] 공통 특징으로 고른 번호: {dims if dims else '없음 (표시된 걸음에만 벌점)'}  (0, 1 = 위치 x, y)")
    print(
        f"[{name}] 시작: {'처음부터 (가중치 새로)' if cfg.scratch else 'bug model 이어서'}, "
        f"학습에 쓰는 걸음 {len(pool_idx) if pool_idx is not None else n:,}, nstep {cfg.nstep}"
        + (f" (처음 {cfg.nstep_until:,}걸음까지만)" if cfg.nstep > 1 and cfg.nstep_until > 0 else "")
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {**asdict(cfg), "rebrac": asdict(rcfg), "start_step": None if cfg.scratch else ck.get("step"),
             "n_transitions": n, "n_train_transitions": len(pool_idx) if pool_idx is not None else n, "labels": lab,
             "n_bug": data["n_bug"], "n_focus": data["n_focus"]},
            indent=2,
        )
    )  # fmt: skip

    clean_only = [False]  # drop_bug_after 를 넘기면 True 로 바뀐다

    def sample():
        if pool_idx is not None:
            return pool_idx[torch.randint(0, len(pool_idx), (rcfg.batch_size,), device=device, generator=gen)]
        if clean_only[0]:
            pool = data["clean_idx"]
            return pool[torch.randint(0, len(pool), (rcfg.batch_size,), device=device, generator=gen)]
        idx = torch.randint(0, n, (rcfg.batch_size - n_focus_batch,), device=device, generator=gen)
        if n_focus_batch > 0:
            pick = torch.randint(0, len(focus_idx), (n_focus_batch,), device=device, generator=gen)
            idx = torch.cat([idx, focus_idx[pick]])
        return idx

    def probe(idx_all: torch.Tensor) -> torch.Tensor:
        return idx_all[torch.randint(0, len(idx_all), (min(2048, len(idx_all)),), device=device, generator=gen)]

    bug_probe = probe(data["bug_idx"]) if len(data["bug_idx"]) else None  # 버그 걸음의 Q값 관찰용
    pre_probe = probe(data["pre_idx"]) if len(data["pre_idx"]) else None  # 벽으로 걸어가는 걸음의 Q값 관찰용

    @torch.no_grad()
    def q_at(idx) -> float:
        if idx is None:
            return float("nan")
        return torch.min(*critic(data["obs"][idx], data["actions"][idx])).mean().item()

    # 3) 평가 준비: 포털이 열린 환경과 벽을 막은 환경
    do_eval = cfg.eval_episodes > 0
    if do_eval:
        ev_open = MazeEvaluator(cfg.eval_episodes, cfg.eval_seed_base, buggy=True)
        ev_patched = MazeEvaluator(cfg.eval_episodes, cfg.eval_seed_base, buggy=False)
    train_log, eval_log = CsvLogger(run_dir / "train_log.csv"), CsvLogger(run_dir / "eval_log.csv")

    @torch.no_grad()
    def policy(obs: np.ndarray) -> np.ndarray:
        return actor(torch.as_tensor((obs - obs_mean) / obs_std, device=device)).cpu().numpy()

    def evaluate_and_save(step: int):
        row = {"step": step}
        if do_eval:
            t0 = time.time()
            m_open, m_pat = ev_open(policy), ev_patched(policy)
            row.update(
                open_success=m_open["success"], bug_usage=m_open["bug_usage"], success_no_bug=m_open["success_no_bug"],
                open_return=m_open["return_mean"], patched_success=m_pat["success"], patched_return=m_pat["return_mean"],
                eval_time=time.time() - t0,
            )  # fmt: skip
            eval_log.log(row)
            print(f"  eval step={step:>6d} | 포털 열림: {format_metrics(m_open)}", flush=True)
            print(
                f"  {'':>16s} | 벽 막음  : success={m_pat['success']:.2f} return={m_pat['return_mean']:7.1f}"
                f"  ({row['eval_time']:.0f}s)",
                flush=True,
            )
        out = {
            "actor": actor.state_dict(), "critic": critic.state_dict(), "obs_mean": obs_mean, "obs_std": obs_std,
            "config": asdict(rcfg), "unlearn_config": asdict(cfg), "step": step, "eval": row,
        }  # fmt: skip
        if step == cfg.n_steps:  # 마지막 체크포인트에는 학습 상태 전체를 저장
            out.update(
                actor_target=actor_t.state_dict(), critic_target=critic_t.state_dict(),
                actor_opt=actor_opt.state_dict(), critic_opt=critic_opt.state_dict(),
            )  # fmt: skip
        path = run_dir / f"actor_{step:07d}.pt"
        try:
            torch.save(out, path)
        except Exception as e:
            # 저장이 실패해도(디스크 문제 등) 학습 전체가 멈추지 않게 한다. 평가 기록(csv)은 이미 남아 있다.
            print(f"  [경고] 체크포인트 저장 실패 ({path.name}): {e}", flush=True)
            print("         체크포인트 없이 학습을 계속합니다.", flush=True)
            try:
                path.unlink(missing_ok=True)  # 쓰다 만 파일은 지운다
            except OSError:
                pass
        return row

    # 4) 추가 학습 (bugmaze/rebrac.py 의 학습 루프와 같은 식)
    last = evaluate_and_save(0)
    stats = {k: [] for k in ("critic_loss", "actor_loss", "q_mean", "actor_bc")}
    t_start = time.time()
    for step in range(1, cfg.n_steps + 1):
        idx = sample()
        s, a = data["obs"][idx], data["actions"][idx]
        if cfg.nstep > 1 and (cfg.nstep_until <= 0 or step <= cfg.nstep_until):
            # 기록을 n걸음 따라가며 미리 더해 둔 보상 + 그 끝에서 이어 붙일 걸음
            r, disc, nxt = data["ret_n"][idx], data["disc_n"][idx], data["last_n"][idx]
        else:
            r, disc, nxt = data["rewards"][idx], rcfg.gamma, idx
        s2, d, a2 = data["next_obs"][nxt], data["dones"][nxt], data["next_actions"][nxt]

        # critic: "여기서 이렇게 움직이면 앞으로 총 몇 점"의 예측을 벌점이 반영된 보상으로 다시 맞춤
        with torch.no_grad():
            noise = (torch.randn_like(a) * rcfg.policy_noise).clamp(-rcfg.noise_clip, rcfg.noise_clip)
            a2_pi = (actor_t(s2) + noise).clamp(-1.0, 1.0)
            q1_t, q2_t = critic_t(s2, a2_pi)
            bc_c = ((a2_pi - a2) ** 2).sum(-1)
            target = r + disc * (1.0 - d) * (torch.min(q1_t, q2_t) - rcfg.critic_bc_coef * bc_c)
        q1, q2 = critic(s, a)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_opt.step()
        stats["critic_loss"].append(critic_loss.detach())

        # actor: 고쳐진 예측에서 점수가 높은 쪽으로 행동을 옮김 (데이터 행동에서 너무 벗어나지 않게)
        if step % rcfg.policy_freq == 0:
            pi = actor(s)
            q = torch.min(*critic(s, pi))
            lmbda = 1.0 / q.abs().mean().detach() if rcfg.normalize_q else 1.0
            bc_a = ((pi - a) ** 2).sum(-1)
            bc_w = 1.0 if cfg.bc_on_bug else (~data["focus"][idx]).float()  # 버그 근처 걸음은 따라 하지 않기
            actor_loss = (rcfg.actor_bc_coef * bc_w * bc_a - lmbda * q).mean()
            actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_opt.step()
            with torch.no_grad():
                for p, pt in zip(actor.parameters(), actor_t.parameters()):
                    pt.lerp_(p, rcfg.tau)
                for p, pt in zip(critic.parameters(), critic_t.parameters()):
                    pt.lerp_(p, rcfg.tau)
            stats["actor_loss"].append(actor_loss.detach())
            stats["q_mean"].append(q.mean().detach())
            stats["actor_bc"].append(bc_a.mean().detach())

        if step % cfg.log_every == 0:
            row = {"step": step, "time": time.time() - t_start}
            row.update({k: torch.stack(v).mean().item() if v else float("nan") for k, v in stats.items()})
            row.update(q_bug=q_at(bug_probe), q_pre=q_at(pre_probe))
            train_log.log(row)
            print(
                f"[{name}] step {step:>6d} ({row['time']:4.0f}s) critic {row['critic_loss']:8.4f} Q {row['q_mean']:7.2f} "
                f"| 버그 걸음 Q {row['q_bug']:8.2f} | 벽으로 가는 걸음 Q {row['q_pre']:8.2f}",
                flush=True,
            )
            stats = {k: [] for k in stats}
        if cfg.drop_bug_after > 0 and step == cfg.drop_bug_after and pool_idx is None:
            clean_only[0] = True
            print(f"[{name}] step {step:>6d}: 여기부터 벌점 받은 궤적을 빼고 학습합니다 (남는 걸음 {len(data['clean_idx']):,}개)", flush=True)
        if cfg.nstep > 1 and step == cfg.nstep_until:
            print(f"[{name}] step {step:>6d}: 여기부터 원래(한 걸음) 방식으로 바꿉니다", flush=True)
        if step % cfg.eval_every == 0 or step == cfg.n_steps:
            last = evaluate_and_save(step)

    if do_eval:
        ev_open.close()
        ev_patched.close()
        (run_dir / "final_eval.json").write_text(json.dumps(last, indent=2))
    print(f"[{name}] 끝. {(time.time() - t_start) / 60:.1f}분 -> {run_dir}")
    return run_dir


if __name__ == "__main__":
    unlearn(parse_args())
