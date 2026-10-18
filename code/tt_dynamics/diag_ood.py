"""진단 D2: 생성한 궤적이 TT 학습 데이터 분포 밖으로 나갔는지 (새로운 물리 영역) 잰다.

step 특징 = S 중 위치·yaw·접지를 뺀 33 개 (높이, 중력 3, 선속도 3, 각속도 3, 관절각 12, 관절속도 12) + A 12 + ΔA 12.
데이터 표준편차로 정규화한 뒤, 각 생성 step 에서 학습 데이터의 가장 가까운 step 까지 거리 (NN 거리) 를 잰다.
기준선 = 검증용으로 뗀 데이터 step 의 NN 거리 99 백분위 (데이터 자신도 1 % 는 이 값을 넘는다).
특징 묶음별로도 잰다: S (상태), A (행동), dA (행동 변화량).
판정 (사전): 생성 step 의 기준선 초과 비율 ≥ 3 % 이고 scale 이 커질수록 늘면 → 실패 원인 = 데이터 밖 영역 (새로운 물리 영역).
            1 % 근처면 → TT 가 데이터 범위 안에서도 틀린다 (TT 학습 문제).

usage (codesign/ 에서): CUDA_VISIBLE_DEVICES=0 ../transformer-transformer/.venv/bin/python code/tt_dynamics/diag_ood.py --out runs/pipeline_v1/diag/D2_ood
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from rollout_data import features

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--runs", nargs="+", default=["track_try1", "zigzag_try1", "track_try2", "zigzag_try2"])
parser.add_argument("--ref-steps", type=int, default=200000, help="NN 검색에 쓰는 데이터 step 수 (무작위 표본)")
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()
R = Path("runs/pipeline_v1")
dev = "cuda"
S_IDX = [2, *range(5, 38)]  # 높이, 중력, 선속도, 각속도, 관절각, 관절속도
GROUPS = {"S": slice(0, 34), "A": slice(34, 46), "dA": slice(46, 58), "all": slice(0, 58)}


def step_feats(S, A, m):
    """S [T,42], A [T,12], mask [T] -> 유효 step 특징 [n,58]. ΔA 는 직전 step 대비 (첫 step 0)."""
    dA = np.concatenate([np.zeros_like(A[:1]), np.diff(A, axis=0)])
    return np.concatenate([S[:, S_IDX], A, dA], 1)[m]


# 학습 데이터 (TT 와 같은 rollout)
D = []
for p in sorted(glob.glob(str(R / "rollouts/e00[0-2]/*_seed0.npz"))):
    z = np.load(p)
    for b in range(len(z["valid"])):
        L = int(z["valid"][b].sum())
        if L < 2:
            continue
        S = features(z["qpos"][b, :L], z["qvel"][b, :L], z["contact"][b, :L])
        D.append(step_feats(S, z["target_q"][b, :L], np.ones(L, bool)))
D = np.concatenate(D).astype(np.float32)
rng = np.random.default_rng(args.seed)
perm = rng.permutation(len(D))
held, ref = D[perm[:5000]], D[perm[5000:5000 + args.ref_steps]]
mu, sd = D.mean(0), D.std(0) + 1e-6
ref_t = torch.tensor((ref - mu) / sd, device=dev)


def nn_dist(X, g):
    """X [n,58] -> 묶음 g 기준 가장 가까운 데이터 step 까지 거리 [n] (정규화 공간, 차원 수의 제곱근으로 나눔)."""
    x = torch.tensor((X - mu) / sd, device=dev)[:, GROUPS[g]]
    r = ref_t[:, GROUPS[g]]
    out = [torch.cdist(x[i:i + 2048], r).min(1).values for i in range(0, len(x), 2048)]
    return (torch.cat(out) / (r.shape[1] ** 0.5)).cpu().numpy()


base = {g: nn_dist(held, g) for g in GROUPS}
thr = {g: float(np.percentile(base[g], 99)) for g in GROUPS}
result = dict(data_steps=len(D), ref_steps=len(ref), threshold_p99=thr,
              data_median={g: float(np.median(base[g])) for g in GROUPS}, runs={})
print("data", len(D), "thr", {g: round(v, 3) for g, v in thr.items()}, flush=True)

for run in args.runs:
    sm = json.load(open(R / f"task/{run}/summary.json"))["runs"]
    result["runs"][run] = {}
    for sc in ("scale_0", "scale_1", "scale_3", "scale_10"):
        z = np.load(R / f"task/{run}/{sc}.npz")
        pred = z["pred"]
        cands = []
        for i in range(len(pred)):
            X = step_feats(pred[i, :, :42], pred[i, :, 42:], np.ones(len(pred[i]), bool))
            d = {g: nn_dist(X, g) for g in GROUPS}
            c = sm[sc][i]
            pk, rk = ("pred_laps", "mjx_laps") if run == "track_try1" else ("pred_progress", "mjx_progress")
            cands.append(dict({f"ood_{g}": float((d[g] > thr[g]).mean()) for g in GROUPS},
                              med_all=float(np.median(d["all"])), gap=abs(c[pk] - c[rk]), fell=c["mjx_fell_step"] >= 0,
                              path_err=c["path_err_m"]))
        agg = {k: float(np.mean([c[k] for c in cands])) for k in cands[0] if k != "fell"}
        agg["fall"] = int(sum(c["fell"] for c in cands))
        result["runs"][run][sc] = dict(mean=agg, candidates=cands)
        print(run, sc, {k: round(v, 3) for k, v in agg.items()}, flush=True)

# 후보 단위 상관: OOD 비율 vs 예측-실제 차이·경로 차이 (모든 run·scale)
allc = [c for r in result["runs"].values() for s in r.values() for c in s["candidates"]]
o = np.array([c["ood_all"] for c in allc])
result["corr_ood_vs"] = {k: float(np.corrcoef(o, [c[k] for c in allc])[0, 1]) for k in ("gap", "path_err")}
gen = float(o.mean())
result["verdict"] = dict(generated_ood_all=round(gen, 4), data_ood=0.01,
                         new_physics_region=bool(gen >= 0.03))
args.out.mkdir(parents=True, exist_ok=True)
json.dump(result, open(args.out / "d2_ood.json", "w"), indent=1, ensure_ascii=False)
print("corr", result["corr_ood_vs"], "verdict", result["verdict"])
