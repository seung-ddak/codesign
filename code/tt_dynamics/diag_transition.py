"""진단 D3: 생성 궤적의 TT 전이 오류. 32 step 구간마다 MJX 를 생성된 상태로 다시 맞추고 생성된 행동을 넣어,
MJX 가 만든 상태와 TT 가 생성한 상태를 비교한다 (open-loop 누적 제거).

구간 = 생성과 같은 32 step (31 step 전진), 구간 시작 state 는 생성 S 를 qpos·qvel 로 변환 (위치·yaw 는 원점),
가장 낮은 발이 바닥에 닿게 높이만 맞춘다 (generate_task.execute 와 같음). 오차는 구간 마지막 step (31) 에서 잰다.
대조군 = 데이터의 실제 상태·행동으로 같은 절차 → 오차가 0 에 가까워야 절차를 믿을 수 있다.
비교 기준 = 검증 데이터에서 잰 TT 32 step 오차 (tt/<ckpt>/diag_window.json 의 전체).
판정 (사전): 생성 궤적 오차 / 기준 ≤ 1.5 → TT 평소 오차. ≥ 2 이고 scale 과 함께 커지면 → guidance 가 TT 오류를 이용.

usage (codesign/ 에서): CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false ../transformer-transformer/.venv/bin/python \
    code/tt_dynamics/diag_transition.py --out runs/pipeline_v1/diag/D3_transition
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

CODESIGN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODESIGN.parent / "transformer-transformer"))
sys.path.insert(0, str(CODESIGN / "code" / "motion_tracking"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mujoco import mjx  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from pd_replay import build_model  # noqa: E402
from rollout_data import S_DIM, features  # noqa: E402
from t2.mjx_env.base_env import make_data, step  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402
from tt_model import rebase  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--out", type=Path, required=True)
parser.add_argument("--runs", nargs="+", default=["track/try1", "zigzag/try1", "track/try2", "zigzag/try2"])
parser.add_argument("--control-episodes", type=int, default=8, help="대조군: 형상당 데이터 episode 수")
args = parser.parse_args()
R = CODESIGN / "runs/pipeline_v1"
H, FOOT_R, TURN_DEG = 32, 0.02, 10.0
CKPT = {"try1": "h32/try4", "try2": "h32/try5"}

expert = R / "experts/walkForwardSlow__r00"
cfg = pickle.load(open(sorted(expert.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
ecfg.update(clips=["walkForwardSlow__r00"], reference_dir=str(R / "retargeting/e000" / choice))
jit = {}


def s_to_q(s):
    """S [42] -> qpos [19], qvel [18] (xy 0, yaw 0). generate_task.s0_to_q 와 같다."""
    g = s[5:8] / np.linalg.norm(s[5:8])
    th, ph = np.arcsin(np.clip(g[0], -1, 1)), np.arctan2(-g[1], -g[2])
    ch, sh, cp, sp = np.cos(th / 2), np.sin(th / 2), np.cos(ph / 2), np.sin(ph / 2)
    return np.concatenate([[0, 0, s[2]], [ch * cp, ch * sp, sh * cp, -sh * sp], s[14:26]]), np.concatenate([s[8:14], s[26:38]])


def windows_mjx(E, S_list, A_list):
    """형상 E 에서 구간마다 시작 state S [W,42] 로 맞추고 행동 A [W,H-1,12] 를 넣는다 -> MJX 구간 특징 [W,H,42], 낙상 [W], 높이 보정 [W]."""
    env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                     torque_limit=cfg.get("torque_limit"), uniforms=E), **ecfg)
    feet = np.asarray(env._foot_ids)
    d = mujoco.MjData(env.mj_model)
    Q0, V0, dz = [], [], []
    for s in S_list:
        q, v = s_to_q(s)
        d.qpos[:] = q
        mujoco.mj_kinematics(env.mj_model, d)
        corr = d.geom_xpos[feet, 2].min() - FOOT_R
        q[2] -= corr
        Q0.append(q), V0.append(v), dz.append(corr)
    data0 = make_data(env.mj_model, qpos=jnp.asarray(Q0[0]), qvel=jnp.asarray(V0[0]), impl=env.impl,
                      nconmax=env._nconmax, njmax=env._njmax)
    if "step" not in jit:
        off = jnp.asarray(env.action_offset)
        jit["init"] = jax.jit(lambda m, d0, q, v: jax.vmap(lambda qq, vv: mjx.forward(m, d0.replace(qpos=qq, qvel=vv)))(q, v))
        jit["step"] = jax.jit(lambda m, dd, tq: jax.vmap(lambda x, a: step(m, x, off.at[:12].set(a), env.n_substeps))(dd, tq))
    m = env.mjx_model
    dat = jit["init"](m, data0, jnp.asarray(np.array(Q0), jnp.float32), jnp.asarray(np.array(V0), jnp.float32))
    qs, vs, cs = [np.asarray(dat.qpos)], [np.asarray(dat.qvel)], []
    A = jnp.asarray(np.array(A_list), jnp.float32)
    for t in range(H - 1):
        dat = jit["step"](m, dat, A[:, t])
        qs.append(np.asarray(dat.qpos)), vs.append(np.asarray(dat.qvel))
    Q, V = np.stack(qs, 1), np.stack(vs, 1)
    up = 1 - 2 * (Q[..., 4] ** 2 + Q[..., 5] ** 2)
    fell = (up < env.termination_fellover_threshold).any(1)
    X = np.stack([features(Q[w], V[w], np.zeros((H, 4))) for w in range(len(Q))])
    return X, fell, np.array(dz)


def errs(P, M):
    """P (TT 생성, 구간 기준) [W,H,42], M (MJX) -> 구간 마지막 step 오차 [W] 묶음."""
    yaw = lambda x: np.arctan2(x[..., 4], x[..., 3])  # noqa: E731
    k = H - 1
    return dict(yaw_deg=np.degrees(np.abs(np.angle(np.exp(1j * (yaw(P[:, k]) - yaw(M[:, k])))))),
                pos_cm=np.linalg.norm(P[:, k, :2] - M[:, k, :2], axis=-1) * 100,
                joint_deg=np.degrees(np.sqrt(((P[:, 1:, 14:26] - M[:, 1:, 14:26]) ** 2).mean((1, 2)))),
                vel=np.linalg.norm(P[:, 1:, 8:10] - M[:, 1:, 8:10], axis=-1).mean(1),
                height_cm=np.abs(P[:, 1:, 2] - M[:, 1:, 2]).mean(1) * 100,
                tt_dyaw=np.degrees(np.abs(yaw(P[:, k]))), mjx_dyaw=np.degrees(np.abs(yaw(M[:, k]))))


def summarize(e, fell, mask=None):
    mask = np.ones(len(fell), bool) if mask is None else mask
    ok = mask & ~fell
    out = {k: round(float(np.mean(v[ok])), 2) if ok.any() else None for k, v in e.items()}
    out.update(windows=int(mask.sum()), fell_in_window=int((mask & fell).sum()))
    return out


result = dict(baseline={}, control={}, runs={})
for tag, ck in CKPT.items():
    b = json.load(open(R / f"tt/{ck}/diag_window.json"))
    result["baseline"][tag] = dict(ckpt=ck, **{k: round(b[c][k], 2) for c, k in
                                                (("전체", "yaw_err_deg"), ("전체", "pos_err_cm"), ("전체", "joint_err_deg"), ("전체", "vel_err"))},
                                   rotate_yaw=round(b["회전"]["yaw_err_deg"], 2))

# 대조군: 데이터 실제 상태·행동
for d in ("e000", "e001", "e002"):
    z = np.load(R / f"rollouts/{d}/seq_seed0.npz")
    eps = np.where(~z["fell"] & z["valid"].all(1))[0][:args.control_episodes]
    S_list, A_list, P = [], [], []
    for b in eps:
        S = features(z["qpos"][b], z["qvel"][b], z["contact"][b])
        for t0 in range(0, 500 - H, H - 1):
            S_list.append(S[t0]), A_list.append(z["target_q"][b, t0:t0 + H - 1])
            P.append(rebase(torch.tensor(S[None, t0:t0 + H]))[0].numpy())
    X, fell, dz = windows_mjx(z["E"], S_list, A_list)
    result["control"][d] = summarize(errs(np.array(P), X), fell)
    result["control"][d]["height_fix_cm"] = round(float(np.abs(dz).mean() * 100), 2)
    print("control", d, result["control"][d], flush=True)

for run in args.runs:
    result["runs"][run] = {}
    for sc in ("scale_0", "scale_1", "scale_3", "scale_10"):
        z = np.load(R / f"task/{run}/{sc}.npz")
        allE, allF, allT, dzs = {k: [] for k in ("yaw_deg", "pos_cm", "joint_deg", "vel", "height_cm", "tt_dyaw", "mjx_dyaw")}, [], [], []
        for i in range(len(z["pred"])):
            pred = z["pred"][i]
            S_list, A_list, P = [], [], []
            for t0 in range(0, 500 - H, H - 1):
                S_list.append(pred[t0, :S_DIM]), A_list.append(pred[t0:t0 + H - 1, S_DIM:])
                P.append(rebase(torch.tensor(pred[None, t0:t0 + H, :S_DIM]))[0].numpy())
            X, fell, dz = windows_mjx(z["E"][i], S_list, A_list)
            e = errs(np.array(P), X)
            for k in allE:
                allE[k].append(e[k])
            allF.append(fell), dzs.append(dz)
        e = {k: np.concatenate(v) for k, v in allE.items()}
        fell = np.concatenate(allF)
        turn = e["tt_dyaw"] > TURN_DEG
        result["runs"][run][sc] = dict(all=summarize(e, fell), turn=summarize(e, fell, turn), straight=summarize(e, fell, ~turn),
                                       height_fix_cm=round(float(np.abs(np.concatenate(dzs)).mean() * 100), 2))
        print(run, sc, result["runs"][run][sc]["all"], "fix", result["runs"][run][sc]["height_fix_cm"], flush=True)

# 판정: 생성 궤적 오차 / 검증 데이터 오차 (yaw·위치)
ratio = {}
for run, r in result["runs"].items():
    b = result["baseline"]["try1" if run.endswith("try1") else "try2"]
    ratio[run] = {sc: dict(yaw=round(v["all"]["yaw_deg"] / b["yaw_err_deg"], 2), pos=round(v["all"]["pos_cm"] / b["pos_err_cm"], 2))
                  for sc, v in r.items() if v["all"]["yaw_deg"] is not None}
result["ratio_vs_val"] = ratio
args.out.mkdir(parents=True, exist_ok=True)
json.dump(result, open(args.out / "d3_transition.json", "w"), indent=1, ensure_ascii=False)
print("ratio", json.dumps(ratio))
