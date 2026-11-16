"""발 미끄러짐 원인 검증 (2026-11-12 사용자 요청): 가설 3 (접촉 설정) · 가설 4 (측정에 구름 성분).
정책: check/slide 형상별 정책 (재학습 없음). 같은 정책을 접촉 설정만 바꾼 환경에서 평가.
  A 기준 (학습과 동일): 피라미드 원뿔, impratio 1, 제약 반복 5 / 선탐색 7
  B 타원 원뿔 + impratio 10 (반복 그대로)
  C 제약 반복 50 / 선탐색 20 (원뿔·impratio 그대로)
  D B + C
미끄러짐 두 가지 (접지 = 발-바닥 거리 < 1 mm):
  center  = 발 구 중심의 월드 수평 속도 (지금까지 쓰던 지표, 구르면 0 이 아님)
  contact = 바닥에 닿은 발 재질점의 수평 속도 = v_중심 + ω_종아리 × (0, 0, -r)  (순수 구름이면 0)
출력: results/slip.json, report.html (check/slip_check)"""
import functools, json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT / 'chain/code'))
import jax, jax.numpy as jnp, mujoco
from omegaconf import OmegaConf
from pd_replay import build_model
from policy_io import load_run
from track_env import TrackMjxEnv
from s3_generalization import ckpt
CONFIGS = {'A 기준': dict(cone=0, impratio=1, it=5, ls=7), 'B 타원+impratio10': dict(cone=1, impratio=10, it=5, ls=7),
           'C 반복 50/20': dict(cone=0, impratio=1, it=50, ls=20), 'D 타원+impratio10+반복': dict(cone=1, impratio=10, it=50, ls=20),
           'E 타원+impratio30+반복': dict(cone=1, impratio=30, it=50, ls=20), 'F 타원+impratio100+반복': dict(cone=1, impratio=100, it=50, ls=20)}
from shapes_grid import token
SHAPES = {'E0': token(0, 0), 'E1': token(-20, 0), 'E2': token(20, 0), 'E3': token(0, -20), 'E4': token(0, 20)}
MOTIONS = ['squat__r00', 'walkForwardSlow__r00', 'trotForwardFast__r00', 'trotCCWSlow__r00', 'walkBackwardSlow__r00']
R = 0.02


def run(K, M, c, keep=False):
    out = run_ck(ckpt(K, M), SHAPES[K], ROOT / 'check/slide/references' / K / '00000011', M, c, keep=keep)
    if keep: out[0]['shape'] = K
    else: out['shape'] = K
    return out


def run_ck(ck, E, ref_dir, M, c, multi=False, keep=False):
    """체크포인트 ck 를 형상 E·레퍼런스 ref_dir 의 모션 M 에 접촉 설정 c 로 실행. multi = 관측에 형상 토큰을 붙이는 공유 정책."""
    cfg, policy = load_run(ck); ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop('_target_')
    if multi:
        e = jnp.asarray(E, jnp.float32); base = policy; policy = lambda o, k: base(jnp.concatenate((o, e)), k)
    ecfg.update(reference_dir=str(ref_dir), clips=[M], reset_qpos=0., reset_qvel=0., velocity_kick=[0., 0.],
                impratio=c['impratio'], solver_iterations=c['it'], solver_ls_iterations=c['ls'])
    ecfg['obs_noise'] = {k: 0. for k in ecfg['obs_noise']}

    def model():
        m = build_model('00000011', kp=cfg.get('kp'), tau_scale=cfg.get('tau_scale'), torque_limit=cfg.get('torque_limit') or 33.5, uniforms=list(E))
        m.opt.cone = c['cone']; return m
    env = TrackMjxEnv(mj_model_fn=model, **ecfg)
    reset, step, pol = jax.jit(env.reset_with_traj), jax.jit(env.step), jax.jit(policy)
    feet = np.array(env._foot_ids); fbody = np.array([env.mj_model.geom_bodyid[g] for g in feet]); dt = float(env.dt)
    start, end = int(env.clip_start[0]), int(env.clip_end[0])
    rng = jax.random.PRNGKey(0); s = reset(rng, jnp.int32(0), jnp.int32(start)); prev = np.asarray(s.data.geom_xpos[feet])
    sc_c, sc_p, jerr, perr, qs, fell = [], [], [], [], [np.asarray(s.data.qpos)], False; pxy = []
    for _ in range(end - start - 1):
        rng, k = jax.random.split(rng); s = step(s, pol(s.obs, k)[0]); i = int(s.info['frame'])
        q = np.asarray(s.data.qpos); rq = np.asarray(env.ref_qpos[i]); foot = np.asarray(s.data.geom_xpos[feet])
        jerr.append(np.sqrt(np.mean((q[7:19] - rq[7:19]) ** 2))); perr.append(np.linalg.norm(q[:3] - rq[:3])); pxy.append(np.linalg.norm(q[:2] - rq[:2]))
        geom, dist = np.asarray(s.data._impl.contact.geom), np.asarray(s.data._impl.contact.dist)
        touch = np.array([((geom == f).any(1) & (dist < 1e-3)).any() for f in feet])
        vc = (foot - prev) / dt; prev = foot
        w = np.asarray(s.data.cvel)[fbody, :3]  # 종아리 몸체 각속도 (월드)
        vp = vc + np.cross(w, np.array([0, 0, -R]))
        if touch.any():
            sc_c.append(float(np.linalg.norm(vc[touch, :2], axis=1).mean())); sc_p.append(float(np.linalg.norm(vp[touch, :2], axis=1).mean()))
        if keep: qs.append(q)
        if float(s.info['done/fellover']) > 0: fell = True; break
        if i >= end - 1: break
    out = dict(shape=None, motion=M, config=None, survived=not fell, seconds=round(len(jerr) * dt, 2), slip_center=float(np.mean(sc_c)), slip_contact=float(np.mean(sc_p)),
               joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(np.square(jerr))))), root_rmse_cm=float(np.sqrt(np.mean(np.square(perr))) * 100))
    out.update(root_err_t=[float(x) for x in perr], root_xy_err_t=[float(x) for x in pxy], joint_err_t=[float(np.degrees(x)) for x in jerr], dt=dt)  # 매 step 오차 (그래프용)
    return (out, np.array(qs)) if keep else out


if __name__ == '__main__':
    # 사용: slip_eval.py [출력 이름] [설정 첫 글자들, 예: AD] [형상들, 예: E0,E1]
    tag = sys.argv[1] if len(sys.argv) > 1 else 'slip'; use = sys.argv[2] if len(sys.argv) > 2 else 'ABCD'
    shapes = sys.argv[3].split(',') if len(sys.argv) > 3 else ['E0']
    motions = MOTIONS if len(sys.argv) > 3 else ['trotCCWSlow__r00', 'trotForwardFast__r00', 'walkForwardSlow__r00']
    rows = []
    for K in shapes:
        for M in motions:
            for name, c in [(n, c) for n, c in CONFIGS.items() if n[0] in use]:
                r, qs = run(K, M, c, keep=True); r['config'] = name; rows.append(r)
                print(json.dumps(r, ensure_ascii=False), flush=True)
                if M == 'trotCCWSlow__r00' and name[0] in use:
                    from render_tt import render_tt
                    render_tt(qs, SHAPES[K], HERE / 'videos' / f'{K}_{M}_{name[0]}.mp4', title=f'{M[:-5]} · {name}', sub='same policy, contact setting changed')
                json.dump(rows, open(HERE / f'results/{tag}.json', 'w'), indent=1, ensure_ascii=False)
