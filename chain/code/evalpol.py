"""정책 평가 공용 함수: 체크포인트 ck 를 형상 E·레퍼런스 폴더 ref_dir 의 모션 M 에 적용 (외란·잡음 없이 모션 처음부터 끝까지).
check/short_leg/eval_slide.py 와 같은 계산. 지표: 생존, 관절 RMSE, 몸통 위치 RMSE, 발 위치 오차, 이동 비율, 발 미끄러짐.
multi=True 면 관측 끝에 형상 토큰 E (13) 를 붙인다 (공유 정책)."""
import functools
from pathlib import Path
import numpy as np
from common import CODE  # noqa: sys.path 설정
import jax, jax.numpy as jnp
from omegaconf import OmegaConf
from pd_replay import build_model
from policy_io import load_run
from track_env import TrackMjxEnv
CONTACT_DIST = 1e-3
_ENV, _POL = {}, {}


def evaluate(ck, E, ref_dir, M, multi=False, keep_qpos=False, max_seconds=None, clips=None):
    """clips: 같은 형상에서 여러 모션을 평가할 때 환경을 한 번만 만들도록 전체 목록을 넘긴다 (M 은 그중 하나)."""
    clips = list(clips) if clips else [M]
    ck = str(ck)
    if ck not in _POL:
        if len(_POL) >= 4: _POL.clear()
        _POL[ck] = load_run(ck)
    cfg, policy = _POL[ck]
    key = (tuple(np.round(E, 6)), str(ref_dir), tuple(clips))
    if key not in _ENV:  # 환경은 하나만 유지 (메모리: 환경·jit 이 쌓이면 CPU LLVM 컴파일 메모리 부족)
        _ENV.clear(); jax.clear_caches(); import gc; gc.collect()
        ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop('_target_')
        ecfg.update(reference_dir=str(ref_dir), clips=clips, reset_qpos=0., reset_qvel=0., velocity_kick=[0., 0.])
        ecfg['obs_noise'] = {k: 0. for k in ecfg['obs_noise']}
        env = TrackMjxEnv(mj_model_fn=functools.partial(build_model, '00000011', kp=cfg.get('kp'), tau_scale=cfg.get('tau_scale'),
                                                        torque_limit=cfg.get('torque_limit') or 33.5, uniforms=list(E)), **ecfg)
        _ENV[key] = (env, jax.jit(env.reset_with_traj), jax.jit(env.step))
    env, reset, step = _ENV[key]
    e = jnp.asarray(E, jnp.float32)
    pol = jax.jit((lambda o, k: policy(jnp.concatenate((o, e)), k)) if multi else policy)
    feet = np.array(env._foot_ids); dt = float(env.dt); ci = clips.index(M); start, end = int(env.clip_start[ci]), int(env.clip_end[ci])
    if max_seconds: end = min(end, start + int(max_seconds / dt) + 1)
    rng = jax.random.PRNGKey(0); s = reset(rng, jnp.int32(ci), jnp.int32(start))
    jerr, perr, ferr, slip, qs = [], [], [], [], [np.asarray(s.data.qpos)]
    prev = np.asarray(s.data.geom_xpos[feet]); fell = False; i = start; q = qs[0]
    for _ in range(end - start - 1):
        rng, k = jax.random.split(rng); s = step(s, pol(s.obs, k)[0])
        i = int(s.info['frame']); q = np.asarray(s.data.qpos); rq = np.asarray(env.ref_qpos[i])
        jerr.append(np.sqrt(np.mean((q[7:19] - rq[7:19]) ** 2))); perr.append(np.linalg.norm(q[:3] - rq[:3]))
        rot = np.asarray(s.data.xmat[env._base_idx]); foot = np.asarray(s.data.geom_xpos[feet])
        ferr.append(np.linalg.norm((foot - np.asarray(s.data.xpos[env._base_idx])) @ rot - np.asarray(env.ref_foot[i]), axis=1).mean())
        geom, dist = np.asarray(s.data._impl.contact.geom), np.asarray(s.data._impl.contact.dist)
        sc = np.array([((geom == f).any(1) & (dist < CONTACT_DIST)).any() for f in feet])
        v = np.linalg.norm((foot[:, :2] - prev[:, :2]) / dt, axis=1); prev = foot
        if sc.any(): slip.append(float(v[sc].mean()))
        if keep_qpos: qs.append(q)
        if float(s.info['done/fellover']) > 0: fell = True; break
        if i >= end - 1: break
    r0, r1 = np.asarray(env.ref_qpos[start, :2]), np.asarray(env.ref_qpos[i, :2]); d = r1 - r0
    out = dict(seconds=round(len(jerr) * dt, 2), clip_seconds=round((end - start - 1) * dt, 2), survived=not fell,
               joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(np.square(jerr))))) if jerr else None,
               root_rmse_cm=float(np.sqrt(np.mean(np.square(perr))) * 100) if perr else None,
               foot_err_cm=float(np.mean(ferr) * 100) if ferr else None,
               travel_ratio=float(np.dot(q[:2] - r0, d) / max(np.dot(d, d), 1e-6)), ref_travel_m=float(np.linalg.norm(d)),
               slip_m_s=float(np.mean(slip)) if slip else None)
    if keep_qpos: out['qpos'] = np.array(qs)
    return out


def passes(r, joint_deg=10., root_cm=10.):
    """통과 기준 (2026-11-12 사용자 결정): 끝까지 생존, 관절 RMSE ≤ 10°, 몸통 위치 RMSE ≤ 10 cm."""
    return bool(r['survived'] and r['joint_rmse_deg'] is not None and r['joint_rmse_deg'] <= joint_deg and r['root_rmse_cm'] <= root_cm)
