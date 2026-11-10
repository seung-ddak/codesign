"""형상 E0~E4 × 모션 5 개 정책 평가 (legacy eval_motions.py 와 같은 계산 + 발 미끄러짐).

각 정책: 모션 0 프레임부터 끝까지 결정적 정책으로 추종 (외란·잡음 없는 eval_env).
지표: 생존, 관절 RMSE, 발 위치 오차 (몸통 기준), 자세 오차, 몸통 위치 RMSE (월드), 접지 일치,
발 미끄러짐 = 접지한 발의 월드 수평 속도 (발 중심, 레퍼런스도 같은 방식으로 계산).
출력: results/<E>/<motion>.npz (qpos 기록, A1 qpos), results/summary.json
"""
import glob, json, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT.parent / 'transformer-transformer')); sys.path.insert(0, str(HERE))
import jax, jax.numpy as jnp, mujoco
from omegaconf import OmegaConf
from pd_replay import build_model
from policy_io import load_run
from track_env import TrackMjxEnv
from shapes import SHAPES, MOTIONS

CONTACT_DIST, FOOT_R = 1e-3, 0.02


def latest_ckpt(run):
    c = sorted(glob.glob(str(run / 'wandb/offline-run-*/files/checkpoints/*')), key=lambda p: int(os.path.basename(p)))
    return c[-1] if c else None


def evaluate(K, M):
    run = HERE / 'runs' / K / M; ck = latest_ckpt(run)
    if ck is None or not (run / 'done').exists():
        return None
    cfg, policy = load_run(ck)
    ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True); ecfg.pop('_target_')
    E = SHAPES[K]['E']
    env = TrackMjxEnv(mj_model_fn=lambda: build_model('00000011', kp=cfg.get('kp'), tau_scale=cfg.get('tau_scale'),
                                                      torque_limit=cfg.get('torque_limit'), uniforms=E), **ecfg)
    reset, step, pol = jax.jit(env.reset_with_traj), jax.jit(env.step), jax.jit(policy)
    feet = np.array(env._foot_ids); dt = float(env.dt)
    start, end = int(env.clip_start[0]), int(env.clip_end[0])
    ref_file = np.load(Path(ecfg['reference_dir']) / f'{M}.npz')
    rng = jax.random.PRNGKey(0); s = reset(rng, jnp.int32(0), jnp.int32(start))
    qpos = [np.asarray(s.data.qpos)]; frames = [start]; jerr, ferr, oerr, perr, slip, sc_all, rc_all = [], [], [], [], [], [], []
    prev_foot = np.asarray(s.data.geom_xpos[feet])
    fell = False
    for _ in range(end - start - 1):
        rng, key = jax.random.split(rng)
        s = step(s, pol(s.obs, key)[0])
        i = int(s.info['frame']); q = np.asarray(s.data.qpos); ref_q = np.asarray(env.ref_qpos[i])
        jerr.append(np.sqrt(np.mean((q[7:19] - ref_q[7:19]) ** 2)))
        rot = np.asarray(s.data.xmat[env._base_idx])
        foot = np.asarray(s.data.geom_xpos[feet])
        ferr.append(np.linalg.norm((foot - np.asarray(s.data.xpos[env._base_idx])) @ rot - np.asarray(env.ref_foot[i]), axis=1).mean())
        oerr.append(float(s.info['orn_err'])); perr.append(np.linalg.norm(q[:3] - ref_q[:3]))
        geom, dist = np.asarray(s.data.contact.geom), np.asarray(s.data.contact.dist)
        sc = np.array([((geom == f).any(1) & (dist < CONTACT_DIST)).any() for f in feet])
        v = np.linalg.norm((foot[:, :2] - prev_foot[:, :2]) / dt, axis=1); prev_foot = foot
        if sc.any(): slip.append(float(v[sc].mean()))
        sc_all.append(sc); rc_all.append(np.asarray(env.ref_contact[i]).astype(bool))
        qpos.append(q); frames.append(i)
        if float(s.info['done/fellover']) > 0:
            fell = True; break
        if i >= end - 1:
            break
    # 레퍼런스 자체의 미끄러짐 (같은 방식: 레퍼런스에서 접지로 표시된 발의 수평 속도)
    rf = ref_file['fix_foot_pos']; rcon = ref_file['fix_contact'].astype(bool)
    rv = np.linalg.norm(np.diff(rf[:, :, :2], axis=0), axis=-1) / dt
    ref_slip = float(np.mean([rv[t][rcon[t + 1]].mean() for t in range(len(rv)) if rcon[t + 1].any()]))
    sc_all, rc_all = np.array(sc_all), np.array(rc_all)
    out = HERE / 'results' / K; out.mkdir(parents=True, exist_ok=True)
    a1 = ref_file['a1_qpos'] if 'a1_qpos' in ref_file.files else None
    np.savez(out / f'{M}.npz', qpos=np.array(qpos), frames=np.array(frames) - start, a1_qpos=a1 if a1 is not None else np.zeros(0),
             ref_qpos=np.asarray(env.ref_qpos[start:end]), dt=dt, E=np.array(E))
    return dict(shape=K, motion=M, seconds=round(len(jerr) * dt, 2), clip_seconds=round((end - start - 1) * dt, 2), survived=not fell,
                joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(np.square(jerr))))), foot_err_cm=float(np.mean(ferr) * 100),
                orn_err_deg=float(np.degrees(np.mean(oerr))), root_rmse_cm=float(np.sqrt(np.mean(np.square(perr))) * 100),
                contact_agree=float((sc_all == rc_all).mean()),
                travel_ratio=float(np.dot(q[:2] - np.asarray(env.ref_qpos[start, :2]), (np.asarray(env.ref_qpos[i, :2]) - np.asarray(env.ref_qpos[start, :2])))
                                   / max(np.linalg.norm(np.asarray(env.ref_qpos[i, :2]) - np.asarray(env.ref_qpos[start, :2])) ** 2, 1e-6)),
                ref_travel_m=float(np.linalg.norm(np.asarray(env.ref_qpos[i, :2]) - np.asarray(env.ref_qpos[start, :2]))), slip_m_s=float(np.mean(slip)) if slip else None, ref_slip_m_s=ref_slip,
                ref_infeasible_frac=float(ref_file['inv_bad_frac']), checkpoint=ck)


if __name__ == '__main__':
    rows = [r for K in SHAPES for M in MOTIONS if (r := evaluate(K, M)) is not None]
    for r in rows:
        print(f"{r['shape']} {r['motion']:22s} surv={r['survived']} {r['seconds']}/{r['clip_seconds']}s joint={r['joint_rmse_deg']:.1f}° root={r['root_rmse_cm']:.1f}cm "
              f"slip={r['slip_m_s']:.3f} (ref {r['ref_slip_m_s']:.3f}) contact={r['contact_agree']:.2f}", flush=True)
    json.dump(rows, open(HERE / 'results' / 'summary.json', 'w'), indent=1, ensure_ascii=False)
