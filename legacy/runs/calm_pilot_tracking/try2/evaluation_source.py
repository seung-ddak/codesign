"""Evaluate native 30Hz CALM references with TT position actuators in MuJoCo."""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')
import imageio.v2 as imageio
import mujoco
import numpy as np

from pd_replay import build_model, LEGS


def foot_contacts(model, data, feet, planes):
    contact = np.zeros(4, dtype=bool)
    slip = np.full(4, np.nan)
    for i, foot in enumerate(feet):
        speeds = []
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_GEOM, int(foot), velocity, 0)
        for j, c in enumerate(data.contact):
            if not ((c.geom1 == foot and c.geom2 in planes) or (c.geom2 == foot and c.geom1 in planes)):
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(model, data, j, force)
            if c.dist <= .001 and force[0] > .001:
                v = velocity[3:] + np.cross(velocity[:3], c.pos - data.geom_xpos[foot])
                speeds.append(np.linalg.norm(v[:2]))
        if speeds:
            contact[i], slip[i] = True, np.mean(speeds)
    return contact, slip


def tracking_errors(sim, ref, sim_feet, ref_feet):
    quat = np.clip(np.abs(np.sum(sim[:, 3:7] * ref[:, 3:7], axis=1)), 0, 1)
    return dict(joint_rmse_deg=float(np.degrees(np.sqrt(np.mean((sim[:, 7:] - ref[:, 7:]) ** 2)))),
                body_position_rmse_m=float(np.sqrt(np.mean(np.sum((sim[:, :3] - ref[:, :3]) ** 2, axis=1)))),
                orientation_mean_deg=float(np.degrees(2 * np.arccos(quat)).mean()),
                foot_world_mean_error_m=float(np.linalg.norm(sim_feet - ref_feet, axis=-1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Use trained PPO actions instead of reference PD targets')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    paths = sorted(args.reference.resolve().glob('*.npz'))
    if len(paths) != 5:
        raise ValueError('Expected five references')
    if args.checkpoint:
        from types import SimpleNamespace
        import jax
        import jax.numpy as jnp
        from omegaconf import OmegaConf
        from policy_io import load_run
        from track_env import TrackMjxEnv
        cfg, policy = load_run(args.checkpoint)
        ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
        ecfg.pop('_target_')
        ecfg['reference_dir'] = str(args.reference.resolve())
        env = TrackMjxEnv(mj_model_fn=lambda: build_model('00000011', uniforms=[.5]*13, torque_limit=33.5), **ecfg)
        assert np.isclose(env.dt, 1/30) and not cfg.multi_design

        @jax.jit
        def policy_action(qpos, qvel, xmat, frame, clip_end, prev_action, rng):
            data = SimpleNamespace(qpos=qpos, qvel=qvel, xmat=xmat)
            info = dict(frame=frame, clip_end=clip_end, prev_action=prev_action, rng=rng, ref_off=jnp.zeros(3))
            return jnp.clip(policy(env._get_obs(data, info), rng)[0], env.clip_actions[0], env.clip_actions[1])
    model = build_model('00000011', uniforms=[.5] * 13, torque_limit=33.5)
    model.opt.timestep = 1 / 300
    planes = set(np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE).tolist())
    assert planes and model.nq == 19 and model.nu == 12
    # Disjoint masks retain floor collisions and exclude robot self-collisions.
    model.geom_contype[:] = 2
    model.geom_conaffinity[:] = 1
    for plane in planes:
        model.geom_contype[plane], model.geom_conaffinity[plane] = 1, 2
    feet = np.array([model.geom(l + '_foot').id for l in LEGS])
    model.vis.global_.offwidth, model.vis.global_.offheight = 960, 540
    renderer = mujoco.Renderer(model, height=540, width=960)
    option = mujoco.MjvOption()
    option.geomgroup[3] = 1
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.azimuth, camera.elevation = 125, -22
    rows = []
    conditions = dict(reference_hz=30, physics_hz=300, physics_substeps=10,
                      control='native TT position PD; reference target held for each interval; no PPO',
                      self_collision=False, ground_collision=True, torque_limit_nm=33.5,
                      kp=model.actuator_gainprm[:, 0].tolist(), kd=(-model.actuator_biasprm[:, 2]).tolist(),
                      initialization='reference first pose and finite-difference initial velocity; no resets or height shifts',
                      survival='body up-axis world Z >= 0.5, tested at 30Hz',
                      slip='horizontal velocity at force-bearing foot-plane contact points; not reference contact labels',
                      source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      mujoco_version=mujoco.__version__)
    (out/'conditions.json').write_text(json.dumps(conditions, indent=2))
    if args.checkpoint:
        conditions.update(control='deterministic PPO policy; native TT position PD actuators',
                          checkpoint=str(args.checkpoint.resolve()), training_backend='MJX', evaluation_backend='MuJoCo CPU',
                          observation_noise=ecfg['obs_noise'])
        (out/'conditions.json').write_text(json.dumps(conditions, indent=2))
    for path in paths:
        with np.load(path) as z:
            ref = np.c_[z['fix_base_pos'], z['fix_base_quat'], z['fix_q']]
            dt = float(z['dt'])
        if ref.shape != (301, 19) or not np.isfinite(ref).all() or not np.isclose(dt, 1/30):
            raise ValueError(f'Invalid native reference: {path}')
        assert np.allclose(np.linalg.norm(ref[:, 3:7], axis=1), 1, atol=1e-6)
        data, ghost = mujoco.MjData(model), mujoco.MjData(model)
        data.qpos[:] = ref[0]
        mujoco.mj_differentiatePos(model, data.qvel, dt, ref[0], ref[1])
        data.ctrl[:] = ref[0, 7:]
        mujoco.mj_forward(model, data)
        initial_penetration = max(0., max((-c.dist for c in data.contact if c.geom1 in planes or c.geom2 in planes), default=0.))
        logs = {k: [] for k in ['time', 'qpos', 'qvel', 'foot_pos', 'reference_foot_pos', 'contact', 'slip_mps', 'up_z']}
        peak_torque = 0.
        fell = False
        video = out/(path.stem+'.mp4')
        if args.checkpoint:
            ci = list(ecfg['clips']).index(path.stem)
            start, clip_end = int(env.clip_start[ci]), int(env.clip_end[ci])
            prev_action = jnp.zeros(12)
            rng = jax.random.PRNGKey(0)
        with imageio.get_writer(video, fps=30, codec='libx264', quality=7, macro_block_size=1) as writer:
            for k in range(300):
                if args.checkpoint:
                    rng, key = jax.random.split(rng)
                    action = policy_action(jnp.asarray(data.qpos), jnp.asarray(data.qvel),
                                           jnp.asarray(data.xmat.reshape(-1, 3, 3)), jnp.int32(start+k), jnp.int32(clip_end), prev_action, key)
                    data.ctrl[:] = np.asarray(env.action_offset) + np.asarray(action)*env.action_scale
                    prev_action = action
                else:
                    data.ctrl[:] = ref[k, 7:]
                for _ in range(10):
                    mujoco.mj_step(model, data)
                    peak_torque = max(peak_torque, float(np.abs(data.actuator_force).max()))
                mujoco.mj_forward(model, data)
                if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                    raise ValueError(f'Nonfinite simulation: {path}')
                ghost.qpos[:] = ref[k+1]
                mujoco.mj_forward(model, ghost)
                contact, slip = foot_contacts(model, data, feet, planes)
                up = 1 - 2*(data.qpos[4]**2 + data.qpos[5]**2)
                values = [data.time, data.qpos.copy(), data.qvel.copy(), data.geom_xpos[feet].copy(),
                          ghost.geom_xpos[feet].copy(), contact, slip, up]
                for key, value in zip(logs, values):
                    logs[key].append(value)
                camera.lookat[:] = (data.qpos[:3]+ghost.qpos[:3])/2
                camera.distance = max(1.5, 1.3+float(np.linalg.norm(data.qpos[:3]-ghost.qpos[:3])))
                renderer.update_scene(data, camera=camera, scene_option=option)
                scene = renderer.scene
                n = scene.ngeom
                mujoco.mjv_addGeoms(model, ghost, option, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC, scene)
                for j in range(scene.ngeom):
                    g = scene.geoms[j]
                    if g.category == mujoco.mjtCatBit.mjCAT_DYNAMIC:
                        g.matid = -1
                        g.rgba[:] = [1., .55, .1, .35] if j >= n else [.25, .5, .95, 1.]
                writer.append_data(renderer.render())
                if up < .5:
                    fell = True
                    break
        logs = {k: np.asarray(v) for k, v in logs.items()}
        n = len(logs['time'])
        assert np.isclose(logs['time'][-1], n/30, atol=1e-8)
        assert np.all(np.abs(model.actuator_forcerange) == 33.5)
        np.savez_compressed(out/(path.stem+'.npz'), **logs, reference_qpos=ref[1:n+1])
        slip = logs['slip_mps'][np.isfinite(logs['slip_mps'])]
        row = dict(motion=path.stem, survived_10s=not fell and n==300,
                   evaluated_seconds=float(logs['time'][-1]), fell=fell,
                   slip_mean_mps=float(slip.mean()) if slip.size else None,
                   slip_p95_mps=float(np.percentile(slip, 95)) if slip.size else None,
                   supported_foot_samples=int(slip.size), peak_actual_torque_nm=peak_torque,
                   initial_ground_penetration_m=initial_penetration,
                   reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   video_absolute_path=str(video), **tracking_errors(logs['qpos'], ref[1:n+1], logs['foot_pos'], logs['reference_foot_pos']))
        rows.append(row)
        (out/'metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False))
        print(json.dumps(row, ensure_ascii=False), flush=True)
    renderer.close()
    if not args.checkpoint:
        report(out, rows)


def report(out, rows):
    table = ''.join(f'<tr><td>{r["motion"]}</td><td>{r["evaluated_seconds"]:.2f}</td><td>{"완주" if r["survived_10s"] else "전도"}</td><td>{r["joint_rmse_deg"]:.2f}</td><td>{r["body_position_rmse_m"]*100:.2f}</td><td>{r["foot_world_mean_error_m"]*100:.2f}</td><td>{r["orientation_mean_deg"]:.2f}</td><td>{r["slip_mean_mps"] if r["slip_mean_mps"] is not None else "접촉 없음"}</td></tr>' for r in rows)
    videos = ''.join(f'<h3>{r["motion"]}</h3><video controls preload="metadata" src="{html.escape(r["video_absolute_path"])}"></video><p>{html.escape(r["video_absolute_path"])}</p>' for r in rows)
    text = f'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TT 30Hz 실제 물리 추종 try1</title><style>body{{font-family:system-ui;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.85;color:#243743}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;border-bottom:1px solid #ccd6db}}video{{width:100%;max-width:960px}}h2{{margin-top:36px}}</style>
    <h1>TT 30Hz 실제 물리 추종 실험 · try1</h1><p>2026년 10월 29일 · MuJoCo CPU · 5개 레퍼런스 · 발 고정 보정 없음</p>
    <h2>목적과 실험 방법</h2><p>계산으로 변환한 레퍼런스를 TT의 위치 PD 제어기로 실제 물리 시뮬레이션에서 따라가게 하고 생존·오차·접지 미끄러짐을 측정했다. 새 PPO 정책 학습은 하지 않았다. 몸통 궤적을 강제로 덮어쓰지 않고, 목표 관절각만 입력했다. 원본 30Hz 관절각은 각 제어 구간 동안 유지했으며 시간 보간은 하지 않았다.</p>
    <p>TT e000/00000011, 치수 중간값 13개, 기존 TT PD 이득, 토크 한계 33.5N·m. 자기 충돌 끔·바닥 충돌 유지. 물리 계산은 300Hz로 제어 1회당 10회다. 첫 레퍼런스 자세와 차분 초기 속도에서 시작하며 높이를 추가 보정하거나 중간 초기화하지 않았다. 각 모션은 최대 10초, 몸통 위쪽 축의 수직 성분이 0.5 미만이면 전도로 종료했다. 전도는 30Hz 경계에서 검사하므로 시각 정밀도는 약 33ms다.</p>
    <h2>측정 결과</h2><p>10초 생존: <strong>{sum(r['survived_10s'] for r in rows)}/5</strong>. 오차는 초기 상태를 제외하고 실행 종료까지 계산했다. 짧게 실행된 모션의 작은 오차를 완주한 모션보다 좋은 성능으로 해석하면 안 된다. 생존과 추종 정확도는 별도로 평가하며, 미끄러짐의 합격 기준은 아직 정하지 않았다.</p>
    <table><tr><th>모션</th><th>실행(s)</th><th>생존</th><th>관절 RMSE(°)</th><th>몸통 위치 RMSE(cm)</th><th>발 월드 평균 오차(cm)</th><th>자세 평균 오차(°)</th><th>접지 미끄러짐(m/s)</th></tr>{table}</table>
    <p>관절 RMSE는 모든 관절·평가 상태의 제곱 평균 제곱근이다. 몸통 위치는 같은 시각의 레퍼런스 대비 3차원 거리이며 발 오차는 월드 좌표계의 네 발 목표 대비 평균 거리다. 접지 미끄러짐은 실제 바닥 접촉에서 수직 힘이 0.001N을 넘는 발의 접촉점 수평 속도로 측정했다. 레퍼런스에 저장된 접촉 표시는 사용하지 않았으며, 접촉이 없는 구간은 평균에 넣지 않았다.</p>
    <h2>실제 실행 영상</h2><p>파랑은 물리 시뮬레이션의 TT, 반투명 주황은 같은 시각의 TT 목표 자세다. 주황은 비교용 표시이며 충돌·힘을 발생시키지 않는다. 각 영상은 실제 종료 시점까지만 담았으며 원본 해상도 960×540, 30fps다. 영상 주소는 절대 경로다.</p>{videos}
    <h2>해석과 한계</h2><p>이 결과는 레퍼런스를 고정된 PD 제어기로 재생한 기준 실험이다. 성공한 경우에도 PPO 학습 성공으로 해석하지 않으며, 실패한 경우에도 학습된 추종 정책으로 수행할 수 없다는 뜻은 아니다. 이득 조정·초기 상태·접촉·레퍼런스의 동적 일관성 중 원인을 확정하려면 조건을 하나씩 바꾼 후속 비교가 필요하다. 역동역학 선별 통과와 실제 동작 성공을 구분해야 한다.</p>
    <p><a href="metrics.json">모션별 측정값</a> · <a href="conditions.json">실행 조건과 식별 정보</a></p></html>'''
    (out/'report_try1.html').write_text(text)


if __name__ == '__main__':
    main()
