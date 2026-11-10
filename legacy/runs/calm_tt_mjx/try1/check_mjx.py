"""(1) MJX 몸체 속도 = CPU mj_objectVelocity 인지, (2) 같은 에피소드의 MJX 궤적이 CPU 파일럿과 얼마나 같은지."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT / 'runs/calm_tt_policy_pilot/try1/code'))
import jax, jax.numpy as jnp, mujoco
from mujoco import mjx
import calm_jax, calm_tt_mjx as ctm
from check_calm_tt_direct import physical_body_states, check_body_mapping
MOT = ROOT / 'runs/calm_motion_10,000/dataset/accepted'
params = calm_jax.load_params()
E = [.5] * 13; motion = 7256
m, idx = ctm.make_model(E); ref = dict(np.load(MOT / f'{motion:05d}.npz'))
# (1) 속도 비교: 임의 자세·속도에서
d = mujoco.MjData(m); d.qpos[:] = ctm.initial_qpos(m, idx, ref['root'][0], ref['q'][0]); d.qvel[:] = np.random.default_rng(0).normal(0, .5, m.nv)
mujoco.mj_forward(m, d)
cpu = physical_body_states(m, d, check_body_mapping(m))
mx = mjx.put_model(m); dx = mjx.forward(mx, mjx.make_data(mx).replace(qpos=jnp.asarray(d.qpos), qvel=jnp.asarray(d.qvel)))
gpu = np.asarray(ctm.body_states(dx, idx))
vel_diff = float(np.abs(cpu - gpu).max()); print('body state max diff (pos, quat, lin, ang):', [float(np.abs(cpu[:, a:b] - gpu[:, a:b]).max()) for a, b in ((0, 3), (3, 7), (7, 10), (10, 13))])
# (2) CPU 파일럿 에피소드와 비교
pil = np.load(ROOT / f'runs/calm_tt_policy_pilot/try1/episodes/e0000_{motion:05d}.npz')
T = 300
roll = ctm.make_rollout(m, idx, params, T)
q0 = ctm.initial_qpos(m, idx, ref['root'][0], ref['q'][0])
SIT = 10; sit = np.zeros(T, bool)
for k, a, b in zip(ref['motion_index'], ref['segment_start'], ref['segment_end']):
    if int(k) == SIT: sit[a:b] = True
out = roll(jnp.asarray(q0)[None], jnp.asarray(ref['command'][:T], jnp.float32)[None], jnp.asarray(ref['latent'][:T], jnp.float32)[None], jnp.asarray(sit)[None])
qm = np.asarray(out['qpos'][0]); qc = pil['qpos']; n = min(len(qc), len(qm))
xy = np.linalg.norm(qm[:n, :2] - qc[:n, :2], axis=1)
res = dict(state_max_diff=vel_diff, cpu_steps=int(pil['steps']), cpu_fell=bool(pil['fell']), mjx_fell=bool(np.asarray(out['fell_step'][0]).any()),
           xy_gap_m_at=[float(xy[min(k, n - 1)]) for k in (30, 90, 150, 299)], cpu_end_xy=qc[n - 1, :2].tolist(), mjx_end_xy=qm[n - 1, :2].tolist())
print(res); (Path(__file__).with_name('check_mjx.json')).write_text(json.dumps(res, indent=1))
