"""JAX 로 옮긴 CALM 이 원본 PyTorch 정책과 같은 관절 목표를 내는지 확인 (TT 형상, CPU MuJoCo, 파일럿과 같은 실행)."""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[3]; WS = ROOT.parent
sys.path.insert(0, str(ROOT / 'code/motion_tracking')); sys.path.insert(0, str(ROOT / 'runs/calm_tt_policy_pilot/try1/code'))
import jax; jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp, mujoco
import calm_jax
from check_menagerie_mjx_policy import load_policy
from check_calm_tt_direct import physical_body_states, CALM_BODIES
import calm_tt_pilot as pilot
WHEEL = ROOT / 'runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl'
pilot.MOTIONS = ROOT / 'runs/calm_motion_10,000/dataset/accepted'
torch_policy, joints = load_policy(WS, WHEEL.resolve())
params = calm_jax.load_params()
jpol = jax.jit(calm_jax.policy)
jpol_lat = jax.jit(lambda p, s, c, z: calm_jax.policy(p, s, c, z))
rows = []
for shape, E, motion in [('e0000', [.5] * 13, 7256), ('e0093', None, 4231)]:
    if E is None:
        E = json.load(open(ROOT / 'runs/shared_calm_v100_48h/plan.json'))
        E = next(r['E'] for r in E['designs'] if r['design'] == shape)
    model, mapping, qa, va = pilot.make_model(E)
    ref = dict(np.load(pilot.MOTIONS / f'{motion:05d}.npz'))
    d = mujoco.MjData(model); mujoco.mj_resetData(model, d)
    d.qpos[:] = model.keyframe('home').qpos; d.qpos[:2] = 0
    d.qpos[3:7] = ref['root'][0, [6, 3, 4, 5]] / np.linalg.norm(ref['root'][0, 3:7]); d.qpos[qa] = ref['q'][0]
    mujoco.mj_forward(model, d)
    feet = np.array([model.geom(f + '_foot').id for f in pilot.FEET])
    d.qpos[2] -= float((d.geom_xpos[feet, 2] - model.geom_size[feet, 0]).min()); mujoco.mj_forward(model, d)
    errs, errs_lat = [], []
    for t in range(90):
        phys = physical_body_states(model, d, mapping)
        tt = torch_policy(phys, CALM_BODIES, ref['command'][t])
        jj = np.asarray(jpol(params, jnp.asarray(phys, jnp.float32), jnp.asarray(ref['command'][t], jnp.float32)))
        errs.append(np.abs(tt - jj).max())
        tl = torch_policy(phys, CALM_BODIES, ref['command'][t], latent=ref['latent'][t])
        jl = np.asarray(jpol_lat(params, jnp.asarray(phys, jnp.float32), jnp.asarray(ref['command'][t], jnp.float32), jnp.asarray(ref['latent'][t], jnp.float32)))
        errs_lat.append(np.abs(tl - jl).max())
        for _ in range(pilot.SUBSTEPS):
            tau = np.clip(pilot.KP * (tt - d.qpos[qa]) - pilot.KD * d.qvel[va], -33.5, 33.5)
            d.qfrc_applied[:] = 0; d.qfrc_applied[va] = tau; mujoco.mj_step(model, d)
    rows.append(dict(shape=shape, motion=motion, steps=90, max_abs_diff_hlc_path=float(max(errs)), max_abs_diff_latent_path=float(max(errs_lat))))
    print(rows[-1], flush=True)
(Path(__file__).with_name('check_port.json')).write_text(json.dumps(rows, indent=1))
