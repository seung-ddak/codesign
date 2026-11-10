"""A1 CALM (calm_jax) 을 TT 형상에 적용하는 MJX 실행기. 오전 CPU 파일럿 (calm_tt_pilot.py) 과 같은 조건.

- 제어 30 Hz, 물리 600 Hz (20 substeps), TT 기본 구동기는 끄고 CALM PD (kp 40, kd 1.5, 토크 ±33.5) 를 직접 건다.
- 시작: A1 레퍼런스 첫 자세 (몸통 자세 + 관절각), 발이 바닥에 닿도록 높이 맞춤. 넘어짐 = 직립도 < 0.5 또는 몸통 높이 < 0.07.
- 몸체 상태 23 개 = 위치, 자세 (xyzw), 몸체 원점의 월드 선속도·각속도 (mj_objectVelocity, flg_local=0 과 같은 값).
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calm_jax  # noqa: E402
from check_calm_tt_direct import CALM_BODIES, check_body_mapping, joint_addresses  # noqa: E402
from pd_replay import build_model  # noqa: E402

CONTROL_DT, SUBSTEPS, KP, KD, TORQUE = 1 / 30, 20, 40.0, 1.5, 33.5
FEET = ("FR", "FL", "RR", "RL")


def make_model(E):
    m = build_model("00000011", uniforms=np.asarray(E, float), torque_limit=33.5, flat_ground=True)
    m.opt.timestep = CONTROL_DT / SUBSTEPS
    m.actuator_gainprm[:] = 0
    m.actuator_biasprm[:] = 0
    mapping = check_body_mapping(m)
    _, qa, va = joint_addresses(m)
    bodies = np.array([m.body(mapping[b]).id for b in CALM_BODIES])
    feet = np.array([m.geom(f + "_foot").id for f in FEET])
    return m, dict(qa=qa, va=va, bodies=bodies, feet=feet, root=m.body_rootid[bodies])


def initial_qpos(m, idx, root, q):
    """CPU 로 시작 자세를 만든다 (파일럿과 같은 식). root = A1 데이터 root[0] (pos3, quat xyzw, ...), q = 관절 12."""
    d = mujoco.MjData(m)
    d.qpos[:] = m.keyframe("home").qpos
    d.qpos[:2] = 0
    d.qpos[3:7] = np.asarray(root)[[6, 3, 4, 5]] / np.linalg.norm(np.asarray(root)[3:7])
    d.qpos[idx["qa"]] = q
    mujoco.mj_forward(m, d)
    d.qpos[2] -= float((d.geom_xpos[idx["feet"], 2] - m.geom_size[idx["feet"], 0]).min())
    return d.qpos.copy()


def body_states(d, idx):
    """[23, 13]: xpos, xquat(xyzw), 몸체 원점 월드 선속도, 월드 각속도."""
    b = idx["bodies"]
    ang = d.cvel[b, :3]
    lin_com = d.cvel[b, 3:]
    lin = lin_com + jnp.cross(ang, d.xpos[b] - d.subtree_com[idx["root"]])
    quat = d.xquat[b][:, [1, 2, 3, 0]]
    return jnp.concatenate([d.xpos[b], quat, lin, ang], -1)


def make_rollout(m, idx, params, steps):
    """jit 된 rollout(qpos0, commands[T,3], latents[T,64], use_latent[T]) → 기록. vmap 해서 쓴다."""
    mx = mjx.put_model(m)
    qa, va = jnp.asarray(idx["qa"]), jnp.asarray(idx["va"])

    def control(d, cmd, z, use_z):
        s = body_states(d, idx)
        obs = calm_jax.calm_obs(s)
        z_h = calm_jax.hlc(params, jnp.concatenate([obs, calm_jax.hlc_task_obs(s[0, 3:7], cmd)], -1))
        z_f = z / jnp.maximum(jnp.linalg.norm(z), 1e-12)
        return calm_jax.llc_target(params, obs, jnp.where(use_z, z_f, z_h))

    def substep(d, target):
        tau = jnp.clip(KP * (target - d.qpos[qa]) - KD * d.qvel[va], -TORQUE, TORQUE)
        d = d.replace(qfrc_applied=jnp.zeros(m.nv).at[va].set(tau))
        return mjx.step(mx, d), None

    def step(carry, x):
        d, fell = carry
        cmd, z, use_z = x
        up = 1 - 2 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)
        fell = fell | (up < 0.5) | (d.qpos[2] < 0.07)
        target = control(d, cmd, z, use_z)
        d2, _ = jax.lax.scan(lambda dd, _: substep(dd, target), d, None, length=SUBSTEPS)
        bad = ~(jnp.isfinite(d2.qpos).all() & jnp.isfinite(d2.qvel).all())
        d = jax.tree_util.tree_map(lambda a, b: jnp.where(fell, a, b), d, d2)  # 넘어진 뒤에는 상태를 멈춘다
        fell = fell | bad
        return (d, fell), (d.qpos, d.qvel, target, fell)

    def rollout(qpos0, commands, latents, use_latent):
        d = mjx.make_data(mx).replace(qpos=qpos0)
        d = mjx.forward(mx, d)
        (_, fell), (qp, qv, tg, fl) = jax.lax.scan(step, (d, jnp.array(False)), (commands, latents, use_latent), length=steps)
        return dict(qpos=jnp.concatenate([qpos0[None], qp]), qvel=qv, target=tg, fell_step=fl)

    return jax.jit(jax.vmap(rollout))
