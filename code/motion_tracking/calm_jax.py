"""A1 CALM (HLC + LLC) 를 JAX 로 옮긴 것. MJX 환경에서 TT 형상에 그대로 적용하거나 PPO 미세조정의 시작점으로 쓴다.

원본: external/a1-calm-main (rl_games, PyTorch). 계산은 원본과 같다.
- 관측: humanoid.compute_humanoid_observations_max (몸체 23 개, local_root_obs=True, root_height_obs=True) → 343
- HLC: [관측 343, heading 명령 2, facing 3, up 3] → 351 → MLP 1024-512 (relu) → mu 64 → tanh → clamp(-1, 1) → 정규화
- LLC: [정규화 관측 343, style(z) 64] → MLP 1024-1024-512 (relu) → mu 12 → clamp(-1, 1) → 관절 목표 = offset + scale · a
- 정규화: rl_games RunningMeanStd = clamp((x - mean) / sqrt(var + 1e-5), -5, 5)
쿼터니언은 원본과 같이 xyzw 순서다.
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp

WORKSPACE = Path(__file__).resolve().parents[3]
CALM_DIR = WORKSPACE / "external/a1-calm-main"
A1_JOINTS = [f"{l}_{p}_joint" for l in ("FL", "FR", "RL", "RR") for p in ("hip", "thigh", "calf")]
EPS = 1e-5


# ---------- quaternion (xyzw), isaacgym.torch_utils 와 같은 식 ----------
def quat_rotate(q, v):
    w, u = q[..., 3:4], q[..., :3]
    return v * (2.0 * w ** 2 - 1.0) + jnp.cross(u, v) * w * 2.0 + u * jnp.sum(u * v, -1, keepdims=True) * 2.0


def quat_rotate_inverse(q, v):
    w, u = q[..., 3:4], q[..., :3]
    return v * (2.0 * w ** 2 - 1.0) - jnp.cross(u, v) * w * 2.0 + u * jnp.sum(u * v, -1, keepdims=True) * 2.0


def quat_mul(a, b):
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return jnp.stack([w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                      w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                      w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                      w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2], -1)


def calc_heading_quat_inv(q):
    d = quat_rotate(q, jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), q[..., :3].shape))
    h = -jnp.arctan2(d[..., 1], d[..., 0])
    q = jnp.stack([jnp.zeros_like(h), jnp.zeros_like(h), jnp.sin(h / 2), jnp.cos(h / 2)], -1)
    return q / jnp.linalg.norm(q, axis=-1, keepdims=True)


def quat_to_tan_norm(q):
    tan = quat_rotate(q, jnp.broadcast_to(jnp.array([1.0, 0.0, 0.0]), q[..., :3].shape))
    norm = quat_rotate(q, jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), q[..., :3].shape))
    return jnp.concatenate([tan, norm], -1)


# ---------- 관측 ----------
def calm_obs(states):
    """states [..., 23, 13] = (위치 3, 자세 xyzw 4, 월드 선속도 3, 월드 각속도 3), 첫 몸체 = 몸통 기준. → [..., 343]"""
    pos, rot, vel, ang = states[..., :3], states[..., 3:7], states[..., 7:10], states[..., 10:13]
    root_pos, root_rot = pos[..., 0, :], rot[..., 0, :]
    heading = calc_heading_quat_inv(root_rot)
    hb = jnp.broadcast_to(heading[..., None, :], rot.shape)
    local_pos = quat_rotate(hb, pos - root_pos[..., None, :])[..., 1:, :]
    local_rot = quat_to_tan_norm(quat_mul(hb, rot))
    local_vel = quat_rotate(hb, vel)
    local_ang = quat_rotate(hb, ang)
    flat = lambda x: x.reshape(x.shape[:-2] + (-1,))
    return jnp.concatenate([root_pos[..., 2:3], flat(local_pos), flat(local_rot), flat(local_vel), flat(local_ang)], -1)


def hlc_task_obs(root_rot, command):
    """HLC 의 명령 부분: heading 기준 이동 방향 2 + 몸통 기준 바라볼 방향 3 + 몸통 기준 위쪽 3. command = (vx, vy, 바라볼 yaw)."""
    cmd = jnp.concatenate([command[..., :2], jnp.zeros_like(command[..., :1])], -1)
    head = quat_rotate(calc_heading_quat_inv(root_rot), cmd)[..., :2]
    face_w = jnp.stack([jnp.cos(command[..., 2]), jnp.sin(command[..., 2]), jnp.zeros_like(command[..., 2])], -1)
    face = quat_rotate_inverse(root_rot, face_w)
    up = quat_rotate_inverse(root_rot, jnp.broadcast_to(jnp.array([0.0, 0.0, 1.0]), face.shape))
    return jnp.concatenate([head, face, up], -1)


# ---------- 가중치 ----------
def load_params():
    """원본 체크포인트를 numpy 로 읽는다 (torch 는 읽을 때만 필요)."""
    import torch
    out = {}
    for tag, fname in (("llc", "lowLevelController.pth"), ("hlc", "highLevelPolicy.pth")):
        ck = torch.load(CALM_DIR / "checkpoints" / fname, map_location="cpu", weights_only=True)
        m = {k[len("a2c_network."):]: v.numpy() for k, v in ck["model"].items()}
        rms = {k: v.numpy() for k, v in ck["running_mean_std"].items()}
        out[tag] = dict(m=m, mean=rms["running_mean"].astype(np.float32), var=rms["running_var"].astype(np.float32))
    xml = ET.parse(CALM_DIR / "assets/urdf/a1_description/urdf/a1.urdf").getroot()
    lim = np.array([[float(xml.find(f"joint[@name='{n}']/limit").get(k)) for k in ("lower", "upper")] for n in A1_JOINTS])
    out["offset"] = lim.mean(1).astype(np.float32)
    out["scale"] = (0.7 * (lim[:, 1] - lim[:, 0])).astype(np.float32)
    return jax.tree_util.tree_map(jnp.asarray, out)


def _lin(m, name, x):
    return x @ m[name + ".weight"].T + m[name + ".bias"]


def normalize_obs(p, x):
    return jnp.clip((x - p["mean"]) / jnp.sqrt(p["var"] + EPS), -5.0, 5.0)


def hlc(params, obs351):
    p = params["hlc"]; m = p["m"]
    h = normalize_obs(p, obs351)
    h = jax.nn.relu(_lin(m, "actor_mlp.0", h))
    h = jax.nn.relu(_lin(m, "actor_mlp.2", h))
    z = jnp.clip(jnp.tanh(_lin(m, "mu", h)), -1.0, 1.0)  # HRL 네트워크 forward 가 mu 에 tanh 를 건다
    return z / jnp.maximum(jnp.linalg.norm(z, axis=-1, keepdims=True), 1e-12)


def llc_mu(params, obs343, z):
    """LLC 평균 행동 (clamp 전). z 는 정규화된 64 차원."""
    p = params["llc"]; m = p["m"]
    s = jax.nn.relu(_lin(m, "actor_mlp._style_mlp.0", z))
    s = jax.nn.relu(_lin(m, "actor_mlp._style_mlp.2", s))
    s = jnp.tanh(_lin(m, "actor_mlp._style_dense", s))
    h = jnp.concatenate([normalize_obs(p, obs343), s], -1)
    for i in range(3):
        h = jax.nn.relu(_lin(m, f"actor_mlp._dense_layers.{i}", h))
    return _lin(m, "mu", h)


def llc_target(params, obs343, z):
    """관절 목표 (A1 관절 순서 FL, FR, RL, RR × hip, thigh, calf)."""
    a = jnp.clip(llc_mu(params, obs343, z), -1.0, 1.0)
    return params["offset"] + params["scale"] * a


def policy(params, states, command, latent=None):
    """원본 check_menagerie_mjx_policy.load_policy 의 policy 와 같은 계산. latent 를 주면 HLC 대신 그 z 를 쓴다."""
    obs = calm_obs(states)
    if latent is None:
        z = hlc(params, jnp.concatenate([obs, hlc_task_obs(states[..., 0, 3:7], command)], -1))
    else:
        z = latent / jnp.maximum(jnp.linalg.norm(latent, axis=-1, keepdims=True), 1e-12)
    return llc_target(params, obs, z)
