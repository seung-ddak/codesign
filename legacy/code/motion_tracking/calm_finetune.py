"""A1 CALM 하위 정책 (LLC) 을 TT 형상에서 PPO 로 미세조정하기 위한 환경과 신경망.

환경 (CalmTrackEnv): TrackMjxEnv 의 레퍼런스·보상·종료·리셋을 그대로 쓰고 세 가지만 바꾼다.
- 관측 = CALM 관측 343 (TT 몸체 23 개, calm_jax.calm_obs) + 이 프레임의 A1 잠재 변수 z 64 (원본 데이터셋 latent).
- 행동 a (12, A1 관절 순서) → 관절 목표 = offset + scale · clip(a, -1, 1)  (CALM 과 같음).
- 물리 = CALM PD: 매 substep 토크 = clip(40 (목표 - q) - 1.5 dq, ±33.5), TT 기본 구동기는 끈다.
신경망 (make_calm_ppo_networks): 정책 = CALM LLC 구조와 가중치 (정규화 통계도 원본 고정값),
행동 분포 = 정규분포 (평균 = LLC mu, 표준편차 = 학습 변수). 가치 함수 = 새 MLP.
학습 전 상태에서 평균 행동은 원본 CALM 과 같다 (check_calm_finetune.py 로 확인).
"""
import sys
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from brax.training import distribution, networks
from brax.training.agents.ppo import networks as ppo_networks

sys.path.insert(0, str(Path(__file__).resolve().parent))
import calm_jax  # noqa: E402
from check_calm_tt_direct import CALM_BODIES, body_mapping, joint_addresses  # noqa: E402
from track_env import TrackMjxEnv  # noqa: E402

LATENT = 64
OBS = 343
MOTIONS = Path(__file__).resolve().parents[2] / "runs/calm_motion_10,000/dataset/accepted"
KP, KD, TORQUE = 40.0, 1.5, 33.5


class CalmTrackEnv(TrackMjxEnv):
    def preprocess_mj_model(self, mj_model):
        super().preprocess_mj_model(mj_model)
        mj_model.actuator_gainprm[:] = 0  # 구동기는 끄고 CALM PD 를 직접 건다
        mj_model.actuator_biasprm[:] = 0
        mapping = body_mapping()
        bodies = np.array([mj_model.body(mapping[b]).id for b in CALM_BODIES])
        _, qa, va = joint_addresses(mj_model)  # A1 관절 순서 (FL, FR, RL, RR) → TT qpos / qvel 주소
        self._calm = dict(bodies=jnp.asarray(bodies), root=jnp.asarray(mj_model.body_rootid[bodies]),
                          qa=jnp.asarray(qa), va=jnp.asarray(va), nv=mj_model.nv)
        p = calm_jax.load_params()
        self._calm_offset, self._calm_scale = p["offset"], p["scale"]
        self.action_size = 12

    def load_references(self, reference_files):
        super().load_references(reference_files)
        lat = []
        for f in reference_files:
            mid = int(Path(f).stem.split("_")[1])
            with np.load(MOTIONS / f"{mid:05d}.npz") as z:
                l = np.asarray(z["latent"], np.float32)
            n = int(self.clip_end[len(lat)] - self.clip_start[len(lat)]) if hasattr(self, "clip_end") else len(l) + 1
            l = np.concatenate([l, np.repeat(l[-1:], max(n - len(l), 0), 0)])[:n]
            lat.append(l / np.maximum(np.linalg.norm(l, axis=1, keepdims=True), 1e-12))
        self.ref_latent = jnp.asarray(np.concatenate(lat))
        assert self.ref_latent.shape[0] == self.ref_qpos.shape[0], (self.ref_latent.shape, self.ref_qpos.shape)

    def body_states(self, data):
        c = self._calm; b = c["bodies"]
        ang, lin_com = data.cvel[b, :3], data.cvel[b, 3:]
        lin = lin_com + jnp.cross(ang, data.xpos[b] - data.subtree_com[c["root"]])
        return jnp.concatenate([data.xpos[b], data.xquat[b][:, [1, 2, 3, 0]], lin, ang], -1)

    def _get_obs(self, data, info):
        return jnp.concatenate([calm_jax.calm_obs(self.body_states(data)), self.ref_latent[info["frame"]]])

    def action_target(self, action, info):
        return self._calm_offset + self._calm_scale * jnp.clip(action, -1.0, 1.0)

    def _physics(self, data, target):
        c = self._calm
        from mujoco import mjx

        def sub(d, _):
            tau = jnp.clip(KP * (target - d.qpos[c["qa"]]) - KD * d.qvel[c["va"]], -TORQUE, TORQUE)
            d = d.replace(qfrc_applied=jnp.zeros(c["nv"]).at[c["va"]].set(tau))
            return mjx.step(self.mjx_model, d), None
        return jax.lax.scan(sub, data, None, length=self.n_substeps)[0]


# ---------------- 신경망 ----------------
class CalmPolicy(nn.Module):
    """CALM LLC (style MLP 512-256 → 64 tanh, 본체 1024-1024-512 relu, mu 12) + 학습되는 log std."""
    init_log_std: float

    @nn.compact
    def __call__(self, x):
        obs, z = x[..., :OBS], x[..., OBS:OBS + LATENT]
        mean = jax.lax.stop_gradient(self.variable("constants", "mean", lambda: jnp.zeros(OBS)).value)  # 원본 통계 고정
        var = jax.lax.stop_gradient(self.variable("constants", "var", lambda: jnp.ones(OBS)).value)
        o = jnp.clip((obs - mean) / jnp.sqrt(var + calm_jax.EPS), -5.0, 5.0)
        s = nn.relu(nn.Dense(512, name="style0")(z))
        s = nn.relu(nn.Dense(256, name="style2")(s))
        s = jnp.tanh(nn.Dense(64, name="style_dense")(s))
        h = jnp.concatenate([o, s], -1)
        for i, k in enumerate((1024, 1024, 512)):
            h = nn.relu(nn.Dense(k, name=f"dense{i}")(h))
        mu = nn.Dense(12, name="mu")(h)
        log_std = self.param("log_std", lambda key: jnp.full((12,), self.init_log_std))
        return mu, jnp.broadcast_to(jnp.exp(log_std), mu.shape)


class CalmValue(nn.Module):
    @nn.compact
    def __call__(self, x):
        obs, z = x[..., :OBS], x[..., OBS:OBS + LATENT]
        mean = jax.lax.stop_gradient(self.variable("constants", "mean", lambda: jnp.zeros(OBS)).value)  # 원본 통계 고정
        var = jax.lax.stop_gradient(self.variable("constants", "var", lambda: jnp.ones(OBS)).value)
        h = jnp.concatenate([jnp.clip((obs - mean) / jnp.sqrt(var + calm_jax.EPS), -5.0, 5.0), z], -1)
        for k in (512, 256):
            h = nn.elu(nn.Dense(k)(h))
        return nn.Dense(1)(h)[..., 0]


def calm_policy_params(init_log_std):
    """원본 LLC 가중치를 flax 파라미터 트리로 옮긴다 (torch Linear: y = x Wᵀ + b → flax kernel = Wᵀ)."""
    p = calm_jax.load_params()["llc"]; m = p["m"]
    dense = lambda n: {"kernel": jnp.asarray(m[n + ".weight"]).T, "bias": jnp.asarray(m[n + ".bias"])}
    return {"params": {"style0": dense("actor_mlp._style_mlp.0"), "style2": dense("actor_mlp._style_mlp.2"),
                       "style_dense": dense("actor_mlp._style_dense"),
                       "dense0": dense("actor_mlp._dense_layers.0"), "dense1": dense("actor_mlp._dense_layers.1"),
                       "dense2": dense("actor_mlp._dense_layers.2"), "mu": dense("mu"),
                       "log_std": jnp.full((12,), init_log_std)},
            "constants": {"mean": p["mean"], "var": p["var"]}}


def make_calm_ppo_networks(observation_size, action_size, preprocess_observations_fn=None, init_log_std=float(np.log(0.1)), **_):
    """brax PPO 의 network_factory. 관측 정규화는 원본 CALM 통계를 신경망 안에서 고정해 쓴다 (normalize_observations=False)."""
    assert action_size == 12
    policy, value = CalmPolicy(init_log_std), CalmValue()
    pol_params = calm_policy_params(init_log_std)
    llc = calm_jax.load_params()["llc"]

    def pol_init(key):
        return pol_params

    def pol_apply(processor_params, params, obs):
        return policy.apply(params, obs)

    def val_init(key):
        v = value.init(key, jnp.zeros((1, OBS + LATENT)))
        return {"params": v["params"], "constants": {"mean": llc["mean"], "var": llc["var"]}}

    def val_apply(processor_params, params, obs):
        return value.apply(params, obs)

    return ppo_networks.PPONetworks(
        policy_network=networks.FeedForwardNetwork(init=pol_init, apply=pol_apply),
        value_network=networks.FeedForwardNetwork(init=val_init, apply=val_apply),
        parametric_action_distribution=distribution.NormalDistribution(event_size=12))
