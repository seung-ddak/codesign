"""track_env 자가 점검: RSI 직후 오차 ≈ 0, reference q 를 PD 목표로 넣으면 추종 보상이 유지되는지."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "transformer-transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from pd_replay import build_model
from track_env import TrackMjxEnv

cfg = OmegaConf.load(Path(__file__).parent / "config/env/a1_track.yaml")
cfg.reference_dir = str(Path(__file__).resolve().parents[2] / "runs/pipeline_v1/retargeting/e000/00000011")
cfg.clips = ["sitDown__r00"]
cfg.switch_exclude = []
cfg.switch_forbid = []
cfg = OmegaConf.to_container(cfg)
cfg.pop("_target_")
cfg.update(reset_qpos=0.0, reset_qvel=0.0, velocity_kick=[0.0, 0.0])
env = TrackMjxEnv(mj_model_fn=lambda: build_model("00000011", torque_limit=33.5), **cfg)
print("obs", env.observation_size, "act", env.action_size, "frames", env.ref_qpos.shape[0])

N, T = 4, 20
reset, step = jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.PRNGKey(0), N))
for k in ("pose", "ee", "root_pos", "root_orn"):
    assert float(s.metrics[f"err/{k}"].max()) < 1e-6, (k, s.metrics[f"err/{k}"].max())


def ref_action(s):  # PD 목표 = 다음 프레임 reference 관절각
    i = jnp.minimum(s.info["frame"] + 1, s.info["clip_end"] - 1)
    return (env.ref_qpos[i, 7:7 + env.n_ctrl] - env.action_offset[: env.n_ctrl]) / env.action_scale


alive, acc = jnp.ones(N), {}
for _ in range(T):
    s = step(s, ref_action(s))
    for k, v in s.metrics.items():
        acc[k] = acc.get(k, 0.0) + float((v * alive).sum())
    acc["n"] = acc.get("n", 0.0) + float(alive.sum())
    alive = alive * (1 - s.done)
n = acc.pop("n")
print(f"alive after {T} steps: {float(alive.mean()):.2f}")
for k, v in acc.items():
    print(f"  {k:22s} {v / n:9.4f}")
