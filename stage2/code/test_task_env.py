"""TaskEnv checks: obs layout, standing on the heightfield, height scan, jump reward, curriculum.

  MUJOCO_GL=egl ../transformer-transformer/.venv/bin/python stage2/code/test_task_env.py
"""

import os
import sys
from pathlib import Path

TT = Path(os.environ.get("CODESIGN_TT", Path(__file__).resolve().parents[3] / "transformer-transformer"))
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path[:0] = [str(TT), str(Path(__file__).resolve().parent)]
os.chdir(TT)

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402
from mujoco import mjx  # noqa: E402

import terrain as T  # noqa: E402
from multi_design_env import MultiDesignEnv  # noqa: E402
from task_env import PRESERVE, TaskEnv  # noqa: E402

CHOICE = [0, 1, 1, 0, 0, 1, 0, 1]


def main():
    env = TaskEnv(CHOICE, 0, domain_rand=False, hold_command=True)
    assert env.observation_size == 257, env.observation_size
    reset, step = jax.jit(env.reset), jax.jit(env.step)
    s = reset(jax.random.PRNGKey(0))
    keep = s.info[PRESERVE]
    origin = np.asarray(env.origins[keep["level"], keep["col"]])
    base = np.asarray(s.data.qpos[env.qadr:env.qadr + 3])
    assert np.abs(base[:2] - origin[:2]).max() <= 1. + 1e-5
    assert abs(base[2] - origin[2] - env.stand_height) < 1e-5

    # Flat tile, home pose, zero action: the robot stays up on the heightfield (the random reset
    # pose can tip it over with zero action, on the plane env as well); scan reads ~ ground.
    col = 0  # smooth slope column; row 0 = difficulty 0 = flat
    target = env.origins[0, col]
    s = reset(jax.random.PRNGKey(0))
    home = env.home.at[env.qadr:env.qadr + 3].add(target)
    data = jax.jit(mjx.forward)(env.mjx_model, s.data.replace(qpos=home, qvel=jp.zeros_like(s.data.qvel)))
    s = s.replace(data=data, info={**s.info, PRESERVE: {"level": jp.int32(0), "col": jp.int32(col)},
                                   "spawn_origin": target, "command": jp.array([0., 0., 0., 0., 0.])})
    for _ in range(100):
        s = step(s, jp.zeros(12))
    base = np.asarray(s.data.qpos[env.qadr:env.qadr + 3])
    assert np.abs(base[:2] - np.asarray(target[:2])).max() < 1.5, base  # moved to the curriculum tile
    assert base[2] - float(target[2]) > 0.5 * env.stand_height, base  # stands on the heightfield
    assert float(s.done) == 0.
    heights = np.asarray(s.obs[49:49 + 187]) / 5.  # after 3 + 3 + 3 + 4 + 12 + 12 + 12
    assert np.allclose(heights, np.clip(base[2] - 0.5 - float(target[2]), -1, 1), atol=0.05), heights[:5]
    assert float(s.metrics["jump_air"]) == 0. and float(s.metrics["jump_height"]) == 0.
    assert float(s.obs[12]) == 0.  # jump flag in the observation

    # Jump command: flag observed, lin_vel_z not penalised.
    s = s.replace(info={**s.info, "command": jp.array([0., 0., 0., 0., 1.])})
    s = step(s, jp.zeros(12))
    assert float(s.obs[12]) == 1. and float(s.metrics["lin_vel_z"]) == 0.

    # Curriculum: an episode that barely moved under a 1 m/s command steps down a level.
    s = s.replace(info={**s.info, PRESERVE: {"level": jp.int32(3), "col": jp.int32(col)},
                        "spawn_origin": env.origins[3, col], "episode_step": jp.int32(999),
                        "command": jp.array([1., 0., 0., 0., 0.])})
    s = s.replace(data=s.data.replace(qpos=s.data.qpos.at[env.qadr:env.qadr + 3].set(
        env.origins[3, col] + jp.array([0., 0., env.stand_height]))))
    s = step(s, jp.zeros(12))
    assert int(s.info[PRESERVE]["level"]) == 2, s.info[PRESERVE]

    # Groups share one heightfield copy.
    m = MultiDesignEnv([[(CHOICE, 0), (CHOICE[:6] + [1, 0], 1)]], 1000, env_cls=TaskEnv,
                       hold_command=True, domain_rand=False)
    inner = m.groups[0].env.env
    assert inner.designs["mjx_model"].hfield_data.shape == (2, 1) and len(inner.envs) == 1
    st = jax.jit(m.reset)(jax.random.split(jax.random.PRNGKey(1), 4))
    st = jax.jit(m.step)(st, jp.zeros((4, 12)))
    assert st.obs.shape == (4, 257) and not np.isnan(np.asarray(st.obs)).any()
    print("task_env ok", T.NUM_ROWS, T.NUM_COLS)


if __name__ == "__main__":
    main()
