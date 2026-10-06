"""Many TT morphologies in one PPO batch (T5).

MJX can only vmap models with identical structure. The 128 choices form 32 structure
groups of 4 (the last two choice bits and the seed only change jax-array fields such as
body_pos / geom_size / body_mass). DesignGroupEnv vmaps the stacked models of one group;
MultiDesignEnv concatenates the groups along the batch axis so a single brax PPO run sees
every design. Env slot i of a group always simulates design i % K.
"""

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as brax_training

from locomotion_env import REMOVE_PARTS, LocomotionEnv
from t2.mjx_env.base_env import State
from t2.mjx_env.wrapper import BraxAutoResetWrapper

# Per-design attributes swapped in while tracing; everything else is shared by the group.
PER_DESIGN = ("mjx_model", "morphology")
SHARED_INFO = ("truncation", "steps", "episode_done", "episode_metrics")
FRICTION_RANGE = (0.5, 1.25)  # LeggedRobotCfg.domain_rand.friction_range


def structure_key(choices):
    """Designs whose choices differ only in the last two bits share one MJX structure. With the
    rail removed (REMOVE_PARTS, D49) bit 5 no longer changes the robot either."""
    # T6 진단: "differs from its group" 또는 vmap shape 오류는 구조가 다른 로봇을 같은
    # 그룹으로 묶었다는 뜻이다. 평가기가 아니라 이 key를 수정한다.
    bits = list(choices[:-2])
    if "/rail/" in REMOVE_PARTS:
        del bits[5]
    return tuple(bits)


class DesignGroupEnv:
    """Batched env over K designs of one structure group (reset/step take a batch)."""

    def __init__(self, designs, env_cls=LocomotionEnv, **env_kwargs):
        # T6 진단: 형상별 모델 생성, body/contact 불일치, 그룹 검증 오류는 정책 rollout 전에
        # 이 생성자에서 발생한다.
        self.envs = [env_cls(list(c), s, **env_kwargs) for c, s in designs]
        base = self.env = self.envs[0]
        self.n_designs = len(self.envs)
        self.stand_heights = np.asarray([getattr(e, "stand_height", 0.) for e in self.envs])
        # TaskEnv: every design shares one terrain heightfield. Stack a placeholder and put the single
        # copy back per call, otherwise stacking and the per-env gather copy it per design / env.
        self.hfield = getattr(base, "hfield", None)
        if self.hfield is not None:
            for e in self.envs:
                e.mjx_model = e.mjx_model.replace(hfield_data=jp.zeros(1))
        for e in self.envs[1:]:
            for name in ("home", "ctrl", "contact_bodies", "geom_body", "dof_pos_limits"):
                if not np.array_equal(np.asarray(getattr(e, name)), np.asarray(getattr(base, name))):
                    raise ValueError(f"Design {e.morphology} differs from its group in {name}")
            if ([x.shape for x in jax.tree.leaves(e.mjx_model)] !=
                    [x.shape for x in jax.tree.leaves(base.mjx_model)]):
                raise ValueError("Designs in a group must share the MJX model structure")
        # Static numpy fields live in the treedef; within a group they differ only in
        # geom_rbound_hfield (hfield collision only), so the first design's treedef is used.
        self.designs = {name: jax.tree.unflatten(
            jax.tree.structure(getattr(base, name)),
            [jp.stack(x) for x in zip(*[jax.tree.leaves(getattr(e, name)) for e in self.envs])])
            for name in PER_DESIGN}
        if self.hfield is not None:
            self.envs = self.envs[:1]  # each MjModel holds its own CPU heightfield copy
            self.env.mjx_model = self.env.mjx_model.replace(hfield_data=self.hfield)
        self.action_size, self.observation_size = base.action_size, base.observation_size

    def _vmapped(self, fn, n, *args):
        index = jp.arange(n) % self.n_designs
        designs = jax.tree.map(lambda x: x[index], self.designs)
        # domain_rand.randomize_friction: one coefficient per env slot, fixed for the run.
        # MuJoCo takes the max of the two geoms, so every geom (floor included) gets it.
        friction = jp.array(np.random.RandomState(0).uniform(*FRICTION_RANGE, n), jp.float32)

        def call(design, mu, *a):
            saved = {name: getattr(self.env, name) for name in PER_DESIGN}
            model = design["mjx_model"]
            if self.hfield is not None:
                model = model.replace(hfield_data=self.hfield)
            if self.env.domain_rand:
                design = {**design, "mjx_model": model.replace(
                    geom_friction=model.geom_friction.at[:, 0].set(mu))}
            else:
                design = {**design, "mjx_model": model}
            try:
                for name, value in design.items():
                    setattr(self.env, name, value)
                return fn(*a)
            finally:
                for name, value in saved.items():
                    setattr(self.env, name, value)

        return jax.vmap(call)(designs, friction, *args)

    def reset(self, rng):
        return self._vmapped(self.env.reset, rng.shape[0], rng)

    def step(self, state, action):
        return self._vmapped(self.env.step, action.shape[0], state, action)


class MultiDesignEnv:
    """Concatenates episode/auto-reset wrapped DesignGroupEnvs along the batch axis."""

    def __init__(self, groups, episode_length, **env_kwargs):
        self.groups = [BraxAutoResetWrapper(brax_training.EpisodeWrapper(
            DesignGroupEnv(g, **env_kwargs), episode_length, 1), full_reset=True) for g in groups]
        self.action_size = self.groups[0].action_size
        self.observation_size = self.groups[0].observation_size

    def _split(self, x):
        if x.shape[0] % len(self.groups):
            raise ValueError(f"Batch {x.shape[0]} is not divisible by {len(self.groups)} groups")
        return jp.split(x, len(self.groups))

    @staticmethod
    def _merge(states):
        cat = lambda *x: jp.concatenate(x)
        info = {k: jax.tree.map(cat, *[s.info[k] for s in states])
                for k in SHARED_INFO if k in states[0].info}
        info["groups"] = tuple(states)
        return State(data=None, obs=cat(*[s.obs for s in states]),
                     reward=cat(*[s.reward for s in states]), done=cat(*[s.done for s in states]),
                     metrics=jax.tree.map(cat, *[s.metrics for s in states]), info=info)

    def reset(self, rng):
        return self._merge([g.reset(r) for g, r in zip(self.groups, self._split(rng))])

    def step(self, state, action):
        return self._merge([g.step(s, a) for g, s, a in
                            zip(self.groups, state.info["groups"], self._split(action))])
