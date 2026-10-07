"""여러 TT 형상을 한 PPO batch 로 (T5).

MJX 는 구조가 같은 모델만 vmap 할 수 있다. 128 개 choice 는 4 개씩 32 개 구조 그룹이 된다
(마지막 두 bit 와 seed 는 body_pos / geom_size / body_mass 같은 jax 배열 값만 바꾼다).
DesignGroupEnv 는 한 그룹의 모델을 쌓아 vmap 하고, MultiDesignEnv 는 그룹들을 batch 축으로
이어 붙여 PPO 한 번이 모든 디자인을 보게 한다. 그룹의 env slot i 는 항상 디자인 i % K.
"""

import jax
import jax.numpy as jp
import numpy as np
from brax.envs.wrappers import training as brax_training

from locomotion_env import REMOVE_PARTS, LocomotionEnv
from t2.mjx_env.base_env import State
from t2.mjx_env.wrapper import BraxAutoResetWrapper

PER_DESIGN = ("mjx_model", "morphology")  # trace 중 디자인별로 바꿔 끼우는 속성. 나머지는 그룹 공유
SHARED_INFO = ("truncation", "steps", "episode_done", "episode_metrics")
FRICTION_RANGE = (0.5, 1.25)  # LeggedRobotCfg.domain_rand.friction_range


def structure_key(choices):
    """마지막 두 bit 만 다른 디자인은 MJX 구조가 같다. rail 제거 시(D49) bit 5 도 무관."""
    # T6 진단: "differs from its group" 나 vmap shape 오류 = 구조가 다른 로봇을 같은 그룹으로
    # 묶었다는 뜻. 평가기가 아니라 이 key 를 고친다.
    bits = list(choices[:-2])
    if "/rail/" in REMOVE_PARTS:
        del bits[5]
    return tuple(bits)


class DesignGroupEnv:
    """한 구조 그룹의 K 개 디자인을 batch 로 돌리는 env (reset/step 이 batch 를 받는다)."""

    def __init__(self, designs, env_cls=LocomotionEnv, **env_kwargs):
        # T6 진단: 형상별 모델 생성, body/contact 불일치, 그룹 검증 오류는 rollout 전 여기서 난다.
        # 1. 디자인별 env 생성
        self.envs = [env_cls(list(c), s, **env_kwargs) for c, s in designs]
        base = self.env = self.envs[0]
        self.n_designs = len(self.envs)
        self.stand_heights = np.asarray([getattr(e, "stand_height", 0.) for e in self.envs])

        # 2. heightfield 는 모든 디자인이 하나를 공유 (TaskEnv). 쌓을 땐 자리표시자로 바꿔
        #    디자인·env 마다 복사되지 않게 하고, 호출 때 한 벌을 다시 넣는다.
        self.hfield = getattr(base, "hfield", None)
        if self.hfield is not None:
            for e in self.envs:
                e.mjx_model = e.mjx_model.replace(hfield_data=jp.zeros(1))

        # 3. 그룹 검증: 공유 속성·MJX 구조가 같아야 한다
        for e in self.envs[1:]:
            for name in ("home", "ctrl", "contact_bodies", "geom_body", "dof_pos_limits"):
                if not np.array_equal(np.asarray(getattr(e, name)), np.asarray(getattr(base, name))):
                    raise ValueError(f"Design {e.morphology} differs from its group in {name}")
            if ([x.shape for x in jax.tree.leaves(e.mjx_model)] !=
                    [x.shape for x in jax.tree.leaves(base.mjx_model)]):
                raise ValueError("Designs in a group must share the MJX model structure")

        # 4. 디자인별 속성 쌓기. 정적 numpy 필드는 treedef 에 들어가고 그룹 안에서는
        #    geom_rbound_hfield (hfield 충돌 전용) 만 달라서 첫 디자인 treedef 를 쓴다.
        self.designs = {name: jax.tree.unflatten(
            jax.tree.structure(getattr(base, name)),
            [jp.stack(x) for x in zip(*[jax.tree.leaves(getattr(e, name)) for e in self.envs])])
            for name in PER_DESIGN}
        if self.hfield is not None:
            self.envs = self.envs[:1]  # MjModel 마다 CPU heightfield 사본이 있어서 하나만 남김
            self.env.mjx_model = self.env.mjx_model.replace(hfield_data=self.hfield)
        self.action_size, self.observation_size = base.action_size, base.observation_size

    def reset(self, rng):
        return self._vmapped(self.env.reset, rng.shape[0], rng)

    def step(self, state, action):
        return self._vmapped(self.env.step, action.shape[0], state, action)

    def _vmapped(self, fn, n, *args):
        index = jp.arange(n) % self.n_designs
        designs = jax.tree.map(lambda x: x[index], self.designs)
        # domain_rand.randomize_friction: env slot 당 계수 1 개, run 동안 고정.
        # MuJoCo 는 두 geom 중 큰 값을 쓰므로 모든 geom(바닥 포함)에 넣는다.
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


class MultiDesignEnv:
    """episode/auto-reset wrapper 를 씌운 DesignGroupEnv 들을 batch 축으로 이어 붙인다."""

    def __init__(self, groups, episode_length, **env_kwargs):
        self.groups = [BraxAutoResetWrapper(brax_training.EpisodeWrapper(
            DesignGroupEnv(g, **env_kwargs), episode_length, 1), full_reset=True) for g in groups]
        self.action_size = self.groups[0].action_size
        self.observation_size = self.groups[0].observation_size

    def reset(self, rng):
        return self._merge([g.reset(r) for g, r in zip(self.groups, self._split(rng))])

    def step(self, state, action):
        return self._merge([g.step(s, a) for g, s, a in
                            zip(self.groups, state.info["groups"], self._split(action))])

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
