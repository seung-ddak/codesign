"""Expert-policy environments with a T5-compatible observation prefix.

The first 69 values exactly follow T5. New inputs are appended:
jump trigger 1 + terrain height scan 187 = 257 values.
"""

import importlib.util
import sys
from pathlib import Path

import jax
import jax.numpy as jp

_LEGACY = Path(__file__).resolve().parents[1] / "stage2" / "code"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

_spec = importlib.util.spec_from_file_location("_codesign_stage2_task_env", _LEGACY / "task_env.py")
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)

ACTION_SCALE = _base.ACTION_SCALE
COMMAND_RANGES = _base.COMMAND_RANGES
EPISODE_STEPS = _base.EPISODE_STEPS
ONLY_POSITIVE_REWARDS = _base.ONLY_POSITIVE_REWARDS
TASK_REWARD_SCALES = _base.TASK_REWARD_SCALES
TERMINATE_AFTER_CONTACTS_ON = _base.TERMINATE_AFTER_CONTACTS_ON
NUM_LEG_ACTUATORS = _base.NUM_LEG_ACTUATORS
T5_OBSERVATION_SIZE = 69
OBSERVATION_SIZE = 257
JUMP_PROBABILITY = 2.0
JUMP_TRIGGER_STEPS = 15  # 0.3 s at 50 Hz
JUMP_LANDING_REWARD = 1.0
JUMP_DIAGNOSTICS = (
    "jump_trigger_fraction_per_step", "jump_episode_fraction",
    "raw_jump_height_reward_per_step", "scaled_jump_height_reward_per_step",
    "raw_jump_air_reward_per_step", "scaled_jump_air_reward_per_step",
    "jump_landing_success",
    "tracking_lin_vel_reward_per_step", "tracking_ang_vel_reward_per_step",
)
MultiDesignEnv = _base.MultiDesignEnv
structure_key = _base.structure_key


class FlatTaskEnv(_base.LocomotionEnv):
    """T5 environment with zero-valued slots for every future observation."""

    def observation(self, data, command, action, rng):
        old = super().observation(data, command, action, rng)
        return jp.pad(old, (0, OBSERVATION_SIZE - T5_OBSERVATION_SIZE))


class FlatJumpEnv(_base.LocomotionEnv):
    """First curriculum stage: T5 flat locomotion plus a short jump trigger."""

    def __init__(self, choices, seed=None, **kwargs):
        super().__init__(choices, seed, **kwargs)
        self.stand_height = float(self.home[self.qadr + 2])

    def sample_command(self, rng):
        command_key, jump_key = jax.random.split(rng)
        command = super().sample_command(command_key)
        jump = jax.random.bernoulli(jump_key, JUMP_PROBABILITY)
        return jp.append(jp.where(jump, command.at[:3].set(0.), command), jump)

    def observation(self, data, command, action, rng):
        old = super().observation(data, command, action, rng)
        return jp.concatenate([old, command[4:5], jp.zeros(187)])

    def reward(self, data, action, command, info):
        reward, done, terms, info = super().reward(data, action, command, info)
        jump = command[4]
        reward -= terms["lin_vel_z"] * jump
        rise = jp.maximum(data.qpos[self.qadr + 2] - self.stand_height, 0.)
        raw_air = jp.all(~info["last_contacts"]) * jump
        raw_height = rise * jump
        first_jump = jump * ~info["jump_seen"]
        airborne_seen = info["airborne_seen"] | (raw_air > 0)
        landing = airborne_seen & jp.any(info["last_contacts"]) & ~info["landed_seen"]
        extra = {
            "jump_air": raw_air * TASK_REWARD_SCALES["jump_air"],
            "jump_height": raw_height * TASK_REWARD_SCALES["jump_height"],
            "jump_landing": landing * JUMP_LANDING_REWARD,
            "jump_trigger_fraction_per_step": jump,
            "jump_episode_fraction": first_jump,
            "raw_jump_height_reward_per_step": raw_height,
            "scaled_jump_height_reward_per_step": raw_height * TASK_REWARD_SCALES["jump_height"],
            "raw_jump_air_reward_per_step": raw_air,
            "scaled_jump_air_reward_per_step": raw_air * TASK_REWARD_SCALES["jump_air"],
            "jump_landing_success": landing.astype(jp.float32),
            "tracking_lin_vel_reward_per_step": terms["tracking_lin_vel"],
            "tracking_ang_vel_reward_per_step": terms["tracking_ang_vel"],
        }
        suppressed = terms["lin_vel_z"] + terms["tracking_lin_vel"] + terms["tracking_ang_vel"]
        terms = {**terms, "lin_vel_z": terms["lin_vel_z"] * (1. - jump),
                 "tracking_lin_vel": terms["tracking_lin_vel"] * (1. - jump),
                 "tracking_ang_vel": terms["tracking_ang_vel"] * (1. - jump), **extra}
        task_reward = extra["jump_air"] + extra["jump_height"] + extra["jump_landing"]
        info = {**info, "jump_seen": info["jump_seen"] | (jump > 0),
                "airborne_seen": airborne_seen, "landed_seen": info["landed_seen"] | landing}
        return reward + task_reward, done, terms, info

    def reset(self, rng):
        state = super().reset(rng)
        zero = jp.zeros(())
        metrics = {**state.metrics, "jump_air": zero, "jump_height": zero,
                   "jump_landing": zero,
                   **{name: zero for name in JUMP_DIAGNOSTICS}}
        return state.replace(metrics=metrics, info={**state.info, "jump_seen": jp.array(False),
                                                     "airborne_seen": jp.array(False),
                                                     "landed_seen": jp.array(False)})

    def step(self, state, action):
        age = state.info["episode_step"] % _base.RESAMPLE_STEPS
        command = state.info["command"].at[4].multiply(age < JUMP_TRIGGER_STEPS)
        return super().step(state.replace(info={**state.info, "command": command}), action)


class TaskEnv(_base.TaskEnv):
    """Jump/stairs environment whose observation starts with the complete T5 vector."""

    def __init__(self, choices, seed=None, **kwargs):
        super().__init__(choices, seed, **kwargs)
        n, nh = NUM_LEG_ACTUATORS, self.height_points.shape[0]
        self.noise_vec = jp.concatenate([
            jp.full(3, _base.NOISE["lin_vel"] * _base.LIN_VEL),
            jp.full(3, _base.NOISE["ang_vel"] * _base.ANG_VEL),
            jp.full(3, _base.NOISE["gravity"]), jp.zeros(3),
            jp.full(n, _base.NOISE["dof_pos"] * _base.DOF_POS),
            jp.full(n, _base.NOISE["dof_vel"] * _base.DOF_VEL),
            jp.zeros(n), jp.zeros(self.morphology.size), jp.zeros(1),
            jp.full(nh, _base.HEIGHT_NOISE * _base.HEIGHT_MEASUREMENTS),
        ])
        if self.noise_vec.size != OBSERVATION_SIZE:
            raise ValueError(f"Expected {OBSERVATION_SIZE} observations, got {self.noise_vec.size}")

    def observation(self, data, command, action, rng):
        lin_vel, ang_vel, gravity = self.base_state(data)
        heights = jp.clip(
            data.qpos[self.qadr + 2] - _base.HEIGHT_OFFSET - self.measured_heights(data), -1., 1.)
        obs = jp.concatenate([
            # Exact T5 order. Never insert new values inside this prefix.
            lin_vel * _base.LIN_VEL, ang_vel * _base.ANG_VEL, gravity,
            command[:3] * jp.array([_base.LIN_VEL, _base.LIN_VEL, _base.ANG_VEL]),
            (data.qpos[self.qids] - self.default_dof_pos) * _base.DOF_POS,
            data.qvel[self.vids] * _base.DOF_VEL, action, self.morphology,
            # New inputs.
            command[4:5], heights * _base.HEIGHT_MEASUREMENTS,
        ])
        if self.add_noise:
            obs += (2 * jax.random.uniform(rng, obs.shape) - 1) * self.noise_vec
        return jp.clip(obs, -_base.CLIP_OBSERVATIONS, _base.CLIP_OBSERVATIONS)


def make_training_env(groups, phase="flat"):
    env_cls = {"flat": FlatTaskEnv, "jump": FlatJumpEnv, "universal": TaskEnv}[phase]
    return MultiDesignEnv(groups, EPISODE_STEPS, env_cls=env_cls)
