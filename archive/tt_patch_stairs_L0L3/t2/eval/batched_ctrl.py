"""설계 N 개를 lockstep 으로 굴리며 ctrl 트랜스포머 forward 를 배치한다.

설계 탐색(L0~L3)은 rollout 을 수만 번 돌려야 하는데, 단일 env in-process 는
11.7 step/s 로 에피소드 1 개에 72 초가 걸린다 (`ctrl_rollout_direct.py`).

실측한 병목 구조 (V100, `d7lra7j0` 66.3 M):

| 항목 | 배치 64 기준 1 step |
|------|---------------------|
| 트랜스포머 (denoising 2 회) | 약 1114 ms |
| 물리 (CPU MuJoCo 64 개) | 약 84 ms (7 %) |

즉 **물리는 병목이 아니므로 멀티프로세싱이 필요 없고**, 트랜스포머만 배치하면 된다.
denoising 은 5 → 2 회로 줄여도 결과가 사실상 동일하다 (이동 1.025 vs 1.027 m).
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import copy
import pickle
import time
from typing import Callable, Optional

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from t2.data.dataset import TimestepSampler  # noqa: F401  (hydra 인스턴스화용)
from t2.eval.ctrl import env_obs_to_batch
from t2.model.t2 import setup_decoder
from t2.robogen.umi_on_legs_plus_plus import umi_on_legs_plus_plus_from_params
from t2.train.augment import AddPositionId

CFG_DIR = f"{_TT}/config"
CHOICES = [0] * 8


class BatchedCtrlRollout:
    def __init__(
        self,
        ckpt: str,
        env_name: str,
        pkl: str,
        n_envs: int = 64,
        inference_steps: int = 2,
        episode_steps: int = 300,
        device: Optional[torch.device] = None,
    ):
        self.n_envs = n_envs
        self.episode_steps = episode_steps
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        policy_cfg = pickle.load(open(ckpt.rsplit("/", 1)[0] + "/cfg.pkl", "rb"))
        OmegaConf.resolve(policy_cfg.ctrl_seq_len)
        self.seq_len_cfg = OmegaConf.to_container(policy_cfg.ctrl_seq_len, resolve=True)
        self.offsets = OmegaConf.to_container(
            policy_cfg.datasets.ctrl_rand.dataset.group_time_offsets, resolve=True
        )
        self.num_past = -self.offsets.get("ctrl", 0)
        self.rollout_steps = self.seq_len_cfg["rollout_steps"]

        self.batch_process_fn = hydra.utils.instantiate(
            policy_cfg.datasets.ctrl_rand.batch_process_fn
        )
        self.batch_process_fn.augmentations = [
            a for a in self.batch_process_fn.augmentations if type(a) is AddPositionId
        ]
        self.decoder = setup_decoder(
            policy_cfg,
            ckpt_path=ckpt,
            device=self.device,
            task_name="ctrl",
            num_inference_steps=inference_steps,
            num_repeats_per_step=1,
            clip_samples_in_guidance=True,
            deterministic=False,
            use_flash_attn=False,  # V100 에는 sm_80+ 전용 flash kernel 이 없다
            use_torch_compile=False,
            use_mixed_precision=False,  # V100 은 bfloat16 미지원
        )

        overrides = [
            "ckpt_path=x",
            f"env@eval_fn.runner.env={env_name}",
            f"eval_fn.runner.env.robot_generator.choices={CHOICES}",
            "eval_fn.runner.env.robot_generator.num_uniforms=13",
            f"eval_fn.runner.env.pickle_path={pkl}",
        ]
        with initialize_config_dir(config_dir=CFG_DIR, version_base="1.3"):
            cfg = compose(config_name="evaluate_ctrl", overrides=overrides)
        self.envs = [
            hydra.utils.instantiate(cfg.eval_fn.runner.env) for _ in range(n_envs)
        ]
        obs_time = np.arange(self.rollout_steps) + self.offsets.get("target_pose", 0)
        for env in self.envs:
            env.obs_time_indices = obs_time
            env.include_obs_time_indices = "relative"

    @staticmethod
    def _design_generator(uniforms: np.ndarray) -> Callable:
        """seed 대신 주어진 uniforms 로 로봇을 만드는 generator."""

        def gen(_seed, _u=list(map(float, uniforms))):
            return umi_on_legs_plus_plus_from_params(
                choices=copy.deepcopy(CHOICES), uniforms=_u
            )

        return gen

    def _reshape_obs(self, env, obs):
        out = {}
        for k, v in obs.items():
            if getattr(v, "ndim", 0) == 2:
                if k.split("/")[0] == "target_pose":
                    v = v.reshape(-1, env.n_end_effectors, v.shape[-1])
                else:
                    v = v[None, :, :]
            out[k] = v
        return out

    def run(
        self,
        designs: np.ndarray,
        traj_idx: int = 0,
        seed: int = 0,
    ) -> list[dict]:
        """(B, 13) 설계 -> 설계별 rollout 결과. B 는 n_envs 이하."""
        n = len(designs)
        assert n <= self.n_envs, f"{n} designs > {self.n_envs} envs"
        envs = self.envs[:n]

        t_setup = time.time()
        obs_list, hw_list = [], []
        for env, design in zip(envs, designs):
            env.robot_generator = self._design_generator(design)
            env.reset_hardware(seed)
            obs, _, hw = env.reset_sim_and_target_traj(
                seed=seed, target_traj_idx=traj_idx
            )
            obs_list.append(obs)
            hw_list.append(hw)
        setup_s = time.time() - t_setup

        prev_actions = [np.zeros(env.d.ctrl.shape[0]) for env in envs]
        alive = np.ones(n, dtype=bool)
        start_x = np.array([float(env.d.qpos[0]) for env in envs])
        max_x = start_x.copy()
        steps_done = np.zeros(n, dtype=int)
        pos_err_sum = np.zeros(n)

        t0 = time.time()
        for step in range(self.episode_steps):
            idx = np.flatnonzero(alive)
            if len(idx) == 0:
                break
            batch = env_obs_to_batch(
                obs_list=[
                    {**self._reshape_obs(envs[i], obs_list[i]), **hw_list[i]}
                    for i in idx
                ],
                prev_action_list=[prev_actions[i] for i in idx],
                seq_len_cfg=self.seq_len_cfg,
                device=self.device,
                group_time_offsets=self.offsets,
                batch_process_fn=self.batch_process_fn,
            )
            with torch.inference_mode():
                out = self.decoder(batch)
            actions = (
                out["ctrl/target_qpos"]
                .float()
                .cpu()
                .numpy()
                .reshape(len(idx), self.rollout_steps, self.seq_len_cfg["actuator"])
            )
            for j, i in enumerate(idx):
                env = envs[i]
                action = actions[j, self.num_past][: env.d.ctrl.shape[0]]
                prev_actions[i] = action
                obs_list[i], _, done, _ = env.step(action)
                max_x[i] = max(max_x[i], float(env.d.qpos[0]))
                pos_err_sum[i] += float(np.max(env.curr_pos_err))
                steps_done[i] += 1
                if done:
                    alive[i] = False
        rollout_s = time.time() - t0

        return [
            dict(
                design=designs[i].tolist(),
                steps=int(steps_done[i]),
                completed=bool(steps_done[i] >= self.episode_steps),
                distance=float(max_x[i] - start_x[i]),
                max_x=float(max_x[i]),
                pos_err_mean=float(pos_err_sum[i] / max(steps_done[i], 1)),
                setup_s=round(setup_s / n, 2),
                rollout_s=round(rollout_s / n, 2),
            )
            for i in range(n)
        ]
