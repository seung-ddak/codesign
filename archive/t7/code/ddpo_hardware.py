"""실제 보행 return으로 TT hardware denoiser 자체를 DDPO 학습한다."""

import math
import pickle
from pathlib import Path
import shutil
import tempfile

import hydra
import numpy as np
import torch
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from t2.eval.hardware_decoding import check_zarr_path, hardware_to_nearest_neighbor_params
from t2.eval.utils import extract_masked_hardware, traj2batch
from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.utils.misc import seed_everything


def ddpo_clipped_loss(current_log_prob, old_log_prob, advantage, clip_range):
    """PPO와 같은 DDPO-IS clipped objective와 기본 진단값을 계산한다."""
    log_ratio = (current_log_prob - old_log_prob).clamp(-20., 20.)
    ratio = log_ratio.exp()
    unclipped = ratio * advantage
    clipped = ratio.clamp(1. - clip_range, 1. + clip_range) * advantage
    loss = -torch.minimum(unclipped, clipped).mean()
    return loss, {
        "approx_kl": float((old_log_prob - current_log_prob).mean().detach()),
        "clip_fraction": float(((ratio - 1.).abs() > clip_range).float().mean().detach()),
    }


class DesignLearner:
    """100-step stochastic DDIM trajectory를 task return으로 직접 fine-tuning한다."""

    def __init__(self, tt, checkpoint, bank, batch_size, device, seed,
                 denoise_steps=100, ddpo_epochs=4, lr=1e-6, clip_range=0.2,
                 generation_microbatch=None, ddpo_minibatch=None, trajectory_dir=None):
        seed_everything(seed)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.generation_microbatch = generation_microbatch or batch_size
        self.ddpo_minibatch = ddpo_minibatch or batch_size
        self.denoise_steps = denoise_steps
        self.ddpo_epochs = ddpo_epochs
        self.clip_range = clip_range
        self._bank_view = tempfile.TemporaryDirectory(prefix="ddpo_bank_")
        self.bank = self._bank_view.name
        for path in bank.glob("*.zarr"):
            if len(path.stem) == 8 and set(path.stem) <= {"0", "1"}:
                (Path(self.bank) / path.name).symlink_to(path.resolve(), target_is_directory=True)

        with checkpoint.with_name("cfg.pkl").open("rb") as f:
            cfg = pickle.load(f)
        # 초기화 실패 시 checkpoint/cfg.pkl, CUDA 메모리, 128개 RoboToken bank 순서로 확인한다.
        OmegaConf.resolve(cfg.seq_len)
        self.decoder = setup_decoder(
            cfg, str(checkpoint), self.device, "hardware_gen",
            num_inference_steps=denoise_steps, eta=1.0, disable_grad=False,
            use_flash_attn=False, use_mixed_precision=False, use_torch_compile=False)
        self.model = self.decoder.model
        self.task = self.decoder.task_model
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

        with initialize_config_dir(config_dir=str(tt / "config"), version_base="1.3"):
            inference = compose(config_name="evaluate_hardware_opt_rl")
        inference.timestep_sampler.num_rollout_steps = int(cfg.seq_len.rollout_steps)
        sampler = hydra.utils.instantiate(inference.timestep_sampler)
        processor = hydra.utils.instantiate(cfg.datasets.clean.batch_process_fn)
        add_ids = next(x for x in processor.augmentations if isinstance(x, AddPositionId))
        poses = np.tile(np.eye(4, dtype=np.float32),
                        (int(cfg.seq_len.rollout_steps), int(cfg.seq_len.track_link), 1, 1))
        batch = add_ids(traj2batch(poses, cfg.seq_len, sampler, center_traj=False))
        batch["hardware_goal/mask"] = torch.ones_like(batch["target_pose/mask"])
        batch["target_pose/mask"] = torch.ones_like(batch["target_pose/mask"])
        self._base_batch = {k: v[:1].to(self.device) for k, v in batch.items()}
        self._batches = {}
        self.trajectory_dir = Path(trajectory_dir or self._bank_view.name) / "buffer"
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)

    def _batch(self, size):
        if size not in self._batches:
            self._batches[size] = {
                k: v.repeat(size, *([1] * (v.ndim - 1))) for k, v in self._base_batch.items()
            }
        return self._batches[size]

    def _state(self, size, seed=None):
        batch = self._batch(size)
        return self.task._init_denoising_state(
            batch, self.model.pos_embs, size, self.device, seed, False)

    def _predict_noise(self, samples, timesteps):
        size = len(timesteps)
        batch = self._batch(size)
        (conds, ins, masks, adapters, _, dont_update, slices, seq_len, _) = self._state(size)
        self.task._prepare_backbone_inputs(batch, self.model.pos_embs, samples, ins)
        t_emb = self.task.timestep_embedding(timesteps)
        self.task._apply_timestep(t_emb, size, ins, masks, conds, self.device)
        cond = self.task._build_cond(t_emb, conds)
        output = self.task._run_backbone(self.model.backbone, ins, masks, cond, seq_len)
        pred = {
            name: self.task.adapters[name].out_proj(x=output[:, slices[name]], cond=cond)
            for name in samples if self.task.adapters[name].loss_weight != 0.0
        }
        return pred, adapters, dont_update

    def _ddim_stats(self, sample, pred_noise, timesteps):
        scheduler = self.task.noise_scheduler
        step = scheduler.config.num_train_timesteps // scheduler.num_inference_steps
        previous = timesteps - step
        alpha_t = scheduler.alphas_cumprod.to(self.device)[timesteps]
        safe_previous = previous.clamp(min=0)
        alpha_previous = scheduler.alphas_cumprod.to(self.device)[safe_previous]
        alpha_previous = torch.where(
            previous >= 0, alpha_previous,
            scheduler.final_alpha_cumprod.to(self.device).expand_as(alpha_previous))
        shape = (-1,) + (1,) * (sample.ndim - 1)
        alpha_t = alpha_t.to(sample.dtype).reshape(shape)
        alpha_previous = alpha_previous.to(sample.dtype).reshape(shape)
        beta_t = 1. - alpha_t
        predicted_x0 = ((sample - beta_t.sqrt() * pred_noise) / alpha_t.sqrt()).clamp(-1., 1.)
        variance = ((1. - alpha_previous) / beta_t
                    * (1. - alpha_t / alpha_previous)).clamp(min=0.)
        std = variance.sqrt()  # eta=1
        direction = (1. - alpha_previous - variance).clamp(min=0.).sqrt() * pred_noise
        return alpha_previous.sqrt() * predicted_x0 + direction, std

    @staticmethod
    def _log_prob(next_samples, means, stds, update_masks):
        # 활성 차원 평균을 사용해 고차원 morphology의 importance ratio 포화를 막는다.
        first = next(iter(next_samples.values()))
        total = torch.zeros(first.shape[0], device=first.device)
        count = torch.zeros_like(total)
        constant = math.log(2. * math.pi)
        for name, value in next_samples.items():
            std = stds[name].clamp_min(1e-6)
            log_prob = -.5 * (((value - means[name]) / std).square()
                              + 2. * std.log() + constant)
            active = (~update_masks[name]).to(log_prob.dtype)
            total += (log_prob * active).flatten(1).sum(1)
            count += active.flatten(1).sum(1)
        return total / count.clamp_min(1.)

    def generate(self, seed):
        scheduler = self.task.noise_scheduler
        scheduler.set_timesteps(self.denoise_steps, device=self.device)
        designs = []
        trajectory = Path(tempfile.mkdtemp(prefix="update_", suffix=".zarr",
                                           dir=self.trajectory_dir))
        root = zarr.open_group(trajectory, mode="w")
        transition_count = len(scheduler.timesteps) - 1

        for offset in range(0, self.batch_size, self.generation_microbatch):
            size = min(self.generation_microbatch, self.batch_size - offset)
            micro_seed = seed + offset
            torch.manual_seed(micro_seed)
            batch = self._batch(size)
            (conds, ins, masks, adapters, samples, dont_update, slices,
             seq_len, generator) = self._state(size, micro_seed)
            names = [name for name in samples if self.task.adapters[name].loss_weight != 0.0]
            if offset == 0:
                root.attrs["names"] = names
                root.create_array("timesteps", shape=(transition_count,), dtype="i8")
                root.create_array("old_log_prob", shape=(self.batch_size, transition_count),
                                  chunks=(min(size, self.batch_size), min(32, transition_count)),
                                  dtype="f4")
                for name in names:
                    shape = samples[name].shape[1:]
                    root.create_array(f"states/{name}",
                                      shape=(self.batch_size, transition_count + 1, *shape),
                                      chunks=(1, 1, *shape), dtype="f4")

            transition = 0
            # 마지막 t=0은 분산이 0인 결정적 step이므로 DDPO 확률비에서 제외한다.
            with torch.no_grad():
                for scalar_t in scheduler.timesteps:
                    timesteps = scalar_t.expand(size)
                    self.task._prepare_backbone_inputs(batch, self.model.pos_embs, samples, ins)
                    t_emb = self.task.timestep_embedding(timesteps)
                    self.task._apply_timestep(t_emb, size, ins, masks, conds, self.device)
                    cond = self.task._build_cond(t_emb, conds)
                    output = self.task._run_backbone(
                        self.model.backbone, ins, masks, cond, seq_len)
                    pred = {name: self.task.adapters[name].out_proj(
                        x=output[:, slices[name]], cond=cond) for name in names}
                    means, stds, next_samples = {}, {}, {}
                    for name, sample in samples.items():
                        mean, std = self._ddim_stats(sample, pred[name], timesteps)
                        proposal = mean + std * torch.randn(
                            sample.shape, device=self.device, dtype=sample.dtype, generator=generator)
                        next_samples[name] = torch.where(dont_update[name], sample, proposal)
                        means[name], stds[name] = mean, std
                    if any(float(stds[name].max()) > 0. for name in names):
                        if transition == 0:
                            for name in names:
                                root[f"states/{name}"][offset:offset + size, 0] = \
                                    samples[name].float().cpu().numpy()
                        old_log_prob = self._log_prob(
                            {name: next_samples[name] for name in names},
                            {name: means[name] for name in names},
                            {name: stds[name] for name in names},
                            {name: dont_update[name] for name in names})
                        for name in names:
                            root[f"states/{name}"][offset:offset + size, transition + 1] = \
                                next_samples[name].float().cpu().numpy()
                        root["old_log_prob"][offset:offset + size, transition] = \
                            old_log_prob.float().cpu().numpy()
                        if offset == 0:
                            root["timesteps"][transition] = int(scalar_t)
                        transition += 1
                    samples = next_samples
                decoded = self.task._decode_samples(adapters, samples, clip_samples=True)

            if transition != transition_count:
                raise RuntimeError(f"Expected {transition_count} stochastic transitions, got {transition}")
            for i in range(size):
                item = {k: v[i].detach().cpu().numpy() for k, v in decoded.items()
                        if k.split("/")[0] in ("link", "actuator", "dyna_joint", "fixed_joint")}
                try:
                    hardware, *counts = extract_masked_hardware(item)
                    choices, hardware_seed = hardware_to_nearest_neighbor_params(
                        hardware, *counts, self.bank, local_only=True)
                    match = check_zarr_path(
                        hardware, *counts, Path(self.bank) / ("".join(map(str, choices)) + ".zarr"))
                    designs.append({"invalid": "No finite-distance hardware match"} if
                                   match is None or not np.isfinite(match[1]) else
                                   {"choices": choices, "seed": int(hardware_seed)})
                except ValueError as exc:
                    if str(exc) not in ("No matching archetype found", "No best seed found"):
                        raise
                    designs.append({"invalid": str(exc)})
        return designs, trajectory

    @staticmethod
    def _paired(array, designs, steps):
        """Read paired [design, step] entries while retaining tensor dimensions."""
        tail = np.indices(array.shape[2:], sparse=True)
        shape = (-1,) + (1,) * len(tail)
        coordinates = (designs.reshape(shape), steps.reshape(shape)) + tuple(
            index.reshape((1,) + index.shape) for index in tail)
        return array.vindex[coordinates]

    def update(self, trajectory, returns, epochs=None):
        rewards = torch.as_tensor(returns, device=self.device, dtype=torch.float32)
        if not torch.isfinite(rewards).all():
            raise ValueError("Non-finite rollout return")
        if not Path(trajectory).is_dir():
            raise ValueError("No stochastic DDIM transitions")
        spread = rewards.std(unbiased=False)
        # 동일 return은 advantage가 없다는 뜻이며 정상적인 안전 중단이다.
        if spread < 1e-6:
            shutil.rmtree(trajectory)
            return {"psi_updated": False, "reason": "equal rollout returns"}
        advantages = (rewards - rewards.mean()) / spread
        epochs = self.ddpo_epochs if epochs is None else epochs
        root = zarr.open_group(trajectory, mode="r")
        names = list(root.attrs["names"])
        transition_count = root["old_log_prob"].shape[1]
        total = self.batch_size * transition_count
        losses, kls, clips, grad_norms = [], [], [], []

        for _ in range(epochs):
            order = torch.randperm(total)
            for start in range(0, total, self.ddpo_minibatch):
                selected = order[start:start + self.ddpo_minibatch].numpy()
                design_indices = selected // transition_count
                step_indices = selected % transition_count
                samples = {name: torch.as_tensor(self._paired(
                    root[f"states/{name}"], design_indices, step_indices), device=self.device)
                    for name in names}
                next_samples = {name: torch.as_tensor(self._paired(
                    root[f"states/{name}"], design_indices, step_indices + 1), device=self.device)
                    for name in names}
                timesteps = torch.as_tensor(
                    root["timesteps"][step_indices], device=self.device, dtype=torch.long)
                old_log_prob = torch.as_tensor(
                    root["old_log_prob"].vindex[design_indices, step_indices], device=self.device)
                batch_advantages = advantages[torch.as_tensor(design_indices, device=self.device)]

                pred, _, update_masks = self._predict_noise(samples, timesteps)
                stats = {name: self._ddim_stats(samples[name], pred[name], timesteps)
                         for name in names}
                current_log_prob = self._log_prob(
                    next_samples, {name: value[0] for name, value in stats.items()},
                    {name: value[1] for name, value in stats.items()},
                    {name: update_masks[name] for name in names})
                loss, diagnostic = ddpo_clipped_loss(
                    current_log_prob, old_log_prob, batch_advantages, self.clip_range)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
                if not torch.isfinite(grad_norm):
                    raise ValueError("Non-finite DDPO gradient")
                self.optimizer.step()
                losses.append(float(loss.detach()))
                kls.append(diagnostic["approx_kl"])
                clips.append(diagnostic["clip_fraction"])
                grad_norms.append(float(grad_norm.detach()))

        result = {
            "psi_updated": True,
            "ddpo_loss": float(np.mean(losses)),
            "approx_kl": float(np.mean(kls)),
            "clip_fraction": float(np.mean(clips)),
            "gradient_norm": float(np.mean(grad_norms)),
            "transitions": total,
        }
        shutil.rmtree(trajectory)
        return result

    def save(self, path):
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "denoise_steps": self.denoise_steps,
            "ddpo_epochs": self.ddpo_epochs,
            "clip_range": self.clip_range,
        }, path)

    def load(self, path):
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
