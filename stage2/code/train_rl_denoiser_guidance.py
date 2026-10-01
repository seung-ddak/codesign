"""T7: keep the pretrained TT denoiser frozen and replace DSG with RL guidance.

The TT reverse process and scheduler stay unchanged. During every denoising step,
pi_phi supplies a residual where DSG supplied an objective gradient. Only pi_phi
and its critic learn from the final frozen-locomotion-policy return.
"""

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from eval_locomotion_policy import LAUNCH_DIR  # noqa: E402  (chdirs into TT)
from train_locomotion_policy import TT  # noqa: E402

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

import t2.eval  # noqa: E402

PATCH_EVAL = ROOT / "tt_patch/t2/eval"
if str(PATCH_EVAL) not in t2.eval.__path__:
    t2.eval.__path__.append(str(PATCH_EVAL))

from t2.eval.hardware_decoding import (  # noqa: E402
    check_zarr_path,
    hardware_to_nearest_neighbor_params,
)
from t2.eval.rl_guidance import (  # noqa: E402
    GuidanceCritic,
    RLResidualGuidance,
    ResidualGuidancePolicy,
)
from t2.eval.utils import extract_masked_hardware, traj2batch  # noqa: E402
from t2.model.t2 import setup_decoder  # noqa: E402
from t2.train.augment import AddPositionId  # noqa: E402
from t2.utils.misc import seed_everything  # noqa: E402

from design_scorer import DesignScorer, log_memory, memory_gib  # noqa: E402

PRIOR = TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt"
BANK = TT / "umi_on_legs_plus_plus_robotokens"


class RLGuidedDesignLearner:
    """Frozen TT denoising plus learned residual guidance on every step."""

    def __init__(
        self,
        checkpoint,
        bank,
        batch_size,
        device,
        seed,
        denoise_steps=100,
        guided_steps=100,
        action_scale=0.02,
        explore_std=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        critic_steps=100,
    ):
        seed_everything(seed)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.guided_steps = guided_steps
        self.critic_steps = critic_steps
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self._bank_view = tempfile.TemporaryDirectory(prefix="rl_guidance_bank_")
        self.bank = self._bank_view.name
        for path in bank.glob("*.zarr"):
            if len(path.stem) == 8 and set(path.stem) <= {"0", "1"}:
                (Path(self.bank) / path.name).symlink_to(
                    path.resolve(), target_is_directory=True
                )

        with checkpoint.with_name("cfg.pkl").open("rb") as file:
            cfg = pickle.load(file)
        OmegaConf.resolve(cfg.seq_len)
        self.decoder = setup_decoder(
            cfg,
            str(checkpoint),
            self.device,
            "hardware_gen",
            num_inference_steps=denoise_steps,
            eta=1.0,
            use_flash_attn=False,
            use_mixed_precision=False,
            use_torch_compile=False,
        )
        for parameter in self.decoder.model.parameters():
            parameter.requires_grad_(False)

        self.policy = ResidualGuidancePolicy(
            guided_steps=guided_steps, action_scale=action_scale
        ).to(self.device)
        self.guidance = RLResidualGuidance(
            self.policy,
            {},
            num_steps=denoise_steps,
            guided_steps=guided_steps,
            stochastic=True,
            explore_std=explore_std,
            record_transitions=True,
        )

        with initialize_config_dir(config_dir=str(TT / "config"), version_base="1.3"):
            inference = compose(config_name="evaluate_hardware_opt_rl")
        inference.timestep_sampler.num_rollout_steps = int(cfg.seq_len.rollout_steps)
        sampler = hydra.utils.instantiate(inference.timestep_sampler)
        processor = hydra.utils.instantiate(cfg.datasets.clean.batch_process_fn)
        add_ids = next(x for x in processor.augmentations if isinstance(x, AddPositionId))
        poses = np.tile(
            np.eye(4, dtype=np.float32),
            (int(cfg.seq_len.rollout_steps), int(cfg.seq_len.track_link), 1, 1),
        )
        batch = add_ids(traj2batch(poses, cfg.seq_len, sampler, center_traj=False))
        batch["hardware_goal/mask"] = torch.ones_like(batch["target_pose/mask"])
        batch["target_pose/mask"] = torch.ones_like(batch["target_pose/mask"])
        self.batch = {
            key: value.repeat(batch_size, *([1] * (value.ndim - 1))).to(self.device)
            for key, value in batch.items()
        }
        self.critic = self.actor_optimizer = self.critic_optimizer = None

    def generate(self, seed):
        torch.manual_seed(seed)
        self.guidance.reset()
        with torch.no_grad():
            decoded = self.decoder(self.batch, seed=seed, guidance_fn=self.guidance)
        transitions = self.guidance.transitions
        if len(transitions) != self.guided_steps:
            raise RuntimeError(
                f"Expected {self.guided_steps} RL guidance transitions, got {len(transitions)}"
            )

        if self.critic is None:
            dimensions = {
                term: value.shape[-1]
                for term, value in transitions[0]["states"].items()
            }
            self.critic = GuidanceCritic(
                dimensions, guided_steps=self.guided_steps
            ).to(self.device)
            self.actor_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self.actor_lr
            )
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.critic_lr
            )

        designs = []
        for index in range(self.batch_size):
            item = {
                key: value[index].detach().cpu().numpy()
                for key, value in decoded.items()
                if key.split("/")[0]
                in ("link", "actuator", "dyna_joint", "fixed_joint")
            }
            try:
                hardware, *counts = extract_masked_hardware(item)
                choices, hardware_seed = hardware_to_nearest_neighbor_params(
                    hardware, *counts, self.bank, local_only=True
                )
                match = check_zarr_path(
                    hardware,
                    *counts,
                    Path(self.bank) / ("".join(map(str, choices)) + ".zarr"),
                )
                designs.append(
                    {"invalid": "No finite-distance hardware match"}
                    if match is None or not np.isfinite(match[1])
                    else {"choices": choices, "seed": int(hardware_seed)}
                )
            except ValueError as error:
                if str(error) not in (
                    "No matching archetype found",
                    "No best seed found",
                ):
                    raise
                designs.append({"invalid": str(error)})
        return designs, transitions

    def update(self, transitions, returns):
        rewards = torch.as_tensor(
            returns, device=self.device, dtype=torch.float32
        )
        if not torch.isfinite(rewards).all():
            raise ValueError("Non-finite rollout return")
        spread = rewards.std(unbiased=False)
        if spread < 1e-6:
            return {"phi_updated": False, "reason": "equal rollout returns"}
        target = (rewards - rewards.mean()) / spread

        before = torch.cat(
            [parameter.detach().flatten().clone() for parameter in self.policy.parameters()]
        )
        critic_losses = []
        order = torch.randperm(len(transitions))
        for index in range(self.critic_steps):
            if index and index % len(transitions) == 0:
                order = torch.randperm(len(transitions))
            transition = transitions[int(order[index % len(transitions)])]
            step = torch.full(
                (len(rewards),),
                transition["k"],
                device=self.device,
                dtype=torch.long,
            )
            critic_loss = torch.nn.functional.mse_loss(
                self.critic(
                    transition["states"], transition["actions"], step
                ),
                target,
            )
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.0)
            self.critic_optimizer.step()
            critic_losses.append(float(critic_loss.detach()))

        for parameter in self.critic.parameters():
            parameter.requires_grad_(False)
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_losses = []
        for transition in transitions:
            step = torch.full(
                (len(rewards),),
                transition["k"],
                device=self.device,
                dtype=torch.long,
            )
            actions = {
                term: self.policy(term, state, transition["k"])[0]
                for term, state in transition["states"].items()
            }
            actor_term = -self.critic(transition["states"], actions, step).mean()
            (actor_term / len(transitions)).backward()
            actor_losses.append(float(actor_term.detach()))
        actor_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), 1.0
        )
        if not torch.isfinite(actor_gradient_norm):
            raise ValueError("Non-finite RL-guidance gradient")
        self.actor_optimizer.step()
        for parameter in self.critic.parameters():
            parameter.requires_grad_(True)

        after = torch.cat(
            [parameter.detach().flatten() for parameter in self.policy.parameters()]
        )
        return {
            "phi_updated": True,
            "critic_loss": float(np.mean(critic_losses)),
            "actor_loss": float(np.mean(actor_losses)),
            "actor_gradient_norm": float(actor_gradient_norm),
            "phi_parameter_delta": float(torch.linalg.vector_norm(after - before)),
            "guided_transitions": len(transitions) * len(rewards),
        }

    def save(self, path):
        torch.save(
            {
                "policy": self.policy,
                "critic": self.critic,
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
            },
            path,
        )


def self_check():
    torch.manual_seed(0)
    policy = ResidualGuidancePolicy(
        hidden=8, step_emb_dim=4, guided_steps=4, action_scale=0.02
    )
    guidance = RLResidualGuidance(
        policy,
        {},
        guidance_terms=("link",),
        num_steps=4,
        guided_steps=4,
        stochastic=True,
        explore_std=0.005,
        record_transitions=True,
    )
    state = {"link": torch.zeros(3, 5, 4)}
    outputs = [guidance({}, state) for _ in range(4)]
    assert len(guidance.transitions) == 4
    assert all(output["link"].shape == state["link"].shape for output in outputs)
    assert any(parameter.requires_grad for parameter in policy.parameters())
    print("TT denoiser + RL guidance self-check: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--pi-theta", type=Path, help="T5 run with frozen policy")
    parser.add_argument("--ckpt", help="pi_theta checkpoint step dir (default: latest)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--denoise-steps", type=int, default=100)
    parser.add_argument("--guided-steps", type=int, default=100)
    parser.add_argument("--critic-steps", type=int, default=100)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--action-scale", type=float, default=0.02)
    parser.add_argument("--explore-std", type=float, default=0.005)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.pi_theta is None or args.out is None:
        parser.error("--pi-theta and --out are required unless --self-check is used")
    if min(
        args.updates,
        args.batch,
        args.slots,
        args.denoise_steps,
        args.guided_steps,
        args.critic_steps,
        args.save_every,
    ) < 1:
        parser.error("counts and sizes must be positive")
    if args.guided_steps > args.denoise_steps:
        parser.error("guided-steps cannot exceed denoise-steps")
    if min(
        args.actor_lr,
        args.critic_lr,
        args.action_scale,
        args.explore_std,
    ) <= 0:
        parser.error("learning rates, action-scale and explore-std must be positive")

    run = (LAUNCH_DIR / args.pi_theta).resolve()
    out = (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = run / "checkpoints" / (
        args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1]
    )
    (out / "config.json").write_text(
        json.dumps(
            {
                **{key: str(value) for key, value in vars(args).items()},
                "pi_theta_ckpt": str(checkpoint),
                "tt_prior": str(PRIOR),
                "tt_denoiser_frozen": True,
                "guidance": "RL residual replaces DSG gradient",
                "bank": str(BANK),
            },
            indent=1,
        )
    )

    started = time.time()
    scorer = DesignScorer(
        json.loads((run / "designs.json").read_text()),
        checkpoint,
        args.slots,
    )
    learner = RLGuidedDesignLearner(
        PRIOR,
        BANK,
        args.batch,
        "cuda",
        args.seed,
        denoise_steps=args.denoise_steps,
        guided_steps=args.guided_steps,
        action_scale=args.action_scale,
        explore_std=args.explore_std,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        critic_steps=args.critic_steps,
    )
    print(f"setup {time.time() - started:.0f}s", flush=True)
    log_memory("after_setup")

    with (out / "history.jsonl").open("a") as log:
        for update in range(args.updates):
            started = time.time()
            designs, transitions = learner.generate(
                seed=args.seed * 100000 + update
            )
            valid = [
                index for index, design in enumerate(designs)
                if "invalid" not in design
            ]
            returns = np.full(len(designs), np.nan)
            if valid:
                returns[valid] = scorer.score(
                    [
                        (designs[index]["choices"], designs[index]["seed"])
                        for index in valid
                    ],
                    seed=update,
                )
            worst = np.nanmin(returns) if valid else 0.0
            rewards = np.where(np.isnan(returns), worst, returns)
            stats = learner.update(transitions, rewards.tolist())
            rss_gib, peak_rss_gib = memory_gib()
            row = {
                "update": update,
                "wall_s": round(time.time() - started, 1),
                "valid": len(valid),
                "rss_gib": round(rss_gib, 2),
                "peak_rss_gib": round(peak_rss_gib, 2),
                "compiled_groups": len(scorer.group_cache),
                "return_mean": float(np.nanmean(returns)) if valid else None,
                "return_max": float(np.nanmax(returns)) if valid else None,
                "designs": [
                    {
                        **design,
                        "return": None if np.isnan(value) else float(value),
                    }
                    for design, value in zip(designs, returns)
                ],
                **stats,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"update {update} valid {len(valid)}/{len(designs)} "
                f"return mean {row['return_mean']} max {row['return_max']} "
                f"{row['wall_s']}s",
                flush=True,
            )
            if (
                (update + 1) % args.save_every == 0
                or update + 1 == args.updates
            ):
                learner.save(out / f"phi_{update + 1:04d}.pt")


if __name__ == "__main__":
    main()
