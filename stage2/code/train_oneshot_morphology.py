"""Train a one-shot morphology policy against the frozen T5 locomotion policy.

The policy samples one valid topology plus 13 bounded generator parameters per episode.
No TT decoder, RoboToken bank projection, or denoising trajectory is used.
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.distributions import Beta, Categorical, Independent

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_locomotion_policy import LAUNCH_DIR  # noqa: E402  (chdirs into TT)
from train_locomotion_policy import TT  # noqa: E402

import numpy as np  # noqa: E402

from ddpo_hardware import ddpo_clipped_loss  # noqa: E402
from design_scorer import DesignScorer  # noqa: E402
from locomotion_env import LocomotionEnv  # noqa: E402
from t2.robogen.components import enumerate_choices  # noqa: E402
from t2.robogen.umi_on_legs_plus_plus import COMPONENTS  # noqa: E402

NUM_UNIFORMS = 13


class MorphologyPolicy(nn.Module):
    """One stochastic action: valid topology index plus bounded continuous parameters."""

    def __init__(self, choices, latent_size=32, hidden_size=128):
        super().__init__()
        self.register_buffer("choices", torch.as_tensor(choices, dtype=torch.int64))
        self.backbone = nn.Sequential(nn.Linear(latent_size, hidden_size), nn.Tanh(),
                                      nn.Linear(hidden_size, hidden_size), nn.Tanh())
        self.head = nn.Linear(hidden_size, len(choices) + 2 * NUM_UNIFORMS)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)  # uniform topology and Beta(1, 1) at update zero

    def distributions(self, latent):
        output = self.head(self.backbone(latent))
        logits, concentrations = output.split((len(self.choices), 2 * NUM_UNIFORMS), dim=-1)
        alpha, beta = concentrations.chunk(2, dim=-1)
        continuous = Independent(Beta(alpha.clamp(-6., 6.).exp(),
                                      beta.clamp(-6., 6.).exp()), 1)
        return Categorical(logits=logits), continuous

    def sample(self, latent):
        topology, continuous = self.distributions(latent)
        indices, uniforms = topology.sample(), continuous.sample()
        log_prob = topology.log_prob(indices) + continuous.log_prob(uniforms)
        return indices, uniforms, log_prob


def valid_choices():
    choices = np.asarray([choice for choice, count in enumerate_choices(COMPONENTS)
                          if len(choice) == 8 and set(choice) <= {0, 1} and count == NUM_UNIFORMS])
    if choices.shape != (128, 8):
        raise ValueError(f"Expected 128 eight-bit TT quadruped choices, got {choices.shape}")
    return choices


def self_check():
    choices = valid_choices()
    policy = MorphologyPolicy(choices, latent_size=4, hidden_size=8)
    latent = torch.randn(5, 4)
    indices, uniforms, log_prob = policy.sample(latent)
    assert indices.shape == (5,) and uniforms.shape == (5, NUM_UNIFORMS)
    assert torch.isfinite(log_prob).all() and ((uniforms > 0.) & (uniforms < 1.)).all()
    direct = np.random.RandomState(0).uniform(size=NUM_UNIFORMS)
    seeded = LocomotionEnv(choices[0].tolist(), seed=0, add_noise=False)
    explicit = LocomotionEnv(choices[0].tolist(), uniforms=direct, add_noise=False)
    assert np.array_equal(np.asarray(seeded.morphology), np.asarray(explicit.morphology))
    assert np.array_equal(seeded.mj_model.geom_size, explicit.mj_model.geom_size)
    print("one-shot morphology self-check: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--pi-theta", type=Path, help="T5 run with frozen policy")
    parser.add_argument("--ckpt", help="pi_theta checkpoint step dir (default: latest)")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--slots", type=int, default=1)
    parser.add_argument("--latent-size", type=int, default=32)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--ppo-epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    if args.pi_theta is None or args.out is None:
        parser.error("--pi-theta and --out are required unless --self-check is used")
    if min(args.updates, args.batch, args.slots, args.latent_size, args.hidden_size,
           args.ppo_epochs, args.save_every) < 1:
        parser.error("sizes, updates, epochs, slots and save-every must be positive")
    if args.lr <= 0 or not 0 < args.clip_range < 1 or args.entropy_coef < 0:
        parser.error("lr must be positive, clip-range in (0, 1), entropy-coef non-negative")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run, out = (LAUNCH_DIR / args.pi_theta).resolve(), (LAUNCH_DIR / args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    checkpoint = run / "checkpoints" / (args.ckpt or sorted(os.listdir(run / "checkpoints"))[-1])
    choices = valid_choices()
    policy = MorphologyPolicy(choices, args.latent_size, args.hidden_size)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)
    scorer = DesignScorer(json.loads((run / "designs.json").read_text()), checkpoint, args.slots)
    (out / "config.json").write_text(json.dumps({
        **{key: str(value) for key, value in vars(args).items()},
        "pi_theta_ckpt": str(checkpoint), "topologies": len(choices),
        "morphology": "one valid choice + 13 direct uniforms",
    }, indent=1))

    with (out / "history.jsonl").open("a") as log:
        for update in range(args.updates):
            started = time.time()
            latent = torch.randn(args.batch, args.latent_size)
            with torch.no_grad():
                indices, uniforms, old_log_prob = policy.sample(latent)
            designs = [(choices[index].tolist(), value.tolist()) for index, value in
                       zip(indices.tolist(), uniforms.numpy())]
            returns = torch.as_tensor(scorer.score(designs, seed=update), dtype=torch.float32)
            if not torch.isfinite(returns).all():
                raise ValueError("Non-finite frozen-policy return")
            spread = returns.std(unbiased=False)
            stats = {"phi_updated": False, "reason": "equal rollout returns"}
            if spread >= 1e-6:
                advantage = (returns - returns.mean()) / spread
                losses, entropies, diagnostics = [], [], []
                for _ in range(args.ppo_epochs):
                    topology, continuous = policy.distributions(latent)
                    log_prob = topology.log_prob(indices) + continuous.log_prob(uniforms)
                    policy_loss, diagnostic = ddpo_clipped_loss(
                        log_prob, old_log_prob, advantage, args.clip_range)
                    entropy = (topology.entropy() / math.log(len(choices))
                               + continuous.entropy() / NUM_UNIFORMS).mean()
                    loss = policy_loss - args.entropy_coef * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.)
                    optimizer.step()
                    losses.append(float(policy_loss.detach()))
                    entropies.append(float(entropy.detach()))
                    diagnostics.append(diagnostic)
                stats = {"phi_updated": True, "policy_loss": float(np.mean(losses)),
                         "entropy": float(np.mean(entropies)),
                         "approx_kl": float(np.mean([x["approx_kl"] for x in diagnostics])),
                         "clip_fraction": float(np.mean([x["clip_fraction"] for x in diagnostics])),
                         "gradient_norm": float(grad_norm)}

            row = {"update": update, "wall_s": round(time.time() - started, 1),
                   "return_mean": float(returns.mean()), "return_max": float(returns.max()),
                   "unique_topologies": int(indices.unique().numel()),
                   "designs": [{"choices": c, "uniforms": u, "return": float(r)}
                               for (c, u), r in zip(designs, returns)], **stats}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"update {update} return mean {row['return_mean']:.3f} "
                  f"max {row['return_max']:.3f} {row['wall_s']}s", flush=True)
            if (update + 1) % args.save_every == 0 or update + 1 == args.updates:
                torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(),
                            "choices": choices, "args": vars(args)},
                           out / f"phi_{update + 1:04d}.pt")


if __name__ == "__main__":
    main()
