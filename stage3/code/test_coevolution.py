"""Run directly: reward invariants, shared morphology interface and rollout-based φ updates."""

import tempfile
import json
from argparse import Namespace
from pathlib import Path

from train_coevolution import resume_state, setup_imports

setup_imports()

import jax
import jax.numpy as jp
import numpy as np
import torch

from coevolution_env import WalkEnv, walking_outcome
from ddpo_hardware import ddpo_clipped_loss


def main():
    reward, progress, fell, success = walking_outcome(0., .2, 0., 1., False, 1., .1)
    assert np.isclose(reward, .2) and not fell and not success
    reward, progress, fell, success = walking_outcome(.2, .4, 0., 1., True, 1., .1)
    assert reward == 0 and progress == .2 and fell and not success
    assert walking_outcome(0., 2., 0., .05, False, 1., .1)[2]
    assert not walking_outcome(.9, 1.1, 0., 1., True, 1., .1)[3]
    assert walking_outcome(.9, 1.1, 0., 1., False, 1., .1)[3]
    previous, total = 0., 0.
    for x in [.1, .3, .1, .3]:
        r, previous, _, _ = walking_outcome(previous, x, 0., 1., False, 1., .1)
        total += r
    assert np.isclose(total, .3), "Back-and-forth motion must not farm reward"
    assert walking_outcome(0., float("nan"), 0., 1., False, 1., .1)[2]

    shapes = []
    for choices, seed in [([0] * 8, 0), ([0, 0, 0, 1, 1, 0, 0, 1], 1298)]:
        env = WalkEnv(choices, seed)
        abstract = jax.eval_shape(env.reset, jax.random.PRNGKey(0))
        shapes.append((abstract.obs.shape, env.action_size))
        assert all("/arx5/" not in env.mj_model.actuator(int(i)).name for i in env.leg)
        assert np.asarray(env.bad_geoms).sum() >= 5
        jax.eval_shape(env.step, abstract, jp.zeros(env.action_size))
    assert shapes[0] == shapes[1] == ((109,), 19)

    current = torch.tensor([0., .1, -.1, .3], requires_grad=True)
    old = torch.zeros(4)
    advantage = torch.tensor([-1., -.5, .5, 1.])
    loss, diagnostic = ddpo_clipped_loss(current, old, advantage, .2)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(current.grad).all()
    assert 0. <= diagnostic["clip_fraction"] <= 1.
    with tempfile.TemporaryDirectory() as tmp:
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"rounds": 2, "num_envs": 128,
                                                       "out": str(run)}))
        finished = run / "round_0000"
        finished.mkdir()
        for name in ("theta.pkl", "psi.pt", "designs.json"):
            (finished / name).touch()
        row = {"round": 0, "checkpoint": str(finished)}
        (finished / "row.json").write_text(json.dumps(row))
        rows, latest = resume_state(run, Namespace(rounds=3, num_envs=64,
                                                   out=str(run), resume=True))
        assert rows == [row] and latest == finished
        assert len((run / "history.jsonl").read_text().splitlines()) == 1
    print("PASS: reward/termination, shared shapes, frozen arm actions, DDPO loss and checkpoint")


if __name__ == "__main__":
    main()
