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
from coevolution_guidance import DesignLearner
from t2.eval.rl_guidance import GuidanceCritic, ResidualGuidancePolicy


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

    torch.manual_seed(42)
    learner = DesignLearner.__new__(DesignLearner)
    learner.device = "cpu"
    learner.policy = ResidualGuidancePolicy(guided_steps=1)
    states = {"link": torch.randn(4, 2, 3)}
    learner.policy("link", states["link"], 0)
    learner.critic = GuidanceCritic({"link": 3}, guided_steps=1)
    learner.actor_opt = torch.optim.Adam(learner.policy.parameters(), lr=1e-4)
    learner.critic_opt = torch.optim.Adam(learner.critic.parameters(), lr=1e-3)
    transitions = [{"k": 0, "states": states, "actions": {"link": torch.randn(4, 2, 3) * .005}}]
    assert not learner.update(transitions, [0.] * 4)["phi_updated"]
    result = learner.update(transitions, [0., .1, .2, .3], critic_steps=2, actor_steps=1)
    assert result["phi_updated"] and result["phi_parameter_delta"] > 0
    with tempfile.TemporaryDirectory() as tmp:
        learner.save(Path(tmp) / "phi.pt")
        assert (Path(tmp) / "phi.pt").stat().st_size > 0
        restored = DesignLearner.__new__(DesignLearner)
        restored.device = "cpu"
        restored.guidance = type("Guidance", (), {})()
        restored.load(Path(tmp) / "phi.pt")
        assert restored.guidance.policy is restored.policy
        assert all(torch.equal(a, b) for a, b in zip(learner.policy.parameters(),
                                                     restored.policy.parameters()))
        run = Path(tmp) / "run"
        run.mkdir()
        (run / "config.json").write_text(json.dumps({"rounds": 2, "num_envs": 128,
                                                       "out": str(run)}))
        finished = run / "round_0000"
        finished.mkdir()
        for name in ("theta.pkl", "phi.pt", "designs.json"):
            (finished / name).touch()
        row = {"round": 0, "checkpoint": str(finished)}
        (finished / "row.json").write_text(json.dumps(row))
        rows, latest = resume_state(run, Namespace(rounds=3, num_envs=64,
                                                   out=str(run), resume=True))
        assert rows == [row] and latest == finished
        assert len((run / "history.jsonl").read_text().splitlines()) == 1
    print("PASS: reward/termination, shared shapes, frozen arm actions, φ update and checkpoint")


if __name__ == "__main__":
    main()
