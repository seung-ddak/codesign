"""Pure CPU-level regression checks for shared-E optimizer and Mode 5 masks."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))

import torch
from shared_e_guidance import optimize_shared_e
from tt_model import condition_masks, rest_s0


def test_mode5_keeps_e_and_rest_only():
    valid = torch.ones(2, 32, dtype=torch.bool)
    kx, ke = condition_masks(torch.tensor([4, 5]), valid)
    assert not ke[0].any() and ke[1].all()
    assert torch.equal(kx[0], kx[1])
    assert (kx[:, 0, 2] == 0).all()
    assert (kx[:, 0, 14:26] == 0).all()
    assert (kx[:, 0, 38:42] == 1).all()
    assert not kx[:, 1:, :].any()
    s0 = rest_s0()
    assert (s0[38:42] == 1).all()


def test_shared_e_optimizes_complete_objective_and_preserves_inert_dims():
    torch.manual_seed(0)
    E = torch.full((8, 13), 0.5)
    target = torch.full_like(E, 0.82)
    target[:, [0, 11, 12]] = 0.5

    def task_score(e):
        return -((e - target) ** 2).sum(dim=1)

    initial = task_score(E)
    final_e, scores, log = optimize_shared_e(
        E, task_score, steps=24, step_size=0.14, perturb=0.04, seed=5)
    assert len(log) == 25
    assert torch.all(scores >= initial)
    assert (scores > initial).any()
    assert torch.allclose(final_e[:, [0, 11, 12]], torch.full((8, 3), 0.5))
    assert ((final_e >= 0.05) & (final_e <= 0.95)).all()


def test_shared_e_zero_steps_is_unoptimized_baseline():
    E = torch.full((2, 13), 0.4)
    e, scores, log = optimize_shared_e(E, lambda x: x.sum(1), steps=0)
    assert (e[:, [0, 11, 12]] == 0.5).all()
    assert len(log) == 1
