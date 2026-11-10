"""CPU-only regression checks for explicit FAST PASS/FAIL reporting."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from check_fast_smoke import evaluate


def _fixture(tmp_path, *, valid=True, e_grad=True, s_grad=True, a_grad=True, projected=False):
    baseline = np.zeros((1, 100, 54), dtype=np.float32)
    guided = baseline.copy()
    guided[:, :, 2] = 0.1
    guided[:, :, 42] = 0.2
    np.savez_compressed(tmp_path / "scale_0.npz", E=np.full((1, 13), 0.5),
                        pred=baseline)
    e = np.full((1, 13), 0.5)
    e[:, 1] = 0.6
    np.savez_compressed(tmp_path / "scale_1.npz", E=e, pred=guided)

    def sample(physical, reward, *, invalid=False, projected=False):
        return {
            "E": [0.5] * 13,
            "physically_valid_s0": physical,
            "execution_skipped": invalid,
            "mjx_fell_step": None if invalid else -1,
            "mjx_valid_s": 0.0 if invalid else 2.0,
            "mjx_total": reward,
            "initial_physics": {"geometry_adjusted": projected,
                                "initial_support_mode": "grounded",
                                "preprojection": {"body_height_m": 0.3},
                                "executed_body_height_m": 0.32 if projected else 0.3,
                                "applied_body_z_shift_m": 0.02 if projected else 0.0},
        }

    def g(on):
        return [0.1] * 5 if on else [0.0] * 5

    report = {
        "task": "zigzag", "ckpt": "some-ckpt",
        "runs": {
            "scale_0": [sample(True, 0.5)],
            "scale_1": [sample(valid, 1.0 if valid else None, invalid=not valid,
                               projected=projected)],
        },
        "joint_full_guidance_audit": {
            "scale_1": {
                "diffusion_steps": 5,
                **{f"{k}_gradient_nonzero": flag
                   for k, flag in (("e", e_grad), ("s", s_grad), ("a", a_grad))},
                **{f"{k}_gradient_norm_per_step": g(flag)
                   for k, flag in (("e", e_grad), ("s", s_grad), ("a", a_grad))},
            }
        },
    }
    (tmp_path / "summary.json").write_text(json.dumps(report))


def test_all_four_checks_can_pass(tmp_path):
    _fixture(tmp_path)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert r["all_pass"], r
    assert r["guided_completed_MJX_count"] == 1
    assert all(r["checks"].values())


def test_rejected_start_cannot_be_counted_as_success(tmp_path):
    _fixture(tmp_path, valid=False)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert not r["all_pass"]
    assert not r["checks"]["at_least_one_valid_initial_state"]
    assert not r["checks"]["at_least_one_complete_MJX_open_loop"]
    assert r["guided_rejected_count"] == 1
    assert r["guided_rewards"] == [None]


def test_missing_state_or_action_gradient_fails(tmp_path):
    _fixture(tmp_path, s_grad=False, a_grad=False)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert r["checks"]["E_reward_gradient_nonzero"]
    assert not r["checks"]["S_reward_gradient_nonzero"]
    assert not r["checks"]["A_reward_gradient_nonzero"]
    assert not r["all_pass"]


def test_guidance_must_change_e_s_a(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "scale_1.npz"
    with np.load(path) as z:
        changed = z["pred"].copy()
        e = z["E"].copy()
    changed[..., 42:] = 0.0
    np.savez_compressed(path, E=e, pred=changed)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert not r["checks"]["guidance_changes_generated_A"]
    assert not r["all_pass"]


def test_geometry_projection_is_not_raw_generation_success(tmp_path):
    _fixture(tmp_path, projected=True)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert r["all_pass"]
    assert r["guided_completed_MJX_count"] == 1
    assert r["guided_completed_unmodified_MJX_count"] == 0
    assert r["guided_completed_projection_assisted_MJX_count"] == 1
    assert not r["strict_raw_TT_to_MJX_verified"]


def test_unmodified_completion_is_strict_raw_success(tmp_path):
    _fixture(tmp_path, projected=False)
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert r["all_pass"]
    assert r["guided_completed_unmodified_MJX_count"] == 1
    assert r["strict_raw_TT_to_MJX_verified"]


def test_missing_provenance_never_claims_raw_generation(tmp_path):
    _fixture(tmp_path)
    p = tmp_path / "summary.json"
    d = json.loads(p.read_text())
    del d["runs"]["scale_1"][0]["initial_physics"]
    p.write_text(json.dumps(d))
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert r["all_pass"]  # MJX replay history still says completion
    assert not r["strict_raw_TT_to_MJX_verified"]
    assert r["guided_completed_unmodified_MJX_count"] == 0
    assert r["guided_completed_unknown_provenance_MJX_count"] == 1


def test_inconsistent_shift_cannot_pass_raw_provenance(tmp_path):
    _fixture(tmp_path)
    p = tmp_path / "summary.json"
    d = json.loads(p.read_text())
    d["runs"]["scale_1"][0]["initial_physics"]["applied_body_z_shift_m"] = 0.01
    p.write_text(json.dumps(d))
    r = evaluate(tmp_path, guided_scale=1.0, seconds=2.0)
    assert not r["strict_raw_TT_to_MJX_verified"]
