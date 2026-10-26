"""Offline regression tests for V7 100-step audit (no GPU / MuJoCo required)."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("eval_v7_ddim100", ROOT / "code/tt_dynamics/eval_v7_ddim100.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def example(e, score, predicted, *, complete=True):
    return {"E": e, "mjx_total": score, "pred_total": predicted, "mjx_progress": score,
            "pred_progress": predicted, "mjx_fell_step": -1 if complete else 12,
            "path_err_m": 0.02,
            "execution_skipped": False, "initial_physics": {"physically_valid": True}}


def test_rank_correlation():
    assert mod.rank_corr([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
    assert mod.rank_corr([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert mod.rank_corr([1, 1, 1], [1, 2, 3]) is None


def test_missing_data_is_not_success(tmp_path):
    outcome = mod.evaluate(tmp_path, "all", 10)
    assert outcome["smoke"]["status"] == "not_run"
    assert outcome["codesign"]["zigzag"]["status"] == "not_run"
    assert outcome["full"]["zigzag"]["status"] == "not_run"


def test_group_from_fixed_E_not_metadata():
    fixed = {"runs": {"scale_10": []}}
    for k in range(16):
        e = [0.5] * 13
        e[1] = round(0.1 + k * 0.05, 5)
        for _ in range(4):
            fixed["runs"]["scale_10"].append(example(e, float(k) / 16, float(k) / 16))
    joint = {"runs": {"scale_10": [example([0.5] * 13, 1, 1)]}}
    res = mod.analyze_fixed(fixed, joint, 10)
    assert res["status"] == "evaluated"
    assert res["n_shapes"] == 16
    assert res["samples_per_shape"] == 4
    assert res["spearman_shape_reward_rho"] == 1.0


def test_noncontiguous_morphology_rejected():
    e1, e2 = [0.5] * 13, [0.6] * 13
    fixed = {"runs": {"scale_10": [example(e, 1, 1) for e in (e1, e2, e1)]}}
    res = mod.analyze_fixed(fixed, {"runs": {"scale_10": []}}, 10)
    assert res["status"] == "invalid"


def test_skipped_does_not_count_as_valid():
    a = example([0.5] * 13, 0.3, 0.4)
    b = example([0.5] * 13, 0.0, 2.0)
    b["execution_skipped"] = True
    c = example([0.5] * 13, 0.0, 1.0)
    c["mjx_total"] = None
    d = mod.summarize([a, b, c])
    assert d["candidates"] == 3 and d["valid_mjx"] == 1
    assert d["completed"] == 1 and d["completion_rate"] == 1 / 3
