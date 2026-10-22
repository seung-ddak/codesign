"""CPU tests for legacy joint gate on new fail-closed JSON summaries."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from check_joint_generation import evaluate


def test_skipped_null_results_are_failures():
    s = {"runs": {"scale_0": [{"initial_physics": {"physically_valid": False},
                                "execution_skipped": True, "mjx_fell_step": None,
                                "path_err_m": None},
                               {"initial_physics": {"physically_valid": True},
                                "execution_skipped": False, "mjx_fell_step": 20,
                                "path_err_m": 0.2}]}}
    out = evaluate(s)
    assert not out["passed"]
    assert out["rates"]["joint_valid"] == 0


def test_complete_candidate_can_pass():
    s = {"runs": {"scale_1": [{"initial_physics": {"physically_valid": True},
                                "execution_skipped": False, "mjx_fell_step": -1,
                                "path_err_m": 0.10}]}}
    assert evaluate(s)["passed"]


def test_missing_data_is_not_success():
    assert not evaluate({"runs": {"scale_0": [{}]}})["passed"]
    assert not evaluate({"runs": {}})["passed"]
