import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code" / "motion_tracking"))
from shared_ppo_gate import assess


def row(design, survived=True, body=0.08, joint=7., orientation=5.):
    return dict(design=design, train_motion=True, survived_10s=survived,
                body_position_rmse_m=body, joint_rmse_deg=joint,
                orientation_mean_deg=orientation)


def test_all_designs_must_pass():
    rows = [row("short") for _ in range(12)] + [row("long", body=.3) for _ in range(12)]
    out = assess(rows, ["short", "long"])
    assert not out["passed"]
    assert out["designs"]["short"]["passed"]
    assert not out["designs"]["long"]["passed"]


def test_missing_trials_and_survival_fail_closed():
    assert not assess([row("short")], ["short"])["passed"]
    rows = [row("short", survived=(i == 0)) for i in range(12)]
    out = assess(rows, ["short"])
    assert not out["passed"]
    assert out["designs"]["short"]["survival_rate"] == 1 / 12


def test_survivors_only_and_permissive_complete():
    rows = [row("short", survived=(i != 0),
                body=9. if i == 0 else .08) for i in range(13)]
    out = assess(rows, ["short"])
    assert out["passed"]
    assert out["designs"]["short"]["survivor_means"]["body_position_rmse_m"] == .08


def test_nonfinite_values_never_promote():
    rows = [row("short", body=float("nan")) for _ in range(12)]
    result = assess(rows, ["short"])
    assert not result["passed"]
    assert any("invalid" in issue for issue in result["designs"]["short"]["issues"])


def test_held_out_motions_do_not_count():
    rows = [row("short") for _ in range(12)] + [dict(row("short", survived=False), train_motion=False)]
    out = assess(rows, ["short"])
    assert out["passed"]
    assert out["designs"]["short"]["n"] == 12
