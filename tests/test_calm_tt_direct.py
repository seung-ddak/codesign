"""Small pure-Python checks for CALM -> TT e000 transfer bookkeeping."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/motion_tracking"))
import check_calm_tt_direct as m


def test_calms_23_body_slots_have_explicit_tt_sources():
    mapping = m.body_mapping()
    assert len(m.CALM_BODIES) == 23
    assert set(mapping) == set(m.CALM_BODIES)
    assert mapping["base"] == mapping["trunk"]
    assert mapping["FL_foot"].endswith("left_leg/foot")
    assert mapping["FR_foot"].endswith("right_leg/foot")
    assert mapping["RL_foot"].endswith("left_leg_1/foot")
    assert mapping["RR_foot"].endswith("right_leg_1/foot")


def test_transition_metrics_keep_t_plus_one_states_and_distinguish_tracking():
    n = 300
    qpos = np.zeros((n + 1, 19))
    qpos[:, 3] = 1
    qpos[:, 2] = .35
    qvel = np.zeros((n + 1, 18))
    command = np.zeros((n, 3))
    foot = np.zeros((n + 1, 4, 3))
    contact = np.ones((n + 1, 4), dtype=bool)
    good = m.metrics(qpos, qvel, contact, foot, command, 1 / 30, "duration", .15)
    assert good["survived_10s"] and good["screening_pass"]
    assert good["transitions"] == 300
    bad = m.metrics(qpos, qvel, contact, foot, command, 1 / 30, "trunk_fall", .15)
    assert not bad["survived_10s"] and not bad["screening_pass"]
    moving_commands = np.ones_like(command)
    poor = m.metrics(qpos, qvel, contact, foot, moving_commands, 1 / 30, "duration", .15)
    assert poor["survived_10s"] and not poor["screening_pass"]


def test_observation_names_and_actions_are_in_calm_order():
    assert m.LEGS == ("FL", "FR", "RL", "RR")
    assert m.CALM_BODIES[0:2] == ["base", "trunk"]
    assert m.CALM_BODIES[-1] == "imu_link"


def test_default_wheel_uses_the_organized_experiment_directory():
    assert m.DEFAULT_WHEEL.relative_to(m.ROOT).as_posix() == (
        "runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl"
    )


@pytest.mark.parametrize("option", ["--speed-threshold", "--torque-limit"])
def test_cli_rejects_nonfinite_limits(option, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["check_calm_tt_direct.py", "--out", str(tmp_path / "new"), option, "nan"])
    with pytest.raises(SystemExit) as exc:
        m.main()
    assert exc.value.code == 2
    assert "Invalid torque or speed threshold" in capsys.readouterr().err
