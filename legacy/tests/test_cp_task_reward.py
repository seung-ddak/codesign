"""Small reward regression tests; run from codesign root with pytest.

No simulator or trained checkpoint required.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))

import torch
from task_reward import StraightReward, ZigzagReward


def test_straight_and_zigzag_have_distinct_objectives():
    # y=0 is perfect for straight motion, but not for the default zigzag.
    x = torch.zeros(1, 101, 54)
    x[0, :, 0] = torch.linspace(0.0, 4.0, 101)
    x[0, :, 3], x[0, :, 7] = 1.0, -1.0
    m = torch.ones(1, 101, dtype=torch.bool)
    straight = StraightReward(half_width=0.15)
    zigzag = ZigzagReward(amp=0.3, period=2.0, half_width=0.15)
    assert torch.allclose(straight.p_lane(x, m), torch.zeros(1))
    assert zigzag.p_lane(x, m).item() > 0.001


def test_zigzag_reference_path_receives_zero_lane_penalty():
    x = torch.zeros(1, 201, 54)
    x[0, :, 0] = torch.linspace(0.0, 4.0, 201)
    x[0, :, 3], x[0, :, 7] = 1.0, -1.0
    zigzag = ZigzagReward(amp=0.3, period=2.0, half_width=0.15)
    x[0, :, 1] = zigzag.y_ref(x[0, :, 0])
    assert zigzag.p_lane(x, torch.ones(1, 201, dtype=torch.bool)).item() < 1e-9
