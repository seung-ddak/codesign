"""Unit tests for the speed-first CPU MuJoCo experiment contract.

Pure-Python tests do not claim that V100 physics or learning has been validated.
"""
import ast
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/motion_tracking"))


def test_new_experiment_scripts_parse():
    for path in ("code/motion_tracking/collect_calm_mujoco.py",
                 "code/tt_dynamics/generate_task_mujoco.py"):
        ast.parse((ROOT / path).read_text())


def test_cpu_reference_interpolates_30hz_to_50hz():
    from collect_calm_mujoco import reference_targets
    ref = {"fix_q": np.repeat(np.arange(100)[:, None], 12, axis=1)}
    a = reference_targets(ref, start=0, count=4)
    assert a.shape == (4, 12)
    np.testing.assert_allclose(a[:, 0], [0, .6, 1.2, 1.8])
    b = reference_targets(ref, start=10, count=3)
    np.testing.assert_allclose(b[:, 0], [10, 10.6, 11.2])


def test_floor_contact_only_and_four_channel_mask():
    from collect_calm_mujoco import contact_flags
    contacts = [
        SimpleNamespace(geom1=1, geom2=5, dist=-.001),
        SimpleNamespace(geom1=6, geom2=1, dist=0),
        SimpleNamespace(geom1=7, geom2=8, dist=-.01),
    ]
    data = SimpleNamespace(contact=contacts, ncon=len(contacts))
    flags = contact_flags(None, data, [5, 6, 7, 8], 1)
    np.testing.assert_array_equal(flags, [1., 1., 0., 0.])


def test_tt_mode_requirements_and_strict_source_coverage():
    text = (ROOT / "code/motion_tracking/collect_calm_mujoco.py").read_text()
    assert "morphologies_per_motion=2" in text
    assert "source_motions=10000" in text
    assert "train_designs=200" in text
    assert '"--modes", "1", "2", "4"' in text
    assert '"--val-frac", ".2"' in text
    assert 'len(source)==10000' in text
    assert 'torch.cuda' not in text
    assert "mujoco.mj_step(model, d, nstep=4)" in text


def test_task_generator_uses_cpu_mujoco_and_gradients():
    text = (ROOT / "code/tt_dynamics/generate_task_mujoco.py").read_text()
    assert "joint_diffusion_full_reward(" in text
    assert "mujoco.mj_step(model, data, nstep=4)" in text
    assert 'choices=("zigzag","circle","straight")' in text
    assert "from mujoco import mjx" not in text
