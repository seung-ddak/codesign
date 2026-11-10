"""Small NPZ fixtures: source-based train/val grouping and stable cached array load."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from rollout_data import load, source_group_keys


def _file(path, *, with_group=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    q = np.zeros((2, 4, 19), np.float32)
    q[..., 2] = 0.3
    q[..., 3] = 1.0
    v = np.zeros((2, 4, 18), np.float32)
    contact = np.zeros((2, 4, 4), bool)
    target = np.zeros((2, 4, 12), np.float32)
    valid = np.ones((2, 4), bool)
    data = dict(qpos=q, qvel=v, contact=contact, target_q=target,
                valid=valid, E=np.full(13, 0.5), motions=np.array(["walk"]),
                clip=np.zeros((2, 4), int))
    if with_group:
        data["group_id"] = np.array([18, 19])
    np.savez_compressed(path, **data)


def test_cross_shape_source_row_is_same_group(tmp_path):
    a = tmp_path / "e000" / "primitive_seed0.npz"
    b = tmp_path / "e001" / "primitive_seed0.npz"
    _file(a)
    _file(b)
    with np.load(a) as za, np.load(b) as zb:
        ga = source_group_keys(a, za, length=2)
        gb = source_group_keys(b, zb, length=2)
    assert ga == gb == ["source:primitive_seed0:0", "source:primitive_seed0:1"]
    x, e, mask, src, clips = load([a, b], 4)
    assert x.shape == (4, 4, 54)
    assert e.shape == (4, 13)
    assert mask.all()
    assert (clips[:, :4] == "walk").all()


def test_explicit_intervention_group_across_files(tmp_path):
    a = tmp_path / "f000" / "first.npz"
    b = tmp_path / "f001" / "other.npz"
    _file(a, with_group=True)
    _file(b, with_group=True)
    with np.load(a) as za, np.load(b) as zb:
        assert source_group_keys(a, za, length=2) == source_group_keys(b, zb, length=2)
