"""PD rollout file safety and source grouping; no simulator is run."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/motion_tracking"))
sys.path.insert(0, str(ROOT / "code/tt_dynamics"))
import collect_pd_rollouts as collector
from rollout_data import load, source_group_keys


def _data():
    qpos = np.zeros((2, 4, 19), np.float32)
    qpos[..., 3] = 1
    return dict(
        qpos=qpos, qvel=np.zeros((2, 4, 18), np.float32), contact=np.zeros((2, 4, 4), bool),
        action=np.zeros((2, 3, 12), np.float32), target_q=np.zeros((2, 3, 12), np.float32),
        valid=np.ones((2, 3), bool), clip=np.zeros((2, 3), int), ref_frame=np.tile(np.arange(3), (2, 1)),
        fell=np.zeros(2, bool), E=np.full(13, .5), motions=np.array(["walk"]), dt=.02,
        design="e000", mode="pd_reference", start_frame=np.array([0, 10]),
        group_id=collector.episode_group_ids(["walk", "walk"], [0, 10]),
    )


def test_group_ids_follow_motion_and_local_frame_not_row_or_clip_order():
    a = collector.episode_group_ids(["walk", "walk", "trot"], [0, 10, 0])
    b = collector.episode_group_ids(["trot", "walk", "walk", "walk"], [0, 5, 10, 0])
    ga = source_group_keys("e000/primitive_seed0.npz", {"group_id": a}, length=len(a))
    gb = source_group_keys("e001/primitive_seed0.npz", {"group_id": b}, length=len(b))
    assert ga[0] == gb[3] and ga[1] == gb[2] and ga[2] == gb[0]
    assert len(set(ga + gb)) == 4


def test_saved_rollout_is_complete_and_loads_in_tt(tmp_path):
    path = tmp_path / "e000/primitive_seed0.npz"
    collector.save_rollout(path, _data())
    assert collector.is_complete_rollout(path)
    x, e, mask, _, _ = load([path], 4)
    assert x.shape == (2, 4, 54) and e.shape == (2, 13)
    assert np.isfinite(x).all() and mask.sum() == 6
    assert list(path.parent.iterdir()) == [path]


@pytest.mark.parametrize("existing", [False, True])
def test_interrupted_write_never_publishes_partial_output(tmp_path, monkeypatch, existing):
    path = tmp_path / "primitive_seed0.npz"
    if existing:
        collector.save_rollout(path, _data())
    before = path.read_bytes() if existing else None

    def interrupted(file, **data):
        file.write(b"PK\x03\x04partial ZIP")
        raise KeyboardInterrupt

    monkeypatch.setattr(collector.np, "savez_compressed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        collector.save_rollout(path, _data())
    assert (path.read_bytes() if path.exists() else None) == before
    assert list(tmp_path.iterdir()) == ([path] if existing else [])


def test_resume_rejects_truncated_legacy_and_mismatched_arrays(tmp_path):
    path = tmp_path / "primitive_seed0.npz"
    assert not collector.is_complete_rollout(path)
    collector.save_rollout(path, _data())
    path.write_bytes(path.read_bytes()[:100])
    assert not collector.is_complete_rollout(path)
    data = _data()
    del data["group_id"]
    collector.save_rollout(path, data)
    assert not collector.is_complete_rollout(path)
    data = _data()
    data["target_q"] = data["target_q"][:, :2]
    collector.save_rollout(path, data)
    assert not collector.is_complete_rollout(path)


@pytest.mark.parametrize("offline", [False, True])
def test_completion_notification_saves_real_counts_and_keeps_report_on_failure(tmp_path, monkeypatch, offline):
    path = tmp_path / "e000/primitive_seed0.npz"
    data = _data()
    data["fell"] = np.array([False, True])
    collector.save_rollout(path, data)
    calls = []

    def post(endpoint, payload, token):
        assert endpoint == "/issues/2/comments" and token == "test-token"
        calls.append(payload["body"])
        if offline:
            raise OSError("offline")
        return {"html_url": "https://github.com/seung-ddak/codesign/issues/2#issuecomment-test"}

    monkeypatch.setattr(collector, "post", post)
    if offline:
        with pytest.raises(OSError, match="offline"):
            collector.notify_completion([path], tmp_path, 2, 120, "test-token")
    else:
        collector.notify_completion([path], tmp_path, 2, 120, "test-token")
    body = (tmp_path / "completion_comment.md").read_text()
    assert calls == [body]
    assert "@seung-ddak" in body and "2.0분" in body
    assert "| e000 | 1 | 2 | 6 | 1 |" in body
    assert "업로드는 하지 않았습니다" in body and "test-token" not in body
