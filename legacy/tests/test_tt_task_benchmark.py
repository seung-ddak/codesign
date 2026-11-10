"""Lightweight CPU regressions for episodic PPO and report compilation."""
import base64
import sys
import json
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
from tt_rl_residual_ppo import EpisodicResidualPPO, ppo_update
from tt_task_report import build_report


def test_rl_residual_has_correct_e_s_a_shapes_and_safe_anchor():
    p = EpisodicResidualPPO(num_windows=8)
    raw, old_lp = p.draw(3)
    assert raw.shape == (3, 13 + 8 * 16)
    assert old_lp.shape == (3,)
    assert torch.isfinite(old_lp).all()
    x = [torch.randn(3, 64, 54) for _ in range(8)]
    ce, cx = p.residual_fn(raw)(x, torch.randn(3, 13), 4)
    assert ce.shape == (3, 13)
    assert len(cx) == 8 and all(a.shape == (3, 64, 54) for a in cx)
    assert torch.all(ce[:, [0, 11, 12]] == 0)
    assert all(torch.all(a[:, 0, :42] == 0) for a in cx)
    assert all(torch.all(a[..., 14:26] == 0) for a in cx)
    assert all(torch.isfinite(a).all() for a in cx)


def test_ppo_update_uses_real_returns_and_preserves_tt_independence():
    p = EpisodicResidualPPO(3)
    opt = torch.optim.Adam(p.parameters(), lr=0.002)
    torch.manual_seed(19)
    raw, lp = p.draw(5)
    before = p.mean.detach().clone()
    result = ppo_update(p, opt, raw, lp,
                        torch.tensor([-24., -14., -32., -10., -18.]))
    assert all(np.isfinite(v) for v in result.values())
    assert not torch.equal(before, p.mean.detach())
    assert torch.isfinite(p.value)


def test_report_marks_unevaluated_tasks_and_reads_physical_npz(tmp_path):
    report = build_report(tmp_path)
    assert report["tasks"]["circle"]["status"] == "not_evaluated"
    folder = tmp_path / "circle" / "gradient_scale_0"
    folder.mkdir(parents=True)
    rows = [{"candidate": 0, "morphology": [0.5]*13,
             "reward": -21.5, "valid_seconds": 3.0,
             "complete_10s": False, "progress": 0.2, "initial_valid": True}]
    (folder / "seed_20000.json").write_text(json.dumps({
        "task": {"task":"circle_track","radius":1.0,"half_width":0.3},
        "candidates": rows
    }), encoding="utf-8")
    path = np.zeros((1, 500, 54), np.float32)
    path[0, :, 0] = np.linspace(0, .3, 500)
    mask = np.zeros((1, 500), bool)
    mask[0, :150] = True
    np.savez_compressed(folder/"seed_20000.npz", pred=path,
                        mujoco=path, E=np.ones((1, 13))*.5, valid=mask)
    report = build_report(tmp_path)
    x = report["tasks"]["circle"]["methods"]["Gradient (scale=0)"]
    assert x["candidates"] == 1 and x["completed_10s"] == 0
    assert x["mean_valid_reward"] == -21.5
    assert "svg" in (tmp_path/"report.html").read_text(encoding="utf-8")
    assert (tmp_path/"report.md").is_file()
    # A previous recorder can produce an MP4 without a video_path field in JSON.
    # Rebuilding the report must find the real file, embed it, and expose a link.
    videos = folder/"videos"
    videos.mkdir()
    mp4 = b"\x00\x00\x00\x18ftyp" + b"x"*1024
    (videos/"seed_20000_candidate_0.mp4").write_bytes(mp4)
    report = build_report(tmp_path)
    assert report["videos"]["available"] == 1
    assert report["videos"]["missing"] == 0
    page = (tmp_path/"report.html").read_text(encoding="utf-8")
    assert '<video controls' in page
    embedded = base64.b64encode(mp4).decode("ascii")
    assert page.count(f'data:video/mp4;base64,{embedded}') == 1
    assert 'href="#video-circle/gradient_scale_0/videos/seed_20000_candidate_0.mp4"' in page
    assert "원본 MP4 열기" in page
    assert report["videos"]["candidates"][0]["status"] == "recorded"

    (videos/"seed_20000_candidate_0.mp4").write_bytes(b"x"*1024)
    report = build_report(tmp_path)
    assert report["videos"]["available"] == 0
    assert "손상되었거나 비어 있음" in report["videos"]["candidates"][0]["status"]
