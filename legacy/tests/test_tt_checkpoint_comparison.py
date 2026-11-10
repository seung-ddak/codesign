"""CPU checkpoint discovery, launch arguments and standalone video report."""
import base64
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code" / "tt_dynamics"))
import compare_tt_checkpoints as comparison


def test_checkpoint_inventory_marks_legacy_and_missing(tmp_path):
    valid, legacy, missing = (tmp_path / label / "checkpoint.pt"
                              for label in ("current", "old", "missing"))
    for version, path in ((4, valid), (2, legacy)):
        path.parent.mkdir()
        torch.save({"args": dict(horizon=64, diff_steps=5, modes=[2, 4],
                                 conditioning_version=version), "step": 10}, path)
    paths = comparison.checkpoint_inventory([valid.parent, valid, legacy, missing], [])
    rows = [comparison.inspect_checkpoint(path) for path in paths]
    assert [row["status"] for row in rows] == ["ready", "incompatible", "missing"]
    assert rows[0]["config"]["training_steps"] == 10
    assert rows[0]["config"]["weights"] == "raw"
    assert len(rows[0]["sha256"]) == 64


def test_checkpoint_comparison_runs_on_cpu_and_embeds_video(tmp_path, monkeypatch):
    out = tmp_path / "report"
    out.mkdir()
    entry = dict(id="h64", label="H64", path="checkpoint.pt", status="ready", config={})
    options = dict(seed=0, eval_seeds=1, candidates=2, rl_updates=2, rl_rollouts=4, cpu_threads=2)
    mp4 = b"\x00\x00\x00\x18ftyp" + b"\x00"*300

    def fake_run(command, **kwargs):
        assert command[command.index("--device") + 1] == "cpu"
        assert command[command.index("--methods") + 1:command.index("--methods") + 3] == ["gradient", "rl"]
        assert kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert "--resume" not in command
        version = Path(command[command.index("--out") + 1])
        for name in ("gradient_scale_0", "gradient_scale_1", "rl_ppo"):
            path = version / "zigzag" / name
            path.mkdir(parents=True)
            result = path / "seed_20000.json"
            result.write_text(json.dumps({"candidates": [
                dict(candidate=0, reward=0.4, predicted_guidance_score=0.8,
                     valid_seconds=3.0, complete_10s=False, task_success=False),
                dict(candidate=1, reward=None, initial_valid=False)]}))
            np.savez_compressed(result.with_suffix(".npz"), pred=np.empty(0))
            if name == "gradient_scale_1":
                videos = path / "videos"
                videos.mkdir()
                (videos / "seed_20000_candidate_0.mp4").write_bytes(mp4)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    comparison.run_checkpoint(out, entry, options, resume=False)
    assert entry["status"] == "complete"
    count = comparison.build_comparison(out, dict(options=options, versions=[entry]))
    assert count == 1
    page = (out / "report.html").read_text()
    assert all(title in page for title in comparison.HEADERS)
    assert "0.800 (1개)" in page and "0.400 (1개)" in page
    assert page.count("data:video/mp4;base64," + base64.b64encode(mp4).decode()) == 1


def test_rerender_uses_only_saved_archives(tmp_path, monkeypatch):
    run = tmp_path / "versions" / "h64" / "circle" / "gradient_scale_1"
    run.mkdir(parents=True)
    (run / "seed_20000.json").write_text('{"candidates": []}')
    np.savez_compressed(run / "seed_20000.npz", pred=np.empty(0))
    calls = []

    def fake_run(command):
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(comparison.subprocess, "run", fake_run)
    index = {"versions": [{"id": "h64", "label": "H64"},
                          {"id": "missing", "label": "missing"}]}
    assert comparison.render_saved_versions(tmp_path, index) == []
    command, = calls
    assert Path(command[1]).name == "render_task_videos.py"
    assert command[command.index("--backend") + 1] == "osmesa"
    assert "--resume" in command
    assert "--ckpt" not in command
