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



def test_invalid_fixed_sample_stays_json_serializable():
    fixed = {'runs': {'scale_10': []}}
    for k in range(16):
        e = [.5] * 13; e[1] = .1 + k*.05
        fixed['runs']['scale_10'].extend(example(e, k, k) for _ in range(4))
    fixed['runs']['scale_10'][0]['mjx_total'] = None
    result = mod.analyze_fixed(fixed, {'runs': {'scale_10': []}}, 10)
    assert result['actual_mjx_reward_by_shape'][0] is None
    assert result['valid_mjx_samples_per_shape'][0] == 3
    json.dumps(result, allow_nan=False)


def test_generation_reuses_only_matching_command_and_checkpoint(tmp_path, monkeypatch):
    import pytest
    ckpt = tmp_path / 'ckpt'; ckpt.mkdir()
    (ckpt / 'checkpoint.pt').write_bytes(b'checkpoint-one')
    out = tmp_path / 'result'
    command = ['python', 'generate.py', '--ckpt', str(ckpt), '--out', str(out), '--seed', '0', '--scales', '10']
    calls = []
    def generate(argv, check):
        calls.append(argv)
        out.mkdir()
        (out / 'summary.json').write_text('{}')
    monkeypatch.setattr(mod.subprocess, 'run', generate)
    mod.run_generation(command)
    mod.run_generation(command)
    assert len(calls) == 1
    with pytest.raises(ValueError, match='different settings'):
        mod.run_generation(command[:-1] + ['3'])
    (ckpt / 'checkpoint.pt').write_bytes(b'checkpoint-two')
    with pytest.raises(ValueError, match='different settings'):
        mod.run_generation(command)
    assert len(calls) == 1


def test_old_or_partial_result_is_never_silently_reused(tmp_path):
    import pytest
    ckpt = tmp_path / 'ckpt'; ckpt.mkdir()
    (ckpt / 'checkpoint.pt').write_bytes(b'checkpoint')
    out = tmp_path / 'old'; out.mkdir()
    (out / 'summary.json').write_text('{}')
    with pytest.raises(ValueError, match='incomplete'):
        mod.run_generation(['python', 'generate.py', '--ckpt', str(ckpt), '--out', str(out)])


def test_smoke_failure_exits_nonzero_but_preserves_audit_and_html(tmp_path):
    import subprocess
    import sys
    directory = tmp_path / 'smoke_zigzag_2s'; directory.mkdir()
    row = example([.5]*13, 1., 1.); row['path_err_m'] = .3
    (directory / 'summary.json').write_text(json.dumps({'runs': {'scale_0': [row]}}))
    run = subprocess.run([sys.executable, str(ROOT / 'code/tt_dynamics/eval_v7_ddim100.py'),
                          '--root', str(tmp_path), '--stage', 'smoke', '--require-smoke-gate'],
                         capture_output=True, text=True)
    assert run.returncode != 0
    assert json.loads((tmp_path / 'audit_smoke.json').read_text())['smoke']['dynamics_gate_pass'] is False
    assert (tmp_path / 'report.html').is_file()


def test_report_selects_predicted_and_real_winners_separately(tmp_path):
    folder = tmp_path / 'codesign/joint_zigzag_4s'; (folder / 'videos').mkdir(parents=True)
    rows = [example([.5]*13, 1., 10.), example([.5]*13, 5., 2.), example([.5]*13, .1, .2)]
    (folder / 'summary.json').write_text(json.dumps({'runs': {'scale_10': rows}}))
    for i in range(3): (folder / f'videos/scale_10_cand{i}.mp4').write_bytes(b'mp4-test')
    mod.write_report(tmp_path, {'smoke': {'dynamics_gate_pass': False}}, 10)
    text = (tmp_path / 'report.html').read_text()
    assert 'TT 예측 보상 1위 · joint 0' in text
    assert 'MJX 실제 보상 1위 · joint 1' in text
    assert text.count('data:video/mp4;base64,') == 3
    assert text.index('joint 1') < text.index('<details>') < text.index('joint 2')
