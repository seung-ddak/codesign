"""Run the physical evaluator with MuJoCo and a small dependency-free model."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

mujoco = pytest.importorskip("mujoco")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "tt_dynamics"))
import benchmark_task_inference as benchmark
import generate_task_mujoco as evaluation
from task_reward import SlalomReward, StraightReward, TrackReward
from tt_task_report import build_report


@pytest.fixture
def physical_model(monkeypatch):
    # A floating base, four supported feet and 12 position actuators exercise
    # the real integrator and contact solver without private robot assets.
    legs = []
    for i, name in enumerate(evaluation.FEET):
        x, y = (0.12 if i < 2 else -0.12), (0.1 if i % 2 else -0.1)
        legs.append(f"""
          <body pos="{x} {y} 0">
            <joint name="j{i*3}" axis="1 0 0"/>
            <joint name="j{i*3+1}" axis="0 1 0"/>
            <joint name="j{i*3+2}" axis="0 0 1"/>
            <geom name="{name}_foot" type="sphere" size=".02" pos="0 0 -.08" margin=".000001"/>
          </body>""")
    actuators = [f'<position joint="j{i}" kp="1" ctrlrange="-.5 .5"/>' for i in range(12)]
    xml = f"""
      <mujoco><option timestep=".005" gravity="0 0 0"/>
        <default><geom friction="0 0 0"/><joint damping="2"/></default>
        <worldbody>
          <geom name="floor" type="plane" size="10 10 .1"/>
          <body><freejoint/><geom type="sphere" size=".06"/>
          {''.join(legs)}</body>
        </worldbody>
        <actuator>{''.join(actuators)}</actuator>
      </mujoco>"""

    def make_model(_e, task=None):
        spec = mujoco.MjSpec.from_string(xml)
        if isinstance(task, SlalomReward):
            evaluation.add_slalom_poles(spec, task)
        return spec.compile()

    monkeypatch.setattr(evaluation, "make_physics", make_model)


def candidate(vx=0):
    pred = np.zeros((evaluation.STEPS, 54), np.float32)
    pred[0, 2] = 0.10
    pred[0, 3] = 1.0
    pred[0, 7] = -1.0
    pred[0, 8] = vx
    return pred


def test_all_500_controls_and_terminal_state(physical_model):
    pred = candidate()
    pred[-1, 42] = 2
    trace = []
    actual, valid, meta = evaluation.replay(np.full(13, .5), pred, state_trace=trace)
    assert len(trace) == valid.sum() == 501
    assert meta["control_steps"] == 500
    assert meta["valid_seconds"] == 10.0
    assert meta["complete_10s"]
    assert meta["termination_reason"] == "completed_horizon"
    assert actual[498, 42] == 0
    assert actual[499, 42] == 0.5
    assert actual[500, 42] == 0.5
    assert actual[500, 14] != actual[499, 14]


def test_terminal_fall_is_not_a_completion(physical_model, monkeypatch):
    step = mujoco.mj_step

    def step_then_fall(model, data, **kwargs):
        step(model, data, **kwargs)
        if data.time > 9.99:
            data.qpos[2] = 0.06

    monkeypatch.setattr(mujoco, "mj_step", step_then_fall)
    _, valid, meta = evaluation.replay(np.full(13, .5), candidate())
    assert valid.sum() == 501
    assert meta["valid_seconds"] == 10.0
    assert meta["fell"]
    assert not meta["complete_10s"]
    assert meta["termination_reason"] == "fall"


def test_invalid_initial_state_never_applies_control(physical_model):
    pred = candidate()
    pred[0, 2] = 0.20
    _, valid, meta = evaluation.replay(np.full(13, .5), pred)
    assert not valid.any()
    assert meta["control_steps"] == 0
    assert meta["invalid_reason"] == "invalid_initial_support"


def test_slalom_records_real_pole_contact(physical_model):
    task = SlalomReward()
    actual, valid, meta = evaluation.replay(np.full(13, .5), candidate(vx=1), task=task)
    result = evaluation.score_replay(task, actual, valid, meta)
    assert meta["pole_contacts"] == [0]
    assert result["reward_terms"]["pole_contacts"] == 1
    assert result["reward_terms"]["p_pole"] > 0.5  # 같은 점수식의 폴 근접 감점
    assert result["gates_passed"] == 0
    assert not result["task_success"]


def test_guidance_and_physical_score_use_one_formula(physical_model):
    import torch
    for task in (TrackReward(), SlalomReward(), StraightReward()):
        actual, valid, meta = evaluation.replay(np.full(13, .5), candidate(vx=.3), task=task)
        result = evaluation.score_replay(task, actual, valid, meta)
        value, _ = task(torch.as_tensor(actual[None]), torch.as_tensor(valid[None]))
        assert abs(result["reward"] - float(value[0])) < 1e-5


def test_slalom_guidance_and_ordered_gate_crossings():
    task = SlalomReward()
    route = np.array(task.route)
    assert len(task.gate_crossings(route)) == len(task.gates)
    assert task.gate_crossings(route[::-1]) == []
    assert task.gate_crossings(np.array([[0., 0.], [6., 0.]])) == []
    missed_first = route.copy()
    missed_first[1, 1] = 0
    assert task.gate_crossings(missed_first) == []
    x = torch.zeros((2, 301, 54), requires_grad=True)
    with torch.no_grad():
        x[:, :, 0] = torch.linspace(0, 6, 301)
        x[0, :, 1] = task.y_ref(x[0, :, 0])
        x[:, :, 7] = -1.0
    score, _ = task(x, torch.ones((2, 301), dtype=torch.bool))
    assert score[0] > score[1]
    score.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_survival_is_separate_from_task_success():
    cases = [
        (TrackReward(), np.column_stack([np.sin(np.linspace(0, 2*np.pi, 20)),
                                        1 - np.cos(np.linspace(0, 2*np.pi, 20))])),
        (StraightReward(), np.column_stack([np.linspace(0, 2, 20), np.zeros(20)])),
        (SlalomReward(), np.array(SlalomReward().route)),
    ]
    for task, xy in cases:
        actual = np.zeros((len(xy), 54), np.float32)
        actual[:, :2], actual[:, 7] = xy, -1
        valid = np.ones(len(xy), bool)
        meta = dict(control_steps=len(xy)-1, complete_10s=True,
                    forward_m=float(xy[-1, 0]), pole_contacts=[])
        assert evaluation.score_replay(task, actual, valid, meta)["task_success"]
        meta["complete_10s"] = False
        assert not evaluation.score_replay(task, actual, valid, meta)["task_success"]


def test_legacy_archive_replays_to_new_output(tmp_path, physical_model):
    source = tmp_path / "old"
    folder = source / "straight" / "gradient_scale_0"
    folder.mkdir(parents=True)
    record = folder / "seed_1.json"
    record.write_text(json.dumps({"seed": 1, "method": "gradient", "candidates": []}))
    np.savez_compressed(record.with_suffix(".npz"), E=np.full((1, 13), .5),
                        pred=np.array([candidate()]))
    out = tmp_path / "new"
    benchmark.replay_archive(source, out)
    result_file = out / record.relative_to(source)
    result = json.loads(result_file.read_text())
    assert result["source_evaluation_version"] == 1
    assert result["candidates"][0]["complete_10s"]
    with np.load(result_file.with_suffix(".npz")) as archive:
        assert archive["mujoco"].shape == (1, 501, 54)
        assert archive["action_valid"].sum() == 500
    stats = build_report(out)["tasks"]["straight"]["methods"]["Gradient (scale=0)"]
    assert stats["archived_candidates_replayed"]
    assert stats["completed_10s"] == 1
    assert stats["task_successes"] == 0
    assert json.loads(record.read_text())["candidates"] == []



def test_frozen_checkpoint_both_guidance_methods_on_cpu(tmp_path, physical_model, monkeypatch):
    from tt_model import make_net
    from tt_rl_residual_ppo import EpisodicResidualPPO
    from joint_diffusion_guidance import _windows

    config = dict(horizon=64, d_model=16, layers=1, diff_steps=1,
                  modes=[2, 4], conditioning_version=4, norm="minmax", clip=1)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(dict(args=config, net=make_net(config).state_dict(),
                    mu=np.zeros(54, np.float32), sd=np.ones(54, np.float32)), checkpoint)
    model, setup = benchmark.load_model(checkpoint, "cpu")
    monkeypatch.setattr(benchmark, "settled_start", lambda _e: candidate()[0, :42])
    policy = EpisodicResidualPPO(len(_windows(evaluation.STEPS, config["horizon"])))
    raw = policy.mean.detach()[None, :]
    methods = [("gradient", dict(scale=1.0)), ("rl", dict(rl_raw=raw, rl_policy=policy))]
    for method, kwargs in methods:
        out = tmp_path / method
        benchmark.run_and_store(out, method, model, setup, StraightReward(),
                                count=1, seed=20000, **kwargs)
        row = json.loads((out / "seed_20000.json").read_text())["candidates"][0]
        assert np.isfinite(row["predicted_guidance_score"])
        assert row["control_steps"] > 0
        assert np.isfinite(row["reward"])
