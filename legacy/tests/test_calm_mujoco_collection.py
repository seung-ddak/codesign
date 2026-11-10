import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code/motion_tracking'))


def test_all_motions_are_assigned_to_two_distinct_training_shapes(tmp_path, monkeypatch):
    import shared_calm_common as common
    from collect_calm_mujoco import make_plan
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'status.json').write_text(json.dumps(dict(accepted=10000)))
    rows = [dict(id=i, accepted=True, file=f'{i}.npz',
                 plan=dict(k=i % 12 + 1, segments=[dict(start=0, end=300)]))
            for i in range(10000)]
    (source / 'records.jsonl').write_text('\n'.join(map(json.dumps, rows)))
    monkeypatch.setattr(common, 'sha', lambda path: path.stem)
    plan = make_plan(source, tmp_path / 'out')
    training = {r['design'] for r in plan['designs']}
    heldout = {r['design'] for r in plan['eval_designs']}
    counts = Counter(i for design in training for i in plan['assignments'][design])
    assert len(training) == 200 and len(heldout) == 20 and not training & heldout
    assert set(counts) == set(range(10000)) and set(counts.values()) == {2}
    assert all(len(set(plan['assignments'][design])) == 100 for design in training)
    assert plan['target_episodes'] == 40000


def test_interpolation_and_floor_contacts():
    from collect_calm_mujoco import reference_targets, contact_flags
    ref = dict(fix_q=np.repeat(np.arange(10)[:, None], 12, axis=1))
    np.testing.assert_allclose(reference_targets(ref, 1, 4)[:, 0], [1, 1.6, 2.2, 2.8])
    data = SimpleNamespace(ncon=3, contact=[
        SimpleNamespace(geom1=1, geom2=5, dist=0),
        SimpleNamespace(geom1=6, geom2=1, dist=-.001),
        SimpleNamespace(geom1=7, geom2=8, dist=-.01)])
    np.testing.assert_equal(contact_flags(None, data, [5, 6, 7, 8], 1), [1, 1, 0, 0])


def test_cpu_physics_saves_real_state_transitions_at_50hz():
    from collect_calm_mujoco import rollout_one
    from pd_replay import build_model
    model = build_model('00000011', uniforms=np.full(13, .5), kp=30,
                        torque_limit=33.5, flat_ground=True)
    model.opt.timestep = .005
    home = model.keyframe('home').qpos.copy()
    ref = dict(fix_q=np.tile(home[7:19], (301, 1)),
               fix_base_pos=np.tile(home[:3], (301, 1)),
               fix_base_quat=np.tile(home[3:7], (301, 1)))
    result = rollout_one(model, ref, 0, 0)
    assert result['valid'].sum() >= 2
    assert result['qpos'].shape == (501, 19)
    assert result['target_q'].shape == (500, 12)
    assert np.isfinite(result['qpos']).all() and np.isfinite(result['qvel']).all()
    assert not np.any(np.diff(result['valid'].astype(int)) > 0)


def test_audit_rejects_nonfinite_velocities(tmp_path):
    from collect_calm_mujoco import audit
    (tmp_path / 'plan.json').write_text(json.dumps(dict(
        motions=[dict(id=0)], designs=[dict(design='e0000')], target_episodes=1)))
    folder = tmp_path / 'rollouts/e0000'
    folder.mkdir(parents=True)
    np.savez(folder / 'variant_0.npz', valid=np.ones((1, 500), bool),
             qpos=np.zeros((1, 501, 19)), qvel=np.full((1, 501, 18), np.nan),
             target_q=np.zeros((1, 500, 12)), action=np.zeros((1, 500, 12)),
             contact=np.zeros((1, 501, 4)), E=np.full(13, .5), dt=.02,
             group_id=np.array(['source']), source_motion_id=np.array([0]),
             fell=np.array([False]), design='e0000')
    result = audit(tmp_path)
    assert not result['complete'] and len(result['invalid_files']) == 1
    assert result['invalid_files'][0]['error'] == 'Nonfinite physics'
