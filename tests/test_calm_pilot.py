import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'code/motion_tracking'))
import calm_pilot as m


def trace():
    d={k:np.zeros(shape) for k,shape in dict(root=(301,13),q=(301,12),qvel=(301,12),foot_pos=(301,4,3),foot_floor_contact=(301,4),target_q=(300,12),llc_action=(300,12),latent=(300,64),command=(300,3),joint_force_sensor=(300,12),valid_transition=(300,),time=(301,)).items()}
    d['root'][:,6]=1;d['foot_floor_contact'][:]=1;d['valid_transition'][:]=1;d['time']=np.arange(301)/30
    return d


def test_trace_alignment_rejects_missing_next_state_and_invalid_data():
    d=trace();assert m.validate_trace(d)==300
    d['qvel']=d['qvel'][:-1]
    with pytest.raises(ValueError):m.validate_trace(d)
    d=trace();d['root'][10,0]=np.nan
    with pytest.raises(ValueError):m.validate_trace(d)


def test_selection_is_distinct_and_refuses_duplicates():
    selected=m.select_diverse(np.arange(60).reshape(10,6),5)
    assert len(set(selected))==5
    with pytest.raises(ValueError):m.select_diverse(np.zeros((10,6)),5)


def test_calm_source_fixed_feet_are_fused_without_changing_fk(tmp_path):
    import mujoco
    from check_calm_mjx import build_model
    old_dir=tmp_path/'old';old_dir.mkdir();source=build_model(m.ROOT.parent/'external/a1-calm-main/assets/urdf/a1_description/urdf/a1.urdf',old_dir)
    fused=m.calm_source_model(tmp_path/'new');assert np.isclose(m.np_length(fused),.4)
    a=mujoco.MjData(source);b=mujoco.MjData(fused)
    for model,data in [(source,a),(fused,b)]:
        data.qpos[:3]=[0,0,.35];data.qpos[7:]=[0,.67,-1.3]*4;mujoco.mj_forward(model,data)
    for leg in ['FR','FL','RR','RL']:
        np.testing.assert_allclose(a.xpos[source.body(leg+'_foot').id],b.geom_xpos[fused.geom(leg+'_foot').id],atol=1e-12)


def test_runner_stops_and_comments_on_first_failure(tmp_path,monkeypatch):
    import shutil
    code=tmp_path/'code/motion_tracking';code.mkdir(parents=True)
    for name in ['calm_pilot.py','pd_replay.py','track_env.py','eval_menagerie_calm.py','train_rl_procedural.py','eval_motions.py']:shutil.copyfile(m.CODE/name,code/name)
    monkeypatch.setattr(m,'CODE',code);monkeypatch.setattr(m,'__file__',str(code/'calm_pilot.py'))
    monkeypatch.setattr(m,'ROOT',tmp_path);out=tmp_path/'runs'/'pilot';comments=[]
    monkeypatch.setattr(m.subprocess,'check_output',lambda *a,**k:'abc123\n')
    monkeypatch.setattr(m,'notify',lambda out,issue,event:comments.append(event['stage']) or 'https://example.test/comment')
    def command(*args):raise RuntimeError('simulated failure')
    monkeypatch.setattr(m,'run_command',command)
    # Runner normally has a worker log even when the worker fails.
    out.mkdir(parents=True);(out/'collect.log').write_text('worker failed')
    monkeypatch.setattr(sys,'argv',['pilot','--out',str(out),'--issue','9']);m.main()
    import json
    state=json.loads((out/'pipeline.json').read_text())
    assert [e['status'] for e in state['events']]==['FAIL'] and comments==['collect']
    assert state['not_run']==m.STAGES[1:]
    m.main();assert comments==['collect']
