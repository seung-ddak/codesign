from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/motion_tracking'))
from calm_sequences import make_plan,make_sequence,valid_order,labels,command,motion_feature,quality


def test_plan_balances_k_and_respects_pose_order_and_variable_duration():
    plans=make_plan(120,seed=7)
    assert np.bincount([p['k'] for p in plans])[1:].tolist()==[10]*12
    assert len({p['signature'] for p in plans})==120
    for p in plans:
        assert valid_order([s['index'] for s in p['segments']])
        frames=600 if p['k']>=6 else 300
        assert labels(p).shape==(frames,)
        assert p['segments'][-1]['end']==frames
        assert p['seconds']==frames/30
        if any(s['index']<10 for s in p['segments']):
            assert all(s['end']-s['start']<=(54 if s['index']==10 else 51) for s in p['segments'] if s['index']>=10)
    assert not valid_order([10,0,11])
    assert not valid_order([0,11])
    assert valid_order([11,0,10])
    assert make_sequence(5,12)==make_sequence(5,12)


def test_commands_have_correct_turn_and_followup_direction():
    for index,sign in [(4,-1),(5,1),(6,1),(7,-1),(8,1),(9,-1)]:
        plan={'segments':[dict(index=index,start=0,end=300,speed=.3,turn_angle=.5,ramp_seconds=.2)]}
        c=command(plan,299,.3,np.zeros(3))
        assert (c[2]-.3)*sign>0
        along=c[:2]@np.array([np.cos(c[2]),np.sin(c[2])])
        assert along==pytest.approx(0 if index in (4,5) else -.3 if index in (8,9) else .3)


def test_duplicates_ignore_global_xy_and_initial_heading():
    root=np.zeros((301,13));root[:,6]=1;root[:,2]=.3;root[:,0]=np.linspace(0,1,301);q=np.zeros((301,12))
    a=motion_feature(root,q)
    other=root.copy();angle=.7;other[:,:2]=root[:,:2]@np.array([[np.cos(angle),np.sin(angle)],[-np.sin(angle),np.cos(angle)]])+[3,5]
    other[:,5]=np.sin(angle/2);other[:,6]=np.cos(angle/2)
    np.testing.assert_allclose(a,motion_feature(other,q),atol=1e-6)
    assert quality(make_sequence(3,1),root[:-1],q[:-1])[0] is False
    with pytest.raises(ValueError):motion_feature(root[:-1],q[:-1])


def test_requested_but_unperformed_motion_is_rejected():
    root=np.zeros((301,13));root[:,6]=1;root[:,2]=.3;q=np.zeros((301,12))
    plan={'segments':[dict(index=0,start=0,end=300)]}
    assert quality(plan,root,q)[0] is False
    root[:,0]=np.linspace(0,.5,301)
    assert quality(plan,root,q)[0] is True


def test_report_separates_planned_and_collected(tmp_path):
    import json
    from calm_sequences import make_report
    plans=make_plan(12)
    (tmp_path/'plan.json').write_text(json.dumps({'motions':plans}))
    counts=make_report(tmp_path)
    assert counts['target']==12 and counts['accepted']==0
    record=dict(id=0,accepted=True,reason='pass',plan=plans[0])
    (tmp_path/'records.jsonl').write_text(json.dumps(record)+'\n')
    counts=make_report(tmp_path)
    assert counts['accepted']==1
    data=np.load(tmp_path/'timeline_labels.npz')
    assert ((data['accepted'][1:]==-1)|(data['accepted'][1:]==-2)).all()
    n=len(labels(plans[0]))
    np.testing.assert_array_equal(data['accepted'][0,:n],labels(plans[0]))
    assert (data['accepted'][0,n:]==-2).all()
    assert data['planned'].shape==(12,600)
    assert '실제 통과 1' in (tmp_path/'report.html').read_text(encoding='utf-8')


def test_atomic_trace_write_keeps_previous_on_failure(tmp_path,monkeypatch):
    from collect_calm_sequences import atomic_npz
    path=tmp_path/'trace.npz';atomic_npz(path,x=np.arange(3))
    def fail(*args,**kwargs):raise OSError('simulated disk failure')
    monkeypatch.setattr(np,'savez_compressed',fail)
    with pytest.raises(OSError):atomic_npz(path,x=np.arange(4))
    np.testing.assert_array_equal(np.load(path)['x'],np.arange(3))


def test_endpoint_change_cannot_hide_long_stationary_tail():
    root=np.zeros((301,13));root[:,6]=1;root[:,2]=.3;q=np.zeros((301,12))
    plan={'segments':[dict(index=0,start=0,end=300)]}
    root[:31,0]=np.linspace(0,.2,31);root[31:,0]=.2
    ok,reason,checks=quality(plan,root,q)
    assert not ok and reason=='segment_not_performed'
    assert checks[0]['longest_inactive_seconds']>1.5
    root[:,0]=np.sin(np.arange(301))*.001
    assert not quality(plan,root,q)[0]


def test_twenty_second_heading_only_and_height_only_activity():
    root=np.zeros((601,13));root[:,6]=1;root[:,2]=.3;q=np.zeros((601,12))
    yaw=np.linspace(0,4,601);root[:,5]=np.sin(yaw/2);root[:,6]=np.cos(yaw/2)
    plan={'segments':[dict(index=5,start=0,end=600)]}
    assert quality(plan,root,q)[0]
    assert motion_feature(root,q).shape==(544,)
    root[:,5]=0;root[:,6]=1;root[:,2]=np.linspace(.3,.1,601)
    assert not quality({'segments':[dict(index=10,start=0,end=600)]},root,q)[0]
    root=root[:61].copy();root[:,2]=np.linspace(.3,.15,61)
    assert quality({'segments':[dict(index=10,start=0,end=60)]},root,q[:61])[0]
