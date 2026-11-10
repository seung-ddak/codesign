from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/motion_tracking'))
from report_calm_motions import validate_trace,select_examples


def trace(n):
    root=np.zeros((n+1,13));root[:,6]=1;root[:,2]=.3;root[:,0]=np.linspace(0,1,n+1)
    return dict(root=root,q=np.zeros((n+1,12)),qvel=np.zeros((n+1,12)),foot_pos=np.zeros((n+1,4,3)),foot_floor_contact=np.zeros((n+1,4),bool),target_q=np.zeros((n,12)),llc_action=np.zeros((n,12)),latent=np.zeros((n,64)),command=np.zeros((n,3)),joint_force_sensor=np.zeros((n,12)),time=np.arange(n+1)/30,valid_transition=np.ones(n,bool),quaternion_order='xyzw',velocity_frame='world')


def test_report_revalidates_actual_length_and_stationarity(tmp_path):
    path=tmp_path/'trace.npz';data=trace(300);plan=dict(k=1,segments=[dict(index=0,start=0,end=300)])
    np.savez(path,**data);result=validate_trace(path,plan)
    assert result['duration_seconds']==10 and result['min_active_fraction']==1
    data['root'][:,0]=0;np.savez(path,**data)
    with pytest.raises(ValueError,match='quality'):validate_trace(path,plan)
    data=trace(300);data['target_q']=data['target_q'][:-1];np.savez(path,**data)
    with pytest.raises(ValueError,match='target_q'):validate_trace(path,plan)


def test_report_requires_twenty_seconds_for_high_k(tmp_path):
    path=tmp_path/'trace.npz';np.savez(path,**trace(300))
    with pytest.raises(ValueError,match='duration'):validate_trace(path,dict(k=6,segments=[dict(index=0,start=0,end=300)]))


def test_five_examples_cover_k_and_pose_transitions_without_reward_selection():
    records=[]
    for i,k in enumerate([1,3,6,9,12]):
        ids=[0] if k==1 else [0,1,4] if k==3 else list(range(k-2))+[10,11]
        records.append(dict(id=i,accepted=True,plan=dict(k=k,segments=[dict(index=j) for j in ids])))
    assert [r['plan']['k'] for r in select_examples(records)]==[1,3,6,9,12]
    with pytest.raises(ValueError,match='K=12'):select_examples(records[:-1])
