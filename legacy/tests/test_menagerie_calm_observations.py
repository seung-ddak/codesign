import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('menagerie_eval',Path(__file__).parents[1]/'code/motion_tracking/eval_menagerie_calm.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_virtual_markers_preserve_real_bodies_and_transport_velocity():
    names=['trunk']+[leg+'_'+part for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh','calf']]
    actual=np.zeros((13,13));actual[:,6]=1;actual[:,7:10]=[1,2,3]
    actual[names.index('FL_hip'),10:13]=[0,0,1]
    actual[names.index('FL_calf'),10:13]=[0,1,0]
    expected=['base','trunk']+[leg+'_'+part for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh_shoulder','thigh','calf','foot']]+['imu_link']
    before=actual.copy();v=m.virtual_body_states(actual,names,expected)
    assert v.shape==(23,13)
    for name in names:np.testing.assert_array_equal(v[expected.index(name)],actual[names.index(name)])
    np.testing.assert_allclose(v[expected.index('FL_thigh_shoulder'),:3],[0,.081,0])
    np.testing.assert_allclose(v[expected.index('FL_thigh_shoulder'),7:10],[.919,2,3])
    np.testing.assert_allclose(v[expected.index('FL_foot'),:3],[0,0,-.2])
    np.testing.assert_allclose(v[expected.index('FL_foot'),7:10],[.8,2,3])
    np.testing.assert_array_equal(actual,before)


def test_original_23_body_observations_are_used_without_reconstruction():
    names=['base','trunk']+[leg+'_'+part for leg in ['FL','FR','RL','RR'] for part in ['hip','thigh_shoulder','thigh','calf','foot']]+['imu_link']
    actual=np.arange(23*13,dtype=float).reshape(23,13)
    result=m.virtual_body_states(actual,names,names)
    np.testing.assert_array_equal(result,actual)
    assert not np.shares_memory(result,actual)


def test_nonfinite_stand_duration_is_rejected_before_creating_output(tmp_path):
    import subprocess,sys
    script=Path(__file__).parents[1]/'code/motion_tracking/eval_menagerie_calm.py'
    output=tmp_path/'unused'
    result=subprocess.run([sys.executable,str(script),'--out',str(output),'--rl-games-wheel','unused.whl','--stand-seconds','nan'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'finite and positive' in result.stderr
    assert not output.exists()


def test_random_commands_are_seeded_smooth_and_bounded():
    a,knots=m.smooth_random_commands(7)
    b,_=m.smooth_random_commands(7)
    c,_=m.smooth_random_commands(8)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(a,c)
    assert a.shape==(300,3) and np.isfinite(a).all()
    np.testing.assert_array_equal(a[0],[0,0,0])
    assert np.linalg.norm(a[:,:2],axis=1).max()<=.6
    assert np.linalg.norm(np.diff(a[:,:2],axis=0)*30,axis=1).max()<=1.8
    assert np.abs(np.diff(a[:,2])*30).max()<=1.05
    assert all(1<=y['time']-x['time']<=2.5 for x,y in zip(knots,knots[1:]))

def test_random_commands_reject_invalid_inputs():
    import pytest
    for args in [(-1,), (0,0), (0,11), (0,10,float('nan')), (0,10,0)]:
        with pytest.raises(ValueError):m.smooth_random_commands(*args)


def test_balanced_commands_have_exact_slot_allocation_and_continuous_boundaries():
    from collections import Counter
    counts=Counter()
    for i in range(10):
        a,segments=m.balanced_commands(0,i)
        b,_=m.balanced_commands(0,i)
        np.testing.assert_array_equal(a,b)
        assert a.shape==(300,3) and np.isfinite(a).all()
        assert np.linalg.norm(a[:,:2],axis=1).max()<=.6
        np.testing.assert_allclose(a[149],a[150])
        counts.update(x['kind'] for x in segments)
    assert counts==dict(forward=6,backward=4,left=3,right=3,turn=3,stop_restart=1)
