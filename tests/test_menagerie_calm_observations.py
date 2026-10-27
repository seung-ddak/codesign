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
