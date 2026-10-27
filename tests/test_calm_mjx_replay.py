import importlib.util
from pathlib import Path
import numpy as np
import mujoco

spec=importlib.util.spec_from_file_location('calm_mjx',Path(__file__).parents[1]/'code/motion_tracking/check_calm_mjx.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_model_and_initial_state_preserve_joint_mapping_and_frames(tmp_path):
    source=Path(__file__).parents[2]/'external/a1-calm-main/assets/urdf/a1_description/urdf/a1.urdf'
    model=m.build_model(source,tmp_path)
    assert (model.nq,model.nv)==(19,18)
    names=[model.joint(i).name for i in range(1,13)][::-1]
    root=np.array([[1,2,.35,0,0,np.sin(.3),np.cos(.3),.1,.2,.3,1,0,0]])
    trace=dict(root=root,q=np.arange(12)[None,:]/100,qvel=np.arange(12)[None,:]/10,joint_names=names)
    qp,qv,qa,va=m.initial_state(trace,model)
    np.testing.assert_allclose(qp[qa],trace['q'][0]);np.testing.assert_allclose(qv[va],trace['qvel'][0])
    np.testing.assert_allclose(qp[3:7],root[0,[6,3,4,5]])
    np.testing.assert_allclose(qv[3:6],[np.cos(.6),-np.sin(.6),0],atol=1e-12)
    assert np.isclose(model.body('FL_thigh_shoulder').mass[0],.166226)
    robot=model.geom_contype==1;floor=model.geom_contype==2
    assert np.all(model.geom_conaffinity[robot]==2) and np.all(model.geom_conaffinity[floor]==1)
    restored=mujoco.MjModel.from_xml_path(str(tmp_path/'model.xml'))
    np.testing.assert_allclose(restored.body_mass,model.body_mass)


def test_menagerie_body_observations_use_world_velocity_and_xyzw():
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'code/motion_tracking'))
    from check_menagerie_mjx_policy import body_states
    scene=Path(__file__).parents[2]/'transformer-transformer/assets/mjcf/unitree_a1/scene.xml'
    model=mujoco.MjModel.from_xml_path(str(scene));data=mujoco.MjData(model)
    data.qpos[:3]=[0,0,.35];data.qpos[3:7]=[np.cos(.3),0,0,np.sin(.3)];data.qvel[:6]=[.1,.2,.3,.4,.5,.6];mujoco.mj_forward(model,data)
    before,names=body_states(model,data)
    assert len(names)==13 and before.shape==(13,13)
    np.testing.assert_allclose(before[0,3:7],[0,0,np.sin(.3),np.cos(.3)])
    mujoco.mj_integratePos(model,data.qpos,data.qvel,1e-7);mujoco.mj_forward(model,data)
    after,_=body_states(model,data)
    np.testing.assert_allclose((after[:,:3]-before[:,:3])/1e-7,before[:,7:10],atol=1e-7)
