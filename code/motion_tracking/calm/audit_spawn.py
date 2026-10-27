"""Compare original A1 creation and Default reset without a learned policy.

The original Humanoid asset loader and PD mapping are reused. AMP, reference
motion, fall-state generation, and policy checkpoints are not instantiated.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--disable-self-collision',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__).resolve();project=source.parents[3]
    root=project.parent/'external/a1-calm-main';os.environ['CALM_ROOT']=str(root)
    os.chdir(root);sys.path.insert(0,str(root/'calm'))
    sys.argv=['audit_spawn.py','--task','QuadrupedAMPGetup','--test','--headless','--num_envs','2','--seed','0',
              '--cfg_env','calm/data/cfg/a1_getup_31psc8n_4096.yaml',
              '--cfg_train','calm/data/cfg/train/rlg/calm_a1_31psc8n.yaml']
    import run
    from utils.config import get_args,load_cfg,parse_sim_params,set_seed
    from env.tasks.humanoid import Humanoid
    from env.tasks.base_task import BaseTask
    from isaacgym import gymapi
    from reset_observations import build_state
    import torch
    import numpy as np
    import imageio.v2 as imageio
    args=get_args();cfg,train,_=load_cfg(args);set_seed(0,False)
    params=parse_sim_params(args,cfg,train)
    original=BaseTask.__init__
    def camera_init(self,cfg,enable_camera_sensors=False):original(self,cfg,True)
    BaseTask.__init__=camera_init
    if a.disable_self_collision:
        Humanoid._get_humanoid_collision_filter=lambda self: 1
    task=Humanoid(cfg,params,args.physics_engine,args.device,args.device_id,True)
    names=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
    joints=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0])
    raw=task._dof_pos[0].clone()
    task.reset(torch.tensor([1],device=task.device,dtype=torch.long))
    targets=torch.stack([raw,task._initial_dof_pos[1]])
    hold=(targets-task._pd_action_offset)/task._pd_action_scale
    # Both targets must pass through the unchanged original action transform exactly.
    if not torch.allclose(task._action_to_pd_targets(hold),targets,atol=1e-6):raise RuntimeError('PD target mapping mismatch')
    urdf=root/'assets/urdf/a1_description/urdf/a1.urdf'
    initial_fk=build_state(urdf,names,joints,task._humanoid_root_states,task._dof_pos,task._dof_vel)
    foot_ids=[names.index(l+'_foot') for l in ['FR','FL','RR','RL']]
    foot_bottom=(initial_fk[:,foot_ids,2]-.02).cpu().numpy()
    initial_q=task._dof_pos.cpu().numpy().copy()
    lo,hi=task.dof_limits_lower,task.dof_limits_upper
    violation=torch.maximum(torch.clamp(lo-task._dof_pos,min=0),torch.clamp(task._dof_pos-hi,min=0)).cpu().numpy()
    props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles[0])
    cameras=[];writers=[];labels=['raw_creation','default_reset']
    camprops=gymapi.CameraProperties();camprops.width=960;camprops.height=544
    for i,label in enumerate(labels):
        camera=task.gym.create_camera_sensor(task.envs[i],camprops)
        if camera<0:raise RuntimeError('Camera creation failed')
        cameras.append(camera);writers.append(imageio.get_writer(str(out/(label+'.mp4')),fps=30))
    states=[];dofs=[];feet=[];forces=[]
    try:
        for k in range(60):
            task.step(hold)
            task.gym.fetch_results(task.sim,True)
            task._refresh_sim_tensors()
            states.append(task._humanoid_root_states.cpu().numpy().copy())
            dofs.append(task._dof_pos.cpu().numpy().copy())
            feet.append(task._rigid_body_pos[:,foot_ids].cpu().numpy().copy())
            forces.append(task._contact_forces.cpu().numpy().copy())
            for i,camera in enumerate(cameras):
                pos=states[-1][i,:3]
                task.gym.set_camera_location(camera,task.envs[i],gymapi.Vec3(*(pos+[1.2,1.2,.65])),gymapi.Vec3(*pos))
            task.gym.step_graphics(task.sim);task.gym.render_all_camera_sensors(task.sim)
            for i,camera in enumerate(cameras):
                rgba=task.gym.get_camera_image(task.sim,task.envs[i],camera,gymapi.IMAGE_COLOR).reshape(544,960,4)
                writers[i].append_data(rgba[:,:,:3])
                if k==0:imageio.imwrite(out/(labels[i]+'_first_step.png'),rgba[:,:,:3])
    finally:
        for w in writers:w.close()
        np.savez_compressed(out/'trace.npz',root=states,q=dofs,foot_pos=feet,contact_force=forces,body_names=names,joint_names=joints,dt=task.dt,targets=targets.cpu().numpy(),initial_q=initial_q,initial_foot_bottom=foot_bottom)
    states=np.asarray(states);quat=states[:,:,3:7];up=1-2*(quat[:,:,0]**2+quat[:,:,1]**2)
    result=dict(disable_self_collision=a.disable_self_collision,experiment='policy_free_spawn_audit',seed=0,seconds=len(states)*float(task.dt),
                policy_loaded=False,reference_loaded=False,fall_states_generated=False,
                git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_loader_sha256=hashlib.sha256((root/'calm/env/tasks/humanoid.py').read_bytes()).hexdigest(),
                cfg_env=cfg,sim_dt=float(params.dt),sim_substeps=params.substeps,
                runtime_joint_properties={key:props[key].tolist() for key in ['stiffness','damping','effort']},cases={})
    for i,label in enumerate(labels):
        result['cases'][label]=dict(initial_joint_angles=initial_q[i].tolist(),
                                    initial_foot_bottom_m=foot_bottom[i].tolist(),
                                    max_initial_joint_limit_violation_rad=float(violation[i].max()),
                                    min_body_up=float(up[:,i].min()),min_root_height_m=float(states[:,i,2].min()),
                                    max_root_height_m=float(states[:,i,2].max()),
                                    final_root_height_m=float(states[-1,i,2]),
                                    finite=bool(np.isfinite(states[:,i]).all()))
    (out/'summary.json').write_text(json.dumps(result,indent=2,default=str));print('SPAWN_AUDIT',json.dumps(result,default=str))
    task.gym.destroy_sim(task.sim)


if __name__=='__main__':main()
