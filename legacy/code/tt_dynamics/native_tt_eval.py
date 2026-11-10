"""Original t2 decoding/guidance and independent 50Hz MuJoCo task execution."""
import json
import types
from pathlib import Path

import mujoco
import numpy as np
import torch

from native_tt_data import save_json


def trajectory(task):
    t = np.arange(500)*.02
    x = .5*t
    if task=='straight':
        y=np.zeros_like(x); heading=np.zeros_like(x)
    elif task=='zigzag':
        y=.3*2/np.pi*np.arcsin(np.sin(np.pi*x))
        heading=np.arctan(.6*np.sign(np.cos(np.pi*x)))
    elif task=='circle':
        x,y=np.sin(.5*t),1-np.cos(.5*t);heading=.5*t
    else:raise ValueError(task)
    poses=np.tile(np.eye(4),(500,1,1))
    poses[:,:3,3]=np.stack([x,y,np.full(500,.3)],axis=1)
    poses[:,0,0]=np.cos(heading);poses[:,0,1]=-np.sin(heading)
    poses[:,1,0]=np.sin(heading);poses[:,1,1]=np.cos(heading)
    return poses


def generation_batch(cfg, task, n, device):
    from t2.data.dataset import EvenTimestepSampler
    from t2.eval.utils import traj2batch
    from t2.train.augment import AddPositionId
    batch=traj2batch(trajectory(task),cfg.seq_len,EvenTimestepSampler(8),center_traj=False)
    batch=AddPositionId(dict(cfg.hardware_pos_seq_lens))(batch)
    return {k:v.repeat(n,*([1]*(v.ndim-1))).to(device) for k,v in batch.items()}


def sample(model,cfg,task,n,seed,scale=0.,controller=None,hardware=None):
    from t2.eval.diffusion_guidance import ObjectiveGuidedDiffusion
    from native_tt_residual import residual_context
    batch=generation_batch(cfg,task,n,next(model.parameters()).device)
    task_model=model.tasks.hardware_gen
    original_probs={}
    original_init=task_model._init_denoising_state
    if hardware is not None:
        from t2.data.pad import pad
        padded=pad(hardware,dict(cfg.seq_len))
        padding={group:torch.tensor(padded[group+'/mask'],device=next(model.parameters()).device)[None]
                 for group in ['link','dyna_joint','fixed_joint','actuator']}
        for key,value in padded.items():
            if key.endswith('/mask'):
                # Preserve the native padding learning semantics in the sampler.
                continue
            batch[key]=torch.tensor(value,device=next(model.parameters()).device)[None].repeat(n,1,1)
        for name in ['link','dyna_joint','fixed_joint','actuator']:
            adapter=task_model.adapters[name]
            original_probs[name]=adapter.default_diffuse_prob
            adapter.default_diffuse_prob=0.
        def known_padding(self,*args,**kwargs):
            state=original_init(*args,**kwargs)
            samples=state[4]
            for name,mask in padding.items():
                adapter=self.adapters[name]
                known=adapter.modality.to_tensor(kwargs['batch'],encoders=adapter.attr_encoders)
                samples[name]=torch.where(mask,torch.full_like(known,adapter.modality.pad_value),known)
                state[5][name]=torch.ones_like(known,dtype=torch.bool)
            return state
        task_model._init_denoising_state=types.MethodType(known_padding,task_model)
    guidance=None
    if scale:
        if controller is not None:raise ValueError('Gradient and RL cannot be combined')
        def surrogate(decoded):
            return -((decoded['track_link_obs/pos']-decoded['target_pose/pos'])**2).mean((1,2))
        guidance=ObjectiveGuidedDiffusion({'task':surrogate},
            list(k for k,a in model.tasks.hardware_gen.adapters.items() if a.loss_weight),True,scale)
    bf16=next(model.parameters()).is_cuda and torch.cuda.is_bf16_supported()
    try:
        with residual_context(task_model,controller), \
             torch.set_grad_enabled(bool(scale)), \
             torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=bf16):
            decoded=task_model.decode(batch,model.backbone,model.pos_embs,seed=seed,
                                     guidance_fn=guidance,use_pbar=False,eta=0.)
    finally:
        for name,prob in original_probs.items():task_model.adapters[name].default_diffuse_prob=prob
        task_model._init_denoising_state=original_init
    result={k:v.detach().float().cpu().numpy() if v.is_floating_point() else v.detach().cpu().numpy()
            for k,v in {**batch,**decoded}.items()}
    if not all(np.isfinite(value).all() for value in result.values()):
        raise FloatingPointError('Nonfinite original sampler output')
    return result


def task_step(previous,position,rotation,delta,torque,task,step,initial_height):
    x,y=position[:2]; velocity=(position[:2]-previous[:2])/.02
    if task=='circle':
        radial=np.array([x,y-1]);old=previous[:2]-[0,1]
        tangent=np.array([-radial[1],radial[0]])
        progress=np.arctan2(old[0]*radial[1]-old[1]*radial[0],old@radial)
        lane=abs(np.linalg.norm(radial)-1)
    elif task=='zigzag':
        tangent=np.array([1.,.6*np.sign(np.cos(np.pi*x))]);progress=position[0]-previous[0]
        lane=abs(y-.3*2/np.pi*np.arcsin(np.sin(np.pi*x)))
    elif task=='straight':
        tangent=np.array([1.,0.]);progress=position[0]-previous[0];lane=abs(y)
    else:raise ValueError(task)
    tangent/=max(np.linalg.norm(tangent),1e-8);along=velocity@tangent
    upright=rotation[2,2]
    terms=dict(speed=2*(np.exp(-((along-.5)/.25)**2)-np.exp(-4)),
        progress=2*np.clip(progress/.02,-1,1),lane=-2*min((lane/.15)**2,4),
        tilt=-.5*max(0,1-upright),heading=-.2*(1-np.clip(rotation[:2,0]@tangent,-1,1)),
        low=-2*(max(0,.7*initial_height-position[2])/initial_height)**2,
        stall=-.5*float(step>25 and along<.05),
        action_rate=-.01*float(np.mean((delta/.25)**2)),
        torque=-.005*float(np.mean((torque/33.5)**2)))
    if not np.isfinite(list(terms.values())).all():raise FloatingPointError('Physical reward')
    return .02*sum(terms.values()),terms,lane


def physical_replay(candidate, task):
    """Invalid generated graphs are recorded, never replaced with a training shape."""
    from dm_control import mjcf
    from t2.eval.utils import extract_masked_hardware
    from t2.robotok.io import deserialize
    from t2.robotok.tokenizer import detokenize,DetokenizeError
    from t2.env.mj_utils import set_up_default_scene
    hardware={k:v for k,v in candidate.items() if k.split('/')[0] in
              ['link','dyna_joint','fixed_joint','actuator']}
    try:
        hw,*_=extract_masked_hardware(hardware)
        robot=deserialize(hw)
        root,_,_,_=detokenize(robot,gravcomp=False)
        root=set_up_default_scene(root,add_plane=True,plane_z_pos=0.)
        physics=mjcf.Physics.from_mjcf_model(root);m,d=physics.model.ptr,physics.data.ptr
        m.opt.timestep=.005
        if m.nu<1 or m.nq<7 or m.nv<6:raise ValueError('No actuated floating robot')
        site=m.site('site_0').id
        # Start from the generated geom-frame state, with the collection's zero
        # initial velocity convention. Actual state is preserved during execution.
        free=next(link for link in robot.links if link.free_link_idx!=-1)
        freebody=m.body(f'link_{free.idx}').id
        qadr=m.jnt_qposadr[m.body_jntadr[freebody]]
        pos=np.asarray(candidate['free_link_obs/pos']).reshape(8,-1,3)[0,0]
        rot=np.asarray(candidate['free_link_obs/rotmat']).reshape(8,-1,3,3)[0,0]
        quat=np.zeros(4);mujoco.mju_mat2Quat(quat,rot.reshape(9))
        d.qpos[qadr:qadr+3]=pos;d.qpos[qadr+3:qadr+7]=quat
        joint_positions=candidate['dyna_joint_obs/qpos'].reshape(8,-1)[0]
        for j in range(len(robot.dynamic_joints)):
            d.qpos[m.joint(f'dyna_joint_{j}').qposadr]=joint_positions[j]
        times=candidate['ctrl/time/id'].reshape(8,-1)[:,0]
        controls=candidate['ctrl/target_qpos'].reshape(8,-1)[:,:m.nu]
        if controls.shape[1]!=m.nu:raise ValueError('Actuator count mismatch')
        order=np.argsort(times);times=times[order];controls=controls[order]
        actions=np.stack([np.interp(np.arange(500),times,controls[:,j]) for j in range(m.nu)],1)
        mujoco.mj_forward(m,d)
        if not np.isfinite(d.qpos).all() or pos[2]<=.07 or d.site_xpos[site,2]<=.07 or d.site_xmat[site].reshape(3,3)[2,2]<.5:
            raise ValueError('Invalid generated initial pose')
        # Reject severe penetration instead of lifting an invalid candidate.
        if d.ncon and min(c.dist for c in d.contact[:d.ncon])<-.03:
            raise ValueError('Initial penetration exceeds 3cm')
    except (ValueError,KeyError,AssertionError,StopIteration,RuntimeError,DetokenizeError,np.linalg.LinAlgError) as error:
        return dict(return_value=-20.,invalid=True,reason=f'{type(error).__name__}: {error}',
                    survived=False,seconds=0.,progress=0.,steps=0),{}
    positions=[d.site_xpos[site].copy()];qraw=[d.qpos.copy()];vraw=[d.qvel.copy()]
    rewards=[];terms=[];applied=[];torques=[];lanes=[];last=actions[0].copy();fell=False
    for t,action in enumerate(actions):
        control=np.where(m.actuator_ctrllimited,np.clip(action,m.actuator_ctrlrange[:,0],m.actuator_ctrlrange[:,1]),action)
        d.ctrl[:]=control;mujoco.mj_step(m,d,nstep=4)
        if not np.isfinite(d.qpos).all() or not np.isfinite(d.qvel).all():
            raise FloatingPointError('Nonfinite physical rollout')
        position=d.site_xpos[site].copy();rotation=d.site_xmat[site].reshape(3,3).copy()
        reward,component,lane=task_step(positions[-1],position,rotation,control-last,
                                      d.actuator_force,task,t,positions[0][2])
        fell=rotation[2,2]<.5 or position[2]<.07
        reward-=20.*fell
        rewards.append(reward);terms.append(component);lanes.append(lane)
        positions.append(position);qraw.append(d.qpos.copy());vraw.append(d.qvel.copy())
        applied.append(control.copy());torques.append(d.actuator_force.copy());last=control
        if fell:break
    row=dict(return_value=float(sum(rewards)),invalid=False,survived=not fell,steps=len(rewards),
             seconds=len(rewards)*.02,progress=float(positions[-1][0]-positions[0][0]),
             lane_fraction=float(np.mean(np.asarray(lanes)<=.15)),fell=bool(fell))
    points=np.asarray(positions)
    if task=='circle':
        angle=np.unwrap(np.arctan2(points[:,0],1-points[:,1]))
        row['progress']=float(angle[-1]-angle[0])
    row['distance_m']=float(np.linalg.norm(np.diff(points[:,:2],axis=0),axis=1).sum())
    row['progress_per_second']=row['progress']/row['seconds']
    state_times=candidate['track_link_obs/time/id'].reshape(8,-1)[:,0]
    state_pos=candidate['track_link_obs/pos'].reshape(8,-1,3)[:,0]
    order=np.argsort(state_times)
    predicted=np.stack([np.interp(np.arange(len(positions)),state_times[order],state_pos[order,j])
                        for j in range(3)],1)
    row['prediction_position_rmse_m']=float(np.sqrt(np.mean((predicted-np.asarray(positions))**2)))
    arrays=dict(qpos=np.array(qraw),qvel=np.array(vraw),positions=np.array(positions),
                controls=np.array(applied),torque=np.array(torques),rewards=np.array(rewards))
    return row,arrays


def evaluate(model,cfg,task,seeds,folder,scale=0.,controller=None,hardware=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    rows=[]
    for seed in seeds:
        marker=folder/f'{seed}.json'
        if marker.exists():
            rows.append(json.loads(marker.read_text()));continue
        decoded=sample(model,cfg,task,1,int(seed),scale,controller,hardware)
        row,arrays=physical_replay({k:v[0] for k,v in decoded.items()},task)
        row['noise_seed']=int(seed)
        tmp=folder/f'{seed}.tmp.npz'
        np.savez_compressed(tmp,**arrays,**{'pred_'+k:v for k,v in decoded.items()})
        tmp.replace(folder/f'{seed}.npz');save_json(marker,row);rows.append(row)
    save_json(folder/'summary.json',dict(task=task,scale=scale,rows=rows,
        mean_return=float(np.mean([r['return_value'] for r in rows])),
        survival_fraction=float(np.mean([r['survived'] for r in rows]))))
    return rows
