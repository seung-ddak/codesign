"""Record six LLC references and four HLC commands on Menagerie A1.

Uses original CALM network builders and observation functions. Missing CALM
fixed links are observation markers, not additional masses or collision bodies.
"""
import argparse
import ast
import base64
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys


def original_functions(path, names, namespace):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected}==set(names)
    for node in selected:node.decorator_list=[]
    exec(compile(ast.Module(body=selected,type_ignores=[]),str(path),'exec'),namespace)


def virtual_body_states(actual, names, expected):
    import numpy as np
    from scipy.spatial.transform import Rotation
    states={name:actual[i].copy() for i,name in enumerate(names)}
    if all(n in states for n in expected):
        result=np.stack([states[n] for n in expected])
        assert result.shape==(23,13) and np.isfinite(result).all()
        return result
    states['base']=states['trunk'].copy();states['imu_link']=states['trunk'].copy()
    for leg in ['FL','FR','RL','RR']:
        hip=states[leg+'_hip'];calf=states[leg+'_calf']
        for suffix,parent,offset in [('thigh_shoulder',hip,[0,.081 if leg.endswith('L') else -.081,0]),('foot',calf,[0,0,-.2])]:
            s=parent.copy();delta=Rotation.from_quat(parent[3:7]).apply(offset)
            s[:3]+=delta;s[7:10]+=np.cross(parent[10:13],delta)
            states[leg+'_'+suffix]=s
    result=np.stack([states[n] for n in expected])
    assert result.shape==(23,13) and np.isfinite(result).all()
    return result


def smooth_random_commands(seed, seconds=10., dt=1/30):
    """Seeded world velocity/yaw knots with continuous speed and turn rate."""
    import numpy as np
    if seed<0 or not (0<seconds<=10) or not math.isfinite(dt) or dt<=0:
        raise ValueError('Invalid seed, duration or control period')
    rng=np.random.default_rng(seed);knots=[(0.,np.zeros(3))]
    while knots[-1][0]<seconds:
        duration=float(rng.uniform(1.,2.5));speed=0. if rng.random()<.15 else float(rng.uniform(.1,.6))
        direction=float(rng.uniform(-np.pi,np.pi));yaw=knots[-1][1][2]+float(rng.uniform(-.7,.7))
        knots.append((knots[-1][0]+duration,np.array([speed*np.cos(direction),speed*np.sin(direction),yaw])))
    commands=[];i=0
    for t in np.arange(int(round(seconds/dt)))*dt:
        while i+1<len(knots)-1 and t>=knots[i+1][0]:i+=1
        u=(t-knots[i][0])/(knots[i+1][0]-knots[i][0]);u=u*u*(3-2*u)
        commands.append(knots[i][1]*(1-u)+knots[i+1][1]*u)
    return np.asarray(commands),[dict(time=t,command=v.tolist()) for t,v in knots]


def balanced_commands(seed, episode, dt=1/30):
    """Twenty equal five-second slots implement the requested time allocation."""
    import numpy as np
    if seed<0 or not 0<=episode<10:raise ValueError('Expected nonnegative seed and episode 0..9')
    rng=np.random.default_rng(seed)
    schedule=np.array(['forward']*6+['backward']*4+['left']*3+['right']*3+['turn']*3+['stop_restart'])
    rng.shuffle(schedule);rng=np.random.default_rng(seed+episode+1000)
    commands=[];segments=[];previous=np.zeros(3)
    for slot,kind in enumerate(schedule[episode*2:episode*2+2]):
        speed=float(rng.uniform(.15,.6));yaw=previous[2]
        angle={'forward':0.,'backward':np.pi,'left':np.pi/2,'right':-np.pi/2,'turn':0.,'stop_restart':0.}[kind]
        transition=float(rng.uniform(1.,2.))
        turn=float(rng.choice([-1,1])*rng.uniform(.15,.45)) if kind=='turn' else 0.
        segments.append(dict(start=slot*5.,end=(slot+1)*5.,kind=str(kind),speed=speed,transition_seconds=transition,yaw_rate=turn))
        for step in range(round(5/dt)):
            t=step*dt;u=min(t/transition,1.);u=u*u*(3-2*u)
            facing=yaw+turn*t
            v=speed*np.array([np.cos(facing+angle),np.sin(facing+angle)])
            if kind=='stop_restart':
                # Stop for one second after deceleration, then restart forward.
                u=min(max((t-2.)/transition,0.),1.);u=u*u*(3-2*u)
                v=previous[:2]*(1-min(t/transition,1.)) if t<transition else v*u
            else:v=previous[:2]*(1-u)+v*u
            commands.append([*v,facing])
        previous=np.asarray(commands[-1])
    return np.asarray(commands),segments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rl-games-wheel',type=Path,required=True)
    p.add_argument('--random-episodes',type=int,default=0,help='Generate independent 10-second smooth random HLC episodes')
    p.add_argument('--balanced',action='store_true',help='Ten episodes with exact 30/20/15/15/15/5 percent five-second slot allocation')
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--stand-seconds',type=float,help='Time-extension diagnostic for LLC stand-up only')
    p.add_argument('--calm-no-self-collision',action='store_true',help='Use untouched original CALM A1 with self-collision filter 1; ground contact remains enabled')
    sys.stdout.reconfigure(encoding='utf-8')
    args=p.parse_args()
    if args.balanced and args.random_episodes!=10:p.error('--balanced requires --random-episodes 10')
    if args.random_episodes<0 or args.seed<0:p.error('Episode count and seed must be nonnegative')
    if args.random_episodes and (not args.calm_no_self_collision or args.stand_seconds is not None):p.error('Random collection requires --calm-no-self-collision and cannot combine with stand diagnostics')
    if args.stand_seconds is not None and (not math.isfinite(args.stand_seconds) or args.stand_seconds<=0):p.error('--stand-seconds must be finite and positive')
    out=args.out.resolve();out.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[2];workspace=project.parent
    calm=workspace/'external/a1-calm-main'
    model=calm/'assets/urdf/a1_description/urdf/a1.urdf' if args.calm_no_self_collision else workspace/'external/unitree_a1_menagerie/a1.urdf'
    collision_filter=1 if args.calm_no_self_collision else 0
    sys.path.insert(0,str(args.rl_games_wheel.resolve()));sys.path.insert(0,str(calm/'calm'))
    from isaacgym import gymapi
    import numpy as np
    if 'float' not in np.__dict__:np.float=float  # Compatibility with unchanged Isaac Gym Preview 4.
    import isaacgym.torch_utils as iq
    import torch
    import numpy as np
    import yaml
    import imageio.v2 as imageio
    from utils import torch_utils as tu
    from utils.rollout_motion_lib import RolloutMotionLib
    from learning.calm_network_builder import CALMBuilder
    from learning.hrl_network_builder import HRLBuilder
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    torch.manual_seed(args.seed);np.random.seed(args.seed);device='cuda:0'
    ns=dict(vars(iq));ns.update(torch=torch,torch_utils=tu)
    original_functions(calm/'calm/env/tasks/humanoid.py',['dof_to_obs','compute_humanoid_observations_max'],ns)
    original_functions(calm/'calm/env/tasks/humanoid_amp.py',['build_amp_observations'],ns)
    llc_ck=torch.load(calm/'checkpoints/lowLevelController.pth',map_location=device,weights_only=True)
    hlc_ck=torch.load(calm/'checkpoints/highLevelPolicy.pth',map_location=device,weights_only=True)
    cfg=yaml.safe_load((calm/'calm/data/cfg/train/rlg/calm_a1_31psc8n.yaml').read_text(encoding='utf-8'))
    builder=CALMBuilder();builder.load(cfg['params']['network'])
    llc=builder.build('calm',actions_num=12,input_shape=(343,),amp_input_shape=(1170,),amp_obs_steps=10,calm_latent_shape=(64,)).to(device).eval()
    llc.load_state_dict({k[len('a2c_network.'):] :v for k,v in llc_ck['model'].items()},strict=True)
    cfg=yaml.safe_load((calm/'calm/data/cfg/train/rlg/hrl_a1_heading_lat31zs.yaml').read_text(encoding='utf-8'))
    builder=HRLBuilder();builder.load(cfg['params']['network'])
    hlc=builder.build('hrl',actions_num=64,input_shape=(351,)).to(device).eval()
    hlc.load_state_dict({k[len('a2c_network.'):] :v for k,v in hlc_ck['model'].items()},strict=True)
    def normalizer(ck,key,size):
        n=RunningMeanStd((size,)).to(device).eval();n.load_state_dict(ck[key],strict=True);return n
    norm_llc=normalizer(llc_ck,'running_mean_std',343);norm_hlc=normalizer(hlc_ck,'running_mean_std',351);norm_amp=normalizer(llc_ck,'amp_input_mean_std',117)
    motion_file=calm/'calm/data/motions_a1/v2v3_31motions_squat2s_bfcut.pt'
    raw=torch.load(motion_file,map_location='cpu',weights_only=True)
    library=RolloutMotionLib(str(motion_file),list(range(12)),list(range(13)),[0,1,2,3],True,device=device)
    selected=[('forward','앞','v3_walkForwardSlow'),('backward','뒤','v3_walkBackwardSlow'),('left','좌','v3_walkLeftSlow'),('right','우','v3_walkRightSlow'),('sit','앉기','v2_sitDown'),('stand','일어나기','v2_standUp')]
    if args.random_episodes:selected=[('random_%03d'%i,'연속 이동 %03d'%i,'v3_walkForwardSlow') for i in range(args.random_episodes)]
    if args.stand_seconds is not None:selected=[x for x in selected if x[0]=='stand']
    g=gymapi.acquire_gym()
    manifest=dict(balanced=args.balanced,allocation_basis="equal five-second command slots including transitions",random_episodes=args.random_episodes,random_command_limits=dict(max_speed_mps=.6,knot_seconds=[1.,2.5],yaw_delta_rad=[-.7,.7],interpolation="smoothstep"),stand_seconds=args.stand_seconds,seed=args.seed,model=str(model),model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                  git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=project,text=True).strip(),
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  checkpoint_sha256={n:hashlib.sha256((calm/'checkpoints'/n).read_bytes()).hexdigest() for n in ['lowLevelController.pth','highLevelPolicy.pth']},
                  inference='Original CALM builders, strict checkpoint loading, original running normalization',
                  observation_adapter='Original 23 physical body observations, no virtual markers' if args.calm_no_self_collision else '13 physical bodies -> 23 CALM body markers; derived base, IMU, shoulder and foot markers',
                  kp=40,kd=1.5,effort_limit_nm=15,self_collision_filter=collision_filter,physics_dt=1/60,substeps=2,control_dt=1/30,
                  warning='Self-collision disabled: physical condition differs from original file defaults' if args.calm_no_self_collision else 'Different mass, geometry and observation markers: transferred policy test, not identical training robot')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    # Original CALM link names/order are read from the untouched source asset.
    tmp=g.create_sim(0,-1,gymapi.SIM_PHYSX,gymapi.SimParams());assert tmp is not None
    original_asset=g.load_asset(tmp,str(calm/'assets/urdf/a1_description/urdf'),'a1.urdf',gymapi.AssetOptions())
    expected=g.get_asset_rigid_body_names(original_asset);original_joints=g.get_asset_dof_names(original_asset)
    g.destroy_sim(tmp)
    assert len(expected)==23 and len(original_joints)==12
    import xml.etree.ElementTree as ET
    xml=ET.parse(calm/'assets/urdf/a1_description/urdf/a1.urdf').getroot()
    limits=np.array([[float(xml.find("joint[@name='%s']/limit"%n).get(k)) for k in ['lower','upper']] for n in original_joints])
    offset=torch.tensor(limits.mean(1),device=device,dtype=torch.float32)
    scale=torch.tensor(.7*np.diff(limits,axis=1)[:,0],device=device,dtype=torch.float32)
    def reference_latent(index):
        end=float(library._motion_lengths[index]);times=(end-torch.arange(60,device=device)/30).clamp(min=0)
        ids=torch.full((60,),index,device=device,dtype=torch.long)
        rp,rq,qp,rv,rw,qv,kp=library.get_motion_state(ids,times)
        amp=ns['build_amp_observations'](rp,rq,rv,rw,qp,qv,kp,True,True,72,list(range(13)),True,True)
        hip=torch.tensor([[.1805,.047,0],[.1805,-.047,0],[-.1805,.047,0],[-.1805,-.047,0]],device=device)
        angles=qp.reshape(-1,4,3);ch,sh=torch.cos(angles[:,:,0]),torch.sin(angles[:,:,0]);st,ct=torch.sin(angles[:,:,1]),torch.cos(angles[:,:,1]);side=torch.tensor([.0838,-.0838,.0838,-.0838],device=device)
        local=torch.stack([-.2*st,ch*side+.2*sh*ct,sh*side-.2*ch*ct],dim=-1)+hip
        knee=rp[:,None]+iq.quat_rotate(rq[:,None].expand(-1,4,-1).reshape(-1,4),local.reshape(-1,3)).reshape(-1,4,3)
        amp=torch.cat([amp,(kp[:,:,2]<.045).float(),(knee[:,:,2]<.06).float()],dim=-1)
        assert amp.shape==(60,117)
        amp[:,[10,11]]=0;amp[:,85:97]=0
        return llc.eval_enc(norm_amp(amp).reshape(1,-1))
    def rb_array(rb):
        return np.column_stack([rb['pose']['p'][a] for a in 'xyz']+[rb['pose']['r'][a] for a in 'xyzw']+[rb['vel']['linear'][a] for a in 'xyz']+[rb['vel']['angular'][a] for a in 'xyz']).astype(np.float32)
    results=[]
    with torch.no_grad():
        latents={} if args.random_episodes else {key:reference_latent(raw['names'].index(name)) for key,_,name in selected}
        for controller in (['HLC'] if args.random_episodes else ['LLC'] if args.stand_seconds is not None else ['LLC','HLC']):
            for key,label,name in selected if (controller=='LLC' or args.random_episodes) else selected[:4]:
                index=raw['names'].index(name);length=float(library._motion_lengths[index]);duration=(args.stand_seconds if args.stand_seconds is not None else length) if controller=='LLC' else 10. if args.random_episodes else 5.
                case=out/(controller.lower()+'_'+key);case.mkdir()
                episode_seed=args.seed+selected.index((key,label,name))
                command_sequence,knots=(balanced_commands(args.seed,selected.index((key,label,name))) if args.balanced else smooth_random_commands(episode_seed)) if args.random_episodes else (None,None)
                if knots is not None:(case/'commands.json').write_text(json.dumps(dict(seed=episode_seed,columns=['world_vx_mps','world_vy_mps','world_facing_yaw_rad'],knots=knots),indent=2),encoding='utf-8')
                params=gymapi.SimParams();params.dt=1/60;params.substeps=2;params.up_axis=gymapi.UP_AXIS_Z;params.gravity=gymapi.Vec3(0,0,-9.81);params.physx.use_gpu=True;params.use_gpu_pipeline=False;params.physx.solver_type=1;params.physx.num_position_iterations=4;params.physx.num_velocity_iterations=0;params.physx.contact_offset=.02;params.physx.rest_offset=0
                sim=g.create_sim(0,0,gymapi.SIM_PHYSX,params);assert sim is not None
                try:
                    plane=gymapi.PlaneParams();plane.normal=gymapi.Vec3(0,0,1);plane.static_friction=1;plane.dynamic_friction=1;g.add_ground(sim,plane)
                    opt=gymapi.AssetOptions();opt.angular_damping=.01;opt.max_angular_velocity=100;opt.default_dof_drive_mode=gymapi.DOF_MODE_NONE
                    opt.flip_visual_attachments=args.calm_no_self_collision
                    asset=g.load_asset(sim,str(model.parent),model.name,opt);assert asset is not None
                    env=g.create_env(sim,gymapi.Vec3(-2,-2,0),gymapi.Vec3(2,2,2),1)
                    ep=raw['episodes'][index];pose=gymapi.Transform()
                    xyz=ep['root_pos'][0].numpy().copy();xyz[:2]=0
                    quat=ep['root_rot'][0].numpy() if controller=='LLC' else np.array([0,0,0,1])
                    pose.p=gymapi.Vec3(*xyz) if controller=='LLC' else gymapi.Vec3(0,0,.35);pose.r=gymapi.Quat(*quat)
                    actor=g.create_actor(env,asset,pose,'evaluated_a1',0,collision_filter)
                    names=g.get_actor_rigid_body_names(env,actor);joint_names=g.get_actor_dof_names(env,actor)
                    assert joint_names==original_joints
                    ds=np.zeros(12,dtype=gymapi.DofState.dtype)
                    ds['pos']=ep['dof_pos'][0].numpy() if controller=='LLC' else np.array([0,.67,-1.3]*4)
                    ds['vel']=ep['dof_vel'][0].numpy() if controller=='LLC' else 0
                    g.set_actor_dof_states(env,actor,ds,gymapi.STATE_ALL)
                    props=g.get_actor_dof_properties(env,actor);props['driveMode'].fill(gymapi.DOF_MODE_POS);props['stiffness'].fill(40);props['damping'].fill(1.5);props['effort'].fill(15);g.set_actor_dof_properties(env,actor,props)
                    if args.random_episodes:g.enable_actor_dof_force_sensors(env,actor)
                    g.set_actor_dof_position_targets(env,actor,ds['pos'].copy())
                    cp=gymapi.CameraProperties();cp.width=960;cp.height=544;camera=g.create_camera_sensor(env,cp);assert camera>=0;g.prepare_sim(sim)
                    root_id=names.index('base' if args.calm_no_self_collision else 'trunk')
                    if controller=='LLC':
                        initial=g.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)
                        for field,values in [('linear',ep['root_vel'][0]),('angular',ep['root_ang_vel'][0])]:
                            for axis,value in zip('xyz',values):initial['vel'][field][root_id][axis]=float(value)
                        g.set_actor_rigid_body_states(env,actor,initial,gymapi.STATE_VEL)
                    command={'forward':[.3,0,0],'backward':[-.3,0,0],'left':[0,.3,0],'right':[0,-.3,0]}.get(key,[0,0,0])
                    states=[];qstates=[];target_log=[];reason='duration';joint_errors=[]
                    roots_all=[];qs_all=[];dqs_all=[];feet_all=[];contacts_all=[];actions_all=[];latents_all=[];commands_all=[];forces_all=[]
                    feet_ids=[names.index(leg+'_foot') for leg in ['FR','FL','RR','RL']]
                    def capture_sample(physical):
                        ds_sample=g.get_actor_dof_states(env,actor,gymapi.STATE_ALL)
                        roots_all.append(physical[root_id].copy());qs_all.append(ds_sample['pos'].copy());dqs_all.append(ds_sample['vel'].copy());feet_all.append(physical[feet_ids,:3].copy())
                        contacts=g.get_env_rigid_contacts(env)
                        contacts_all.append([any((int(c['body0'])==-1 and int(c['body1'])==f) or (int(c['body1'])==-1 and int(c['body0'])==f) for c in contacts) for f in feet_ids])
                    if args.random_episodes:capture_sample(rb_array(g.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL)))
                    with imageio.get_writer(str(case/'rollout.mp4'),fps=30) as writer:
                        for step in range(max(1,int(np.ceil(duration*30)))):
                            physical=rb_array(g.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL))
                            virtual=torch.tensor(virtual_body_states(physical,names,expected),device=device).unsqueeze(0)
                            obs=ns['compute_humanoid_observations_max'](virtual[:,:,:3],virtual[:,:,3:7],virtual[:,:,7:10],virtual[:,:,10:13],True,True)
                            assert obs.shape==(1,343) and torch.isfinite(obs).all()
                            if command_sequence is not None:command=command_sequence[step].tolist()
                            z=latents.get(key)
                            if controller=='HLC':
                                r=torch.tensor(physical[root_id],device=device).unsqueeze(0)
                                cmd=torch.tensor([[command[0],command[1],0]],device=device,dtype=torch.float32)
                                head=iq.quat_rotate(tu.calc_heading_quat_inv(r[:,3:7]),cmd)[:,:2]
                                face=iq.quat_rotate_inverse(r[:,3:7],torch.tensor([[math.cos(command[2]),math.sin(command[2]),0.]],device=device))
                                up=iq.quat_rotate_inverse(r[:,3:7],torch.tensor([[0.,0,1]],device=device))
                                z=hlc({'obs':norm_hlc(torch.cat([obs,head,face,up],dim=-1))})[0].clamp(-1,1)
                                z=torch.nn.functional.normalize(z,dim=-1)
                            mu,_=llc.eval_actor(norm_llc(obs),z)
                            target=offset+scale*mu.clamp(-1,1)[0]
                            if not torch.isfinite(target).all():reason='nonfinite_policy';break
                            g.set_actor_dof_position_targets(env,actor,target.cpu().numpy());target_log.append(target.cpu().numpy())
                            if args.random_episodes:
                                actions_all.append(mu.clamp(-1,1)[0].cpu().numpy());latents_all.append(z[0].cpu().numpy());commands_all.append(command)
                            for _ in range(2):g.simulate(sim);g.fetch_results(sim,True)
                            physical=rb_array(g.get_actor_rigid_body_states(env,actor,gymapi.STATE_ALL));r=physical[root_id]
                            ds_now=g.get_actor_dof_states(env,actor,gymapi.STATE_ALL);states.append(r.copy());qstates.append(ds_now['pos'].copy())
                            if args.random_episodes:
                                capture_sample(physical);forces_all.append(g.get_actor_dof_forces(env,actor).copy())
                            if controller=='LLC':ref=library.get_motion_state(torch.tensor([index],device=device),torch.tensor([min((step+1)/30,length)],device=device))[2][0]
                            if controller=='LLC':joint_errors.append(float(torch.mean((torch.tensor(ds_now['pos'].copy(),device=device)-ref)**2)))
                            if not np.isfinite(physical).all():reason='nonfinite';break
                            pos=r[:3];g.set_camera_location(camera,env,gymapi.Vec3(*(pos+[1,1,.6])),gymapi.Vec3(*pos));g.step_graphics(sim);g.render_all_camera_sensors(sim)
                            image=g.get_camera_image(sim,env,camera,gymapi.IMAGE_COLOR).reshape(544,960,4);writer.append_data(image[:,:,:3])
                            if not np.isfinite(physical).all():reason='nonfinite';break
                            if 1-2*(r[3]**2+r[4]**2)<0 or r[2]<.06:reason='trunk_fall';break
                    states=np.asarray(states);elapsed=len(states)/30
                    last=states[-1] if len(states) else physical[root_id]
                    direction=np.array(command[:2]);progress=float(last[:2]@direction/np.linalg.norm(direction)) if np.linalg.norm(direction)>0 else None
                    result=dict(controller=controller,motion=label,reference=None if args.random_episodes else name,seed=episode_seed,requested_seconds=duration,elapsed_seconds=elapsed,termination=reason,
                                resets=0,joint_rmse_deg=float(np.degrees(np.sqrt(np.mean(joint_errors)))) if controller=='LLC' else None,
                                final_root_height_m=float(last[2]),reference_start_height_m=None if args.random_episodes else float(ep['root_pos'][0,2]),reference_end_height_m=None if args.random_episodes else float(ep['root_pos'][-1,2]),
                                xy_displacement_m=last[:2].tolist(),command_velocity_mps=command[:2] if controller=='HLC' else None,directional_progress_m=None if args.random_episodes else progress,
                                initial_root_velocity_mps=ep['root_vel'][0].tolist() if controller=='LLC' else [0,0,0],
                                reference_delta_xy_m=(ep['root_pos'][-1,:2]-ep['root_pos'][0,:2]).tolist() if controller=='LLC' else None,
                                status='FAIL' if reason!='duration' else 'COMPLETED_NOT_TRACKING_APPROVAL')
                    if args.random_episodes:
                        roots=np.asarray(roots_all);target_array=np.asarray(target_log).reshape(-1,12);commands_array=np.asarray(commands_all).reshape(-1,3)
                        assert len(roots)==len(target_array)+1
                        valid=np.isfinite(roots[:-1]).all(axis=1)&np.isfinite(roots[1:]).all(axis=1)&np.isfinite(target_array).all(axis=1)
                        for samples in [qs_all,dqs_all,feet_all]:
                            finite=np.isfinite(np.asarray(samples).reshape(len(roots),-1)).all(axis=1);valid &= finite[:-1]&finite[1:]
                        for samples in [actions_all,latents_all,commands_all,forces_all]:
                            valid &= np.isfinite(np.asarray(samples).reshape(len(valid),-1)).all(axis=1)
                        np.savez_compressed(case/'trace.npz',root=roots,q=qs_all,qvel=dqs_all,foot_pos=feet_all,foot_floor_contact=contacts_all,
                                            target_q=target_array,llc_action=actions_all,latent=latents_all,command=commands_array,
                                            joint_force_sensor=forces_all,valid_transition=valid,time=np.arange(len(roots))/30,dt=1/30,
                                            joint_names=joint_names,foot_names=['FR_foot','FL_foot','RR_foot','RL_foot'],quaternion_order='xyzw',velocity_frame='world')
                        result.update(valid_transitions=int(valid.sum()),total_transitions=len(valid),
                                      actual_path_length_m=float(np.linalg.norm(np.diff(roots[:,:2],axis=0),axis=1).sum()),
                                      mean_speed_mps=float(np.linalg.norm(roots[1:,7:9],axis=1).mean()),
                                      velocity_rmse_mps=float(np.sqrt(np.mean(np.sum((roots[1:,7:9]-commands_array[:,:2])**2,axis=1)))),
                                      source_contact_definition='foot-floor contacts only',data_kind='Isaac Gym reference rollout; not MJX TT dynamics data')
                    else:np.savez_compressed(case/'trace.npz',root=states,q=qstates,target_q=target_log,dt=1/30)
                    (case/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');results.append(result);print('CASE',json.dumps(result,ensure_ascii=False),flush=True)
                finally:g.destroy_sim(sim)
    (out/'summary.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    rows=''.join('<tr><td>'+r['controller']+'</td><td>'+r['motion']+'</td><td>'+f"{r['elapsed_seconds']:.2f}초"+'</td><td>'+r['termination']+'</td><td>'+str([round(v,3) for v in r['xy_displacement_m']])+'</td><td>'+('미평가' if r['joint_rmse_deg'] is None else f"{r['joint_rmse_deg']:.2f}도")+'</td></tr>' for r in results)
    overview='<table><tr><th>정책</th><th>동작</th><th>길이</th><th>종료 사유</th><th>실제 이동량 [앞뒤, 좌우] m</th><th>관절각 오차</th></tr>'+rows+'</table>'
    cards=[]
    for result in results:
        key=next(k for k,l,_ in selected if l==result['motion']);video=out/(result['controller'].lower()+'_'+key)/'rollout.mp4'
        payload=base64.b64encode(video.read_bytes()).decode()
        cards.append('<section><h2>'+html.escape(result['controller']+' '+result['motion'])+'</h2><video controls muted playsinline src="data:video/mp4;base64,'+payload+'"></video><pre>'+html.escape(json.dumps(result,ensure_ascii=False,indent=2))+'</pre></section>')
    title=('CALM A1 자기 충돌 끔' if args.calm_no_self_collision else 'Menagerie A1')
    heading='비율을 지정한 연속 이동' if args.balanced else '부드러운 무작위 연속 이동' if args.random_episodes else 'LLC 일어나기 시간 연장 진단' if args.stand_seconds is not None else 'LLC 여섯 동작과 HLC 네 방향'
    details='기존 CALM A1의 실제 몸체 23개 관측을 그대로 사용했습니다. 바닥 충돌은 유지하고 로봇 부품 사이 자기 충돌만 껐습니다.' if args.calm_no_self_collision else '실제 몸체 13개에서 원래 관측의 가상 지점 23개를 구성했습니다. 질량·충돌 형상은 Menagerie 모델입니다.'
    task_details='속도 목표는 최대 0.6 m/s이며 방향과 몸통 방향 목표를 무작위로 바꿉니다. 전환 구간은 1~2.5초이고 smoothstep으로 연결합니다. 상태는 T+1개, 행동과 명령은 T개를 저장합니다. 자기 관통 여부와 MJX 재현성은 검사하지 않았습니다.' if args.random_episodes else 'HLC에는 앉기·일어나기 명령이 없어 해당 두 동작은 LLC로 검사했습니다. LLC의 앉기·일어나기는 참고 모션의 최종 높이와 함께 확인해야 합니다.'
    if args.balanced:task_details='동일 길이 5초 구간 20개를 전진 30%, 후진 20%, 왼쪽 15%, 오른쪽 15%, 전진 회전 15%, 정지·재출발 5%로 배정했습니다. 전환 시간도 해당 구간에 포함됩니다. 실제 동작 비율은 명령 비율과 다를 수 있습니다. 자기 충돌을 끈 조건이며 관통과 MJX 재현성은 미검증입니다.'
    (out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>'+title+' 정책 검증</title><style>body{max-width:1100px;margin:30px auto;font:17px system-ui;padding:15px}video{width:100%}pre{white-space:pre-wrap}section{margin:30px 0}</style><h1>'+title+': '+heading+'</h1><p>원본 신경망 가중치를 엄격하게 로딩했습니다. '+details+' 원래 학습 환경과 물리 조건이 완전히 같은지는 별도 검증이 필요합니다. 기록 종료 없이 실행됐다는 것만으로 정상 추종이라고 판정하지 않습니다. '+task_details+'</p>'+overview+''.join(cards)+'</html>',encoding='utf-8')


if __name__=='__main__':main()
