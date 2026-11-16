"""[chain 사본] 조이스틱 계획 (calm_joystick.py) 으로 CALM 수집. 원본: legacy/code/motion_tracking/collect_calm_sequences.py.
변경: 명령 = 매 step 실제 몸통 yaw 기준 조이스틱 (vx, vy, wz) → [월드 vx, vy, 목표 yaw = 현재 yaw + wz·0.5 s], 구간 latent 혼합 없음,
품질 = 넘어짐·비유한값·활동 검사 (구간 동작 검사 대신), 재시도 = 새 조이스틱 seed.
Headless batched CALM collection with resumable accepted/failed journal.

Use the rlgpu Python (Isaac Gym must be imported before Torch).
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
import time

_Q=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(_Q/'legacy/code/motion_tracking'));sys.path.insert(0,str(_Q/'code/motion_tracking'));sys.path.insert(0,str(Path(__file__).resolve().parent))
from calm_sequences import FPS, motion_feature
from calm_joystick import commands_for, make_plan as joystick_plan
from eval_menagerie_calm import original_functions
LEAD=0.5  # 목표 yaw = 현재 yaw + wz × LEAD (HEADING='lead')
# 2026-11-12 진단: HLC 는 방향 오차가 있으면 오차 크기와 상관없이 약 2 rad/s 로 돈다 (wz -0.5/-1.0/-1.5 → 실제 -2.0).
# 'integrate' = 목표 yaw 를 wz 로 적분 (목표 방향이 wz 속도로 움직임), 실제 방향과의 차이는 MAX_ERR 로 제한.
HEADING=os.environ.get('CALM_HEADING','step')  # 2026-11-12 사용자 결정: step 0.5 초 (12 모션 비교, check/calm_example/heading_sweep); MAX_ERR=float(os.environ.get('CALM_MAX_ERR','0.3'))
# 'stepabs' = step 과 같은 주기로 목표를 바꾸되, 목표 = 시작 방향 + (다음 갱신 시점까지의 명령 wz 적분) (현재 방향 기준이 아니라 누적 오차가 안 쌓임).
# 'step' = CALM HLC 학습 때처럼 계단형 목표: 구간 시작과 그 뒤 STEP_PERIOD 초마다 목표 = 현재 방향 + (다음 STEP_PERIOD 초 동안의 wz 합)/FPS, 그 사이 고정.
STEP_PERIOD=float(os.environ.get('CALM_STEP_PERIOD','0.5'))

def resegment(p):
    from calm_joystick import make_joystick
    keys=make_joystick(p['seed'],p['seconds'],p.get('weights'),p.get('refs'),p.get('main'))[1]
    return dict(p,segments=[dict(index=0,start=k['start'],end=k['end'],label=k['label']) for k in keys])

def quality(plan,root,q):
    frames=plan['frames']
    if root.shape!=(frames+1,13) or q.shape!=(frames+1,12) or not np.isfinite(root).all() or not np.isfinite(q).all():return False,'incomplete_or_nonfinite',[]
    upright=1-2*(root[:,3]**2+root[:,4]**2)
    if upright.min()<.35 or root[:,2].min()<.055:return False,'fall',[]
    yaw=np.unwrap(np.arctan2(2*(root[:,6]*root[:,5]+root[:,3]*root[:,4]),1-2*(root[:,4]**2+root[:,5]**2)))
    lag=FPS;position=np.linalg.norm(root[lag:,:2]-root[:-lag,:2],axis=1);heading=np.abs(yaw[lag:]-yaw[:-lag])
    active=(position>=.03)|(heading>=np.deg2rad(5));idle=longest=0
    for m in active:
        idle=0 if m else idle+1;longest=max(longest,idle)
    ok=bool(active.mean()>=.6 and longest/FPS<=3.0)
    return ok,'pass' if ok else 'inactive',[dict(active_fraction=float(active.mean()),longest_inactive_seconds=longest/FPS)]

import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CALM=ROOT.parent/'external/a1-calm-main'


def load_policy(wheel):
    sys.path.insert(0,str(wheel.resolve()));sys.path.insert(0,str(CALM/'calm'))
    from isaacgym import gymapi, gymtorch
    import numpy as np
    if 'float' not in np.__dict__:np.float=float
    import isaacgym.torch_utils as iq
    import torch
    import yaml
    from utils import torch_utils as tu
    from utils.rollout_motion_lib import RolloutMotionLib
    from learning.calm_network_builder import CALMBuilder
    from learning.hrl_network_builder import HRLBuilder
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    if not torch.cuda.is_available():raise RuntimeError('CALM collection requires a working CUDA device')
    device='cuda:0';torch.set_num_threads(2);torch.manual_seed(0)
    ns=dict(vars(iq));ns.update(torch=torch,torch_utils=tu)
    original_functions(CALM/'calm/env/tasks/humanoid.py',['dof_to_obs','compute_humanoid_observations_max'],ns)
    original_functions(CALM/'calm/env/tasks/humanoid_amp.py',['build_amp_observations'],ns)
    def network(filename,config,builder,name,kwargs):
        ck=torch.load(CALM/'checkpoints'/filename,map_location=device,weights_only=True)
        b=builder();b.load(yaml.safe_load((CALM/'calm/data/cfg/train/rlg'/config).read_text())['params']['network'])
        net=b.build(name,**kwargs).to(device).eval();net.load_state_dict({k[len('a2c_network.'):]:v for k,v in ck['model'].items()},strict=True)
        return net,ck
    llc,lc=network('lowLevelController.pth','calm_a1_31psc8n.yaml',CALMBuilder,'calm',dict(actions_num=12,input_shape=(343,),amp_input_shape=(1170,),amp_obs_steps=10,calm_latent_shape=(64,)))
    hlc,hc=network('highLevelPolicy.pth','hrl_a1_heading_lat31zs.yaml',HRLBuilder,'hrl',dict(actions_num=64,input_shape=(351,)))
    def norm(ck,key,size):
        n=RunningMeanStd((size,)).to(device).eval();n.load_state_dict(ck[key],strict=True);return n
    norm_llc=norm(lc,'running_mean_std',343);norm_hlc=norm(hc,'running_mean_std',351);norm_amp=norm(lc,'amp_input_mean_std',117)
    mf=CALM/'calm/data/motions_a1/v2v3_31motions_squat2s_bfcut.pt'
    raw=torch.load(mf,map_location='cpu',weights_only=True)
    library=RolloutMotionLib(str(mf),list(range(12)),list(range(13)),[0,1,2,3],True,device=device)
    def latent(name):
        index=raw['names'].index(name);end=float(library._motion_lengths[index])
        times=(end-torch.arange(60,device=device)/30).clamp(min=0);ids=torch.full((60,),index,device=device,dtype=torch.long)
        rp,rq,qp,rv,rw,qv,kp=library.get_motion_state(ids,times)
        amp=ns['build_amp_observations'](rp,rq,rv,rw,qp,qv,kp,True,True,72,list(range(13)),True,True)
        hip=torch.tensor([[.1805,.047,0],[.1805,-.047,0],[-.1805,.047,0],[-.1805,-.047,0]],device=device)
        angles=qp.reshape(-1,4,3);ch,sh=torch.cos(angles[:,:,0]),torch.sin(angles[:,:,0]);st,ct=torch.sin(angles[:,:,1]),torch.cos(angles[:,:,1]);side=torch.tensor([.0838,-.0838,.0838,-.0838],device=device)
        local=torch.stack([-.2*st,ch*side+.2*sh*ct,sh*side-.2*ch*ct],dim=-1)+hip
        knee=rp[:,None]+iq.quat_rotate(rq[:,None].expand(-1,4,-1).reshape(-1,4),local.reshape(-1,3)).reshape(-1,4,3)
        amp=torch.cat([amp,(kp[:,:,2]<.045).float(),(knee[:,:,2]<.06).float()],dim=-1)
        assert amp.shape==(60,117)
        amp[:,[10,11]]=0;amp[:,85:97]=0
        return llc.eval_enc(norm_amp(amp).reshape(1,-1))[0]
    with torch.no_grad():latents={10:latent('v2_sitDown')}
    return locals()


class Simulator:
    def __init__(self,r,n):
        self.r,self.n=r,n;g=r['gymapi'].acquire_gym();self.g=g
        api=r['gymapi'];np=r['np'];torch=r['torch']
        p=api.SimParams();p.dt=1/60;p.substeps=2;p.up_axis=api.UP_AXIS_Z;p.gravity=api.Vec3(0,0,-9.81)
        p.physx.use_gpu=True;p.use_gpu_pipeline=True;p.physx.solver_type=1;p.physx.num_position_iterations=4;p.physx.num_velocity_iterations=0;p.physx.contact_offset=.02;p.physx.rest_offset=0
        self.sim=g.create_sim(0,-1,api.SIM_PHYSX,p)
        if self.sim is None:raise RuntimeError('Isaac Gym create_sim failed')
        plane=api.PlaneParams();plane.normal=api.Vec3(0,0,1);plane.static_friction=plane.dynamic_friction=1.;g.add_ground(self.sim,plane)
        assetdir=CALM/'assets/urdf/a1_description/urdf';opt=api.AssetOptions();opt.angular_damping=.01;opt.max_angular_velocity=100;opt.default_dof_drive_mode=api.DOF_MODE_NONE;opt.flip_visual_attachments=True
        asset=g.load_asset(self.sim,str(assetdir),'a1.urdf',opt)
        self.names=g.get_asset_rigid_body_names(asset);self.joints=g.get_asset_dof_names(asset)
        assert len(self.names)==23 and len(self.joints)==12
        self.feet=[self.names.index(leg+'_foot') for leg in ['FR','FL','RR','RL']]
        origins=[]
        for i in range(n):
            env=g.create_env(self.sim,api.Vec3(-3,-3,0),api.Vec3(3,3,2),int(np.ceil(np.sqrt(n))))
            pose=api.Transform();pose.p=api.Vec3(0,0,.35)
            actor=g.create_actor(env,asset,pose,'calm',i,1)
            props=g.get_actor_dof_properties(env,actor);props['driveMode'].fill(api.DOF_MODE_POS);props['stiffness'].fill(40);props['damping'].fill(1.5);props['effort'].fill(15)
            g.set_actor_dof_properties(env,actor,props);g.enable_actor_dof_force_sensors(env,actor)
            o=g.get_env_origin(env);origins.append([o.x,o.y,o.z])
        g.prepare_sim(self.sim)
        wrap=r['gymtorch'].wrap_tensor
        self.root=wrap(g.acquire_actor_root_state_tensor(self.sim)).view(n,13)
        self.dof=wrap(g.acquire_dof_state_tensor(self.sim)).view(n,12,2)
        self.body=wrap(g.acquire_rigid_body_state_tensor(self.sim)).view(n,23,13)
        self.contact=wrap(g.acquire_net_contact_force_tensor(self.sim)).view(n,23,3)
        self.force=wrap(g.acquire_dof_force_tensor(self.sim)).view(n,12)
        self.origins=torch.tensor(origins,device='cuda:0')
        import xml.etree.ElementTree as ET
        xml=ET.parse(assetdir/'a1.urdf').getroot()
        limits=np.array([[float(xml.find("joint[@name='%s']/limit"%j).get(k)) for k in ['lower','upper']] for j in self.joints])
        self.offset=torch.tensor(limits.mean(1),device='cuda:0',dtype=torch.float32)
        self.scale=torch.tensor(.7*np.diff(limits,axis=1)[:,0],device='cuda:0',dtype=torch.float32)

    def refresh(self):
        for name in ['actor_root_state','dof_state','rigid_body_state','net_contact_force','dof_force']:
            getattr(self.g,'refresh_'+name+'_tensor')(self.sim)

    def rollout(self,plans):
        r=self.r;torch=r['torch'];np=r['np'];iq=r['iq'];tu=r['tu'];g=self.g;unwrap=r['gymtorch'].unwrap_tensor
        self.root.zero_();self.root[:,:3]=self.origins;self.root[:,2]+=.35;self.root[:,6]=1
        self.dof.zero_();self.dof[:,:,0]=torch.tensor([0,.67,-1.3]*4,device='cuda:0')
        for i,plan in enumerate(plans):
            if plan['initial_pose']=='seated':
                ep=r['raw']['episodes'][r['raw']['names'].index('v2_standUp')]
                self.root[i,2]=self.origins[i,2]+float(ep['root_pos'][0,2]);self.root[i,3:7]=ep['root_rot'][0].to('cuda:0');self.dof[i,:,0]=ep['dof_pos'][0].to('cuda:0')
        g.set_actor_root_state_tensor(self.sim,unwrap(self.root));g.set_dof_state_tensor(self.sim,unwrap(self.dof))
        targets=self.dof[:,:,0].contiguous();g.set_dof_position_target_tensor(self.sim,unwrap(targets))
        # Refresh rigid-body FK after state setters before the first policy observation.
        for _ in range(2):g.simulate(self.sim);g.fetch_results(self.sim,True)
        self.refresh()
        arrays={k:[] for k in ['root','q','qvel','foot_pos','foot_floor_contact','target_q','llc_action','latent','command','joint_force_sensor']}
        def capture():
            root=self.root.clone();root[:,:3]-=self.origins;arrays['root'].append(root.cpu().numpy())
            arrays['q'].append(self.dof[:,:,0].cpu().numpy().copy());arrays['qvel'].append(self.dof[:,:,1].cpu().numpy().copy())
            arrays['foot_pos'].append((self.body[:,self.feet,:3]-self.origins[:,None,:]).cpu().numpy())
            arrays['foot_floor_contact'].append((self.contact[:,self.feet,2]>1.).cpu().numpy())
        capture();previous=np.zeros((self.n,3));entry=previous.copy();yaw_entry=np.zeros(self.n);last_z=None;entry_z=None
        frames=plans[0]['frames']
        assert all(p['frames']==frames for p in plans)
        joy=[commands_for(p) for p in plans];tyaw=None
        for step in range(frames):
            physical=self.body
            obs=r['ns']['compute_humanoid_observations_max'](physical[:,:,:3],physical[:,:,3:7],physical[:,:,7:10],physical[:,:,10:13],True,True)
            if not torch.isfinite(obs).all():raise FloatingPointError('nonfinite physical observation')
            kinds=[];blend=[];commands=[]
            roots=self.root.cpu().numpy()
            for i,plan in enumerate(plans):
                w=roots[i,6];x,y,z=roots[i,3:6];yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z));vx,vy,wz=joy[i][step]
                if HEADING=='lead':face_yaw=yaw+wz*LEAD
                elif HEADING=='stepabs':
                    if tyaw is None:tyaw=np.zeros(len(plans));yaw0=np.zeros(len(plans))
                    k=max(1,round(STEP_PERIOD*FPS))
                    if step==0:yaw0[i]=yaw
                    if step%k==0:
                        goal=yaw0[i]+joy[i][:min(step+k,len(joy[i])),2].sum()/FPS
                        tyaw[i]=yaw+np.clip(np.angle(np.exp(1j*(goal-yaw))),-2.5,2.5)
                    face_yaw=tyaw[i]
                elif HEADING=='step':
                    if tyaw is None:tyaw=np.zeros(len(plans))
                    starts={sg['start'] for sg in plan['segments']}
                    last=max(st for st in starts|{0} if st<=step)
                    if step==0 or (step-last)%max(1,round(STEP_PERIOD*FPS))==0:
                        k=max(1,round(STEP_PERIOD*FPS));nxt=min([st for st in starts if st>step]+[len(joy[i])]);e=min(step+k,nxt)
                        tyaw[i]=yaw+joy[i][step:e,2].sum()/FPS
                    face_yaw=tyaw[i]
                else:
                    if tyaw is None:tyaw=np.zeros(len(plans))
                    if step==0:tyaw[i]=yaw
                    tyaw[i]+=wz/FPS;err=np.angle(np.exp(1j*(tyaw[i]-yaw)));tyaw[i]=yaw+np.clip(err,-MAX_ERR,MAX_ERR);face_yaw=tyaw[i]
                commands.append([vx*np.cos(yaw)-vy*np.sin(yaw),vx*np.sin(yaw)+vy*np.cos(yaw),face_yaw]);kinds.append(0);blend.append(1.)
            previous=np.asarray(commands);cmd=torch.tensor(previous,device='cuda:0',dtype=torch.float32)
            velocity=torch.cat([cmd[:,:2],torch.zeros(self.n,1,device='cuda:0')],dim=1)
            head=iq.quat_rotate(tu.calc_heading_quat_inv(self.root[:,3:7]),velocity)[:,:2]
            face=iq.quat_rotate_inverse(self.root[:,3:7],torch.stack([torch.cos(cmd[:,2]),torch.sin(cmd[:,2]),torch.zeros(self.n,device='cuda:0')],dim=1))
            up=iq.quat_rotate_inverse(self.root[:,3:7],torch.tensor([[0.,0.,1.]],device='cuda:0').expand(self.n,3))
            z=r['hlc']({'obs':r['norm_hlc'](torch.cat([obs,head,face,up],dim=1))})[0].clamp(-1,1)
            z=torch.nn.functional.normalize(z,dim=-1)
            for i,kind in enumerate(kinds):
                if kind==10:z[i]=r['latents'][kind]
                # The fixed stand-up latent stayed seated in the pilot; HLC's zero-speed upright target is tested instead.
            if last_z is not None:
                if entry_z is None:entry_z=last_z.clone()
                for i,plan in enumerate(plans):
                    if any(s['start']==step for s in plan['segments']):entry_z[i]=last_z[i]
                alpha=torch.tensor(blend,device='cuda:0',dtype=torch.float32)[:,None]
                z=torch.nn.functional.normalize(entry_z*(1-alpha)+z*alpha,dim=-1)
            last_z=z.clone()
            mu,_=r['llc'].eval_actor(r['norm_llc'](obs),z)
            targets=(self.offset+self.scale*mu.clamp(-1,1)).contiguous()
            if not torch.isfinite(targets).all():raise FloatingPointError('nonfinite CALM target')
            arrays['target_q'].append(targets.cpu().numpy());arrays['llc_action'].append(mu.clamp(-1,1).cpu().numpy());arrays['latent'].append(z.cpu().numpy());arrays['command'].append(previous.copy())
            g.set_dof_position_target_tensor(self.sim,unwrap(targets))
            for _ in range(2):g.simulate(self.sim);g.fetch_results(self.sim,True)
            self.refresh();arrays['joint_force_sensor'].append(self.force.cpu().numpy().copy());capture()
        return {k:np.asarray(v).swapaxes(0,1) for k,v in arrays.items()}


def atomic_npz(path,**arrays):
    import numpy as np
    temporary=path.with_suffix('.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f,**arrays);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--batch',type=int,default=16);p.add_argument('--max-attempts',type=int,default=50000);p.add_argument('--duplicate-distance',type=float,default=.03);p.add_argument('--rl-games-wheel',type=Path,default=ROOT/'runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl');a=p.parse_args()
    if a.batch<1 or a.max_attempts<1 or not 0<a.duplicate_distance<1:p.error('invalid batch, attempts, or duplicate threshold')
    out=a.out.resolve();planfile=out/'plan.json'
    if not planfile.exists():p.error('create plan.json with chain/calm_joystick.py first')
    lock=(out/'collection.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plans=json.loads(planfile.read_text())['motions'];(out/'accepted').mkdir(exist_ok=True)
    records=[];journal=out/'records.jsonl'
    if journal.exists():
        lines=journal.read_text().splitlines()
        for i,line in enumerate(lines):
            try:records.append(json.loads(line))
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise
        journal.write_text(''.join(json.dumps(r)+'\n' for r in records))
    accepted={r['id']:r for r in records if r['accepted']};attempts=[0]*len(plans)
    for r in records:attempts[r['id']]+=1
    import numpy as np
    features=np.empty((len(plans),544),np.float32);feature_count=0;usage=np.zeros(12)
    for record in accepted.values():
        with np.load(out/record['file']) as d:features[feature_count]=d['feature']
        feature_count+=1
        for s in record['plan']['segments']:usage[s['index']]+=s['end']-s['start']
    config=dict(plan_sha256=hashlib.sha256(planfile.read_bytes()).hexdigest(),duplicate_distance=a.duplicate_distance,
                source_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('calm_joystick.py')]},
                checkpoint_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (CALM/'checkpoints').glob('*.pth')},
                duration_seconds=sorted({p['seconds'] for p in plans}),command=('joystick body-frame vx,vy,wz; facing=yaw+wz*%.1f'%LEAD) if HEADING=='lead' else ('joystick body-frame vx,vy,wz; facing=step target every %.1fs within segments'%STEP_PERIOD if HEADING=='step' else 'joystick body-frame vx,vy,wz; facing=integrated wz, |facing-yaw|<=%.2f'%MAX_ERR),
                temporal_validation=dict(lag_seconds=1,position_m=.03,heading_degrees=5,height_m=.02,min_active_fraction=.5,max_idle_seconds=1.5,pose_hold_allowed=False),
                physics=dict(dt=1/60,substeps=2,control_dt=1/30,kp=40,kd=1.5,effort=15,self_collision=False),
                stand_up_controller='HLC zero-speed upright target; actual body rise required',
                data_kind='CALM A1 PhysX rollout; not TT dynamics training data')
    configpath=out/'collection_config.json'
    if configpath.exists() and json.loads(configpath.read_text())!=config:raise ValueError('resume configuration/code/checkpoint mismatch')
    configpath.write_text(json.dumps(config,indent=2))
    r=load_policy(a.rl_games_wheel);n=min(a.batch,len(plans));sim=Simulator(r,n);start=time.monotonic()
    try:
        with r['torch'].no_grad():
            while len(accepted)<len(plans) and len(records)<a.max_attempts:
                remaining=sorted((i for i in range(len(plans)) if i not in accepted),key=lambda i:(attempts[i],i))
                duration=plans[remaining[0]]['frames']
                chosen=[i for i in remaining if plans[i]['frames']==duration][:min(n,a.max_attempts-len(records))];batch=[]
                for i in chosen:
                    plan=plans[i] if attempts[i]==0 else resegment(dict(plans[i],seed=plans[i]['seed']+attempts[i]*104729))  # 재시도 = 새 조이스틱 seed
                    plan=dict(plan,id=i);batch.append(plan)
                padded=batch+[batch[-1]]*(n-len(batch));traces=sim.rollout(padded)
                for j,plan in enumerate(batch):
                    i=plan['id'];attempts[i]+=1;ok,reason,checks=quality(plan,traces['root'][j],traces['q'][j]);distance=None
                    if not all(np.isfinite(v[j]).all() for v in traces.values()):ok=False;reason='nonfinite'
                    feature=motion_feature(traces['root'][j],traces['q'][j])
                    if ok and feature_count:
                        distance=float(np.sqrt(np.maximum(np.mean((features[:feature_count]-feature)**2,axis=1),0)).min())
                        if distance<a.duplicate_distance:ok=False;reason='near_duplicate'
                    record=dict(id=i,attempt=attempts[i],accepted=bool(ok),reason=reason,nearest_distance=distance,plan=plan,checks=checks)
                    arrays={key:value[j] for key,value in traces.items()}
                    frames=plan['segments'][-1]['end']
                    if ok:
                        record['file']='accepted/%05d.npz'%i
                        atomic_npz(out/record['file'],**arrays,feature=feature,time=np.arange(frames+1)/FPS,dt=1/FPS,
                                   valid_transition=np.ones(frames,dtype=bool),joint_names=np.array(sim.joints),
                                   foot_names=np.array(['FR_foot','FL_foot','RR_foot','RL_foot']),
                                   quaternion_order='xyzw',velocity_frame='world',
                                   motion_index=np.array([s['index'] for s in plan['segments']]),segment_start=np.array([s['start'] for s in plan['segments']]),segment_end=np.array([s['end'] for s in plan['segments']]))
                        accepted[i]=record;features[feature_count]=feature;feature_count+=1
                        for s in plan['segments']:usage[s['index']]+=s['end']-s['start']
                    elif len(records)<16:
                        (out/'diagnostics').mkdir(exist_ok=True);atomic_npz(out/'diagnostics'/('%05d_attempt%d.npz'%(i,attempts[i])),**arrays)
                    with journal.open('a') as f:f.write(json.dumps(record)+'\n');f.flush();os.fsync(f.fileno())
                    records.append(record)
                progress=dict(target=len(plans),accepted=len(accepted),attempts=len(records),seconds=time.monotonic()-start,status='running')
                (out/'status.json').write_text(json.dumps(progress,indent=2));print(json.dumps(progress),flush=True)
    finally:
        sim.g.destroy_sim(sim.sim)
        counts=dict(target=len(plans),accepted=len(accepted),attempts=len(records));counts['status']='complete' if counts['accepted']==counts['target'] else 'incomplete'
        (out/'status.json').write_text(json.dumps(counts,indent=2));print(counts,flush=True)


if __name__=='__main__':main()
