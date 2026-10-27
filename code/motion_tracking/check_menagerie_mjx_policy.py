"""Closed-loop CALM transfer test on the native Menagerie MJX model."""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET
import numpy as np
from eval_menagerie_calm import original_functions,virtual_body_states,balanced_commands


def load_policy(workspace,wheel):
    import torch,yaml
    torch.set_num_threads(1)
    # Load only the original pure tensor math; Isaac Gym binary bindings are not used.
    if 'float' not in np.__dict__:np.float=float
    spec=importlib.util.spec_from_file_location('isaacgym.torch_utils',Path.home()/'software/isaacgym/python/isaacgym/torch_utils.py')
    iq=importlib.util.module_from_spec(spec);spec.loader.exec_module(iq)
    gym=types.ModuleType('isaacgym');gym.torch_utils=iq;sys.modules['isaacgym']=gym;sys.modules['isaacgym.torch_utils']=iq
    calm=workspace/'external/a1-calm-main';sys.path.insert(0,str(wheel));sys.path.insert(0,str(calm/'calm'))
    from utils import torch_utils as tu
    from learning.calm_network_builder import CALMBuilder
    from learning.hrl_network_builder import HRLBuilder
    from rl_games.algos_torch.running_mean_std import RunningMeanStd
    networks=[];norms=[]
    for filename,cfgname,builder,size,extra in [('lowLevelController.pth','calm_a1_31psc8n.yaml',CALMBuilder(),343,dict(actions_num=12,amp_input_shape=(1170,),amp_obs_steps=10,calm_latent_shape=(64,))),('highLevelPolicy.pth','hrl_a1_heading_lat31zs.yaml',HRLBuilder(),351,dict(actions_num=64))]:
        ck=torch.load(calm/'checkpoints'/filename,map_location='cpu',weights_only=True)
        cfg=yaml.safe_load((calm/'calm/data/cfg/train/rlg'/cfgname).read_text());builder.load(cfg['params']['network'])
        net=builder.build('calm',input_shape=(size,),**extra).eval();net.load_state_dict({k[len('a2c_network.'):]:v for k,v in ck['model'].items()},strict=True);networks.append(net)
        norm=RunningMeanStd((size,)).eval();norm.load_state_dict(ck['running_mean_std'],strict=True);norms.append(norm)
    ns=dict(vars(iq));ns.update(torch=torch,torch_utils=tu)
    original_functions(calm/'calm/env/tasks/humanoid.py',['dof_to_obs','compute_humanoid_observations_max'],ns)
    legs=['FL','FR','RL','RR'];expected=['base','trunk']+[l+'_'+p for l in legs for p in ['hip','thigh_shoulder','thigh','calf','foot']]+['imu_link'];joints=[l+'_'+p+'_joint' for l in legs for p in ['hip','thigh','calf']]
    xml=ET.parse(calm/'assets/urdf/a1_description/urdf/a1.urdf').getroot();limits=np.array([[float(xml.find("joint[@name='%s']/limit"%n).get(k)) for k in ['lower','upper']] for n in joints]);offset=torch.tensor(limits.mean(1),dtype=torch.float32);scale=torch.tensor(.7*np.diff(limits,axis=1)[:,0],dtype=torch.float32)
    @torch.no_grad()
    def policy(physical,names,command):
        v=torch.tensor(virtual_body_states(physical,names,expected),dtype=torch.float32)[None]
        obs=ns['compute_humanoid_observations_max'](v[:,:,:3],v[:,:,3:7],v[:,:,7:10],v[:,:,10:13],True,True)
        r=v[:,0];cmd=torch.tensor([[*command[:2],0]],dtype=torch.float32)
        head=iq.quat_rotate(tu.calc_heading_quat_inv(r[:,3:7]),cmd)[:,:2]
        face=iq.quat_rotate_inverse(r[:,3:7],torch.tensor([[math.cos(command[2]),math.sin(command[2]),0]],dtype=torch.float32));up=iq.quat_rotate_inverse(r[:,3:7],torch.tensor([[0.,0,1.]]))
        z=networks[1]({'obs':norms[1](torch.cat([obs,head,face,up],-1))})[0].clamp(-1,1);z=torch.nn.functional.normalize(z,dim=-1)
        action=networks[0].eval_actor(norms[0](obs),z)[0].clamp(-1,1)[0]
        target=(offset+scale*action).numpy();assert np.isfinite(target).all()
        return target
    return policy,joints


def body_states(model,data):
    import mujoco
    names=['trunk']+[l+'_'+p for l in ['FL','FR','RL','RR'] for p in ['hip','thigh','calf']]
    states=[]
    for name in names:
        b=model.body(name).id;vel=np.zeros(6);mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_XBODY,b,vel,0)
        states.append(np.r_[data.xpos[b],data.xquat[b][[1,2,3,0]],vel[3:],vel[:3]])
    return np.asarray(states),names


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--wheel',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    import mujoco,jax,jax.numpy as jnp,imageio.v2 as imageio
    from mujoco import mjx
    workspace=Path(__file__).resolve().parents[3];policy,joints=load_policy(workspace,a.wheel.resolve())
    spec=mujoco.MjSpec.from_file(str(workspace/'transformer-transformer/assets/mjcf/unitree_a1/scene.xml'))
    spec.compiler.meshdir=str(workspace/'transformer-transformer/assets/mjcf/unitree_a1/assets');spec.compiler.texturedir=spec.compiler.meshdir
    spec.option.cone=mujoco.mjtCone.mjCONE_PYRAMIDAL;spec.option.impratio=1
    spec.option.timestep=1/600;spec.visual.global_.offwidth=960;spec.visual.global_.offheight=544
    # Match the tested CALM PD controller, rather than stacking it with Menagerie actuators.
    for actuator in list(spec.actuators):spec.delete(actuator)
    for key in list(spec.keys):spec.delete(key)
    for joint in spec.joints:
        joint.damping=[0,0,0];joint.armature=0;joint.frictionloss=0
    for g in spec.geoms:
        if g.contype or g.conaffinity:
            g.contype=2 if g.name=='floor' else 1;g.conaffinity=1 if g.name=='floor' else 2;g.friction=[1,.005,.0001]
    model=spec.compile();(a.out/'model.xml').write_text(spec.to_xml());mm=mjx.put_model(model)
    qa=np.array([model.jnt_qposadr[model.joint(n).id] for n in joints]);va=np.array([model.jnt_dofadr[model.joint(n).id] for n in joints])
    @jax.jit
    def advance(d,target):
        def substep(_,d):
            force=jnp.zeros(model.nv).at[va].set(jnp.clip(40*(target-d.qpos[qa])-1.5*d.qvel[va],-15,15))
            return mjx.step(mm,d.replace(qfrc_applied=force))
        return jax.lax.fori_loop(0,20,substep,d)
    renderer=mujoco.Renderer(model,544,960);results=[];cards=[]
    for episode in range(10):
        commands,segments=balanced_commands(0,episode);data=mujoco.MjData(model);data.qpos[:3]=[0,0,.35];data.qpos[3:7]=[1,0,0,0];data.qpos[qa]=[0,.67,-1.3]*4;mujoco.mj_forward(model,data)
        d=mjx.put_data(model,data);poses=[data.qpos.copy()];targets=[];velocities=[];reason='duration'
        video=a.out/f'episode_{episode:03d}.mp4'
        with imageio.get_writer(video,fps=30) as writer:
            for cmd in commands:
                physical,names=body_states(model,data);target=policy(physical,names,cmd);d=advance(d,jnp.array(target));data.qpos[:]=np.asarray(d.qpos);data.qvel[:]=np.asarray(d.qvel);mujoco.mj_forward(model,data)
                poses.append(data.qpos.copy());targets.append(target);velocities.append(data.qvel[:3].copy())
                if not np.isfinite(data.qpos).all():reason='nonfinite';break
                camera=mujoco.MjvCamera();camera.lookat[:]=data.qpos[:3];camera.distance=1.6;camera.azimuth=45;camera.elevation=-25;renderer.update_scene(data,camera);writer.append_data(renderer.render())
                if data.qpos[2]<.06 or 1-2*(data.qpos[4]**2+data.qpos[5]**2)<0:reason='trunk_fall';break
        n=len(targets);poses=np.array(poses);velocities=np.array(velocities)
        result=dict(episode=episode,seconds=n/30,termination=reason,resets=0,path_length_m=float(np.linalg.norm(np.diff(poses[:,:2],axis=0),axis=1).sum()),velocity_rmse_mps=float(np.sqrt(np.mean(np.sum((velocities[:,:2]-commands[:n,:2])**2,axis=1)))),final_height_m=float(poses[-1,2]));results.append(result)
        np.savez_compressed(a.out/f'episode_{episode:03d}.npz',qpos=poses,target_q=targets,command=commands[:n],velocity=velocities,dt=1/30,joint_names=joints)
        cards.append('<h2>구간 '+str(episode)+'</h2><video style="width:100%" controls muted playsinline src="data:video/mp4;base64,'+base64.b64encode(video.read_bytes()).decode()+'"></video><pre>'+json.dumps(result,indent=2)+'</pre>');print(json.dumps(result),flush=True)
    renderer.close();summary=dict(contact_cone='pyramidal',impratio=1,joint_damping=0,joint_armature=0,joint_frictionloss=0,physics_dt=1/600,control_dt=1/30,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),checkpoint_sha256={n:hashlib.sha256((workspace/'external/a1-calm-main/checkpoints'/n).read_bytes()).hexdigest() for n in ['lowLevelController.pth','highLevelPolicy.pth']},backend=str(jax.devices()),model='Menagerie native A1 MJCF',mass_kg=float(model.body_mass.sum()),self_collision=False,kp=40,kd=1.5,torque_limit_nm=15,policy='closed loop original HLC and LLC; 13 physical bodies mapped to 23 observations',results=results)
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2));(a.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><body style="max-width:1000px;margin:auto"><h1>Menagerie A1 MJX 폐쇄 루프 무작위 이동</h1><p>매 제어 단계에서 MJX 현재 상태를 기존 HLC·LLC에 입력했습니다. 기존 비율 배정 명령 10개를 사용했습니다. 물리 간격은 1/600초, 제어 간격은 1/30초입니다. 접촉 원뿔은 pyramidal, impratio는 1이며 기본 관절 감쇠·마찰·armature를 제거하고 별도 PD를 사용했습니다. 자기 충돌은 껐으며 PD와 마찰 조건을 맞추기 위해 기본 Menagerie 구동 설정을 변경했습니다. 원래 학습 형상과 다르며 관측 지점 13개를 23개로 변환했습니다. 중간 초기화 없이 전도 시 해당 구간을 종료합니다.</p>'+''.join(cards)+'</body></html>')


if __name__=='__main__':main()
