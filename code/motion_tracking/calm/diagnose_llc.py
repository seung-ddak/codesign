"""Run original fixed-latent LLC, recording first termination without auto-reset."""
import argparse
import hashlib
import subprocess
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--start',choices=['first','normal','alternate'],default='first')
    p.add_argument('--sync-reset',action='store_true')
    p.add_argument('--warmup',action='store_true')
    p.add_argument('--cpu-physics',action='store_true')
    p.add_argument('--reset-probe',action='store_true')
    a=p.parse_args();out=a.out.resolve();out.mkdir(parents=True,exist_ok=False)
    source_file=Path(__file__).resolve()
    git_root=source_file.parents[3]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=git_root,text=True).strip()
    root=Path(__file__).resolve().parents[4]/'external/a1-calm-main'
    os.environ['CALM_ROOT']=str(root);os.environ['START_MODE']=a.start
    os.chdir(root);sys.path.insert(0,str(root/'calm'))
    import run
    if a.cpu_physics:
        original_load = run.load_cfg
        def cpu_load(args):
            cfg,train,log=original_load(args)
            train['params']['config']['device_name']='cpu'
            return cfg,train,log
        run.load_cfg=cpu_load
    from learning.calm_players import CALMPlayer
    import numpy as np
    import torch
    if a.sync_reset:
        from reset_observations import install
        install(root/'assets/urdf/a1_description/urdf/a1.urdf')

    def bounded(self):
        self._reset_latent_step_count();self.model.a2c_network._env=self.env
        task=self.env.task;n=task.num_envs
        obs=self.env_reset()
        if a.reset_probe:
            def snapshot():
                quat=task._humanoid_root_states[:,3:7]
                return dict(motion_ids=task._reset_ref_motion_ids.cpu().tolist(),
                            body_up=(1-2*(quat[:,0]**2+quat[:,1]**2)).cpu().tolist())
            first=snapshot()
            self.env_reset(torch.arange(n,device=self.device))
            second=snapshot()
            reproduced = first['motion_ids']==list(range(n)) and first['motion_ids']!=second['motion_ids']
            result=dict(status='PASS' if reproduced else 'FAIL',meaning='Reset behavior reproduced, not a policy or G0 pass',
                        start=a.start,first=first,second=second,
                        git_commit=commit,source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                        note='Explicit reset calls only, no policy or physical motion executed')
            (out/'reset_probe.json').write_text(json.dumps(result,indent=2));print('RESET_PROBE',json.dumps(result));return
        if a.warmup:
            hold=(task._dof_pos-task._pd_action_offset)/task._pd_action_scale
            native,_,_,_=self.env.step(hold.clamp(-1,1))
            obs=self.obs_to_torch(native)
        if self.is_rnn:self.init_rnn()
        expected=task._knee_world_pos(task._dof_pos,task._humanoid_root_states[:,:3],task._humanoid_root_states[:,3:7])
        fk_error=(expected-task._rigid_body_pos[:,task._knee_body_ids]).norm(dim=-1).max(dim=1).values.cpu().numpy()
        from reset_observations import build_state
        names_body=task.gym.get_actor_rigid_body_names(task.envs[0],task.humanoid_handles[0])
        names_joint=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0])
        fk=build_state(root/'assets/urdf/a1_description/urdf/a1.urdf',names_body,names_joint,
                       task._humanoid_root_states,task._dof_pos,task._dof_vel)
        pose_error=float((fk[:,:,:3]-task._rigid_body_pos).norm(dim=-1).max())
        rotation_error=float((1-(fk[:,:,3:7]*task._rigid_body_rot).sum(dim=-1).abs()).abs().max())
        props=task.gym.get_actor_dof_properties(task.envs[0],task.humanoid_handles[0])
        runtime={k:props[k].tolist() for k in ['stiffness','damping','effort','lower','upper']}
        first_done=np.full(n,-1);knee=np.zeros(n,dtype=bool);up=np.full(n,np.nan);state=[];actions=[];targets=[];latent=[]
        state.append(torch.cat([task._humanoid_root_states,task._dof_pos,task._dof_vel],dim=1).cpu().numpy().copy())
        with torch.no_grad():
            for k in range(60):
                action=self.get_action(obs,True).reshape(n,12)
                actions.append(action.cpu().numpy().copy());targets.append(task._action_to_pd_targets(action).cpu().numpy().copy());latent.append(self._calm_latents.cpu().numpy().copy())
                native,_,done,_=self.env.step(action)
                obs=self.obs_to_torch(native)
                state.append(torch.cat([task._humanoid_root_states,task._dof_pos,task._dof_vel],dim=1).cpu().numpy().copy())
                new=(first_done<0)&done.bool().cpu().numpy()
                first_done[new]=k+1;knee[new]=task._knee_term_buf.cpu().numpy()[new]
                quat=task._humanoid_root_states[:,3:7]
                up[new]=(1-2*(quat[:,0]**2+quat[:,1]**2)).cpu().numpy()[new]
                if (first_done>=0).all():break
        names=task.gym.get_actor_dof_names(task.envs[0],task.humanoid_handles[0])
        np.savez_compressed(out/'trace.npz',state=state,action=actions,target_q=targets,latent=latent,joint_names=names,dt=task.dt,first_done=first_done)
        result=dict(git_commit=commit,source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
                    full_body_fk_position_error_m=pose_error,full_body_fk_quaternion_error=rotation_error,
                    terminal_body_up=[None if not np.isfinite(x) else float(x) for x in up],runtime_joint_properties=runtime,
                    cpu_physics=a.cpu_physics,warmup=a.warmup,start=a.start,sync_reset=a.sync_reset,dt=float(task.dt),first_done_steps=first_done.tolist(),knee_termination=knee.tolist(),reset_fk_max_error_m=fk_error.tolist(),survived_2s=int((first_done<0).sum()),note='No resets after initialization; only steps preceding each first_done are valid episode data')
        (out/'summary.json').write_text(json.dumps(result,indent=2));print('DIAG_RESULT',json.dumps(result))
    CALMPlayer.run=bounded
    sys.argv=['fixed_latent_per_env_test.py','--task','QuadrupedAMPGetup','--test','--headless','--seed','0','--latents_on_reset',
              '--cfg_env','calm/data/cfg/a1_getup_31psc8n_4096.yaml','--cfg_train','calm/data/cfg/train/rlg/calm_a1_31psc8n.yaml',
              '--motion_file','calm/data/motions_a1/v2v3_31motions_squat2s_bfcut.pt','--num_envs','31','--checkpoint','checkpoints/lowLevelController.pth']
    if a.cpu_physics:sys.argv+=['--pipeline','cpu','--sim_device','cpu','--rl_device','cpu']
    runpy.run_path(str(root/'calm/fixed_latent_per_env_test.py'),run_name='__main__')


if __name__=='__main__':main()
