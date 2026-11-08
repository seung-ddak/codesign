"""Stage 1: compare identical E/S0/motor targets in CPU training and MJX execution."""
import argparse
import ast
import hashlib
import html
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from train_diffusion_rl import replay_generated,physical_rest_s0
from train_task_rl import build_task_model,mujoco
from s0_geometry import s0_to_q
from rollout_data import features
from initial_geometry_gate import bounded_lift,classify_initial_support

ROOT=Path(__file__).resolve().parents[2]


def compare_traces(q1,q2,ctrl1,ctrl2):
    n=min(len(q1),len(q2));c=min(len(ctrl1),len(ctrl2))
    if not n: return dict(common_states=0,common_controls=c)
    q1,q2=np.asarray(q1)[:n],np.asarray(q2)[:n]
    a=q1[:,3:7]/np.linalg.norm(q1[:,3:7],axis=1,keepdims=True)
    b=q2[:,3:7]/np.linalg.norm(q2[:,3:7],axis=1,keepdims=True)
    angles=2*np.arccos(np.clip(np.abs((a*b).sum(1)),0,1))
    body=np.linalg.norm(q1[:,:3]-q2[:,:3],axis=1)
    joint=np.rad2deg(np.sqrt(np.mean((q1[:,7:19]-q2[:,7:19])**2,axis=1)))
    return dict(common_states=n,common_controls=c,
                initial_qpos_max_error=float(np.max(np.abs(q1[0]-q2[0]))),
                control_max_error=float(np.max(np.abs(np.asarray(ctrl1)[:c]-np.asarray(ctrl2)[:c]))) if c else None,
                first_step_body_error_m=float(body[1]) if n>1 else None,
                first_step_joint_rmse_deg=float(joint[1]) if n>1 else None,
                first_step_orientation_error_deg=float(np.rad2deg(angles[1])) if n>1 else None,
                prefix_2s_body_max_error_m=float(body[:101].max()),
                prefix_2s_joint_max_rmse_deg=float(joint[:101].max()),
                prefix_2s_orientation_max_error_deg=float(np.rad2deg(angles[:101]).max()),
                body_max_error_m=float(body.max()),joint_max_rmse_deg=float(joint.max()))


def model_differences(one,two):
    fields=['body_mass','body_inertia','body_pos','geom_size','geom_pos','geom_friction',
            'geom_contype','geom_conaffinity','jnt_range','dof_damping','dof_armature',
            'actuator_gainprm','actuator_biasprm','actuator_forcerange']
    differences=[]
    for name in fields:
        if not np.array_equal(getattr(one,name),getattr(two,name)):differences.append(name)
    for name in ['timestep','impratio','iterations','ls_iterations','gravity','solver','integrator']:
        if not np.array_equal(getattr(one.opt,name),getattr(two.opt,name)):differences.append('opt.'+name)
    return differences


def load_execution(expert,reference):
    """Load the actual production execute function without running its CLI/generator."""
    import jax,jax.numpy as jnp
    from mujoco import mjx
    from omegaconf import OmegaConf
    from t2.mjx_env.base_env import make_data,step
    from pd_replay import build_model
    from track_env import TrackMjxEnv
    p=ROOT/'code/tt_dynamics/generate_task.py'
    node=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='execute')
    cfg=pickle.loads(sorted(expert.glob('wandb/offline-run-*/files/config.pkl'))[-1].read_bytes())
    ecfg=OmegaConf.to_container(cfg.eval_env,resolve=True);ecfg.pop('_target_')
    ecfg.update(reference_dir=str(reference),clips=['walkForwardSlow__r00'],impl='jax')
    # Exactly the production RL deployment overrides; keep the saved expert physics.
    trace={'controls':[],'qvel':[]}
    def recording_jit(fn):
        compiled=jax.jit(fn)
        def call(model,data,ctrl):
            trace['controls'].append(np.asarray(ctrl).copy())
            if not trace['qvel']:trace['qvel'].append(np.asarray(data.qvel).copy())
            result=compiled(model,data,ctrl)
            trace['qvel'].append(np.asarray(result.qvel).copy())
            return result
        return call
    ns=dict(np=np,mujoco=mujoco,mjx=mjx,jnp=jnp,jax=SimpleNamespace(jit=recording_jit),
            build_model=build_model,TrackMjxEnv=TrackMjxEnv,make_data=make_data,step=step,
            s0_to_q=s0_to_q,features=features,bounded_lift=bounded_lift,classify_initial_support=classify_initial_support,
            args=SimpleNamespace(task='zigzag',max_geometry_lift_m=.04,geometry_s0=False,no_s0_align=True,
                                 allow_airborne_s0=False,replan=0),
            GEN_S0=True,choice=cfg.get('choice'),KP=30.,cfg=cfg,ecfg=ecfg,jit_step=None,
            EP=501,X_DIM=54,S_DIM=42,A=slice(42,54))
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
    return ns,trace,dict(kp=30.,tau_scale=cfg.get('tau_scale'),torque_limit=cfg.get('torque_limit'),
                         ctrl_dt=ecfg['ctrl_dt'],sim_dt=ecfg['sim_dt'],action_scale=ecfg['action_scale'],
                         clip_actions=ecfg['clip_actions'],topology=str(cfg.get('choice')),
                         execute_sha256=hashlib.sha256(ast.get_source_segment(p.read_text(),node).encode()).hexdigest())


def make_cases(root):
    cases=[]
    e0=np.full(13,.5,dtype=np.float32)
    with np.load(root/'shared_window_zigzag_10s_100iter/rl_after.npz') as z:e1=z['E'][2].copy()
    for label,e in [('e000',e0),('generated_E',e1)]:
        s=physical_rest_s0(torch.tensor(e[None]))[0].numpy()
        home=build_task_model(e).keyframe('home').ctrl.copy()
        for count in [101,501]:
            pred=np.tile(np.r_[s,home],(count,1)).astype(np.float32)
            cases.append((f'home_{label}_{(count-1)*.02:g}s',e.copy(),pred))
    for name,indices in [('zigzag_1000iter',[0,6]),('fixed_tt_zigzag_100iter',[0]),('shared_window_zigzag_10s_100iter',[2])]:
        with np.load(root/name/'rl_after.npz') as z:
            for i in indices:cases.append((f'{name}_candidate{i}',z['E'][i].copy(),z['pred'][i].copy()))
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=ROOT/'runs/pipeline_v1/rl_denoising_v1')
    p.add_argument('--expert',type=Path,default=ROOT/'runs/pipeline_v1/experts_wide16/walkForwardSlow__r00')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    a.out.mkdir(parents=True,exist_ok=False)
    cases=make_cases(a.runs_root)
    ns,trace,settings=load_execution(a.expert,ROOT/'runs/pipeline_v1/retargeting/e000/00000011')
    limits=dict(initial_qpos_max=1e-6,control_max=1e-6,first_step_body_m=.001,first_step_joint_deg=.5,
                prefix_2s_body_m=.02,prefix_2s_joint_deg=2.,prefix_2s_orientation_deg=2.,fall_time_difference_s=.04)
    protocol=dict(stage=1,settings=settings,limits=limits,cases=[n for n,_,_ in cases],
                  note='CPU training and current report evaluation share replay_generated. MJX is a separate deployment path.',
                  mujoco_version=mujoco.__version__)
    (a.out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2))
    results=[]
    for name,e,pred in cases:
        print('환경 비교:',name,flush=True)
        x=torch.tensor(pred[None]);E=torch.tensor(e[None]);_,rows,cpu=replay_generated(x,E,'zigzag')
        _,again,repeat=replay_generated(x,E,'zigzag')
        same_cpu=np.array_equal(cpu['qpos_0'],repeat['qpos_0']) and rows[0]['terminal_reward']==again[0]['terminal_reward']
        trace['controls'].clear();trace['qvel'].clear();ns['EP']=len(pred)
        tokens,mask,fell,q,model,initial=ns['execute'](e,pred[:,42:],pred[0,:42])
        ctrl=np.asarray(trace['controls']).reshape(-1,12)
        metrics=compare_traces(cpu['qpos_0'],q,cpu['ctrl_0'],ctrl)
        differences=model_differences(build_task_model(e),model)
        train_valid=not rows[0]['invalid_initial'];eval_valid=not initial.get('execution_skipped',False)
        reasons=[]
        if not same_cpu:reasons.append('cpu_training_report_replay')
        if differences:reasons.append('model_settings')
        if train_valid!=eval_valid:reasons.append('initial_gate')
        if metrics.get('initial_qpos_max_error',0)>limits['initial_qpos_max']:reasons.append('initial_state')
        if metrics.get('control_max_error') is not None and metrics['control_max_error']>limits['control_max']:reasons.append('motor_targets')
        if train_valid and eval_valid:
            for field,key in [('first_step_body_error_m','first_step_body_m'),('first_step_joint_rmse_deg','first_step_joint_deg'),
                              ('prefix_2s_body_max_error_m','prefix_2s_body_m'),('prefix_2s_joint_max_rmse_deg','prefix_2s_joint_deg'),
                              ('prefix_2s_orientation_max_error_deg','prefix_2s_orientation_deg')]:
                if metrics[field] is not None and metrics[field]>limits[key]:reasons.append(field)
            cpu_fell=rows[0]['termination_reason']=='fall';mjx_fell=fell>=0
            if cpu_fell!=mjx_fell:reasons.append('fall_outcome')
            elif cpu_fell and abs(rows[0]['steps']-fell)*.02>limits['fall_time_difference_s']:reasons.append('fall_time')
        result=dict(case=name,cpu_training_report_identical=bool(same_cpu),model_differences=differences,
                    cpu=rows[0],mjx=dict(execution_valid=eval_valid,steps=max(0,len(q)-1),fell_step=int(fell),initial=initial),
                    metrics=metrics,status='pass' if not reasons else 'different',differences=reasons,
                    scope='initial rejection only' if not train_valid and not eval_valid else 'trajectory comparison')
        results.append(result)
        np.savez_compressed(a.out/f'{name}.npz',E=e,pred=pred,cpu_qpos=cpu['qpos_0'],cpu_qvel=cpu['qvel_0'],
                            cpu_ctrl=cpu['ctrl_0'],mjx_qpos=q,mjx_qvel=np.asarray(trace['qvel']),mjx_ctrl=ctrl)
        (a.out/'results.json').write_text(json.dumps(dict(protocol=protocol,results=results),ensure_ascii=False,indent=2))
        print(name,result['status'],reasons,flush=True)
    blocks=[]
    for r in results:
        blocks.append(f'<h2>{html.escape(r["case"])}: {"일치" if r["status"]=="pass" else "차이 발견"}</h2><p>CPU 학습·보고서 경로 재현: {r["cpu_training_report_identical"]}. 학습 초기 유효: {not r["cpu"]["invalid_initial"]}, MJX 초기 유효: {r["mjx"]["execution_valid"]}. 실제 실행 시간: CPU {r["cpu"]["execution_seconds"]:.3f}초, MJX {r["mjx"]["steps"]*.02:.3f}초.</p><pre>{html.escape(json.dumps(dict(model=r["model_differences"],differences=r["differences"],metrics=r["metrics"]),ensure_ascii=False,indent=2))}</pre>')
    (a.out/'report.html').write_text('<!doctype html><html lang="ko"><meta charset="utf-8"><title>1단계 물리 환경 비교</title><style>body{font:17px/1.8 system-ui;max-width:1100px;margin:35px auto;padding:0 20px}pre{white-space:pre-wrap}</style><h1>1단계 물리 환경 일치 확인</h1><p>같은 E·초기 상태·생성 행동을 CPU 학습 경로와 별도 MJX 평가 경로에 입력했습니다. 현재 보고서 평가는 CPU 학습과 같은 실행 함수를 쓰므로 MJX 차이가 현재 CPU 평가 실패의 원인이라는 뜻은 아닙니다. 정지 유지와 저장된 실제 생성 행동을 따로 비교했습니다. 동작을 동일하게 재현하는 것과 모션을 잘 수행하는 것은 별개입니다.</p><p>허용 오차는 실행 전에 protocol.json에 저장했습니다. 초기 무효 판정만 일치한 사례는 물리 궤적 검증으로 세지 않습니다. 긴 시간 차이는 접촉 계산과 수치 정밀도 차이의 영향을 받을 수 있어 곧바로 설정 오류로 단정하지 않습니다.</p>'+''.join(blocks)+'<p><a href="results.json">전체 결과</a> · <a href="protocol.json">사전 설정과 허용 오차</a></p></html>')
    print(a.out/'report.html',flush=True)

if __name__=='__main__':main()
