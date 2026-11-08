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


def render_report(out):
    """Present the decision and evidence; keep raw diagnostics in results.json."""
    data=json.loads((out/'results.json').read_text());rows=data['results']
    executed=[r for r in rows if not r['cpu']['invalid_initial'] and r['mjx']['execution_valid']]
    different=[r for r in executed if r['status']!='pass']
    rejected=[r for r in rows if r['cpu']['invalid_initial'] and not r['mjx']['execution_valid']]
    cpu_same=all(r['cpu_training_report_identical'] for r in rows)
    shared=next((r for r in rows if r['case'].startswith('shared_window_') and r['cpu']['termination_reason']=='fall' and r['mjx']['fell_step']>=0),None)
    if shared and cpu_same:
        verdict='이번 즉시 낙상은 학습·평가 환경 차이로 설명되지 않습니다.'
        meaning='같은 형상에서 기본 관절 목표는 두 환경 모두 10초를 유지했지만, 생성 행동은 두 환경에서 거의 즉시 넘어졌습니다.' if any(r['case']=='home_generated_E_10s' and r['cpu']['survived'] and r['mjx']['fell_step']==-1 for r in rows) else '생성 행동을 두 환경에 넣었을 때 같은 낙상이 재현됐습니다.'
        action='다음에는 생성된 첫 관절 목표와 그 변화량을 확인해야 합니다. 환경을 바꾸거나 학습 횟수를 늘리기 전에, 왜 이 행동이 로봇을 넘어뜨리는지 확인하는 것이 우선입니다.'
    elif different:
        verdict='별도 MJX 평가 경로와의 완전한 일치는 확인되지 않았습니다.'
        meaning='같은 입력을 주어도 일부 움직임에서 사전에 정한 차이 기준을 넘었습니다. 이 결과만으로 정책이나 평가 코드 중 어느 쪽이 잘못됐는지 확정할 수는 없습니다.'
        action='차이가 시작되는 제어 시점과 접촉·물리 계산을 추가로 비교해야 합니다.'
    else:
        verdict='확인한 범위에서는 학습·평가 경로의 동작이 일치했습니다.'
        meaning='동작의 재현 여부를 확인한 결과이며, 모션 추종이나 과제 수행 성공을 뜻하지는 않습니다.'
        action='다음 단계에서 보상과 생성 행동이 원하는 과제를 유도하는지 확인합니다.'
    evidence=[]
    evidence.append('현재 CPU 학습·보고서 평가 경로는 모든 비교 사례에서 동일하게 재현됐습니다.' if cpu_same else '현재 CPU 학습·보고서 평가 경로에서도 재현되지 않은 사례가 있습니다.')
    holds=[r for r in rows if r['case'].startswith('home_') and r['cpu']['max_seconds']==10 and r['cpu']['survived'] and r['mjx']['fell_step']==-1 and r['mjx']['steps']==500]
    if holds:evidence.append(f'기본 관절 목표 유지: 확인한 형상 {len(holds)}개에서 두 환경 모두 10초를 버텼습니다.')
    if shared:evidence.append(f'최근 생성 행동: CPU 학습 경로에서는 {shared["cpu"]["execution_seconds"]:.2f}초, MJX 경로에서는 {shared["mjx"]["steps"]*.02:.2f}초에 낙상했습니다.')
    evidence.append(f'별도 MJX 경로의 실제 움직임 비교 {len(executed)}개 중 {len(executed)-len(different)}개는 허용 오차 이내였고, {len(different)}개는 차이 기준을 넘었습니다.')
    if rejected:evidence.append(f'초기 상태를 두 경로 모두 거부한 {len(rejected)}개는 실제 움직임 비교에서 제외했습니다.')
    caveat=''
    if different:
        r=different[0];angle=r['metrics'].get('prefix_2s_orientation_max_error_deg')
        detail=f'이전 생성 행동 한 사례는 처음 2초 동안 몸통 자세 차이가 최대 {angle:.2f}도로, 사전 기준 2도를 넘었습니다.' if angle is not None else '한 사례 이상에서 실행 차이가 사전 기준을 넘었습니다.'
        precision=out.parent/'stage1_physics_precision_check/results.json'
        if precision.exists() and precision.resolve()!=(out/'results.json').resolve():
            probe=json.loads(precision.read_text())['results']
            match=next((p for p in probe if p['case']==r['case']),None)
            if match and match['status']!='pass':detail+=' 64비트 추가 검사에서도 차이가 남아, 계산 정밀도만 바꾸는 것으로 해결되지는 않았습니다.'
        caveat='<section class="caution"><h2>남아 있는 문제</h2><p>'+detail+'</p><p>따라서 별도 MJX 평가 경로까지 완전히 같다고 결론 내리지는 않습니다. 현재 보고서의 CPU 평가에서 발생한 즉시 낙상과는 구분해야 합니다.</p></section>'
    labels={'home_e000_2s':'기준 형상 · 2초 정지 유지','home_e000_10s':'기준 형상 · 10초 정지 유지',
            'home_generated_E_2s':'생성 형상 · 2초 정지 유지','home_generated_E_10s':'생성 형상 · 10초 정지 유지',
            'zigzag_1000iter_candidate0':'이전 생성 행동 · 후보 0','zigzag_1000iter_candidate6':'이전 생성 행동 · 후보 6',
            'fixed_tt_zigzag_100iter_candidate0':'초기 상태가 무효인 후보','shared_window_zigzag_10s_100iter_candidate2':'최근 공유 정책의 생성 행동'}
    cases=[]
    for r in rows:
        if r in rejected:description='두 경로 모두 실행 전에 거부했습니다. 움직임이 일치한 사례로 세지 않습니다.'
        else:
            cpu=r['cpu']['execution_seconds'];mjx=r['mjx']['steps']*.02
            event='낙상' if r['cpu']['termination_reason']=='fall' and r['mjx']['fell_step']>=0 else '실행'
            description=f'CPU {cpu:.2f}초, MJX {mjx:.2f}초 {event}. '
            description+='비교 결과는 허용 오차 이내입니다.' if r['status']=='pass' else '움직임 차이가 허용 기준을 넘었습니다.'
        cases.append(f'<article><h3>{html.escape(labels.get(r["case"],"추가 비교 사례"))}</h3><p>{description}</p></article>')
    figure=''
    if (out/'comparison.svg').exists():figure='<h2>그래프로 확인하기</h2><img src="comparison.svg" alt="기본 관절 목표 유지와 생성 행동, 환경 사이의 자세 차이 비교"><p>왼쪽은 기본 관절 목표로 버티는 모습, 가운데는 생성 행동을 적용했을 때의 낙상입니다. 오른쪽은 별도로 남아 있는 두 실행 경로의 자세 차이입니다.</p>'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>1단계 결과 · 환경 차이와 즉시 낙상 구분</title><style>body{font:17px/1.85 system-ui;max-width:1050px;margin:35px auto;padding:0 22px;color:#223047}h1{font-size:1.6em}h2{font-size:1.25em}h3{font-size:1em}section{padding:20px 24px;margin:20px 0;border-radius:10px}.verdict{background:#edf4fc}.caution{background:#fff2df}img{width:100%}article{border-bottom:1px solid #dbe2eb;padding:8px 0}details{margin:24px 0}li{margin:10px 0}.small{color:#596575;font-size:.9em}</style><h1>1단계: 학습·평가 환경 비교</h1><section class="verdict"><h2>'+verdict+'</h2><p>'+meaning+'</p></section><h2>그렇게 판단한 근거</h2><ul>'+''.join('<li>'+x+'</li>' for x in evidence)+'</ul><h2>그래서 다음에 할 일</h2><p>'+action+'</p>'+caveat+figure+'<details><summary>각 비교 사례의 결과 보기</summary>'+''.join(cases)+'</details><p class="small">같은 E·초기 자세와 속도·행동을 사용했습니다. 허용 오차는 실험 전에 정했습니다. 이번 결과는 비교한 사례에 한정되며, 모든 형상과 행동에서의 일치나 과제 성공을 보장하지 않습니다.</p><p class="small"><a href="results.json">검증 원자료</a> · <a href="protocol.json">실험 조건</a></p></html>'
    (out/'report.html').write_text(page)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root',type=Path,default=ROOT/'runs/pipeline_v1/rl_denoising_v1')
    p.add_argument('--expert',type=Path,default=ROOT/'runs/pipeline_v1/experts_wide16/walkForwardSlow__r00')
    p.add_argument('--report-only',action='store_true',help='Rewrite the report from saved results without physics replay')
    p.add_argument('--cases',nargs='+',help='Run only selected named cases')
    p.add_argument('--x64',action='store_true',help='Precision diagnostic only; keep separate from deployment defaults')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();torch.set_num_threads(2)
    if a.report_only:
        render_report(a.out)
        print(a.out/'report.html')
        return
    import jax
    if a.x64:jax.config.update('jax_enable_x64',True)
    a.out.mkdir(parents=True,exist_ok=False)
    cases=make_cases(a.runs_root)
    if a.cases:
        missing=set(a.cases)-{name for name,_,_ in cases}
        if missing:raise ValueError('Unknown cases: '+str(missing))
        cases=[case for case in cases if case[0] in a.cases]
    (a.out/'auditor_source.py').write_bytes(Path(__file__).read_bytes())
    ns,trace,settings=load_execution(a.expert,ROOT/'runs/pipeline_v1/retargeting/e000/00000011')
    limits=dict(initial_qpos_max=1e-6,control_max=1e-6,first_step_body_m=.001,first_step_joint_deg=.5,
                prefix_2s_body_m=.02,prefix_2s_joint_deg=2.,prefix_2s_orientation_deg=2.,fall_time_difference_s=.04)
    protocol=dict(stage=1,settings=settings,limits=limits,cases=[n for n,_,_ in cases],
                  note='CPU training and current report evaluation share replay_generated. MJX is a separate deployment path.',
                  mujoco_version=mujoco.__version__,jax_version=jax.__version__,jax_x64=bool(jax.config.jax_enable_x64),jax_backend=jax.default_backend())
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
        initial_v_error=float(np.max(np.abs(cpu['qvel_0'][0]-trace['qvel'][0]))) if trace['qvel'] else None
        metrics['initial_qvel_max_error']=initial_v_error
        train_valid=not rows[0]['invalid_initial'];eval_valid=not initial.get('execution_skipped',False)
        reasons=[]
        if not same_cpu:reasons.append('cpu_training_report_replay')
        if differences:reasons.append('model_settings')
        if train_valid!=eval_valid:reasons.append('initial_gate')
        if metrics.get('initial_qpos_max_error',0)>limits['initial_qpos_max']:reasons.append('initial_state')
        if initial_v_error is not None and initial_v_error>1e-6:reasons.append('initial_velocity')
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
    render_report(a.out)
    print(a.out/'report.html',flush=True)

if __name__=='__main__':main()
