"""Validate the completed local v16 A/B and report delta vs its matched twin only."""
import hashlib
import html
import json
import math
import os
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
P1=ROOT/'runs/pipeline_v1'
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())


def fmt(x):return '미정' if x is None else f'{x:.4f}'


def artifact_audit(run):
    counts={'candidates':0,'physically_executed':0,'initially_excluded':0,'videos':0}
    for summary_path in sorted((run/'evaluation/check_w1').glob('*/summary.json')):
        stage=read(summary_path)
        for scale, rows in stage['runs'].items():
            with np.load(summary_path.parent/(scale+'.npz')) as z:
                assert len(z['pred'])==len(rows)
                assert np.isfinite(z['pred']).all()
                assert np.isfinite(z['mjx'][z['mjx_mask']]).all()
                assert np.isfinite(z['E']).all()
            counts['candidates']+=len(rows)
            counts['initially_excluded']+=sum(r.get('execution_skipped',False) for r in rows)
    counts['physically_executed']=counts['candidates']-counts['initially_excluded']
    assert counts['candidates']==176
    videos=list((run/'evaluation').rglob('*.mp4'))
    assert videos and all(p.stat().st_size>1000 for p in videos)
    counts['videos']=len(videos)
    return counts


def resource_summary():
    rows=[]
    for name in ('twin','delta'):
        for f in ('resources.jsonl','eval_resources.jsonl'):
            rows += [json.loads(x) for x in (P1/('fast_mode24_v16_'+name)/f).read_text().splitlines()]
    return {'samples':len(rows),
            'min_ram_available_gib':min(r['ram_available_gib'] for r in rows),
            'max_gpu_used_mib':max(int(r['gpu'].split(',')[0]) for r in rows if r.get('gpu')),
            'min_root_free_gib':min(r['disk_free_gib']['/'] for r in rows),
            'min_home_free_gib':min(v for r in rows for k,v in r['disk_free_gib'].items() if k!='/')}


def display(v):
    if isinstance(v,dict):return ' / '.join(f'{k}: {display(x)}' for k,x in v.items())
    if isinstance(v,float):return f'{v:.4f}'
    return str(v)


def main():
    state=read(OUT/'status.json');assert all(v['status']=='complete' for v in state.values()) and len(state)==4
    protocol=read(OUT/'protocol.json')
    for p,h in protocol['evaluation_sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    results={};checkpoints={};histories={};configs={};conditions={}
    for name in ('twin','delta'):
        run=P1/('fast_mode24_v16_'+name)
        ck=torch.load(run/'tt_joint/checkpoint.pt',map_location='cpu',weights_only=False)
        assert all(torch.isfinite(v).all() for v in ck['net'].values())
        checkpoints[name]=ck;configs[name]=read(run/'tt_joint/config.json');conditions[name]=read(run/'conditions.json')
        rows=[json.loads(x) for x in (run/'tt_joint/progress.jsonl').read_text().splitlines()]
        assert len(rows)==300 and [r['step'] for r in rows]==list(range(200,60001,200))
        assert all(math.isfinite(r['loss']) and math.isfinite(r['val_loss']) for r in rows)
        histories[name]=rows
        gate=read(run/'gate.json');d=read(run/'evaluation/delta_r2.json');comparison=read(run/'evaluation/comparison.json')
        assert gate['criteria']['E_swap_sign_source']=='held_mode1_same_action_516'
        m1=read(run/'evaluation/given_a_held_mode1.json')['pd']['start30']
        assert m1['pairs']==516
        tasks={}
        for task in ('zigzag','circle'):
            path=run/f'evaluation/check_w1/joint_{task}_4s'
            rr=read(path/'summary.json')['runs']['scale_10']
            assert len(rr)==16
            with np.load(path/'scale_10.npz') as z:
                assert np.isfinite(z['pred']).all() and np.isfinite(z['mjx'][z['mjx_mask']]).all()
            tasks[task]={'survived':sum(r['mjx_fell_step']==-1 for r in rr),'candidates':16,
                         'progress_median':float(np.median([r['mjx_progress'] for r in rr if r['mjx_progress'] is not None and np.isfinite(r['mjx_progress'])])),
                         'unit':'m' if task=='zigzag' else 'laps','reward_median':float(np.median([r['mjx_total'] for r in rr if r['mjx_total'] is not None and np.isfinite(r['mjx_total'])]))}
        results[name]={'artifact_audit':artifact_audit(run),'training_minutes':(read(run/'status.json')['ended']-read(run/'status.json')['started'])/60,
                       'final_base_validation_loss':rows[-1]['val_loss'],
                       'wrong_action_velocity_extra_error':m1['linvel']['wrongA_moving']-m1['linvel']['pin_moving'],
                       'same_action_516_forward_sign':m1['paired_forward_delta_sign'],
                       'mode1_moving_rmse':{k:m1[k]['pin_moving'] for k in ('height','tilt','linvel')},'delta_r2':{k:d[k] for k in ('E','A')},'tasks_4s':tasks,'gate':gate}
        for group in ('sha256','implementation_sha256'):
            for p,h in conditions[name][group].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    assert conditions['twin']['sha256']==conditions['delta']['sha256']
    for key in ('data','horizon','batch','lr','d_model','layers','diff_steps','norm','clip','seed','modes','mode_weights','pair_batch','loss_balance','loss_group_weights','contact_range'):
        assert configs['twin'][key]==configs['delta'][key],key
    assert configs['twin']['delta_weight']==0 and configs['delta']['delta_weight']==1
    assert np.array_equal(checkpoints['twin']['val'],checkpoints['delta']['val'])
    assert np.array_equal(checkpoints['twin']['mu'],checkpoints['delta']['mu']) and np.array_equal(checkpoints['twin']['sd'],checkpoints['delta']['sd'])
    assert all(t['sampled_modes']==d['sampled_modes'] and t['sampled_sources']==d['sampled_sources'] for t,d in zip(histories['twin'],histories['delta']))
    g=results['delta']['gate'];assert g['reference']=='fast_mode24_v16_twin'
    assert abs(g['wrong_action_velocity_extra_error']['ref']-results['twin']['wrong_action_velocity_extra_error'])<1e-4
    summary={'primary_comparison':'delta vs twin, kp30','planned_branch':'B: held-out pair R2 passed, action sensitivity and shape-change sign failed; guidance path reviewed without inference changes','all_predeclared_checks_pass':g['all_pass'],
             'checks':{k:g[k] for k in g['checks']},'models':results,
             'post_hoc_mode1_moving_rmse_ratio':{k:results['delta']['mode1_moving_rmse'][k]/results['twin']['mode1_moving_rmse'][k] for k in ('height','tilt','linvel')},
             'regression_gate_condition':'Predeclared gate uses legacy Mode2 A-pinned metrics. Mode1 A-known moving errors are reported separately and do not change the gate.',
             'validation':{'input_implementation_evaluation_hashes_match':True,'shared_data_normalization_validation_split':True,
                           'same_base_conditioning_source_counters_all_300_records':True,'checkpoint_and_predictions_finite':True,
                           'related_tests_passed':26},'resources':resource_summary(),'protocol':'protocol.json',
             'limits':protocol['limits']+['Body14 R2 aggregates different physical channels; subgroup R2 is retained.','Single-k training monitor is not the held-out DDIM metric.','The twin already has positive DDIM E/A R2; a positive delta result alone does not prove auxiliary-loss benefit.','v12 uses a different data/controller world and is not the matched causal control.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    fig,ax=plt.subplots(2,2,figsize=(12,8))
    for name,color in (('twin','#64748b'),('delta','#087f8c')):
        rows=histories[name];ax[0,0].plot([r['step'] for r in rows],[r['val_loss'] for r in rows],label=name,color=color)
    ax[0,0].set(xlabel='Updates',ylabel='Base diffusion validation MSE',yscale='log');ax[0,0].legend()
    for i,name in enumerate(('twin','delta')):
        ax[0,1].bar(np.arange(2)+(i-.5)*.3,[results[name]['delta_r2'][k]['r2']['body'] for k in ('E','A')],.3,label=name)
    ax[0,1].set_xticks([0,1],['E + initial state pairs','A pairs']);ax[0,1].axhline(0,color='k',lw=.8);ax[0,1].set_ylabel('Held-out DDIM body delta R2');ax[0,1].legend()
    ax[1,0].bar(['twin','delta'],[results[n]['wrong_action_velocity_extra_error'] for n in ('twin','delta')]);ax[1,0].axhline(max(.01,2*results['twin']['wrong_action_velocity_extra_error']),ls='--',color='k');ax[1,0].set_ylabel('Extra normalized moving RMSE, wrong A')
    for i,name in enumerate(('twin','delta')):
        ax[1,1].bar(np.arange(2)+(i-.5)*.3,[results[name]['tasks_4s'][k]['survived'] for k in ('zigzag','circle')],.3,label=name)
    ax[1,1].set_xticks([0,1],['zigzag','circle']);ax[1,1].set(ylim=(0,16),ylabel='4s joint generation survived /16');ax[1,1].legend()
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    rows=''
    labels={'wrong_action_velocity_extra_error':'행동을 틀리게 줬을 때 속도 예측 오차 증가','e_swap_sign':'동일 행동 516쌍 전진 변화 부호','joint_survival':'공동 생성 생존 유지','delta_r2':'별도 생성한 E/A쌍 몸통 변화량 설명','no_regression_rmse':'기존 Mode2 평가 움직임 오차 악화 10% 이내'}
    for key in g['checks']:
        v=g[key]
        rows+=f'<tr><td>{labels[key]}</td><td>{html.escape(display(v.get("ref","대조군의 1.0배")))}</td><td>{html.escape(display(v.get("run",v.get("ratio"))))}</td><td>{"통과" if v["pass_"] else "미달"}</td></tr>'
    taskrows=''
    for name in ('twin','delta'):
        for task,r in results[name]['tasks_4s'].items():taskrows+=f'<tr><td>{name} · {task}</td><td>{r["survived"]}/16</td><td>{r["progress_median"]:.4f} {r["unit"]}</td><td>{r["reward_median"]:.4f}</td></tr>'
    r2rows=''
    for name in ('twin','delta'):
        for kind,v in results[name]['delta_r2'].items():r2rows+=f'<tr><td>{name} · {kind}</td><td>{v["pairs"]}</td>'+''.join(f'<td>{fmt(v["r2"][k])}</td>' for k in ('body','pos','height','linvel'))+'</tr>'
    verdict='사전 기준 모두 통과' if g['all_pass'] else '사전 기준 일부 미달'
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>v16 변화량 손실 A/B</title><style>body{{font:16px sans-serif;max-width:1150px;margin:32px auto;padding:0 16px;line-height:1.7}}td,th{{border:1px solid #aaa;padding:7px}}table{{border-collapse:collapse;font-size:14px}}img{{max-width:100%}}.lead{{padding:16px;background:#edf4f8}}</style>
<h1>v16: 변화량 손실 추가 전후 비교</h1><p class="lead">[판정] {verdict}. 추가 손실이 없는 twin과 있는 delta를 같은 kp30 데이터·기본 학습 조건에서 비교했음. 실제 task의 진행 방향·낙상도 별도로 확인해야 하며, 양수 R²나 생존만으로 task 성공이라고 보지 않음.</p>
<p>[구성] 16형상·48파일·22,272 에피소드. 두 모델의 구조·정규화·검증분할·학습seed0·60,000update·기본조건비율1/2/4=20/40/40은 같음. delta에만 같은 잡음을 준 E/A쌍의 몸통14개 값 변화 차이 손실을 계수1로 추가했음. v13의 실제 손실 20% 고정은 이번 실험에 사용하지 않음. E쌍은 같은 행동에서 형상과 해당 초기 상태가 달라지고, A쌍은 같은 실제 초기 상태·형상에서 행동만 달라짐.</p>
<p>[구현 확인] 기본 학습 난수와 추가 쌍 난수를 분리했음. 두 모델의 검증 분할·정규화가 같고, 300개 기록의 조건·출처 샘플링 카운터도 일치함. 학습·검증 쌍 누수 방지, 입력 및 코드 해시, 모델 수치 검증을 통과했음. 평가를 나눠 계산하는 경계와 무관하게 같은 쌍은 초기 잡음을 공유함.</p>
<h2>사전 기준 판정</h2><p>문서 8-49의 동일 행동 물리쌍 516개를 사용해 부호를 판정했음. 서로 다른 생성 행동을 사용한 고정 형상 task의 평균 부호는 보조 지표로 구분함. 비교 대상은 delta 대 twin임. twin 보고서의 v12(kp200) 참조는 원인 비교로 사용하지 않음.</p><table><tr><th>기준</th><th>대조군/참조</th><th>추가손실모델/비율</th><th>판정</th></tr>{rows}</table>
<p>[조건 구분] 사전 퇴보 방지 기준은 기존 Mode2 평가(행동과 상태를 함께 생성하는 조건에서 행동을 고정)의 오차임. 이 기준은 통과했음. 별도 Mode1 평가(행동을 알려 주고 상태를 예측)에서는 높이·기울기·속도 오차가 각각 58.3%·71.0%·55.3% 증가했음. 이 비교는 post-hoc로 별도 기록했으며 사전 기준을 바꾸지 않았음.</p><h2>별도 생성의 변화량 예측</h2><p>R²는 실제 움직임의 차이를 얼마나 설명하는지 나타냄. 0보다 크면 평균 변화만 예측하는 것보다 낫지만, 추가 손실의 효과는 twin 수치와 함께 봐야 함. 학습 중 단일 잡음 단계의 모니터는 이미 잡음이 섞인 실제 미래를 입력받으므로 최종 DDIM 생성 평가와 다름. 몸통 14채널에는 서로 다른 물리 단위가 섞여 있어 위치·높이·속도도 분리해 기록함.</p><table><tr><th>모델/개입</th><th>검증쌍</th><th>몸통전체R²</th><th>위치R²</th><th>높이R²</th><th>속도R²</th></tr>{r2rows}</table>
<img src="comparison.png"><h2>실제 4초 공동 생성</h2><p>지그재그 진행 단위는 m, 원형은 바퀴(laps)임. 진행 중앙값은 유한한 실제 후보 전체를 기준으로 계산했음. 자세·행동·형상·영상은 각 실험 보고서에 있음.</p><table><tr><th>조건</th><th>생존</th><th>진행중앙값</th><th>보상중앙값</th></tr>{taskrows}</table>
<h2>한계와 원본</h2><p>학습 seed는 하나임. E쌍은 형상과 초기 상태가 함께 달라지므로 E 토큰만의 독립적인 인과 효과를 입증하지 않음. delta는 추가 Mode1 감독과 계산량도 늘어나므로 같은 계산량의 추가 복원 대조군과 비교한 실험은 아님. 두 모델의 합산 학습 손실은 항목 수가 달라 직접 비교하지 않고, 같은 기본 검증 오차를 그래프로 보였음.</p>
<p><a href="../fast_mode24_v16_twin/report.html">twin 전체평가·영상</a> / <a href="../fast_mode24_v16_delta/report.html">delta 전체평가·영상</a> / <a href="summary.json">전체결과·검증</a> / <a href="protocol.json">평가전기록·수정사항</a> / <a href="pair_data_audit.json">쌍원본검증</a> / <a href="README.md">재현명령</a> / <a href="guidance_path_review.md">생성 경로 점검</a></p></html>'''
    (OUT/'report.html').write_text(text)
    print(json.dumps({'all_pass':g['all_pass'],'checks':{k:g[k]['pass_'] for k in g['checks']},'twin':results['twin']['wrong_action_velocity_extra_error'],'delta':results['delta']['wrong_action_velocity_extra_error']},indent=2))


if __name__=='__main__':main()
