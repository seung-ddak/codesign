"""Validate all actual rollouts and compare task rewards without changing historical gates."""
import hashlib
import html
import json
import os
from pathlib import Path
import sys
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0,str(OUT))
from run_tasks import STAGES,spec,V12


def read(path):return json.loads(path.read_text())


def finite_median(values):
    a=np.array([v for v in values if v is not None and np.isfinite(v)])
    return float(np.median(a)) if len(a) else None


def stage_result(folder,scale,expected,task):
    meta=read(folder/'summary.json');rows=meta['runs'][f'scale_{scale}']
    assert len(rows)==expected
    with np.load(folder/f'scale_{scale}.npz',allow_pickle=False) as z:
        pred,mjx,mask,e=[z[k] for k in ('pred','mjx','mjx_mask','E')]
        assert pred.shape==mjx.shape==(expected,round(meta['seconds']/.02),54)
        assert np.isfinite(pred).all() and np.isfinite(e).all()
        assert np.isfinite(mjx[mask]).all()
        assert mask.shape==pred.shape[:2]
        lane=[]
        for i,row in enumerate(rows):
            assert int(mask[i].sum())==round(row['mjx_valid_s']/.02)
            pts=mjx[i,mask[i],:2]
            if len(pts)==0:lane.append(None);continue
            reward=meta['reward']
            if task=='zigzag':
                ref=reward['amp']*(2/np.pi)*np.arcsin(np.sin(2*np.pi*pts[:,0]/reward['period']).clip(-1+1e-6,1-1e-6))
                inside=np.abs(pts[:,1]-ref)<=reward['half_width']
            else:
                radius=np.linalg.norm(pts-np.array(reward['center']),axis=1)
                inside=np.abs(radius-reward['radius'])<=reward['half_width']
            lane.append(float(inside.mean()))
        if expected==64:
            assert e.shape==(64,13)
            assert all(np.allclose(e[i:i+4],e[i]) for i in range(0,64,4))
    survive=np.array([r['mjx_fell_step']==-1 and not r.get('execution_skipped',False) and r['mjx_valid_s']>=meta['seconds']-.020001 for r in rows])
    positive=np.array([r['mjx_progress'] is not None and np.isfinite(r['mjx_progress']) and r['mjx_progress']>0 for r in rows])
    tt_pick=int(np.argmax([r['pred_total'] for r in rows]))
    actual=np.array([r['mjx_total'] if r['mjx_total'] is not None and np.isfinite(r['mjx_total']) else -np.inf for r in rows])
    best=int(np.argmax(actual)) if np.isfinite(actual).any() else None
    return {'candidates':expected,'survived':int(survive.sum()),'positive_progress':int(positive.sum()),'survived_and_positive_progress':int((survive&positive).sum()),
            'initial_physics_valid':sum(r.get('physically_valid_s0',True) for r in rows),'execution_skipped':sum(r.get('execution_skipped',False) for r in rows),
            'reward_median':finite_median([r['mjx_total'] for r in rows]),'progress_median':finite_median([r['mjx_progress'] for r in rows]),
            'survivor_progress_median':finite_median([r['mjx_progress'] for r,ok in zip(rows,survive) if ok]),
            'unit':'m' if task=='zigzag' else 'laps','path_error_median_m':finite_median([r['path_err_m'] for r in rows]),
            'lane_inside_fraction_median_while_valid':finite_median(lane),
            'predicted_reward_pick':{'index':tt_pick,'row':rows[tt_pick]},'oracle_actual_best':{'index':best,'row':rows[best] if best is not None else None},
            'reward_description':meta['reward'],'seconds':meta['seconds'],'rows':rows}


def eta(values,groups):
    den=((values-values.mean())**2).sum()
    return float(sum((groups==k).sum()*(values[groups==k].mean()-values.mean())**2 for k in np.unique(groups))/den) if den else 0.


def morphology_gate(fixed,smoke):
    rows=fixed['rows'];assert len(rows)==64
    pred=np.array([r['pred_total'] for r in rows]);groups=np.repeat(np.arange(16),4)
    truth=np.array([r['mjx_total'] if r['mjx_total'] is not None and np.isfinite(r['mjx_total']) else np.nan for r in rows])
    pm=np.array([pred[groups==i].mean() for i in range(16)])
    am=np.array([np.nanmean(truth[groups==i]) if np.isfinite(truth[groups==i]).any() else np.nan for i in range(16)])
    effect=eta(pred,groups);rng=np.random.default_rng(0)
    p=float(np.mean([eta(pred,rng.permutation(groups))>=effect for _ in range(10000)]))
    rho=float(spearmanr(pm,am).statistic);rank=rho if np.isfinite(rho) else None
    ok=np.isfinite(am)
    secondary=float(spearmanr(pm[ok],am[ok]).statistic) if ok.sum()>=3 else np.nan
    return {'predicted_effect_p':p,'predicted_effect_eta2':effect,'reward_rank_correlation_all16':rank,
            'finite_designs':int(ok.sum()),'secondary_finite_design_rank':secondary if np.isfinite(secondary) else None,
            'smoke_path_error_m':smoke['path_error_median_m'],
            'historical_gate':bool(p<.05 and rank is not None and rank>.6 and smoke['path_error_median_m'] is not None and smoke['path_error_median_m']<.15)}


def main():
    conditions=read(OUT/'conditions.json');status=read(OUT/'status.json')
    assert set(status)==set(STAGES) and all(s['status']=='complete' for s in status.values())
    for p,h in conditions['sha256'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    result={'conditions':'conditions.json','new_mjx_candidates':0,'stages':{},'baselines':{},'morphology_gates':{},'limitations':[
        'Same two historical tasks; no new obstacle or other task benchmark.',
        'Seed0 only; candidate variation is not an independent training seed study.',
        'Survival and small positive progress alone do not demonstrate task success.',
        'Lane adherence is calculated only over valid physics frames; falls are separately retained.',
        '4s v12 results reused from committed outputs; 10s v12 joint results newly executed. No v12 fixed-E 10s grid.']}
    for stage in STAGES:
        s=spec(stage);n=64 if s['kind']=='fixedE' else 16
        result['stages'][stage]=stage_result(OUT/stage,s['scale'],n,s['task']);result['new_mjx_candidates']+=n
    assert result['new_mjx_candidates']==conditions['new_mjx_candidate_count']==368
    baselinefolder=V12/'evaluation/check_w1'
    result['baselines']['v12_smoke_2s']=stage_result(baselinefolder/'smoke_zigzag_joint_c16',0,16,'zigzag')
    for task in ('zigzag','circle'):
        for kind in ('fixedE','joint'):
            key=f'v12_{kind}_{task}_4s';result['baselines'][key]=stage_result(baselinefolder/f'{kind}_{task}_4s',10,64 if kind=='fixedE' else 16,task)
            assert result['baselines'][key]['reward_description']==result['stages'][f'v13_{kind}_{task}_4s']['reward_description']
        for model,duration in (('v12',4),('v13',4),('v13',10)):
            fixed=(result['baselines'] if model=='v12' else result['stages'])[f'{model}_fixedE_{task}_{duration}s']
            smoke=(result['baselines'] if model=='v12' else result['stages'])[f'{model}_smoke_2s']
            result['morphology_gates'][f'{model}_{task}_{duration}s']=morphology_gate(fixed,smoke)
        assert result['stages'][f'v12_joint_{task}_10s']['reward_description']==result['stages'][f'v13_joint_{task}_10s']['reward_description']
    resources=[json.loads(x) for x in (OUT/'resources.jsonl').read_text().splitlines()]
    result['resources']={'min_ram_available_gib':min(x['ram_available_gib'] for x in resources),'max_overall_gpu_used_mib':max(float(x['gpu'].split(',')[0]) for x in resources),'min_root_free_gib':min(x['disk_free_gib']['/'] for x in resources),'min_home_free_gib':min(x['disk_free_gib']['/home'] for x in resources)}
    result['total_worker_minutes']=sum(s['ended']-s['started'] for s in status.values())/60
    result['videos']=len(list(OUT.glob('v13_*/videos/*.mp4')))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    compact={'new_mjx_candidates':368,'videos':result['videos'],'v13_4s_gates':{t:result['morphology_gates'][f'v13_{t}_4s'] for t in ('zigzag','circle')},
             'joint_results':{k:{n:v[n] for n in ('survived','candidates','progress_median','reward_median','survived_and_positive_progress','lane_inside_fraction_median_while_valid','unit')} for k,v in {**result['baselines'],**result['stages']}.items() if '_joint_' in k},
             'resources':result['resources'],'total_worker_minutes':result['total_worker_minutes']}
    (OUT/'summary.json').write_text(json.dumps(compact,indent=2,ensure_ascii=False))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for row,task in enumerate(('zigzag','circle')):
        keys=[f'{m}_joint_{task}_{s}s' for s in (4,10) for m in ('v12','v13')]
        vals=[{**result['baselines'],**result['stages']}[k] for k in keys]
        labels=['v12 4s','v13 4s','v12 10s','v13 10s']
        axes[row,0].bar(labels,[v['survived'] for v in vals]);axes[row,0].set(ylim=(0,16),ylabel='Survived /16',title=task)
        axes[row,1].bar(labels,[v['progress_median'] if v['progress_median'] is not None else np.nan for v in vals]);axes[row,1].axhline(0,color='k',lw=.8);axes[row,1].set(ylabel='Signed progress ('+vals[0]['unit']+')',title='All finite candidates: median')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    taskrows=''
    for key,r in compact['joint_results'].items():
        taskrows+='<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in (key,f"{r['survived']}/16",f"{r['survived_and_positive_progress']}/16",r['progress_median'],r['unit'],r['reward_median'],r['lane_inside_fraction_median_while_valid']))+'</tr>'
    gates=''.join(f'<li>{task}: {"통과" if v["historical_gate"] else "미달"}, 형상 보상 순위 상관 {v["reward_rank_correlation_all16"]}, 예측 형상 효과 p={v["predicted_effect_p"]}, 2초 경로 차 {v["smoke_path_error_m"]}m</li>' for task,v in compact['v13_4s_gates'].items())
    videohtml=''
    for stage in STAGES:
        s=spec(stage)
        if not s['render']:continue
        r=result['stages'][stage]
        for label,pick in [('TT 예측 보상으로 선택한 후보',r['predicted_reward_pick']),('실제 보상 최고 후보 (사후 선택)',r['oracle_actual_best'])]:
            index=pick['index']
            if index is None:continue
            path=OUT/stage/'videos'/f'scale_10_cand{index}.mp4'
            if path.exists():videohtml+=f'<h3>{stage} · {label} cand{index}</h3><p>{html.escape(str(pick["row"]))}</p><video controls preload="none" src="{path.relative_to(OUT)}"></video>'
        videohtml+='<details><summary>'+stage+' 전체 영상</summary>'+''.join(f'<p>{p.stem}</p><video controls preload="none" src="{p.relative_to(OUT)}"></video>' for p in sorted((OUT/stage/'videos').glob('*.mp4')))+'</details>'
    passed=all(g['historical_gate'] for g in compact['v13_4s_gates'].values())
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>v13 실제 task 성능 평가</title><style>body{{font:16px sans-serif;max-width:1200px;margin:32px auto;padding:0 16px;line-height:1.7}}table{{border-collapse:collapse;font-size:14px}}td,th{{border:1px solid #aaa;padding:6px}}img,video{{max-width:100%}}.lead{{background:#fff3cd;padding:16px}}</style>
<h1>v13: 실제 MJX task 성능</h1><p class="lead">[판정] 기존 4초 Co-Design 기준 (= 형상에 따른 보상 차이를 예측하고 실제 형상 순위를 맞히며 예측 경로와 실제 경로도 가까운지)은 {'통과' if passed else '미달'}임. 생존만으로 task 성공이라고 판단하지 않음.</p>
<p>[사실] 기존 지그재그·원형 task, 같은 seed0·보상·guidance10·kp200·home1초 실제 시작 상태로 비교했음. v13은 형상 고정16×4후보와 E·A 공동 생성16후보를 4초·10초에서 모두 실행. v12는 기존4초 결과를 해시 확인해 재사용하고10초 공동 생성은 새로 실행했음. 새 실제 MJX 후보 총368개. 2초 unguided smoke는 v13 16개. 재계획·추가 자세 보정·재학습 없음.</p>
<h2>공동 생성 실제 결과</h2><p>진행 중앙값은 유한한 실제 결과 전체를 사용하며 낙상 후보를 임의로 제외하지 않음. '생존+정방향'은 보조 진단이며 성공률 정의가 아님. 트랙 내 비율은 실제 유효 프레임에 대해서만 계산했으므로 생존과 함께 봐야 함. 원형 진행 단위는 바퀴(laps), 지그재그는m임.</p>
<table><tr><th>조건</th><th>생존</th><th>생존+정방향</th><th>진행 중앙값</th><th>단위</th><th>보상 중앙값</th><th>유효 프레임 트랙 내 비율 중앙값</th></tr>{taskrows}</table><img src="comparison.png">
<h2>기존 기준</h2><ul>{gates}</ul><p>원래 기준 p&lt;.05·모든16형상 순위상관&gt;.6·2초 경로 차&lt;.15m를 변경하지 않았음. 일부 형상의 유한 보상이 없으면 전체 순위는 계산 불가로 남김. 보조 유한형상 순위로 통과 판정을 대체하지 않음. 10초에는 새 성공 문턱을 만들지 않고 생존·진행·보상·이탈을 그대로 보고함.</p>
<h2>실제 실행 영상</h2><p>흰 점은 트랙 경계, 파란 점은 TT 예측 경로임. 예측으로 선택한 후보와 실제 최고 후보를 분리해 표시함. 실제 최고 후보는 사후 선택이므로 생성기의 선택 성능으로 해석하지 않음.</p>{videohtml}
<h2>범위와 재현</h2><p>[한계] 기존 두 task·seed0 평가임. 새 장애물 task와 독립 학습 seed 반복은 이번 범위 밖임. 고정 형상별 보상·후보별 초기 물리 상태·낙상 시점·원래 예측과 실제 궤적은 모두 보관했음. 컴파일 .mjb 중복본만 제거했고 형상 JSON으로 재생성할 수 있음.</p><p><a href="conditions.json">사전 조건·해시</a> / <a href="summary.json">핵심 결과</a> / <a href="results.json">전체 후보·고정형상 결과</a> / <a href="status.json">실행 상태</a> / <a href="README.md">실행 방법</a></p></html>'''
    (OUT/'report.html').write_text(text)
    artifacts=[p for stage in STAGES for p in (OUT/stage).rglob('*') if p.is_file()]
    (OUT/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},indent=2))
    print(json.dumps(compact,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
