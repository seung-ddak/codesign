"""Held-out morphology regression on a matched 2x2 MJX intervention dataset."""
import ast
import hashlib
import html
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code/tt_dynamics'))
from rollout_data import features
from measure_e_signal_lib import outcomes,NAMES


def crossed_contrasts(values):
    x=np.asarray(values).reshape(2,2)
    return {'rest_minus_reference':float((x[1]-x[0]).mean()),
            'ppo_minus_reference_action':float((x[:,1]-x[:,0]).mean()),
            'interaction':float((x[1,1]-x[1,0])-(x[0,1]-x[0,0]))}


def main():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    assert torch.cuda.is_available(),'GPU unavailable; run outside sandbox with approved GPU access'
    torch.cuda.set_per_process_memory_fraction(.30)
    c=json.loads((OUT/'conditions.json').read_text())
    fingerprint=hashlib.sha256((OUT/'conditions.json').read_bytes()).hexdigest()
    dev='cuda'; start=time.monotonic()
    source=ROOT/'code/tt_dynamics/regression_gate.py'
    names={'MLP','fit_predict','metrics'}
    nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names]
    assert len(nodes)==3
    ns=dict(np=np,torch=torch,nn=nn,args=SimpleNamespace(hidden=c['hidden'],epochs=c['epochs'],wd=c['wd']),dev=dev,NAMES=NAMES)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),ns)
    fit=ns['fit_predict'];metrics=ns['metrics']
    bank=dict(np.load(OUT/'action_bank.npz'))
    arrays=[];stats={}
    for d in c['designs']:
        path=OUT/'rollouts'/d/'factorial.npz'
        with np.load(path) as z:
            assert str(z['fingerprint'])==fingerprint
            a={k:z[k].copy() for k in ('qpos','qvel','contact','valid','fell','E','target_q')}
        assert np.array_equal(a['target_q'][0],bank['target_q']) and np.array_equal(a['target_q'][1],bank['target_q'])
        assert np.array_equal(a['qpos'][:,0,:,0],a['qpos'][:,1,:,0])
        assert np.array_equal(a['qvel'][:,0,:,0],a['qvel'][:,1,:,0])
        arrays.append(a)
    G=c['groups'];D=len(arrays);T=c['steps']
    alive=np.stack([~a['fell'] for a in arrays])  # D, initial, action, G
    common=alive.all(axis=(0,1,2))
    common_motions=len(np.unique(bank['clip'][common]))
    enough=bool(common.sum()>=c['minimum_common_groups'] and common_motions>=c['minimum_common_motions'])
    result={'status':'running','common_groups':int(common.sum()),'common_motions':common_motions,
            'primary_population_sufficient':enough,'conditions_sha256':fingerprint,'conditions':c,
            'survival':{arm:{'episodes':D*G,'survived':int(alive[:,i,j].sum()),'all_design_surviving_groups':int(alive[:,i,j].all(0).sum())} for i in range(2) for j,arm in enumerate(c['arms'][i*2:i*2+2])},
            'populations':{},'versions':{'torch':torch.__version__,'numpy':np.__version__,'gpu':torch.cuda.get_device_name(0)}}
    print('Common surviving groups',int(common.sum()),'of',G,'motions',common_motions,'sufficient',enough,flush=True)
    allpred={}
    idx=np.linspace(0,T-1,c['a_steps']).astype(int)
    for population in ('common','arm_specific'):
        result['populations'][population]={}
        for i in range(2):
            for j in range(2):
                arm=c['arms'][i*2+j]
                use=common if population=='common' else alive[:,i,j].all(0)
                if population=='arm_specific' and np.array_equal(use,common):
                    result['populations'][population][arm]=result['populations']['common'][arm]
                    continue
                if use.sum()<8:
                    result['populations'][population][arm]={'inconclusive':'fewer than 8 full-group observations','groups':int(use.sum())}
                    continue
                xe=[];xs=[];xa=[];yy=[];gi=[];di=[]
                for d,a in enumerate(arrays):
                    z={k:a[k][i,j] for k in ('qpos','qvel','contact','valid','fell')}
                    y,ok=outcomes(z,np.full(G,T))
                    assert ok[use].all()
                    state=np.stack([features(z['qpos'][g,:1],z['qvel'][g,:1],z['contact'][g,:1])[0] for g in np.flatnonzero(use)])
                    xe.append(np.repeat(a['E'][None],int(use.sum()),0));xs.append(state)
                    xa.append(a['target_q'][i,j,use][:,idx].reshape(int(use.sum()),-1))
                    yy.append(y[use]);gi.append(np.flatnonzero(use));di.append(np.full(int(use.sum()),d))
                E,S,A,Y,GI,DI=[np.concatenate(v).astype(np.float32) if k<4 else np.concatenate(v) for k,v in enumerate((xe,xs,xa,yy,gi,di))]
                feats={'E_s0_A':np.concatenate([E,S,A],1),'s0_A':np.concatenate([S,A],1)}
                P={m:np.zeros((c['seeds'],*Y.shape),np.float32) for m in feats};B0=np.zeros_like(Y)
                fs=[]
                for fold in range(4):
                    te=DI%4==fold;tr=~te
                    assert not set(DI[te])&set(DI[tr])
                    B0[te]=np.stack([Y[tr&(GI==g)].mean(0) for g in GI[te]])
                    for model,x in feats.items():
                        for seed in range(c['seeds']):
                            cache=OUT/'predictions'/f'{population}_{arm}_{fold}_{model}_{seed}.npy'
                            cache.parent.mkdir(exist_ok=True)
                            if cache.exists():v=np.load(cache,allow_pickle=False)
                            else:
                                v=fit(x[tr],Y[tr],x[te],seed);np.save(cache,v)
                            assert v.shape==Y[te].shape and np.isfinite(v).all()
                            P[model][seed,te]=v
                        print(population,arm,'fold',fold+1,model,'done',round(time.monotonic()-start),'s',flush=True)
                    fs.append({m:metrics(v[:,te].mean(0),Y[te],B0[te],np.maximum(np.abs(Y).mean(0),1e-9)) for m,v in P.items()})
                mag=np.maximum(np.abs(Y).mean(0),1e-9)
                score={m:metrics(v.mean(0),Y,B0,mag) for m,v in P.items()}
                vx=score['E_s0_A'][NAMES[0]]
                gates={'gain_ge_20pct':vx['gain_vs_b0']>=c['gate']['vx_gain_vs_B0_min'],
                       'sign_ge_80pct':vx['sign_agree']>=c['gate']['vx_effect_sign_min'],
                       'enough_effect_cells':vx['n_big']>=c['minimum_effect_cells'],
                       'enough_common_groups_and_motions':enough if population=='common' else False}
                r={'groups':int(use.sum()),'motions':len(np.unique(bank['clip'][use])),'samples':len(Y),
                   'ensemble':score,'B0':metrics(B0,Y,None,mag),'per_seed':{m:[metrics(vv,Y,B0,mag) for vv in v] for m,v in P.items()},
                   'per_fold':fs,'gates':gates,'primary_pass':bool(all(gates.values())),'movement_mag':dict(zip(NAMES,mag.tolist()))}
                result['populations'][population][arm]=r
                allpred.update({f'{population}_{arm}_Y':Y,f'{population}_{arm}_GI':GI,f'{population}_{arm}_DI':DI,
                                f'{population}_{arm}_B0':B0,**{f'{population}_{arm}_{m}':v for m,v in P.items()}})
                (OUT/'results.partial.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    result['contrasts']={}
    for population,arms in result['populations'].items():
        if all('ensemble' in arms[a] for a in c['arms']):
            for model in ('E_s0_A','s0_A'):
                vals=[arms[a]['ensemble'][model][NAMES[0]]['gain_vs_b0'] for a in c['arms']]
                result['contrasts'][population+'_'+model]=crossed_contrasts(vals)
    result['elapsed_min']=(time.monotonic()-start)/60;result['status']='complete'
    np.savez_compressed(OUT/'predictions.npz',**allpred)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    report(result)
    print('FACTORIAL_COMPLETE',json.dumps(result['contrasts']),flush=True)


def report(r):
    rows=''
    primary=r['populations']['common']
    for arm,v in primary.items():
        survived=r['survival'][arm]['survived']
        if 'ensemble' not in v:
            rows+=f'<tr><td>{arm}</td><td>{survived}</td><td colspan="5">공통 표본 부족, 판정 불가</td></tr>';continue
        x=v['ensemble']['E_s0_A'][NAMES[0]];n=v['ensemble']['s0_A'][NAMES[0]]
        vals=(arm,survived,f"{x['rmse']:.4f}",f"{v['B0'][NAMES[0]]['rmse']:.4f}",f"{x['gain_vs_b0']*100:+.1f}%",f"{x['sign_agree']*100:.1f}%",f"{n['gain_vs_b0']*100:+.1f}%",'통과' if v['primary_pass'] else '미달/표본 부족')
        rows+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in vals)+'</tr>'
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    labels=['ref/ref','ref/PPO','rest/ref','rest/PPO']
    for model in ('E_s0_A','s0_A'):
        vals=[primary[a]['ensemble'][model][NAMES[0]]['gain_vs_b0']*100 if 'ensemble' in primary[a] else np.nan for a in r['conditions']['arms']]
        axes[0].plot(labels,vals,'o-',label=model)
    axes[0].axhline(20,color='k',ls='--',label='20% gate');axes[0].axhline(0,color='gray',lw=.5)
    axes[0].set(ylabel='vx RMSE reduction vs B0 (%)',title='Same surviving groups, held-out designs');axes[0].legend()
    axes[1].bar(labels,[r['survival'][a]['survived']/r['survival'][a]['episodes']*100 for a in r['conditions']['arms']])
    axes[1].set(ylabel='Survival (%)',ylim=(0,100),title='All collected trials, before survivor filtering')
    fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
    text=f'''<!doctype html><meta charset="utf-8"><title>출발 상태 × 행동 출처 비교</title><style>body{{font:16px sans-serif;max-width:1200px;margin:30px auto;line-height:1.6}}table{{border-collapse:collapse}}td,th{{border:1px solid #999;padding:5px}}</style>
<h1>실제 출발 상태 × 행동 출처: 2×2 MJX 비교</h1>
<p>[사실] 실제 MJX GPU에서 16형상·19모션·모션당4구간·1초·kp200으로 수집했음. 총 4,864회. reference_reference=레퍼런스 출발/레퍼런스 목표각, reference_ppo=레퍼런스 출발/PPO 목표각, rest_reference=home 1초 출발/레퍼런스 목표각, rest_ppo=home 1초 출발/PPO 목표각.</p>
<p>[사실] 각 행동 열은 모든 형상과 출발 조건에서 동일함. 출발 상태는 같은 형상·같은 구간에서 행동 출처 사이에 동일함. PPO 행동은 기존 kp30 전문가에서 가져와 공통 kp200에서 재생했음. PPO를 kp200으로 새로 학습한 실험은 아님.</p>
<p>[사실] 네 조건·모든 형상에서 생존한 공통 {r['common_groups']}개 그룹, {r['common_motions']}개 모션으로 주 비교했음. 최소24그룹·8모션 충족 여부: {r['primary_population_sufficient']}. 조건별 생존 그룹을 별도로 사용한 결과는 보조 결과임.</p>
<p>안 본 형상 4-fold, seed3개. E=형상 좌표, s0=실제 출발 상태, A=실제 동일 목표각 열. B0=동일 행동 그룹 내 학습 형상 평균. 성공 기준은 B0 대비 vx 오차20% 이상 감소·형상 효과 부호80% 이상 일치·효과 표본32개 이상·공통 그룹 수 충족임.</p>
<table><tr><th>출발/행동</th><th>생존/1216</th><th>E+s0+A 오차(m/s)</th><th>B0 오차</th><th>B0 대비 개선</th><th>효과 부호</th><th>s0+A 개선</th><th>주 판정</th></tr>{rows}</table>
<img src="comparison.png" style="max-width:100%"><p>[한계] 안착 출발은 실제 초기 속도·접촉도 함께 바꾸므로 관절 자세만의 효과가 아님. 공통 생존 그룹 평가는 낙상 이후 효과를 제외함. e000의 행동 구간만 사용했으므로 다른 행동 출처·새 모션에 대한 일반화는 미검증. 입력 s0만으로 형상을 추정할 수 있으므로 E 없는 모델도 함께 비교함. 회귀 성공은 TT 공동 생성·10초 task 성공의 증거가 아님.</p>
<p><a href="results.json">전체 결과·seed·분할·요인별 차이</a> / <a href="conditions.json">사전 조건</a> / <a href="collection_audit.json">수집 점검</a> / <a href="predictions.npz">원단위 예측</a></p>'''
    (OUT/'report.html').write_text(text)


if __name__=='__main__':main()
