"""Validate the completed v13 run and produce a portable training report."""
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.runtime/mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    status=json.loads((OUT/'status.json').read_text());assert status['status']=='complete'
    c=json.loads((OUT/'conditions.json').read_text())
    rows=[json.loads(x) for x in (OUT/'tt_joint/progress.jsonl').read_text().splitlines()]
    assert len(rows)==300 and [r['step'] for r in rows]==list(range(200,60001,200))
    assert all(math.isfinite(r[k]) for r in rows for k in ('loss','val_loss','raw_loss','val_raw_loss'))
    names=['action','joint_position','joint_velocity','body','contact','E']
    weights=np.array([.2,.2,.2,.2,.1,.1])
    train_share=np.array([[r['loss_group_fractions'][n] for n in names] for r in rows])
    val_share=np.array([[r['val_loss_group_fractions'][n] for n in names] for r in rows])
    assert np.allclose(train_share,weights,atol=1e-6) and np.allclose(val_share,weights,atol=1e-6)
    for r in rows:
        assert math.isclose(sum(r['loss_group_terms'].values()),r['loss'],abs_tol=1e-6)
        assert math.isclose(sum(r['val_loss_group_terms'].values()),r['val_loss'],abs_tol=1e-6)
    for manifest in (c['implementation_sha256'],c['sha256']):
        for p,h in manifest.items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
    config=json.loads((OUT/'tt_joint/config.json').read_text())
    ck=torch.load(OUT/'tt_joint/checkpoint.pt',map_location='cpu',weights_only=False)
    assert ck['args']['loss_balance']=='actual' and ck['args']['loss_group_weights']==weights.tolist()
    assert all(torch.isfinite(v).all() for v in ck['net'].values())
    assert config['steps']==60000 and config['episodes']==2432 and config['train']==2192 and config['val']==240
    assert config['modes']==[1,2,4] and config['mode_weights']==[.2,.4,.4]
    assert config['horizon']==32 and config['diff_steps']==5
    diag=json.loads((OUT/'tt_joint/eval_dynamics_ddim.json').read_text())
    with np.load(OUT/'tt_joint/eval_dynamics_ddim_samples.npz',allow_pickle=False) as z:
        assert np.isfinite(z['pred']).all() and np.isfinite(z['true']).all()
        sample={k:z[k].copy() for k in z.files}
    v12=ROOT/'runs/pipeline_v1/fast_mode24_v12_pd_kp200_mode1/tt_joint'
    baseline={'same_cases':False}
    with np.load(v12/'eval_dynamics_ddim_samples.npz',allow_pickle=False) as z:
        baseline['same_cases']=all(np.array_equal(z[k],sample[k]) for k in ('true','mask','src'))
    if baseline['same_cases']:baseline['metrics']=json.loads((v12/'eval_dynamics_ddim.json').read_text())
    resources=[json.loads(x) for x in (OUT/'resources.jsonl').read_text().splitlines()]
    summary={'status':'training_complete','steps':60000,'episodes':2432,'train_episodes':2192,'validation_episodes':240,
             'training_minutes':(status['ended']-status['started'])/60,'first_val_raw_loss':rows[0]['val_raw_loss'],
             'final_val_raw_loss':rows[-1]['val_raw_loss'],'min_val_raw_loss':min(r['val_raw_loss'] for r in rows),
             'min_val_raw_loss_step':min(rows,key=lambda r:r['val_raw_loss'])['step'],
             'group_order':names,'target_shares':weights.tolist(),
             'max_actual_share_deviation':float(max(np.abs(train_share-weights).max(),np.abs(val_share-weights).max())),
             'final_val_group_raw_mse':rows[-1]['val_raw_group_mse'],
             'dynamics_diagnostic':diag,'v12_default_diagnostic_comparison':baseline,
             'min_ram_available_gib':min(x['ram_available_gib'] for x in resources),
             'max_overall_gpu_used_mib':max(float(x['gpu'].split(',')[0]) for x in resources),
             'min_root_free_gib':min(x['disk_free_gib']['/'] for x in resources),
             'min_home_free_gib':min(x['disk_free_gib']['/home'] for x in resources),
             'checks':{'300_progress_records_finite':True,'actual_shares_verified':True,'input_and_implementation_hashes_verified':True,'checkpoint_parameters_finite':True,'default_predictions_finite':True},
             'limitations':['Balanced loss is essentially constant; raw loss monitors learning.','Loss contribution shares do not fix parameter-gradient shares.','Default dynamics diagnostic uses 8 validation episodes; not the full morphology-swap or wrong-action benchmark.','New physical task generation and 10-second MJX validation have not been run.','No performance-success claim from training completion alone.']}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    x=[r['step'] for r in rows]
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for k,label in [('raw_loss','train raw weighted MSE'),('val_raw_loss','validation raw weighted MSE')]:ax[0].plot(x,[r[k] for r in rows],label=label)
    ax[0].set(xlabel='Training steps',ylabel='Unbalanced weighted noise MSE',yscale='log',title='Learning progress: use raw errors');ax[0].legend()
    for n in names:ax[1].plot(x,[r['val_raw_group_mse'][n] for r in rows],label=n)
    ax[1].set(xlabel='Training steps',ylabel='Raw group noise MSE',yscale='log',title='Validation errors before balancing');ax[1].legend(fontsize=8)
    fig.tight_layout();fig.savefig(OUT/'training.png',dpi=150);plt.close(fig)
    shares=', '.join(f'{n} {100*w:g}%' for n,w in zip(names,weights))
    comparison='[사실] v12와 기본 진단 평가 표본이 동일하지 않아 직접 수치 비교를 제시하지 않음.'
    if baseline['same_cases']:
        old=baseline['metrics']['all'];new=diag['all']
        comparison=f"[사실] 같은8개 검증 episode의 기본 Mode1 (= 실제 E·초기 상태·행동을 주고 움직임 예측) 진단: 관절 오차 {old['joint_rmse_deg']:.2f}→{new['joint_rmse_deg']:.2f}도, 위치 오차 {old['pos_err_cm']:.2f}→{new['pos_err_cm']:.2f}cm. 이 작은 진단은 형상 교체 일반화·task 성공 평가를 대체하지 않음."
    text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>v13 실제 손실 비율 학습 완료</title><style>body{{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.65}}img{{max-width:100%}}.result{{padding:16px;background:#e9f1f7}}</style>
<h1>v13: 실제 손실 비율을 맞춘 TT 학습</h1>
<p class="result">[사실] 60,000step 학습과 기본 예측 진단 완료. 학습·검증 기록300개에서 실제 합산 손실 비율20/20/20/20/10/10을 확인했음. [판단] 학습 완료를 형상·행동 공동 생성 성공으로 해석하지 않음.</p>
<p>[구성] PD kp200 데이터16형상·2432episode, train2192/validation240. 기존 v12의 모델·정규화·학습 조건 비율·seed·학습 횟수를 유지하고 목적식만 변경했음. 실제 손실 비중: {shares}. 몸통은 위치·높이·자세·선속도·각속도14개를 포함함.</p>
<p>[방식] 매 학습 묶음에서 그룹별 평균 오차 크기에 반비례하도록 계수를 조절함. 계수는 역전파에서 상수로 취급해 학습 신호를 유지함. 이미 조건으로 주어진 항목은 생성 손실0이며, 정확히 오차0인 항목도 기여0임. 매우 작은 오차 확대는 분모1e-12 하한으로 제한함. 이 비율은 스칼라 손실 비율이며 파라미터 업데이트 비율을 고정하는 것은 아님.</p>
<p>[사실] 조절한 합산 loss/val_loss는 거의1로 일정하므로 성능 지표로 쓰지 않음. 조정 전 검증 오차 val_raw_loss는 첫 기록 {rows[0]['val_raw_loss']:.6f} → 마지막 {rows[-1]['val_raw_loss']:.6f}. 자체 최저값은 {summary['min_val_raw_loss']:.6f}, {summary['min_val_raw_loss_step']}step. 최종 체크포인트는60000step이며 최저값 step과 다를 수 있음.</p>
<img src="training.png" alt="Training and validation raw errors before balancing">
<p>{comparison}</p>
<p>[미검증] 틀린 행동 추가 오차·형상 교체 시 전진 변화 부호·새 task 생성·10초 MJX 성공은 이번 요청에서 실행하지 않았음. 기존 가중 전 val_loss와 새 가중 손실의 숫자를 직접 비교하지 않음. validation_by_source_mode.json의 균형화 손실값도 성능 비교에 쓰지 않음.</p>
<p>[검증] 데이터·구현 해시, 체크포인트 전체 파라미터 유한성, 예측 유한성, 실제 비중과 손실 항의 합 검증 통과. GPU 전체 사용 관측 최대 {summary['max_overall_gpu_used_mib']:.0f}MiB, RAM 여유 최소 {summary['min_ram_available_gib']:.1f}GiB. 소요 {summary['training_minutes']:.1f}분.</p>
<p><a href="summary.json">전체 요약·검증</a> / <a href="conditions.json">조건·입력 해시</a> / <a href="tt_joint/progress.jsonl">전체 학습 기록</a> / <a href="tt_joint/eval_dynamics_ddim.json">기본 예측 진단</a> / <a href="tt_joint/checkpoint.pt">체크포인트</a> / <a href="README.md">재현 방법</a></p></html>'''
    (OUT/'report.html').write_text(text)
    (OUT/'artifact_sha256.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((OUT/'tt_joint').iterdir()) if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('dynamics_diagnostic','v12_default_diagnostic_comparison')},indent=2))


if __name__=='__main__':main()
