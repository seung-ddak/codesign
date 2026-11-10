"""Present the registered inconclusive result and the separately labeled replay check."""
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
r=json.loads((OUT/'results.json').read_text())
c=r['conditions']
p=json.loads((OUT/'posthoc_probe_results.json').read_text())
probe=json.loads((OUT/'posthoc_probe_conditions.json').read_text())
for manifest in (c['sha256'],probe['sha256']):
    for path,digest in manifest.items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
bank=dict(np.load(OUT/'action_bank.npz'))
falls=[];common=np.ones(c['groups'],bool);rollout_hashes={}
for name in c['designs']:
    path=OUT/'rollouts'/name/'factorial.npz'
    with np.load(path) as z:
        assert z['qpos'].shape==(2,2,76,51,19)
        assert z['qvel'].shape==(2,2,76,51,18)
        assert z['valid'].shape==(2,2,76,50)
        assert np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()
        assert np.array_equal(z['target_q'][0],bank['target_q']) and np.array_equal(z['target_q'][1],bank['target_q'])
        assert np.array_equal(z['qpos'][:,0,:,0],z['qpos'][:,1,:,0])
        assert np.array_equal(z['qvel'][:,0,:,0],z['qvel'][:,1,:,0])
        assert np.array_equal(z['valid'].all(-1),~z['fell'])
        common &= (~z['fell']).all((0,1))
        falls.append(z['fell'])
    rollout_hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
falls=np.stack(falls)
assert common.sum()==r['common_groups']==0
survival=(~falls).sum((0,3));rates=survival/(16*76)
ref30=p['kp']['30']['actual_state_replay']
summary={'primary_verdict':'inconclusive: no common surviving groups; regression not trained',
         'episodes':4864,'designs':16,'groups':76,'common_groups':0,
         'survival_counts':survival.tolist(),'survival_rates':rates.tolist(),
         'survival_contrasts_pp':{'rest_minus_reference':float((rates[1]-rates[0]).mean()*100),
                                  'ppo_minus_reference_actions':float((rates[:,1]-rates[:,0]).mean()*100),
                                  'interaction':float((rates[1,1]-rates[1,0]-rates[0,1]+rates[0,0])*100)},
         'posthoc_replay':p,'limitations':['PPO targets originated at kp30; primary cross-shape replay uses kp200',
                 'No inference about learned E/A dynamics; primary common-survivor set empty',
                 'Posthoc kp30 vs200 check only e000; not a 16-shape kp30 comparison',
                 'Reference vs home1s changes full initial state including velocities and contacts',
                 'All target shapes receive action windows from e000, not all source designs'],
         'validation':{'rollout_shape_finiteness_common_controls_paired_initial_states':True,'source_and_rollout_hashes_verified':True,
                       'unit_tests_passed':17,'data_complete':True,'TT_training_run':False,'regression_training_run':False}}
resources=[json.loads(x) for x in (OUT/'resources.jsonl').read_text().splitlines()]
summary['resources']={'min_ram_available_gib':min(x['ram_available_gib'] for x in resources),'max_overall_gpu_used_mib':max(float(x['gpu'].split(',')[0]) for x in resources),
                      'min_root_free_gib':min(x['disk_free_gib']['/'] for x in resources),'min_home_free_gib':min(x['disk_free_gib']['/home'] for x in resources)}
summary['stage_seconds']={k:v['ended']-v['started'] for k,v in json.loads((OUT/'status.json').read_text()).items()}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False))
(OUT/'rollout_sha256.json').write_text(json.dumps(rollout_hashes,indent=2))
labels=['ref / ref','ref / PPO','home1s / ref','home1s / PPO']
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].bar(labels,rates.reshape(-1)*100,color=['#3978ad','#b6675c','#67aab0','#b7926c'])
for i,(n,v) in enumerate(zip(survival.reshape(-1),rates.reshape(-1))):ax[0].text(i,v*100+1,f'{n}/1216',ha='center',fontsize=9)
ax[0].set(ylabel='Survival (%)',ylim=(0,100),title='Registered 2x2 replay, 16 shapes, kp200')
vals=[p['kp'][str(k)]['survived']['actual_PPO_state'] for k in (30,200)]
ax[1].bar(['kp30','kp200'],np.array(vals)/76*100,color=['#43996b','#b6675c'])
for i,n in enumerate(vals):ax[1].text(i,n/76*100+1,f'{n}/76',ha='center')
ax[1].set(ylabel='Survival (%)',ylim=(0,112),title='Post-hoc e000: same PPO state + actions')
fig.tight_layout();fig.savefig(OUT/'final_comparison.png',dpi=150);plt.close(fig)
rows=''
for label,n,v in zip(('레퍼런스 출발 / 레퍼런스 행동','레퍼런스 출발 / PPO 행동','home 1초 출발 / 레퍼런스 행동','home 1초 출발 / PPO 행동'),survival.reshape(-1),rates.reshape(-1)):
    rows+=f'<tr><td>{label}</td><td>{n}/1216</td><td>{v*100:.1f}%</td></tr>'
text=f'''<!doctype html><html lang="ko"><meta charset="utf-8"><title>출발 상태·행동 출처 비교: 공통 표본 부족</title><style>body{{font:16px sans-serif;max-width:1100px;margin:32px auto;padding:0 16px;line-height:1.65}}td,th{{border:1px solid #aaa;padding:7px}}table{{border-collapse:collapse}}.result{{padding:16px;background:#fff3cd}}img{{max-width:100%}}</style>
<h1>출발 상태 × 행동 출처 비교</h1>
<p class="result"><b>[판정] 형상 예측 학습의 우열은 판정 불가.</b> 네 조건·16형상 모두에서 1초간 생존한 공통 행동 구간이 0개였음. 사전 최소24그룹·8모션을 충족하지 못해 회귀 학습은 실행하지 않았음. 주 기준을 바꾸거나 낙상 표본을 임의로 바꿔 성공 판정을 만들지 않았음.</p>
<h2>실행한 비교</h2><p>[사실] 실제 MJX GPU에서 16형상 × 19모션 × 4구간 × 4조건 = 4,864회 수집했음. 길이1초, kp200, dt0.02s. e000에서 고른 같은 행동 열을 모든 형상과 두 출발 상태에서 그대로 재생했음. 레퍼런스 출발 상태는 형상별 리타겟 상태이며, home 출발은 실제 home 목표각으로 1초 시뮬레이션한 상태임. 같은 형상·구간의 두 행동 조건은 정확히 같은 초기 qpos/qvel을 사용했음.</p>
<table><tr><th>조건</th><th>1초 생존</th><th>생존율</th></tr>{rows}</table>
<p>[사실] 레퍼런스 행동 조건도 모든 형상에서 살아남은 그룹은 3개뿐이었고, 다른 조건은 0개였음. 조건별 전체 생존율과 공통 생존 표본은 다른 지표임. 표본별 실제 궤적은 전부 보관했음.</p>
<h2>별도 사후 분석: 재생 코드와 kp 불일치 점검</h2>
<p>[사실] 주 실험을 본 뒤 e000에서만 추가 점검했음. 기존 PPO 행동은 kp30에서 수집된 목표각임. 실제 원본 시작 상태·같은 행동·kp30으로 재생하면 76/76 생존했고, 원본과 전 구간 관절각 RMSE는 {ref30['all_step_joint_rmse_rad']:.8f}rad, 마지막 XY 오차 중앙값은 {ref30['end_xy_error_median_m']:.8f}m였음. 재생 코드가 원본 궤적을 복원함을 확인했음.</p>
<p>[사실] 같은 원본 상태·행동에서 kp만200으로 바꾸면 생존은37/76으로 줄었음. e000의 레퍼런스 출발·home 출발도 kp30에서는 각각76/76 생존했음. [판단] 이번 kp200에서의 많은 PPO 행동 낙상을 PPO 데이터의 형상 학습 한계로 해석할 수 없음. kp가 다른 곳에서 만들어진 목표각을 이식한 영향이 확인됐음. 다른15형상의 kp30 결과는 미검증임.</p>
<img src="final_comparison.png" alt="Registered survival results and separately labeled posthoc replay check">
<h2>범위와 검증</h2><p>[한계] 이 실험으로 초기 상태와 행동 출처 중 무엇이 안 본 형상 예측을 개선하는지 결론내릴 수 없음. home1초는 완전 정지·안착을 보장하지 않음. e000 행동만 사용했으며, 원본 재생 점검도 한 형상만 수행했음. 회귀 학습·TT 재학습·10초 task 평가는 실행하지 않았음. 주 실험과 사후 분석은 분리했음.</p>
<p>[사실] 입력 해시, 전체 궤적의 유한성·크기, 형상 간 동일 행동, 행동 조건 간 동일 실제 초기 상태, valid/fell 일치 검증을 통과했음. 관련 코드 검증17개 통과. 결과·캐시·임시 파일은 /home에 저장했음.</p>
<p><a href="conditions.json">사전 조건·성공 기준</a> / <a href="results.json">주 평가 결과</a> / <a href="summary.json">요약·자원·검증</a> / <a href="posthoc_probe_conditions.json">사후 검증 조건</a> / <a href="posthoc_probe_results.json">사후 검증 결과</a> / <a href="collection_audit.json">형상별 낙상</a> / <a href="rollout_sha256.json">궤적 파일 해시</a> / <a href="README.md">재현 방법</a></p></html>'''
(OUT/'report.html').write_text(text)
for target in re.findall(r'href="([^"]+)"',text):
    if target!='README.md':assert (OUT/target).is_file(),target
print(json.dumps(summary,indent=2,ensure_ascii=False))
