"""Render real FAST smoke artifacts; no simulation or result modification."""
from pathlib import Path
import json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent; out=root/'smoke_zigzag_joint';repo=root.parents[2]
s=json.loads((out/'summary.json').read_text());check=json.loads((out/'smoke_check.json').read_text())
g=s['joint_full_guidance_audit']['scale_1'];r0=s['runs']['scale_0'][0];r1=s['runs']['scale_1'][0]
z0=np.load(out/'scale_0.npz');z1=np.load(out/'scale_1.npz')
assert z0['pred'].shape==z1['pred'].shape==(1,100,54)
assert all(np.isfinite(z[k]).all() for z in (z0,z1) for k in ('pred','E'))
fig,ax=plt.subplots(2,2,figsize=(12,8));steps=np.arange(1,6)
for key,label in [('e','E'),('s','S'),('a','A')]: ax[0,0].plot(steps,g[f'{key}_gradient_norm_per_step'],marker='o',label=label)
ax[0,0].set(yscale='log',xlabel='Reverse diffusion step (execution order)',ylabel='Gradient norm',title='Reward gradients reach E / S / A');ax[0,0].legend()
for r,label in [(r0,'Scale 0'),(r1,'Scale 1')]:ax[0,1].plot(np.arange(13),r['E'],marker='o',label=label)
ax[0,1].set(xlabel='Morphology coordinate',ylabel='E',ylim=(0,1),title='Generated morphology');ax[0,1].legend()
for z,label in [(z0,'Scale 0'),(z1,'Scale 1')]:ax[1,0].plot(np.arange(100)*.02,z['pred'][0,:,2],label=label)
ax[1,0].set(xlabel='Time (s)',ylabel='Predicted body height (m)',title='TT predictions (MJX execution skipped)');ax[1,0].legend()
i=np.arange(4)
for shift,r,label in [(-.18,r0,'Scale 0'),(.18,r1,'Scale 1')]: ax[1,1].bar(i+shift,np.array(r['initial_physics']['raw_foot_z_m'])*1000,width=.36,label=label)
ax[1,1].axhline(-2,color='red',linestyle='--',label='Allowed minimum -2 mm');ax[1,1].set(xticks=i,xlabel='Foot index',ylabel='Foot collision clearance (mm)',title='Initial feet penetrate ground');ax[1,1].legend()
fig.tight_layout();fig.savefig(root/'smoke_diagnostics.png',dpi=160);plt.close(fig)
style=(repo/'runs/pipeline_v1/final/report.html').read_text().split('<style>',1)[1].split('</style>',1)[0]
img=base64.b64encode((root/'smoke_diagnostics.png').read_bytes()).decode()
physics=''
for label,r in [('Scale 0',r0),('Scale 1',r1)]:
 p=r['initial_physics'];physics+=f"<tr><td>{label}</td><td>{p['raw_foot_contact_count']} / {p['mjx_foot_contact_count']}</td><td>{-min(p['raw_foot_z_m'])*1000:.2f} mm</td><td>{p['raw_nonfoot_penetration_m']*1000:.2f} mm</td><td>{p['raw_self_penetration_m']*1000:.2f} mm</td><td>{'위반' if p['raw_joint_limit_violation'] else '정상'}</td><td class='badc'>탈락</td></tr>"
checks=''.join(f"<tr><td>{key}</td><td class={'ok' if val else 'badc'}>{'PASS' if val else 'FAIL'}</td></tr>" for key,val in check['checks'].items())
gradrows=''.join(f"<tr><td>{key.upper()}</td><td>{', '.join(f'{v:.6f}' for v in g[key+'_gradient_norm_per_step'])}</td><td class='ok'>5/5 nonzero</td></tr>" for key in ('e','s','a'))
page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>FAST 2초 공동 생성 · MJX 검증</title><style>{style}a{{color:var(--acc)}}.pill{{padding:4px 9px;border:1px solid var(--line);border-radius:4px}}</style></head><body><main>
<header><h1>FAST 2초 E·S·A 공동 생성 검증</h1><div class="sub">2026-10-22 · 실행 코드 f0c84d29f · 학습 checkpoint fast_mode24_v4/tt_joint · conditioning v4</div></header>
<nav><a href="#summary">판정</a><a href="#setup">실험 조건</a><a href="#grad">Gradient</a><a href="#physics">물리 검사</a><a href="#checks">전체 검사</a><a href="#next">해석·다음 단계</a><a href="#files">파일</a></nav>
<h2 id="summary">판정</h2><div class="box bad"><b>전체 Smoke 검증 미통과 · 생성 S₀의 지면 관통으로 MJX 실행 제외</b>E·S·A gradient 전달 및 유한한 2초 공동 생성은 통과했습니다. 그러나 baseline과 guided 후보 모두 초기 물리 검사를 통과하지 못했습니다. 실제 2초 동역학 실행·생존·reward는 평가하지 못했습니다.</div>
<div class="tw"><table><thead><tr><th>요청 항목</th><th>결과</th><th>확인 내용</th></tr></thead><tbody><tr><td>Reward Gradient → E·S·A</td><td class="ok">PASS</td><td>실제 학습 모델에서 세 gradient norm 모두 5개 denoising 단계에 걸쳐 nonzero</td></tr><tr><td>2초 E·S·A 공동 생성</td><td class="ok">PASS</td><td>E (1,13), S·A (1,100,54), 유한값. Guidance에 따라 E·S·A 모두 변화</td></tr><tr><td>생성 E·S₀ 물리 유효성</td><td class="badc">FAIL</td><td>Guided 후보의 발 관통 최대 51.83 mm, 비발 지면 관통 35.76 mm</td></tr><tr><td>2초 MJX Open-loop</td><td class="mid">미실행 / gate FAIL</td><td>초기 물리 검사 탈락으로 stepping 제외. 실제 reward는 null</td></tr></tbody></table></div>
<h2 id="setup">실험 조건</h2><ul><li>TT 재학습 없음. 기존 16개 형상·19종 primitive Mode 2·4 / 3,000-update checkpoint 사용.</li><li>Zigzag reward · seed 0 · scale 0 대 scale 1 · 각 후보 1개 · 동일 초기 diffusion noise.</li><li>H=32, 겹치는 window 4개, diffusion 5단계. 하나의 E를 공유하며 전체 예측 구간 reward를 guidance에 사용.</li><li>초기 XY·Yaw만 고정. E·나머지 S₀·S·A 공동 생성. 몸통 높이 보정 없음.</li><li>100개 state/action 토큰, 0.02초 간격의 2초 실험 계약. 첫/마지막 state 시각 차이는 1.98초.</li><li>관련 단위 테스트 14개 통과. 단위 테스트 결과와 실제 후보의 물리 실패를 구분.</li></ul>
<h2 id="grad">Gradient 및 생성 결과</h2><div class="tw"><table><thead><tr><th>성분</th><th>Gradient norm · 실행 순서</th><th>판정</th></tr></thead><tbody>{gradrows}</tbody></table></div>
<p>Guidance 전후 L2 변화량: E = {check['guidance_change_l2']['E_l2']:.6f}, S = {check['guidance_change_l2']['S_l2']:.6f}, A = {check['guidance_change_l2']['A_l2']:.6f}. 서로 다른 단위·차원 수를 가진 전체 배열의 norm이므로 크기를 직접 비교하지 않습니다.</p>
<p>TT 예측 reward: scale 0 <b>{r0['pred_total']:.6f}</b> → scale 1 <b>{r1['pred_total']:.6f}</b>. 예측 reward가 증가했지만 실제 MJX reward는 양쪽 모두 평가하지 못했습니다.</p>
<figure class="wide"><img alt="E S A gradient, 생성 형상, 예측 높이, 발 지면 관통 비교" src="data:image/png;base64,{img}"><figcaption>상태 곡선은 TT 예측입니다. 발 clearance는 초기 CPU MuJoCo geometry에서 계산했으며 음수는 지면 관통입니다. Gradient norm의 단위와 차원 수가 성분마다 다릅니다.</figcaption></figure>
<h2 id="physics">생성 E·S₀ 물리 유효성</h2><div class="tw"><table><thead><tr><th>후보</th><th>접촉 발 수 CPU / MJX</th><th>최대 발 관통</th><th>비발 지면 관통</th><th>자체 관통</th><th>관절 한계</th><th>판정</th></tr></thead><tbody>{physics}</tbody></table></div>
<p>허용 관통은 발·비발·자체 충돌 각각 2 mm 이내입니다. 두 후보 모두 접촉 발 수는 최소 2개를 만족하고, 자세 quaternion과 관절 한계도 통과했습니다. 그러나 지면을 수 cm 관통하여 초기 상태가 유효하지 않습니다. 이때 접촉 개수가 많다는 것은 안정적인 지지를 의미하지 않습니다.</p>
<p>E는 유한하며 활성 좌표는 [0.05,0.95] 범위, 비활성 좌표 0·11·12는 0.5입니다. 이 범위 검사는 E와 S₀의 물리적 정합성을 보장하지 않습니다. 실제 실패는 두 값을 함께 적용한 geometry 검사에서 발견됐습니다.</p>
<div class="box note"><b>MJX 결과를 0 reward나 낙상으로 기록하지 않음</b>MJX 초기 forward 접촉은 확인했지만 시간 stepping은 수행하지 않았습니다. 실제 reward=null, valid time=0이며, 이는 0초 만에 낙상했다는 의미가 아닙니다.</div>
<h2 id="checks">자동 검사 원문</h2><div class="tw"><table><thead><tr><th>검사</th><th>판정</th></tr></thead><tbody>{checks}</tbody></table></div>
<h2 id="next">해석과 다음 단계</h2><div class="box note"><b>이번에 좁혀진 병목: 생성 E와 초기 자세 S₀의 물리적 정합성</b>Reward 신호가 E에 도달하지 않는 배선 문제는 이번 후보에서 관측되지 않았습니다. Guidance가 없는 baseline부터 관통하므로 현재 실패를 reward guidance만의 문제로 설명할 수 없습니다.</div>
<ul><li>이번은 seed 0의 한 쌍 결과입니다. 전체 생성 분포의 유효율이나 원인을 확정하지 않습니다.</li><li>낮은 denoising loss만으로 geometry 제약이 충족되지는 않음을 실제 후보에서 확인했습니다. 학습 부족·생성 오차·표현 변환 중 원인을 분리해야 합니다.</li><li>다음 진단은 학습 데이터의 실제 S₀를 동일한 state 변환·physics audit에 통과시켜 검사기/변환의 정합성을 확인하고, 생성 S₀와 비교하는 것입니다.</li><li>그 후 여러 seed의 baseline 유효율을 측정해 국소 실패인지 반복되는 실패인지 확인합니다. 허용 관통을 완화하거나 높이를 보정해 이번 gate를 통과시키지는 않았습니다.</li><li>10초 검증 전에 최소 하나의 물리적으로 유효한 생성 시작 상태와 2초 MJX replay 결과가 필요합니다.</li></ul>
<h2 id="files">원자료와 재현</h2><ul><li><a href="smoke_zigzag_joint/summary.json">후보별 요약·초기 물리 감사·gradient</a></li><li><a href="smoke_zigzag_joint/smoke_check.json">자동 PASS/FAIL</a></li><li><a href="smoke_zigzag_joint/scale_0.npz">Baseline NPZ</a> · <a href="smoke_zigzag_joint/scale_1.npz">Guided NPZ</a></li><li><a href="smoke_diagnostics.png">검증 그래프</a> · <a href="../fast_mode24_v4/report.html">선행 학습 보고서</a> · <a href="make_report.py">보고서 생성 코드</a></li></ul>
<pre><code>QUICK_CKPT=runs/pipeline_v1/fast_mode24_v4/tt_joint \
QUICK_ROOT=runs/pipeline_v1/fast_mode24_v4_smoke_audit \
bash code/tt_dynamics/run_quick_validation.sh smoke</code></pre><p class="sub">기존 출력 덮어쓰기 방지로 재실행 시 새 QUICK_ROOT를 사용합니다. 실행기는 physics gate 실패를 기록하고 exit code 1로 정상적인 실패 판정을 반환했습니다.</p>
</main></body></html>'''
(root/'report.html').write_text(page);print(root/'report.html')
