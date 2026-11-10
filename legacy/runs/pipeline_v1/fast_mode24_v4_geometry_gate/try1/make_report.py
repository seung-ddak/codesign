from pathlib import Path
import json,base64
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path(__file__).resolve().parent;out=p/'smoke_zigzag_joint';repo=p.parents[2]
s=json.loads((out/'summary.json').read_text());c=json.loads((out/'smoke_check.json').read_text())
fig,ax=plt.subplots(2,1,figsize=(10,7),sharex=True)
for tag,color in [('scale_0','C0'),('scale_1','C1')]:
 with np.load(out/f'{tag}.npz') as z:
  i=3;mask=z['mjx_mask'][i].astype(bool);t=np.arange(z['pred'].shape[1])*.02
  assert mask.sum()>0
  for a,idx,label in [(ax[0],2,'Body height (m)'),(ax[1],0,'Forward position (m)')]:
   a.plot(t,z['pred'][i,:,idx],color=color,ls='--',label=tag+' TT prediction');a.plot(t[mask],z['mjx'][i,mask,idx],color=color,label=tag+' MJX');a.set_ylabel(label);a.grid(alpha=.2);a.legend(fontsize=9)
ax[0].set_title('Candidate 3: unmodified airborne start, MJX ends at fall');ax[1].set_xlabel('Time (s)');fig.tight_layout();fig.savefig(p/'replay_comparison.png',dpi=150);plt.close(fig)
style=(repo/'runs/pipeline_v1/final/report.html').read_text().split('<style>',1)[1].split('</style>',1)[0]
rows=''
for tag,rr in s['runs'].items():
 for i,r in enumerate(rr):
  f=r['initial_physics'];pr=f['geometry_projection'];mj='실행 제외' if r['execution_skipped'] else f"낙상 step {r['mjx_fell_step']}, 유효 {r['mjx_valid_s']:.2f}s"
  rows+=f"<tr><td>{tag} / {i}</td><td>{pr['requested_lift_m']*1000:.2f}</td><td>{pr['applied_lift_m']*1000:.2f}</td><td>{f['initial_support_mode']}</td><td>{r['physically_valid_s0']}</td><td>{mj}</td><td>{r['mjx_total'] if r['mjx_total'] is not None else '미평가'}</td></tr>"
img=base64.b64encode((p/'replay_comparison.png').read_bytes()).decode()
page=f'''<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>제한 Geometry Gate · 2초 MJX 실행</title><style>{style}a{{color:var(--acc)}}</style></head><body><main><header><h1>제한 Geometry Gate · 실제 2초 MJX 검증</h1><div class="sub">2026-10-22 · 코드 d10ca76cb · 기존 checkpoint 재사용 · 문서 전체 단계 실행 완료</div></header><nav><a href="#summary">결과</a><a href="#method">실행 조건</a><a href="#samples">후보별 결과</a><a href="#replay">실제 궤적</a><a href="#interpret">해석</a><a href="#files">원자료</a></nav>
<h2 id="summary">결과</h2><div class="box bad"><b>실제 MJX 실행에는 진입했지만, 2초 완주 후보는 없음</b>각 scale에서 후보 3이 높이 보정 없이 공중 시작 조건을 통과했습니다. Baseline은 유효 시간 1.68초, guided는 0.50초에서 낙상했습니다. 물리적으로 가능한 시작 상태를 하나 확보하는 것과 생성 명령으로 안정적으로 움직이는 것은 별도 조건입니다.</div><div class="tw"><table><tr><th>검증</th><th>결과</th></tr><tr><td>CPU 회귀 테스트</td><td class="ok">14개 통과</td></tr><tr><td>E·S·A 유한한 공동 생성 / gradient / guidance 변화</td><td class="ok">모두 통과</td></tr><tr><td>초기 물리 Gate</td><td>Baseline 1/4, guided 1/4 통과</td></tr><tr><td>2초 완주</td><td class="badc">Baseline 0/4, guided 0/4</td></tr><tr><td>보정 없이 완주 / 보정 후 완주</td><td>Guided 모두 0</td></tr><tr><td>strict_raw_TT_to_MJX_verified</td><td class="badc">false</td></tr></table></div>
<h2 id="method">전체 실행 조건</h2><ol><li>GitHub 최신 코드 d10ca76cb를 fast-forward 반영.</li><li>CPU geometry·원본/보정 결과 구분 테스트 14개 실행.</li><li>문서 실행기로 2초 Zigzag 후보 4개씩 scale 0·1 생성. 동일 checkpoint·seed 0 사용.</li><li>생성 E에 맞춰 MuJoCo 지면 관통을 측정하고, 필요량+1mm의 높이 이동이 4cm 이내일 때만 적용. 이후 접촉과 geometry 재검사.</li><li>CPU·MJX 무접촉 및 발 clearance 최소 5mm인 상태는 공중 시작으로 허용.</li><li>통과 후보에 한해 실제 MJX Open-loop stepping. 실패 결과 저장 후 exit code 1로 종료. 10초 실행 없음.</li></ol><p>원본 TT pred와 predicted reward는 변경하지 않았습니다. 생성 상태와 실행 상태 보정은 summary의 preprojection·geometry_projection·executed_body_height_m에 별도로 기록됩니다.</p>
<h2 id="samples">후보별 결과</h2><div class="tw"><table><tr><th>Scale / 후보</th><th>요청 높이 이동 mm</th><th>적용 mm</th><th>재검사 지지 조건</th><th>초기 물리 유효</th><th>MJX</th><th>실제 reward</th></tr>{rows}</table></div><p>요청 이동이 40mm를 초과한 후보는 상한으로 잘라 적용하지 않고 제외했습니다. 상한 이내 보정 후보도 접촉 조건을 재검사하여 탈락했습니다. 정상 stepping에 진입한 후보는 양쪽 모두 보정 없는 후보 3입니다.</p>
<h2 id="replay">실제 궤적과 TT 예측</h2><figure class="wide"><img alt="후보 3의 TT 예측과 실제 MJX 높이 및 전진 위치 비교" src="data:image/png;base64,{img}"><figcaption>실선은 낙상까지의 실제 MJX 유효 구간, 점선은 수정하지 않은 TT 예측입니다. 다른 후보는 시간 stepping에서 제외됐으므로 실제 곡선이 없습니다.</figcaption></figure>
<h2 id="interpret">해석과 한계</h2><ul><li>기존의 무조건적인 2발 접촉 조건 때문에 제외되던 후보 하나가 이번 공중 시작 규칙으로 실제 실행에 진입했습니다.</li><li>정적 geometry를 통과한 원본 생성 상태에서도 두 rollout 모두 낙상했습니다. 초기 지면 관통만 해결하면 2초 동작이 성공한다는 가설은 이 후보에서 지지되지 않습니다.</li><li>Guided 유효 구간 reward는 -2.2987, baseline은 -2.7454지만 유효 시간이 0.50초와 1.68초로 다릅니다. 이를 guidance 성능 개선으로 해석하면 안 됩니다. 현재 reward는 일부 항목의 구간 평균과 진행량을 합칩니다.</li><li>이전 실험과 같은 seed·checkpoint의 재사용이며 독립적인 성공률 추정이 아닙니다. Gate 미통과 후 임의로 후보 수·높이 상한을 늘리지 않았습니다.</li><li>다음 판단에는 생성 행동과 실제 상태 변화의 불일치, 초기 속도/자세 및 학습·sampling 품질을 분리한 진단이 필요합니다. 이번에는 요청된 실행기 범위에서 중단했습니다.</li></ul>
<h2 id="files">실행 원자료</h2><ul><li><a href="execution.log">전체 실행 로그</a></li><li><a href="smoke_zigzag_joint/summary.json">후보별 원본·보정 상태와 reward</a> · <a href="smoke_zigzag_joint/smoke_check.json">PASS/FAIL</a></li><li><a href="smoke_zigzag_joint/scale_0.npz">Baseline NPZ</a> · <a href="smoke_zigzag_joint/scale_1.npz">Guided NPZ</a></li><li><a href="replay_comparison.png">실제 실행 그래프</a> · <a href="make_report.py">보고서 재현 코드</a></li></ul><pre><code>QUICK_CKPT=runs/pipeline_v1/fast_mode24_v4/tt_joint QUICK_ROOT=runs/pipeline_v1/fast_mode24_v4_geometry_gate/try1 QUICK_CANDIDATES=4 bash code/tt_dynamics/run_minimal_geometry_gate.sh</code></pre></main></body></html>'''
(p/'report.html').write_text(page);print(p/'report.html')
