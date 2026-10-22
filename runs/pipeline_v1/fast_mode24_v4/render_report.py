"""Render the FAST validation report using the existing pipeline HTML style."""
from pathlib import Path
import base64
import json
import numpy as np

root = Path(__file__).resolve().parent
repo = root.parents[2]
original = (repo / 'runs/pipeline_v1/final/report.html').read_text()
style = original.split('<style>', 1)[1].split('</style>', 1)[0]
logs = [json.loads(line) for line in (root / 'tt_joint/progress.jsonl').read_text().splitlines()]
validation = json.loads((root / 'validation_by_mode.json').read_text())
config = json.loads((root / 'tt_joint/config.json').read_text())
first, last = logs[0], logs[-1]
best = min(logs, key=lambda row: row['val_loss'])
reduction = (1 - last['val_loss'] / first['val_loss']) * 100
plot = base64.b64encode((root / 'loss_curve.png').read_bytes()).decode()
rows = ''
for mode in ('2', '4'):
    scores = validation[mode]['validation']
    vals = [v['total'] for v in scores]
    train = validation[mode]['train_reference']['total']
    state = np.mean([v['trajectory_noise_mse'] for v in scores])
    morphology = np.mean([v['morphology_noise_term'] for v in scores])
    rows += f'<tr><td>Mode {mode}</td><td>{train:.6f}</td><td>{np.mean(vals):.6f} ± {np.std(vals):.6f}</td><td>{state:.6f}</td><td>{morphology:.6f}</td></tr>'
body = f'''
<header><h1>Mode 2·4 FAST 학습 보고서</h1><div class="sub">2026-10-22 · 코드 dd3e248d8 · conditioning version 4 · 16개 형상 · 19종 primitive</div></header>
<nav><a href="#summary">요약</a><a href="#data">데이터·설정</a><a href="#curve">학습 곡선</a><a href="#validation">전체 검증</a><a href="#interpretation">수렴 판단</a><a href="#next">다음 단계</a><a href="#files">파일</a><button id="theme" type="button">테마 전환</button></nav>
<h2 id="summary">요약</h2>
<div class="box note"><b>학습 완료 · 2초 생성 검증에 사용할 checkpoint 확보</b>3,000 update 동안 train·validation loss가 감소했습니다. 후반에는 감소가 둔화됐으며, 전체 validation의 train reference 대비 큰 차이는 관측되지 않았습니다. 충분한 수렴과 물리적 생성 품질은 아직 확정하지 않습니다.</div>
<div class="metrics"><div><span>고정 validation 감소</span><strong>{reduction:.1f}%</strong><small>200 → 3,000 step</small></div><div><span>최종 고정 validation</span><strong>{last['val_loss']:.5f}</strong><small>첫 16개 validation episode</small></div><div><span>검증 episode</span><strong>{config['val']:,}</strong><small>전체 검증 × seed 3개</small></div><div><span>완료 update</span><strong>3,000</strong><small>최종 가중치 저장 완료</small></div></div>
<h2 id="data">데이터와 학습 설정</h2>
<div class="flow"><div>기존 MJX primitive<br>16개 형상 × 19종 모션</div><span>→</span><div>Mode 2·4<br>H=32 joint diffusion 학습</div><span>→</span><div>전체 validation<br>모드별 seed 3개 평가</div></div>
<div class="tw"><table><thead><tr><th>항목</th><th>실제 사용 설정</th></tr></thead><tbody>
<tr><td>전체 / train / validation</td><td>4,864 / 4,378 / 486 episode</td></tr><tr><td>형상</td><td>e000~e016 중 e003 제외, 총 16개</td></tr><tr><td>데이터 종류</td><td>19종 primitive의 실제 MJX rollout, target_q 포함. Transition 미포함.</td></tr><tr><td>분할</td><td>파일별 episode 무작위 분할, seed 0. 미학습 형상 평가는 아님.</td></tr><tr><td>네트워크</td><td>기존 TT Net · d_model=256 · layers=6 · 약 4.93M parameter</td></tr><tr><td>학습</td><td>3,000 update · batch 16 · H=32 · AdamW · 초기 LR 0.0002 · cosine schedule</td></tr><tr><td>Diffusion</td><td>5 step · minmax 정규화 · clip=1 · DDIM 생성 설정</td></tr><tr><td>정규화</td><td>Train 데이터에서만 궤적 통계 계산</td></tr><tr><td>시간</td><td>로그의 optimizer 구간 {last['min']:.2f}분. NPZ 초기 로딩과 사후 평가는 제외.</td></tr></tbody></table></div>
<div class="grid"><div class="qa"><div class="q">Mode 2 · E와 S₀ 조건</div><div class="a">형상 E와 초기 상태 S₀를 제공하고 이후 상태 S와 관절 명령 A를 생성합니다.</div></div><div class="qa"><div class="q">Mode 4 · 좌표 기준만 조건</div><div class="a">초기 XY=(0,0), Yaw=0만 고정합니다. E·높이·관절각·속도·접촉·행동을 데이터 분포에서 함께 생성합니다.</div></div></div>
<h2 id="curve">학습 곡선</h2>
<figure class="wide"><img alt="200에서 3000 update까지 train 단일 배치 loss와 고정 16 episode validation loss의 로그 축 그래프" src="data:image/png;base64,{plot}"><figcaption>Train은 해당 step의 단일 무작위 batch입니다. Validation은 첫 16개 검증 episode와 고정 seed를 사용합니다. 전체 486개 검증 결과는 아래 별도 표에 있습니다.</figcaption></figure>
<div class="tw"><table><thead><tr><th>기록</th><th>Step</th><th>Train loss</th><th>고정 validation loss</th></tr></thead><tbody><tr><td>첫 기록</td><td>{first['step']}</td><td>{first['loss']:.6f}</td><td>{first['val_loss']:.6f}</td></tr><tr><td>최저 validation</td><td>{best['step']}</td><td>{best['loss']:.6f}</td><td>{best['val_loss']:.6f}</td></tr><tr><td>최종</td><td>{last['step']}</td><td>{last['loss']:.6f}</td><td>{last['val_loss']:.6f}</td></tr></tbody></table></div>
<p>Checkpoint는 3,000 step의 마지막 가중치입니다. 2,400 step의 최저 validation 가중치는 별도로 저장하지 않았습니다.</p>
<h2 id="validation">전체 validation · 모드별 결과</h2>
<p>검증 episode 486개 각각에서 무작위 window 하나를 선택하고, seed 101·102·103으로 반복했습니다. ±는 seed 간 표준편차이며 신뢰구간이 아닙니다. Train reference는 train에서 뽑은 최대 486개 episode의 seed 101 평가입니다.</p>
<div class="tw"><table><thead><tr><th>모드</th><th>Train reference</th><th>Validation total 평균 ± 표준편차</th><th>궤적 noise MSE</th><th>E noise 항</th></tr></thead><tbody>{rows}</tbody></table></div>
<div class="box note"><b>Loss의 의미</b>이 값은 diffusion noise 복원 오차입니다. 실제 위치·관절각 오차나 MJX task reward가 아닙니다. Mode 2는 E가 알려져 E noise 항이 0이고, Mode 4는 E까지 복원하므로 total loss의 크기로 두 모드의 성능 순위를 정할 수 없습니다.</div>
<h2 id="interpretation">수렴 판단과 해석 범위</h2>
<div class="tw"><table><thead><tr><th>확인 항목</th><th>판단</th><th>근거와 한계</th></tr></thead><tbody><tr><td>학습 진행</td><td class="ok">확인</td><td>첫 기록 대비 고정 validation loss {reduction:.1f}% 감소, 기록된 train·val loss 모두 유한함.</td></tr><tr><td>후반 안정화</td><td class="mid">감소 둔화 관측</td><td>마지막 4개 validation 기록 0.05745~0.06134. 학습률이 0까지 감소했으므로 평탄화만으로 충분한 수렴을 확정할 수 없음.</td></tr><tr><td>Train–validation 격차</td><td class="ok">큰 격차 미관측</td><td>최종 모델의 모드별 평가 기준. 단일 학습 seed와 episode 분할에 한정.</td></tr><tr><td>미학습 형상 일반화</td><td class="mid">미검증</td><td>같은 16개 형상이 train과 validation에 포함됨.</td></tr><tr><td>형상 간 독립성</td><td class="mid">분할 한계</td><td>같은 원본 episode 번호가 형상별로 다른 split에 포함될 수 있음. 공통 난수·모션의 상관성을 고려한 공통 분할은 미적용.</td></tr><tr><td>물리적 S₀·MJX 실행</td><td class="mid">미검증</td><td>생성 및 물리 replay를 이번 학습 보고서에 포함하지 않음.</td></tr></tbody></table></div>
<h2 id="next">다음 검증</h2><ol><li>이 checkpoint로 2초 공동 생성과 전체 horizon reward guidance를 실행합니다.</li><li>실제 E reward gradient, 생성 S₀의 물리적 유효성, MJX 생존과 reward를 확인합니다.</li><li>2초 검증 결과에 따라 10초 원형·지그재그 평가로 확장합니다.</li></ol>
<h2 id="files">결과 파일과 재현</h2><ul><li><a href="tt_joint/checkpoint.pt">최종 checkpoint</a> · <a href="tt_joint/config.json">학습 설정</a></li><li><a href="tt_joint/progress.jsonl">학습 기록</a> · <a href="train.log">전체 학습 로그</a></li><li><a href="validation_by_mode.json">모드·형상별 검증 원자료</a> · <a href="loss_curve.png">Loss 그래프 PNG</a></li><li><a href="evaluate_report.py">검증 재현 코드</a> · <a href="render_report.py">HTML 보고서 생성 코드</a></li></ul><p class="sub">보고서: runs/pipeline_v1/fast_mode24_v4/report.html · 평가 수치와 학습 원자료에서 생성</p>
'''
extra = '''.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:20px 0}.metrics>div{padding:16px;border:1px solid var(--line);background:var(--card);border-radius:6px}.metrics span,.metrics small{display:block;color:var(--mut);font-size:12px}.metrics strong{display:block;font-size:30px;color:var(--acc);margin:4px 0}a{color:var(--acc)}button{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:4px 10px;cursor:pointer}nav button{margin-left:auto}'''
script = '''document.getElementById('theme').onclick=()=>{const r=document.documentElement;const dark=r.dataset.theme?r.dataset.theme==='dark':matchMedia('(prefers-color-scheme: dark)').matches;r.dataset.theme=dark?'light':'dark'};'''
page = f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Mode 2·4 FAST 학습 보고서</title><style>{style}{extra}</style></head><body><main>{body}</main><script>{script}</script></body></html>'
assert all(f'id="{key}"' in page for key in ('summary','data','curve','validation','interpretation','next','files'))
assert '<img ' in page and 'data:image/png;base64,' in page
(root / 'report.html').write_text(page)
print(root / 'report.html')
