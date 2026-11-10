"""Compare original reward/100 updates, dense/100, and dense/1000."""
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'task_rl_v1'
rows = []
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
conclusions = []
comparison = {}
for ax, task in zip(axes, ('zigzag', 'circle')):
    run = ROOT / f'{task}_seed0'
    new = json.loads((run / 'summary.json').read_text())
    short = json.loads((run / 'evaluation_update100.json').read_text())
    old = json.loads((OLD / ('zigzag_seed0_retry' if task == 'zigzag' else 'circle_seed0') / 'summary.json').read_text())
    methods = {'학습 전': new['untrained'], '기존 보상·100 update': old['rl'],
               '새 보상·100 update': short['rl'], '새 보상·1000 update': new['rl']}
    comparison[task] = methods
    for split, indices in [('전체', range(16)), ('학습 제외', range(12,16))]:
        for name, candidates in methods.items():
            selected = [candidates[i] for i in indices]
            median = lambda key: float(np.median([r[key] for r in selected]))
            rows.append(f'<tr><td>{task}</td><td>{split}</td><td>{name}</td>'
                        f'<td>{median("reward"):.4f}</td><td>{median("progress"):.4f}</td>'
                        f'<td>{sum(r["survived"] for r in selected)}/{len(selected)}</td>'
                        f'<td>{np.sqrt(median("p_lane")):.4f}</td></tr>')
    held = new['heldout']['rl']
    conclusions.append(f'{task}: 학습 제외 4형상 진행 중앙값 {held["progress_median"]:.4f} '
                       f'{"m" if task == "zigzag" else "바퀴"}, 생존 {held["survived"]}/4.')
    for filename, color, label in [('rl_update100_cpu.npz', '#d69535', 'dense 100'),
                                    ('rl_cpu.npz', '#2280c6', 'dense 1000')]:
        trajectories = np.load(run / filename)
        for i in range(16):
            xy = trajectories[f'tokens_{i}'][:, :2]
            ax.plot(xy[:,0], xy[:,1], color=color, alpha=.55, label=label if i == 0 else None)
    if task == 'circle':
        t = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.sin(t), 1-np.cos(t), 'k--', alpha=.4, label='target')
    else:
        x = np.linspace(0, 2, 200)
        ax.plot(x, .3*2/np.pi*np.arcsin(np.sin(np.pi*x)), 'k--', alpha=.4, label='target')
    ax.set_title(task); ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_aspect('equal', adjustable='datalim'); ax.grid(alpha=.2); ax.legend()
fig.tight_layout(); fig.savefig(ROOT / 'paths.png', dpi=150); plt.close(fig)
video_sections = []
for task in ('zigzag', 'circle'):
    path = html.escape(str(ROOT / f'videos/{task}_heldout12_comparison.mp4'))
    video_sections.append(f'<h3>{task}</h3><video controls preload="metadata" src="{path}"></video><p><a href="{path}">{path}</a></p>')
page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Task RL v2 보상 재설계</title>
<style>body{font:16px/1.8 system-ui;max-width:1150px;margin:35px auto;padding:0 20px;color:#20303c}table{width:100%;border-collapse:collapse}td,th{border:1px solid #ccd3dd;padding:7px}video,img{width:100%}a{color:#1565c0}</style>
<h1>Task RL v2: 보상 재설계 후 실제 실행</h1>
<p><strong>학습·평가 완료. 진행량·낙상과 기존 보상을 함께 보고 판단해야 한다.</strong></p>
<p>'''+html.escape(' '.join(conclusions))+'''</p>
<p>같은 E·S0·seed0·PPO 구조를 사용했다. 새 학습 보상은 목표 접선속도 0.5m/s, 실제 전진, 경로 유지 보상과 정지·역주행·자세·행동 변화·토크 패널티다. 0.5초 출발 유예, 낙상 시 -20을 적용했다. 표의 보상은 변경하지 않은 기존 평가식이다.</p>
<p><strong>해석:</strong> 같은 100 update에서는 두 task 모두 진행 개선이 확인되지 않았다. 1000 update 결과는 학습량도 10배이므로 개선을 보상 변경 하나의 효과로 단정할 수 없다. 지그재그 전체 전진은 약 0.224m/3.98초(평균 약 0.056m/s)로, 목표 속도 0.5m/s에 크게 못 미친다.</p>
<p>기존 실험과 같은 100 update(76,800 전이)에서 보상 변경을 비교하고, 추가로 1000 update(768,000 전이) 결과를 분리했다. E는 고정, S는 물리 실행 결과다. 새 학습 보상값의 크기를 구 보상과 직접 비교하지 않는다.</p>
<table><tr><th>Task</th><th>분할</th><th>정책</th><th>기존 보상 중앙값</th><th>진행 중앙값</th><th>생존</th><th>폭 밖 오차 RMS 중앙값(m)</th></tr>'''+''.join(rows)+'''</table>
<p>지그재그 진행 단위 m, 원형은 바퀴. 폭 밖 오차는 기존 p_lane의 제곱근이며 경로 폭 내부 오차는 0으로 기록된다. 생존만으로 task 성공을 선언하지 않는다.</p>
<h2>실제 실행 영상</h2><p>첫 학습 제외 형상 12번, 왼쪽 기존 guidance, 오른쪽 새 보상으로 1000 update 학습한 RL. 저장된 qpos 재생, 4초 실제 속도. 낙상 이후는 마지막 기록을 유지한다.</p>'''+''.join(video_sections)+'''
<h2>실제 이동 경로</h2><img src="paths.png" alt="Physical trajectories and target centerlines">
<p>단일 seed·4초 평가이며 장기 보행을 입증하지 않는다. 형상 E 최적화와 diffusion 공동 생성 통합은 수행하지 않았다.</p>
<p><a href="README.md">보상식·조건·재현 명령</a> · <a href="comparison.json">후보별 비교 데이터</a></p></html>'''
(ROOT / 'report.html').write_text(page)
(ROOT / 'comparison.json').write_text(json.dumps(comparison, indent=2))
print(ROOT / 'report.html')
