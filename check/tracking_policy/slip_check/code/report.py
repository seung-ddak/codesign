"""results/<tag>.json → report_<tag>.html (+ 이슈 게시). 사용: report.py <tag> [--publish "요약"]"""
import json, sys
from pathlib import Path
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(ROOT / 'chain/code'))
from common import report, table, img, video, publish
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt; import numpy as np
plt.rcParams['font.family'] = 'Noto Sans CJK TC'
tag = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith('--') else 'slip'
rows = json.load(open(HERE / f'results/{tag}.json'))
ref = {(r['shape'], r['motion']): r['ref_slip_m_s'] for r in json.load(open(ROOT / 'check/slide/results/summary.json'))}
ms = list(dict.fromkeys(r['motion'] for r in rows)); cs = list(dict.fromkeys(r['config'] for r in rows)); ks = list(dict.fromkeys(r['shape'] for r in rows))
get = lambda k, m, c, key: next((r[key] for r in rows if r['shape'] == k and r['motion'] == m and r['config'] == c), np.nan)
fig, ax = plt.subplots(1, len(ms), figsize=(4.3 * len(ms), 4.2), sharey=True, squeeze=False); ax = ax[0]
for i, m in enumerate(ms):
    for j, c in enumerate(cs):
        ax[i].bar(np.arange(len(ks)) + (j - (len(cs) - 1) / 2) * 0.8 / len(cs), [get(k, m, c, 'slip_contact') for k in ks], 0.8 / len(cs), label=c)
    ax[i].scatter(np.arange(len(ks)), [ref.get((k, m), np.nan) for k in ks], marker='_', s=500, c='k', label='레퍼런스', zorder=5)
    ax[i].set_xticks(range(len(ks)), ks); ax[i].set_title(m.replace('__r00', '')); ax[i].grid(alpha=.3, axis='y')
ax[0].set_ylabel('접촉점 수평 속도 (m/s, 구름 제외)'); ax[-1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(HERE / f'{tag}.png', dpi=105); plt.close(fig)
srows = []
for k in ks:
    for m in ms:
        a, d = get(k, m, cs[0], 'slip_contact'), get(k, m, cs[-1], 'slip_contact')
        srows.append((m.replace('__r00', ''), k, f'{ref.get((k, m), np.nan):.3f}', f'{a:.3f}', f'{d:.3f}', f'{(d - a) / a * 100:+.0f} %',
                      '/'.join('○' if get(k, m, c, 'survived') else '×' for c in cs),
                      ' / '.join(f"{get(k, m, c, 'root_rmse_cm'):.1f}" for c in cs)))
tab = table(['모션', '형상', '레퍼런스 (중심)', f'{cs[0]} 접촉점', f'{cs[-1]} 접촉점', '변화', '생존 (' + '/'.join(c[0] for c in cs) + ')', '몸통 RMSE cm (' + '/'.join(c[0] for c in cs) + ')'],
            srows, filter_col=0)
vids = ''.join(video(f'videos/{p.name}', p.stem, tag=p.stem.split('_')[1]) for p in sorted((HERE / 'videos').glob('*.mp4')))
ov = sorted((HERE / 'videos_overlay').glob('*.mp4')); ovh = ''
for k in ks:  # 겹친 비교: 형상마다 묶고, 모션으로 거르기
    fs = [p for p in ov if p.stem.startswith(k + '_')]
    if fs: ovh += f'<h3>{k}</h3><div class="grid">' + ''.join(video(f'videos_overlay/{p.name}', p.stem.replace('__r00', ''), tag=p.stem.split('_')[1]) for p in fs) + '</div>'
report(HERE / f'report_{tag}.html', f'발 미끄러짐: 접촉 설정 비교 ({tag})', [('형상 × 모션 미끄러짐', img(HERE / f'{tag}.png')), ('수치', tab),
       ('겹친 비교 영상 (원래 색 = A 학습 설정, 반투명 주황 = 테스트 설정)', ovh or '<p>렌더링 중</p>'),
       ('영상 (trotCCWSlow, A·D 단독)', f'<div class="grid">{vids}</div>')],
       [('정책', 'check/slide 형상별 정책 (재학습 없음)'), ('평가', '외란·잡음 없이 모션 처음부터 끝까지'),
        ('A 기준', '피라미드 원뿔, impratio 1, 제약 반복 5 / 선탐색 7 (학습과 동일)'), ('D', '타원 원뿔, impratio 10, 제약 반복 50 / 선탐색 20'),
        ('접촉점 속도', 'v_발중심 + ω_종아리 × (0, 0, -0.02 m), 접지 = 발-바닥 거리 < 1 mm'), ('마찰', '발·바닥 0.8')])
def stage_dir():
    import shutil
    out = HERE / f'_pub_{tag}'; shutil.rmtree(out, ignore_errors=True); (out / 'videos').mkdir(parents=True); (out / 'videos_overlay').mkdir()
    shutil.copy(HERE / f'report_{tag}.html', out / 'report.html'); shutil.copy(HERE / f'{tag}.png', out / f'{tag}.png')
    for d in ('videos', 'videos_overlay'):
        for p in (HERE / d).glob('*.mp4'): shutil.copy(p, out / d / p.name)
    return out


if '--update' in sys.argv:  # 기존 보고서 경로에 덮어쓰고 기존 댓글 수정: --update <댓글 id> "<바꿀 문장>=><새 문장>"
    import gh, shutil
    from common import state
    cid = int(sys.argv[sys.argv.index('--update') + 1]); out = stage_dir()
    gh.upload_dir(out, f"{state().get('run_tag')}/slip_check_{tag}"); shutil.rmtree(out)
    body = gh.get_comment(cid)['body']
    for rep in sys.argv[sys.argv.index('--update') + 2:]:
        old, new = rep.split('=>'); body = body.replace(old, new)
    gh.edit_comment(cid, body); print('updated', cid)
if '--publish' in sys.argv:
    import shutil
    out = HERE / f'_pub_{tag}'; shutil.rmtree(out, ignore_errors=True); (out / 'videos').mkdir(parents=True)
    shutil.copy(HERE / f'report_{tag}.html', out / 'report.html'); shutil.copy(HERE / f'{tag}.png', out / f'{tag}.png')
    for p in (HERE / 'videos').glob('*.mp4'): shutil.copy(p, out / 'videos' / p.name)
    print(publish(f'slip_check_{tag}', out, sys.argv[-1], [out / f'{tag}.png'])); shutil.rmtree(out)
