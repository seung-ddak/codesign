"""s8z 수집 결과 보고서 + 영상. 영상 1 개 = 프로세스 1 개 (X BadWindow 회피). python s8z_report.py [--video <세트> <chunk> <i> <이름>]"""
import sys, json, glob, subprocess, collections
import numpy as np
from common import RUNS, report, table, img, video
OUT = RUNS / 's8z_collect'

def one(g, chunk, i, nm):
    from render_tt import render_tt
    from s7h_eval100 import to50
    z = np.load(OUT / 'rollouts' / g / f'chunk_{int(chunk):03d}.npz'); i = int(i); n = int(z['valid'][i].sum())
    render_tt(to50(z['qpos'][i, :n], 100), list(z['E'][i]), OUT / 'videos' / nm,
              title=f"{g.replace('__r00','')} · {z['motion'][i]} · {z['shape'][i]}", fell=bool(z['fell'][i]))

def main():
    p = json.load(open(OUT / 'plan.json')); seen = {k for k, v in p['shapes'].items() if v['seen']}
    idx = collections.defaultdict(list); per = collections.defaultdict(lambda: [0, 0, 0]); shp = collections.defaultdict(lambda: [0, 0]); kind = collections.defaultdict(lambda: [0, 0, 0])
    for f in sorted(glob.glob(str(OUT / 'rollouts/*/chunk_*.npz'))):
        g = f.split('/')[-2]; c = int(f[-7:-4]); z = np.load(f); v = z['valid'].sum(1)
        for i, (sh, m, n, fe) in enumerate(zip(z['shape'], z['motion'], v, z['fell'])):
            sh, m = str(sh), str(m); idx[m].append((g, c, i, sh, bool(fe))); per[g][0] += 1; per[g][1] += n; per[g][2] += fe
            shp[sh][0] += 1; shp[sh][1] += fe; k = '학습 형상' if sh in seen else '새 형상'; kind[k][0] += 1; kind[k][1] += n; kind[k][2] += fe
    (OUT / 'videos').mkdir(exist_ok=True); vids = []
    calm = sorted((m for m in idx if m.startswith('calm')), key=lambda m: -len(idx[m]))[0]  # 형상 수가 가장 많은 CALM 모션 하나
    for g, c, i, sh, fe in idx[calm][:4]: vids.append((g, c, i, f'same_motion_{calm}_{sh}.mp4', f'같은 모션 · {sh}'))
    sh0 = idx[calm][0][3]
    for m in [m for m in idx if m.startswith('trot')][:3]:
        for g, c, i, sh, fe in idx[m]:
            if sh == sh0: vids.append((g, c, i, f'same_shape_{sh0}_{m}.mp4', f'같은 형상 · {m}')); break
    fell = [(g, c, i, sh) for m in idx for g, c, i, sh, fe in idx[m] if fe][:2]
    for g, c, i, sh in fell: vids.append((g, c, i, f'fell_{g}_{c}_{i}.mp4', f'넘어짐 · {sh}'))
    for g, c, i, nm, _ in vids:
        if not (OUT / 'videos' / nm).exists(): subprocess.run([sys.executable, '-u', __file__, '--video', g, str(c), str(i), nm])
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    fig, ax = plt.subplots(figsize=(6.5, 5.5)); L = lambda s: (int(s[3:6]), int(s[8:11]))
    xs = [L(s) for s in shp]; r = [shp[s][1] / shp[s][0] * 100 for s in shp]
    sc = ax.scatter([x[0] for x in xs], [x[1] for x in xs], c=r, s=[60 if s in seen else 14 for s in shp], cmap='magma_r', vmin=0, vmax=max(5, max(r)))
    plt.colorbar(sc, label='넘어짐 비율 (%)'); ax.set_xlabel('다리 길이 변화 (%)'); ax.set_ylabel('몸통 길이 변화 (%)'); ax.set_title('형상 509 개 (큰 점 = 학습 형상)')
    fig.tight_layout(); fig.savefig(OUT / 'shapes.png', dpi=110); plt.close(fig)
    pct = lambda a: f'{a[2] / a[0] * 100:.2f} %'
    report(OUT / 'report.html', '8z 단계: 형상 509 개 × 모션 60 개 rollout 수집 (100 Hz)', [
        ('형상 구분별', table(['형상', '에피소드', '유효 step', '넘어짐'], [(k, v[0], int(v[1]), pct(v)) for k, v in kind.items()])),
        ('세트별', table(['세트', '에피소드', '유효 step', '넘어짐'], [(g.replace('__r00', ''), v[0], int(v[1]), pct(v)) for g, v in sorted(per.items())])),
        ('형상 분포 · 넘어짐', img(OUT / 'shapes.png')),
        ('영상', '<div class="grid">' + ''.join(video(f'videos/{nm}', cap) for *_, nm, cap in vids if (OUT / 'videos' / nm).exists()) + '</div>')],
        [('형상', '학습 9 + 학습 범위 안 새 형상 500 (farthest-point, 정수 %)'), ('모션', '형상당 60 = 세트 6 × (레퍼런스 + CALM 9), CALM 6,000 개 각 4~5 형상'),
         ('정책', 's7h 선택 100 Hz 체크포인트'), ('실행', 'MJX, 결정적, 외란·잡음 없음, 넘어지면 에피소드 끝 (fell)'),
         ('저장', 'rollouts/<세트>/chunk_XXX.npz: qpos, qvel, contact, target_q, valid, fell, E·shape·motion (에피소드별)')])

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--video': one(*sys.argv[2:6])
    else: main()
