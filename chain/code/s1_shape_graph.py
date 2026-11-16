"""1 단계: 지표 대 형상 그래프 + 다리 하한 자동 판정.
데이터: check/slide/results/summary.json (E0~E4 × 모션), check/short_leg/results/summary.json (L05~L25 × squat·후진), check/short_leg/results/ckpts.json.
하한 규칙 (2026-11-12 사용자 결정): squat·후진 중 하나라도 실패 (끝까지 생존 못 함) 한 가장 작은 축소 비율의 바로 앞 단계. 들쭉날쭉하면 판정하지 않고 표시.
출력: runs/s1_shape_graph/ (report.html, 그림, videos/), 반환 dict(lower_leg, irregular)."""
import json, shutil
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, log, report, table, img, video
OUT = RUNS / 's1_shape_graph'
SLIDE, SHORT = ROOT / 'check/slide', ROOT / 'check/short_leg'
LEG = {'E0': 0, 'E1': -20, 'E2': 20, 'L05': -5, 'L10': -10, 'L15': -15, 'L20': -20, 'L25': -25}
BODY = {'E3': -20, 'E0': 0, 'E4': 20}
METRICS = [('survival', '생존 비율 (실행/모션 길이)'), ('joint_rmse_deg', '관절 RMSE (°)'), ('root_rmse_cm', '몸통 위치 RMSE (cm)'),
           ('travel_ratio', '이동 비율 (실제/레퍼런스)'), ('slip_m_s', '발 미끄러짐 (m/s)')]


def rows():
    r = []
    for f in [SLIDE / 'results/summary.json', SHORT / 'results/summary.json']:
        if f.exists():
            for x in json.load(open(f)):
                x = dict(x, survival=x['seconds'] / max(x['clip_seconds'], 1e-6)); r.append(x)
    return r


def lower_bound(rs):
    lv = sorted([k for k in LEG if k.startswith('L')], key=lambda k: -LEG[k])  # L05, L10, ...
    fail = {k: any(not x['survived'] for x in rs if x['shape'] == k) for k in lv}
    have = [k for k in lv if any(x['shape'] == k for x in rs)]
    first = next((k for k in have if fail[k]), None)
    irregular = first is not None and any(not fail[k] for k in have[have.index(first) + 1:])
    if first is None: lower = have[-1] if have else None
    else:
        i = lv.index(first); lower = lv[i - 1] if i > 0 else 'E0'
    return lower, irregular, fail


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
    OUT.mkdir(parents=True, exist_ok=True); rs = rows(); rs.sort(key=lambda x: (x['shape'] != 'E1', x['shape']))  # E1 을 먼저 넣어 L20 이 덮어쓰게
    motions = sorted({x['motion'] for x in rs})
    figs = []
    for axis, name in [(LEG, '다리 길이 변화 (%)'), (BODY, '몸통 길이 변화 (%)')]:
        fig, ax = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4))
        for m in motions:
            byx = {}
            for x in rs:  # 같은 위치에 E1 과 L20 이 겹치면 L20 (짧은 다리 실험) 을 쓴다
                if x['motion'] == m and x['shape'] in axis and not (x['shape'] == 'E1' and axis[x['shape']] in byx):
                    byx[axis[x['shape']]] = x
            pts = sorted(byx.items(), key=lambda t: t[0])
            if len(pts) < 2: continue
            for j, (key, _) in enumerate(METRICS):
                ax[j].plot([p[0] for p in pts], [np.nan if p[1].get(key) is None else p[1][key] for p in pts], 'o-', label=m.replace('__r00', ''))
        for j, (_, t) in enumerate(METRICS): ax[j].set_title(t); ax[j].set_xlabel(name); ax[j].grid(alpha=.3)
        ax[1].set_yscale('log'); ax[2].set_yscale('log'); ax[0].legend(fontsize=8)
        fig.tight_layout(); f = OUT / f"metric_vs_{'leg' if axis is LEG else 'body'}.png"; fig.savefig(f, dpi=105); plt.close(fig); figs.append(f)
    lower, irregular, fail = lower_bound(rs)
    vids = OUT / 'videos'; vids.mkdir(exist_ok=True); vh = ''
    for src in [SHORT / 'video', SLIDE / 'video']:
        for mp4 in sorted(src.glob('*/*.mp4')):
            dst = vids / f'{mp4.parent.name}_{mp4.name}'
            if not dst.exists(): shutil.copy(mp4, dst)
    for mp4 in sorted(vids.glob('L*.mp4')): vh += video(f'videos/{mp4.name}', mp4.stem, tag=mp4.stem.split('_', 1)[1].replace('__r00', ''))
    tab = table(['형상', '모션', '생존', '관절 RMSE (°)', '몸통 RMSE (cm)', '이동 비율', '미끄러짐 (m/s)'],
                [(x['shape'], x['motion'].replace('__r00', ''), f"{'○' if x['survived'] else '×'} {x['seconds']:.1f}/{x['clip_seconds']:.1f}s",
                  f"{x['joint_rmse_deg']:.1f}", f"{x['root_rmse_cm']:.1f}", f"{x['travel_ratio']:.2f}", '-' if x.get('slip_m_s') is None else f"{x['slip_m_s']:.3f}") for x in rs],
                lambda i: 'bad' if not rs[i]['survived'] else '', filter_col=1)
    ck = SHORT / 'video/ckpt_curves.png'
    secs = [('다리 길이 축', img(figs[0], 'E0 = 0 %, E2 = +20 %, L05~L25 = -5~-25 % (허벅지·종아리 같은 비율)')),
            ('몸통 길이 축', img(figs[1], 'E3 = -20 %, E0, E4 = +20 %')),
            ('다리 하한 판정', f"<p>실패 여부 (squat·후진 중 하나라도 생존 실패): {', '.join(f'{k} {chr(215) if v else chr(9675)}' for k, v in fail.items())}</p>"
                            f"<p><b>하한 = {lower}</b>{' · <b>들쭉날쭉: 자동 판정 보류</b>' if irregular else ''}</p>")]
    if ck.exists(): secs.append(('짧은 다리 중간 체크포인트 평가', img(ck)))
    secs += [('전체 수치', tab), ('짧은 다리 영상 (A1 왼쪽, TT 오른쪽)', f'<div class="grid">{vh}</div>')]
    report(OUT / 'report.html', '1 단계: 지표 대 형상', secs,
           [('정책', '형상 1 × 모션 1 PPO (동시 환경 32, 5.24M step, lr 1e-3, 미끄러짐 감점 없음, 넘어지면 감점 없이 조기 종료, step 보상 ≥ 0)'),
            ('평가', '외란·잡음 없이 모션 처음부터 끝까지, 결정적 정책'), ('형상 토큰', 'check/slide/shapes.py, check/short_leg/shapes.py')])
    log(f's1 하한 {lower} irregular={irregular}')
    return dict(lower_leg=lower, irregular=irregular, images=figs)


if __name__ == '__main__':
    print(main())
