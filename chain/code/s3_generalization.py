"""3 단계: 학습하지 않은 형상에 기존 정책 적용 (일반화 범위 확인).
정책: check/slide 의 형상별 정책 (E0~E4 × 모션 5 개, 형상 정보 입력 없음). 평가 형상: 축 사이 (다리 ±10 %, 몸통 ±10 %) + 대각선 (다리 ±20 % × 몸통 ±20 %).
평가 형상마다 모션을 그 형상으로 리타겟하고, (a) 토큰 거리 최근접 학습 형상의 정책, (b) E0 정책을 적용.
출력: runs/s3_generalization/ (report.html, 그림, results.json, references/)."""
import json, sys
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, CODE, log, report, table, img, video, publish
from shapes_grid import token, name, dims
import evalpol
OUT = RUNS / 's3_generalization'
SLIDE = ROOT / 'check/slide'
TRAINED = {'E0': (0, 0), 'E1': (-20, 0), 'E2': (20, 0), 'E3': (0, -20), 'E4': (0, 20)}
MOTIONS = ['squat__r00', 'walkForwardSlow__r00', 'trotForwardFast__r00', 'trotCCWSlow__r00', 'walkBackwardSlow__r00']
TESTS = [(-10, 0), (10, 0), (0, -10), (0, 10), (-20, -20), (-20, 20), (20, -20), (20, 20), (-10, 10), (10, -10)]
SRC = ROOT / 'legacy/artifacts/randomized_reference'


def ckpt(K, M):
    import glob, os
    c = sorted(glob.glob(str(SLIDE / 'runs' / K / M / 'wandb/offline-run-*/files/checkpoints/*')), key=lambda p: int(os.path.basename(p)))
    return c[-1] if c else None


def retarget(leg, body):
    import pd_replay as pd
    dst = OUT / 'references' / name(leg, body)
    todo = [m for m in MOTIONS if not (dst / '00000011' / f'{m}.npz').exists()]
    if todo: pd.retarget_tt(str(SRC), str(dst), only=todo, choices=['00000011'], uniforms=token(leg, body), torque_limit=33.5)
    return dst / '00000011'


def nearest(leg, body):
    E = np.array(token(leg, body))
    return min(TRAINED, key=lambda k: np.linalg.norm(np.array(token(*TRAINED[k])) - E))


def main():
    OUT.mkdir(parents=True, exist_ok=True); res_f = OUT / 'results.json'
    res = json.loads(res_f.read_text()) if res_f.exists() else []
    done = {(r['leg'], r['body'], r['motion'], r['policy']) for r in res}
    # 기준선: 학습 형상에서 자기 정책
    for K, (l, b) in TRAINED.items():
        for M in MOTIONS:
            if (l, b, M, K) in done or ckpt(K, M) is None: continue
            r = evalpol.evaluate(ckpt(K, M), token(l, b), SLIDE / 'references' / K / '00000011', M)
            res.append(dict(r, leg=l, body=b, motion=M, policy=K, kind='학습 형상')); res_f.write_text(json.dumps(res, indent=1))
    for l, b in TESTS:
        rd = retarget(l, b); nk = nearest(l, b)
        for M in MOTIONS:
            for K in dict.fromkeys([nk, 'E0']):
                if (l, b, M, K) in done or ckpt(K, M) is None: continue
                r = evalpol.evaluate(ckpt(K, M), token(l, b), rd, M)
                res.append(dict(r, leg=l, body=b, motion=M, policy=K, kind='최근접' if K == nk else 'E0'))
                res_f.write_text(json.dumps(res, indent=1))
                log(f"s3 {name(l, b)} {M} policy {K}: surv {r['survived']} joint {r['joint_rmse_deg']} root {r['root_rmse_cm']}")
    render_videos()
    return make_report(res)


def render_videos(M='walkForwardSlow__r00'):
    """대각선 형상 4 개 × 모션 M, 최근접 정책 영상."""
    from render_tt import render_tt
    (OUT / 'videos').mkdir(exist_ok=True)
    for l, b in [(-20, -20), (-20, 20), (20, -20), (20, 20)]:
        dst = OUT / 'videos' / f'{name(l, b)}_{M}.mp4'
        if dst.exists(): continue
        K = nearest(l, b); r = evalpol.evaluate(ckpt(K, M), token(l, b), retarget(l, b), M, keep_qpos=True)
        render_tt(r['qpos'], token(l, b), dst, title=f'leg {l:+d}% body {b:+d}% · policy {K}', sub='unseen shape, nearest trained-shape policy', fell=not r['survived'])


def make_report(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, kind in zip(axs, ['최근접', 'E0']):
        pts = {}
        for r in res:
            if r['kind'] in (kind, '학습 형상'):
                pts.setdefault((r['leg'], r['body']), []).append(evalpol.passes(r))
        for (l, b), v in pts.items():
            frac = np.mean(v); trained = (l, b) in TRAINED.values()
            ax.scatter(l, b, s=900, c=[[1 - frac, frac * .7, .2]], marker='s' if trained else 'o', edgecolors='k')
            ax.text(l, b, f'{int(sum(v))}/{len(v)}', ha='center', va='center', color='w', fontsize=10, weight='bold')
        ax.set_title(f"통과 모션 수 ({'최근접 학습 형상 정책' if kind == '최근접' else 'E0 정책'})\n■ 학습 형상 (자기 정책) · ● 학습하지 않은 형상")
        ax.set_xlabel('다리 길이 변화 (%)'); ax.set_ylabel('몸통 길이 변화 (%)'); ax.set_xlim(-28, 28); ax.set_ylim(-28, 28); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'pass_map.png', dpi=105); plt.close(fig)
    rows = sorted(res, key=lambda r: (r['kind'] != '학습 형상', r['leg'], r['body'], r['motion']))
    tab = table(['형상 (다리 %, 몸통 %)', '치수', '모션', '정책', '구분', '생존', '관절 RMSE (°)', '몸통 RMSE (cm)', '이동 비율', '통과'],
                [(f"{r['leg']:+d}, {r['body']:+d}", dims(r['leg'], r['body']), r['motion'].replace('__r00', ''), r['policy'], r['kind'],
                  f"{'○' if r['survived'] else '×'} {r['seconds']:.1f}/{r['clip_seconds']:.1f}s", f"{r['joint_rmse_deg']:.1f}" if r['joint_rmse_deg'] is not None else '-',
                  f"{r['root_rmse_cm']:.1f}" if r['root_rmse_cm'] is not None else '-', f"{r['travel_ratio']:.2f}", '○' if evalpol.passes(r) else '×') for r in rows],
                lambda i: 'good' if evalpol.passes(rows[i]) else 'bad', filter_col=2)
    report(OUT / 'report.html', '3 단계: 학습하지 않은 형상에서 기존 정책 성능', [
        ('통과 지도', img(OUT / 'pass_map.png', '색: 초록 = 모션 전부 통과, 빨강 = 전부 실패')),
        ('전체 수치', tab),
        *[(f'영상: 대각선 형상 × {m} (최근접 정책)', '<div class="grid">' + ''.join(video(f'videos/{p.name}', p.stem, tag=m) for p in sorted((OUT / 'videos').glob(f'*_{m}__r00.mp4'))) + '</div>') for m in sorted({p.stem.split('_', 3)[3].replace('__r00', '') for p in (OUT / 'videos').glob('*.mp4')})]],
        [('정책', 'check/slide 형상별 PPO (E0~E4 × 모션 5, 형상 정보 입력 없음, 동시 환경 32, 5.24M step)'),
         ('평가 형상', '축 사이 다리 ±10 %, 몸통 ±10 %, 대각선 (다리, 몸통) = (±20, ±20), (-10, +10), (+10, -10)'),
         ('적용 정책', '토큰 거리 최근접 학습 형상의 정책, E0 정책'), ('레퍼런스', '평가 형상마다 새로 리타겟 (pd_replay.retarget_tt, 토크 33.5)'),
         ('평가', '외란·잡음 없이 모션 처음부터 끝까지'), ('통과 기준', '끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm')])
    return OUT / 'pass_map.png'


if __name__ == '__main__':
    import sys
    png = main()
    if '--publish' in sys.argv:
        res = json.loads((OUT / 'results.json').read_text()); test = [r for r in res if r['kind'] == '최근접']
        import evalpol as ev
        publish('s3_generalization', OUT, f"기존 형상별 정책 (E0~E4) 을 학습하지 않은 형상 {len(TESTS)} 개에 적용. 최근접 정책 통과 {sum(map(ev.passes, test))}/{len(test)}.", [png])
