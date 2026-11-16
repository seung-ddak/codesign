"""s6c50 단계 (2026-11-14 사용자 결정): CALM 단독 정책을 50M 으로 다시 학습 (s6c 5M 은 CALM 에 부족했음).
대상: 묶음마다 s7 (s6 100M 재평가) 에서 실패한 CALM 2 개 + 통과한 CALM 2 개 = 24 개 (seed 0 무작위). 형상 9 개, 설정 = s6c 와 같고 학습량만 50M.
목적: 모션 문제 (단독 50M 도 실패) vs 공유 정책 문제 (단독 50M 은 통과) 구분. s6c 5M · s7 공유 정책과 모션별 비교.
출력: runs/s6c50_calm/<묶음>/<모션>/, results.json, report.html, scatter.png, videos/"""
import json
import numpy as np
import s6_train as s6, s6c_calm as s6c, evalpol
from common import RUNS, log, report, table, img, video
OUT = RUNS / 's6c50_calm'; STEPS = 50_000_000


def pick(rng=np.random.default_rng(0), n=2):
    r = json.load(open(s6c.S7 / 'results.json')); out = []
    for g, a in r.items():
        by = {}
        for x in a['results']:
            if x['clip'] != g: by.setdefault(x['clip'], []).append(evalpol.passes(x))
        fail = sorted(c for c, v in by.items() if not any(v)); ok = sorted(c for c, v in by.items() if all(v))
        out += [(g, c, 's7 실패') for c in rng.choice(fail, min(n, len(fail)), replace=False)]
        out += [(g, c, 's7 통과') for c in rng.choice(ok, min(n, len(ok)), replace=False)]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True); shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; T = pick()
    json.dump(T, open(OUT / 'targets.json', 'w'), indent=1)
    s6c.prepare_refs(shapes, sorted({c for _, c, _ in T}))
    rf = OUT / 'results.json'; res = json.loads(rf.read_text()) if rf.exists() else {}
    for i, (g, c, why) in enumerate(T):
        key = f'{g}/{c}'
        if key in res: continue
        f = s6c.build(g, c, shapes, out=OUT, steps=STEPS); ok = (f / 'done').exists() or s6.train(f)
        rr = s6c.evaluate(f, c, shapes) if ok else []
        res[key] = dict(group=g, clip=c, why=why, failed=not ok, results=rr)
        log(f"s6c50 [{i + 1}/{len(T)}] {key} ({why}) 통과 {sum(r['passed'] for r in rr)}/{len(rr)}")
        rf.write_text(json.dumps(res, indent=1, default=float))
    keep = s6c.OUT; s6c.OUT = OUT
    try: s6c.videos(res, shapes)
    finally: s6c.OUT = keep
    return make_report(res)


def make_report(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    s7r = json.load(open(s6c.S7 / 'results.json')); old = json.loads((s6c.OUT / 'results.json').read_text())
    def med(rr): return float(np.nanmedian([r['joint_rmse_deg'] or np.nan for r in rr])) if rr else np.nan
    rows, pts = [], []; cnt = dict(motion=0, policy=0, both=0)
    for key, v in sorted(res.items()):
        g, c = v['group'], v['clip']; a = [r for r in s7r[g]['results'] if r['clip'] == c]; b = v['results']; o = old.get(key, {}).get('results', [])
        pa, pb = sum(map(evalpol.passes, a)), sum(r['passed'] for r in b)
        verdict = '모션 문제' if b and pb / len(b) < .5 else ('공유 정책 문제' if a and pa / len(a) < .5 else '둘 다 통과')
        cnt['motion' if verdict == '모션 문제' else 'policy' if verdict == '공유 정책 문제' else 'both'] += 1
        rows.append((g.replace('__r00', ''), c, v['why'], f'{pa}/{len(a)}', f'{med(a):.1f}',
                     f"{sum(r['passed'] for r in o)}/{len(o)}" if o else '-', f'{med(o):.1f}' if o else '-', f'{pb}/{len(b)}', f'{med(b):.1f}', verdict))
        pts.append((v['why'], med(a), med(b)))
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for w, col in [('s7 실패', '#d62728'), ('s7 통과', '#2a78d6')]:
        p = np.array([(x, y) for ww, x, y in pts if ww == w]); ax.scatter(p[:, 0], p[:, 1], s=24, color=col, label=w) if len(p) else None
    ax.plot([0, 40], [0, 40], 'k:', lw=1); ax.axhline(10, color='gray', lw=.8); ax.axvline(10, color='gray', lw=.8)
    ax.set_xlabel('s7 공유 정책 (100M) 관절 RMSE 중앙 (°)'); ax.set_ylabel('단독 정책 50M 관절 RMSE 중앙 (°)'); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'scatter.png', dpi=110); plt.close(fig)
    vh = ''.join(video(f'videos/{p.parent.name}/{p.name}', f'{p.parent.name} · {p.stem}', tag=p.parent.name) for p in sorted((OUT / 'videos').glob('*/*.mp4')))
    report(OUT / 'report.html', 's6c50: CALM 단독 정책 50M (s7 공유 정책 · s6c 5M 과 비교)', [
        ('판정 요약', table(['모션 문제 (단독 50M 통과 < 50 %)', '공유 정책 문제 (단독 통과, s7 < 50 %)', '둘 다 통과'], [(cnt['motion'], cnt['policy'], cnt['both'])])),
        ('모션별 관절 오차', img(OUT / 'scatter.png', '점 = 모션. 위쪽 = 혼자 50M 학습해도 안 됨, 오른쪽 아래 = 혼자 학습하면 됨')),
        ('모션별', table(['묶음', '모션', '선정', 's7 통과', 's7 관절', 's6c 5M 통과', 's6c 5M 관절', '50M 통과 (형상 9)', '50M 관절', '판정'], rows, filter_col=0)),
        ('영상 (기준 형상, 묶음마다 오차 최소 2 · 최대 2)', f'<div class="grid">{vh}</div>' if vh else '<p>영상 없음</p>')],
        [('정책', '모션 1 개 × 학습 형상 9 개, s6_train.base_config, 학습량 50M (체크포인트 5M 간격), 마지막 체크포인트 평가'),
         ('대상', '묶음마다 s7 (s6 100M) 에서 실패 CALM 2 + 통과 CALM 2 (seed 0), 24 개'),
         ('평가', '외란·잡음 없음, 최대 10 초, 형상 9 개, 통과 = 끝까지 생존 · 관절 ≤ 10° · 몸통 ≤ 10 cm')])
    return f"CALM 단독 정책 50M {len(res)} 개 완료. 모션 문제 {cnt['motion']}, 공유 정책 문제 {cnt['policy']}, 둘 다 통과 {cnt['both']}."
