"""7 단계: 공유 정책 체크포인트 선택 + 평가.
묶음마다: 마지막 5 개 체크포인트를 표본 (형상 전부 × [레퍼런스 + 학습 CALM 1 개]) 으로 평가해 통과 수가 가장 많은 것 (동률이면 관절 RMSE 작은 것) 선택.
선택한 체크포인트로 전체 표본 평가: 형상마다 레퍼런스 + 학습 CALM 3 개 (학습 짝) + 다른 형상에만 배정된 CALM 2 개 (학습하지 않은 짝, 이 형상으로 새로 리타겟).
통과 기준: 끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm (모션 최대 10 초까지 평가). 영상: 묶음마다 대표 1 개.
출력: runs/s7_eval/ (results.json, selected.json {묶음: 체크포인트}, report.html, videos/)"""
import glob, json, os
from pathlib import Path
import numpy as np
from common import ROOT, RUNS, log, report, table, img, video
import evalpol
OUT = RUNS / 's7_eval'; S5 = RUNS / 's5_retarget'; S6 = RUNS / 's6_train'
MAXS = 10.0


def ckpts(f):
    return sorted(glob.glob(str(f / 'stages/*/files/checkpoints/*')), key=lambda p: int(os.path.basename(p)))


def unpaired_ref_dir(shape, E, clips):
    import pd_replay as pd
    dst = OUT / 'unpaired_refs' / shape; todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(S5 / 'a1_reference'), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return dst / '00000011'


def sample(plan, rng, n_train=3, n_unpaired=2):
    ref = plan['reference']; shapes = {d['design']: d['E'] for d in plan['designs']}; out = []
    allc = {c for v in plan['assignments'].values() for c in v[1:]}
    for s, E in shapes.items():
        mine = plan['assignments'][s][1:]; other = sorted(allc - set(mine))
        pick = list(rng.choice(mine, min(n_train, len(mine)), replace=False)) if mine else []
        un = list(rng.choice(other, min(n_unpaired, len(other)), replace=False)) if other else []
        out.append((s, E, [ref] + pick, un))
    return out


def run_sample(ck, smp, quick=False):
    res = []
    for s, E, train_clips, un in smp:
        td = S5 / 'references' / s / '00000011'; tc = train_clips[:2] if quick else train_clips
        for M in tc:
            r = evalpol.evaluate(ck, E, td, M, multi=True, max_seconds=MAXS, clips=tc); res.append(dict(r, shape=s, clip=M, pair='학습 짝'))
        if not quick and un:
            ud = unpaired_ref_dir(s, E, un)
            for M in un:
                r = evalpol.evaluate(ck, E, ud, M, multi=True, max_seconds=MAXS, clips=un); res.append(dict(r, shape=s, clip=M, pair='학습하지 않은 짝'))
    return res


def score(res):
    p = [evalpol.passes(r) for r in res]; j = [r['joint_rmse_deg'] or 99 for r in res]
    return (sum(p), -float(np.mean(j)))


def main():
    OUT.mkdir(parents=True, exist_ok=True); rf = OUT / 'results.json'
    allres = json.loads(rf.read_text()) if rf.exists() else {}; sel = {}
    for f in sorted(p for p in S6.iterdir() if (p / 'done').exists()):
        plan = json.load(open(f / 'plan.json')); rng = np.random.default_rng(0); smp = sample(plan, rng)
        if f.name not in allres:
            cands = ckpts(f)[-5:]; scores = {}
            for ck in cands:
                scores[ck] = score(run_sample(ck, smp, quick=True)); log(f's7 {f.name} {Path(ck).name} {scores[ck]}')
            best = max(cands, key=lambda c: scores[c])
            allres[f.name] = dict(checkpoint=best, candidates={c: list(v) for c, v in scores.items()}, results=run_sample(best, smp))
            rf.write_text(json.dumps(allres, indent=1, default=float))
        sel[f.name] = allres[f.name]['checkpoint']
    json.dump(sel, open(OUT / 'selected.json', 'w'), indent=1)
    videos(allres); make_report(allres)
    return sel


def videos(allres):
    """묶음마다 대표 (기준 형상, 레퍼런스 모션) 1 개 영상 (code/viz render_three 의 TT 단독 렌더)."""
    (OUT / 'videos' / 'center_ref').mkdir(parents=True, exist_ok=True)
    try:
        from render_tt import render_tt
    except Exception as e:
        log(f's7 영상 생략: {e}'); return
    for g, a in allres.items():
        dst = OUT / 'videos' / 'center_ref' / f'{g}.mp4'
        if dst.exists(): continue
        plan = json.load(open(S6 / g / 'plan.json')); d = plan['designs'][len(plan['designs']) // 2]
        try:
            r = evalpol.evaluate(a['checkpoint'], d['E'], S5 / 'references' / d['design'] / '00000011', g, multi=True, keep_qpos=True, max_seconds=MAXS)
            render_tt(r['qpos'], d['E'], dst, title=f"{g.replace('__r00', '')} · {d['design']}", fell=not r['survived'])
        except Exception as e:
            log(f's7 영상 실패 {g}: {e}')


def make_report(allres):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    groups = list(allres); shapes = sorted({r['shape'] for a in allres.values() for r in a['results']})
    fig, ax = plt.subplots(1, 2, figsize=(14, 0.5 * len(groups) + 2.5))
    for k, pair in enumerate(['학습 짝', '학습하지 않은 짝']):
        M = np.full((len(groups), len(shapes)), np.nan)
        for i, g in enumerate(groups):
            for j, s in enumerate(shapes):
                v = [evalpol.passes(r) for r in allres[g]['results'] if r['shape'] == s and r['pair'] == pair]
                if v: M[i, j] = np.mean(v)
        ax[k].imshow(M, vmin=0, vmax=1, cmap='RdYlGn'); ax[k].set_title(f'통과 비율 ({pair})')
        ax[k].set_xticks(range(len(shapes)), shapes, rotation=45, ha='right', fontsize=8); ax[k].set_yticks(range(len(groups)), [g.replace('__r00', '') for g in groups])
    fig.tight_layout(); fig.savefig(OUT / 'pass_map.png', dpi=105); plt.close(fig)
    rows = []
    for g, a in allres.items():
        for pair in ['학습 짝', '학습하지 않은 짝']:
            rr = [r for r in a['results'] if r['pair'] == pair]
            if rr: rows.append((g.replace('__r00', ''), Path(a['checkpoint']).name, pair, f"{sum(map(evalpol.passes, rr))}/{len(rr)}",
                                f"{np.mean([r['seconds'] / r['clip_seconds'] for r in rr]):.2f}", f"{np.nanmean([r['joint_rmse_deg'] or np.nan for r in rr]):.1f}",
                                f"{np.nanmean([r['root_rmse_cm'] or np.nan for r in rr]):.1f}", f"{np.nanmean([r['slip_m_s'] or np.nan for r in rr]):.3f}"))
    def kind(g, r): return '레퍼런스' if r['clip'] == g else ('학습 CALM' if r['pair'] == '학습 짝' else '학습하지 않은 CALM')
    def stat(rr): return (f"{sum(map(evalpol.passes, rr))}/{len(rr)}", f"{np.mean([r['survived'] for r in rr]) * 100:.0f}%",
                          f"{np.nanmedian([r['joint_rmse_deg'] or np.nan for r in rr]):.1f}", f"{np.nanmedian([r['root_rmse_cm'] or np.nan for r in rr]):.1f}")
    krows = [(g.replace('__r00', ''), k) + stat(rr) for g, a in allres.items() for k in ['레퍼런스', '학습 CALM', '학습하지 않은 CALM']
             if (rr := [r for r in a['results'] if kind(g, r) == k])]
    srows = [(g.replace('__r00', ''), s, k) + stat(rr) for g, a in allres.items() for s in shapes for k in ['레퍼런스', '학습 CALM', '학습하지 않은 CALM']
             if (rr := [r for r in a['results'] if r['shape'] == s and kind(g, r) == k])]
    vh = ''.join(video(f'videos/center_ref/{p.name}', p.stem, tag=p.stem.replace('__r00', '')) for p in sorted((OUT / 'videos/center_ref').glob('*.mp4')))
    va = ''.join(video(f'videos/all_shapes/{p.parent.name}/{p.name}', f'{p.parent.name} · {p.stem}', tag=p.parent.name)
                 for p in sorted((OUT / 'videos/all_shapes').glob('*/*.mp4')))
    report(OUT / 'report.html', '7 단계: 공유 정책 체크포인트 선택·평가', [
        ('통과 지도', img(OUT / 'pass_map.png', '행 = 레퍼런스 묶음 정책, 열 = 학습 형상')),
        ('묶음별 요약', table(['묶음', '선택 체크포인트 (step)', '구분', '통과', '생존 비율', '관절 RMSE (°)', '몸통 RMSE (cm)', '미끄러짐 (m/s)'], rows, filter_col=0)),
        ('레퍼런스 vs CALM (학습 형상 9 개)', table(['묶음', '모션 종류', '통과', '생존', '관절 RMSE 중앙 (°)', '몸통 RMSE 중앙 (cm)'], krows, filter_col=0)),
        ('형상별 · 모션 종류별', table(['묶음', '형상', '모션 종류', '통과', '생존', '관절 RMSE 중앙 (°)', '몸통 RMSE 중앙 (cm)'], srows, filter_col=0)),
        ('영상: 학습 형상 전부 (형상마다 레퍼런스 1 + 학습 CALM 1, videos/all_shapes/)', f'<div class="grid">{va}</div>' if va else '<p>영상 없음</p>'),
        ('영상: 기준 형상, 레퍼런스 모션 (기존, videos/center_ref/)', f'<div class="grid">{vh}</div>' if vh else '<p>영상 없음</p>')],
        [('선택', '마지막 5 개 체크포인트 중 표본 통과 수 최대 (동률 = 관절 RMSE 최소)'),
         ('표본', '형상마다 레퍼런스 + 학습 CALM 3 + 학습하지 않은 CALM 2 (새로 리타겟)'), ('평가', f'외란·잡음 없음, 모션 처음부터 최대 {MAXS:.0f} 초'),
         ('통과 기준', '끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm'),
         ('모션 종류', '레퍼런스 = 묶음 trot 레퍼런스 · 학습 CALM = 그 형상이 학습한 CALM · 학습하지 않은 CALM = 다른 형상에만 배정된 CALM (이 형상으로 새로 리타겟)')])
