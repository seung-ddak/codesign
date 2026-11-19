"""7h 단계: 100 Hz 추종 정책 (s6h_train100) 평가 + 30 Hz 기존 정책 (s6_train, s7_eval 선택 체크포인트) 기준값.
모션 세트마다:
  - 100 Hz 체크포인트 선택: 마지막 5 개를 표본 (학습 형상 9 × [레퍼런스 + CALM 1]) 으로 평가, 통과 수 최대 (동률 = 관절 RMSE 최소). s7_eval 과 같은 방식.
  - 평가 모션 = 레퍼런스 + CALM 3 개 (seed 0 고정, 세트의 CALM 중 30 Hz 정책도 학습한 것에서 뽑아 두 정책이 같은 모션).
  - 학습 형상 9 개: 100 Hz 만 (s5h 리타겟 그대로). 30 Hz 학습 형상 결과는 s7_eval 에 있음.
  - 처음 보는 형상 10 개 (UNSEEN, s8 형상 중 격자와 겹치지 않는 것): 100 Hz · 30 Hz 둘 다. 각 주기로 새로 리타겟 (100 Hz = s5h a1_100, 30 Hz = s5 a1_reference).
통과 기준: 끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm (모션 최대 10 초, evalpol.passes).
--wait 이면 chain.log 의 's6h END' 를 기다린 뒤 리타겟 → 평가 (학습과 GPU 를 겹치지 않음).
이어하기: results.json 에 (세트, 정책, 형상, 모션) 단위로 저장, 다시 실행하면 남은 것만.
출력: runs/s7h_eval100/ (results.json, selected.json, unseen_refs/, report.html, videos/)"""
import argparse, glob, json, os, time
from pathlib import Path
import numpy as np
from common import RUNS, CHAIN, log, report, table, img, video
OUT = RUNS / 's7h_eval100'; S5 = RUNS / 's5_retarget'; S5H = RUNS / 's5h_retarget100'; S6H = RUNS / 's6h_train100'; S6 = RUNS / 's6_train'
MAXS = 10.0; N_CALM = 3
UNSEEN = ['G_l+10_b-10', 'G_l+10_b+10', 'G_l-10_b-10', 'G_l-10_b+09', 'G_l-05_b+15',
          'G_l-15_b-15', 'G_l+15_b+15', 'G_l+19_b-10', 'G_l-11_b-01', 'G_l+10_b-19']
SRC = {'100Hz': S5H / 'a1_100', '30Hz': S5 / 'a1_reference'}
# 학습 환경 레퍼런스 속도 평활 (track_env.SG_SEC) 을 각 정책의 학습 값과 같게: 100 Hz = 70 ms (s6h, TRACK_SG_SEC=0.07), 30 Hz = 7 프레임 (기본값 7/30 s).
SG = {'100Hz': 0.07, '30Hz': 7 / 30}


def ev(hz, *args, **kw):
    import evalpol, track_env
    track_env.SG_SEC = SG[hz]  # 환경을 새로 만들 때 읽힘 (evalpol 은 ref_dir 별로 환경을 새로 만든다 — 두 주기의 ref_dir 은 다름)
    return evalpol.evaluate(*args, **kw)


def ckpts(f):
    return sorted(glob.glob(str(f / 'stages/*/files/checkpoints/*')), key=lambda p: int(os.path.basename(p)))


def motions(g, sets):
    """레퍼런스 + CALM N_CALM 개 (30 Hz 정책이 어느 형상에서든 학습한 CALM 중에서)."""
    p30 = json.load(open(S6 / g / 'plan.json')); seen30 = {c for v in p30['assignments'].values() for c in v[1:]}
    pool = sorted(set(sets[g][1:]) & seen30)
    return [g] + [str(c) for c in np.random.default_rng(0).choice(pool, N_CALM, replace=False)]


def unseen_dir(hz, shape, E, clips):
    import pd_replay as pd
    dst = OUT / 'unseen_refs' / hz / shape; todo = [c for c in clips if not (dst / '00000011' / f'{c}.npz').exists()]
    if todo: pd.retarget_tt(str(SRC[hz]), str(dst), only=todo, choices=['00000011'], uniforms=E, torque_limit=33.5)
    return dst / '00000011'


def retarget_all(sets, s8):
    for g in sets:
        M = motions(g, sets)
        for hz in SRC:
            for s in UNSEEN: unseen_dir(hz, s, s8[s]['E'], M)
    log('s7h 처음 보는 형상 리타겟 완료')


def wait_training():
    while 's6h END' not in (CHAIN / 'logs/chain.log').read_text(): time.sleep(300)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--wait', action='store_true'); ap.add_argument('--sets', nargs='*'); ap.add_argument('--video', nargs='+'); a = ap.parse_args()
    if a.video: return one_video(*a.video)
    OUT.mkdir(parents=True, exist_ok=True)
    A = json.load(open(S5H / 'assignment.json')); shapes, sets = A['shapes'], A['sets']
    s8 = json.load(open(RUNS / 's8_collect/shapes.json')); assert not set(UNSEEN) & set(shapes)
    if a.wait: wait_training()  # a1_100 은 s5h 가 모션 세트 순서로 만들므로 리타겟도 학습 (= s5h) 이 끝난 뒤
    retarget_all(sets, s8)
    import evalpol
    rf = OUT / 'results.json'; R = json.loads(rf.read_text()) if rf.exists() else {}
    sel30 = json.load(open(RUNS / 's7_eval/selected.json'))
    for g in a.sets or list(sets):
        f = S6H / g
        if not (f / 'done').exists(): log(f's7h {g} 학습 미완료 — 건너뜀'); continue
        M = motions(g, sets); r = R.setdefault(g, dict(motions=M, results=[]))
        if 'checkpoint' not in r:  # 100 Hz 체크포인트 선택
            sc = {}
            for ck in ckpts(f)[-5:]:
                res = [ev('100Hz', ck, shapes[s]['E'], S5H / 'references' / s / '00000011', m, multi=True, max_seconds=MAXS, clips=M[:2])
                       for s in shapes for m in M[:2]]
                sc[ck] = (sum(map(evalpol.passes, res)), -float(np.mean([x['joint_rmse_deg'] or 99 for x in res]))); log(f's7h {g} {Path(ck).name} {sc[ck]}')
            r['checkpoint'] = max(sc, key=lambda c: sc[c]); r['candidates'] = {c: list(v) for c, v in sc.items()}; rf.write_text(json.dumps(R, indent=1, default=float))
        done = {(x['policy'], x['shape'], x['clip']) for x in r['results']}
        jobs = [('100Hz', r['checkpoint'], s, shapes[s]['E'], S5H / 'references' / s / '00000011', '학습 형상') for s in shapes]
        jobs += [(hz, r['checkpoint'] if hz == '100Hz' else sel30[g], s, s8[s]['E'], OUT / 'unseen_refs' / hz / s / '00000011', '처음 보는 형상')
                 for hz in SRC for s in UNSEEN]
        for hz, ck, s, E, d, kind in jobs:
            for m in M:
                if (hz, s, m) in done: continue
                x = ev(hz, ck, E, d, m, multi=True, max_seconds=MAXS, clips=M)
                r['results'].append(dict(x, policy=hz, shape=s, clip=m, kind=kind, motion='레퍼런스' if m == g else 'CALM'))
            rf.write_text(json.dumps(R, indent=1, default=float))
        log(f's7h {g} 평가 완료')
    json.dump({g: r['checkpoint'] for g, r in R.items() if 'checkpoint' in r}, open(OUT / 'selected.json', 'w'), indent=1)
    videos(R, s8, sel30); make_report(R)
    log('s7h END')


def to50(q, hz):
    """render_tt 는 50 Hz 입력을 가정 → 시간축 선형 보간으로 50 Hz 로 맞춘다 (영상 속도 일치)."""
    t = np.arange(len(q)) / hz; t50 = np.arange(0, t[-1], 1 / 50)
    return np.stack([np.interp(t50, t, q[:, j]) for j in range(q.shape[1])], 1)


def videos(R, s8, sel30):
    """세트마다 처음 보는 형상 1 개 (UNSEEN[0]) 의 레퍼런스 모션: 100 Hz · 30 Hz 나란히 비교용 2 개."""
    from render_tt import render_tt
    import evalpol
    (OUT / 'videos').mkdir(exist_ok=True); s = UNSEEN[0]; E = s8[s]['E']
    for g, r in R.items():
        if 'checkpoint' not in r: continue
        for hz, ck, rate in [('100Hz', r['checkpoint'], 100), ('30Hz', sel30[g], 30)]:
            dst = OUT / 'videos' / f'{g}_{hz}.mp4'
            if dst.exists(): continue
            # 영상 1 개 = 프로세스 1 개 (한 프로세스에서 두 번째 렌더러를 만들면 X BadWindow 로 프로세스가 죽는다, 2026-11-18)
            import subprocess, sys
            rc = subprocess.run([sys.executable, '-u', __file__, '--video', g, hz]).returncode
            if rc: log(f's7h 영상 실패 {g} {hz}: exit {rc}')


def one_video(g, hz, clip=None):
    """clip 없으면 레퍼런스. clip = CALM 모션 이름이면 videos/<세트>_<clip>_<hz>.mp4."""
    from render_tt import render_tt
    R = json.loads((OUT / 'results.json').read_text()); r = R[g]; s8 = json.load(open(RUNS / 's8_collect/shapes.json'))
    sel30 = json.load(open(RUNS / 's7_eval/selected.json')); s = UNSEEN[0]; E = s8[s]['E']
    ck, rate = (r['checkpoint'], 100) if hz == '100Hz' else (sel30[g], 30)
    m = clip or g; dst = OUT / 'videos' / (f'{g}_{hz}.mp4' if m == g else f'{g}_{m}_{hz}.mp4')
    x = ev(hz, ck, E, OUT / 'unseen_refs' / hz / s / '00000011', m, multi=True, keep_qpos=True, max_seconds=MAXS, clips=r['motions'])
    render_tt(to50(x['qpos'], rate), E, dst, title=f"{g.replace('__r00', '')} · {m if m != g else 'reference'} · {s} · {hz}", fell=not x['survived'])


def make_report(R):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import evalpol
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    def stat(rr):
        return (f"{sum(map(evalpol.passes, rr))}/{len(rr)}", f"{np.mean([x['survived'] for x in rr]) * 100:.0f}%",
                f"{np.nanmedian([x['joint_rmse_deg'] or np.nan for x in rr]):.1f}", f"{np.nanmedian([x['root_rmse_cm'] or np.nan for x in rr]):.1f}")
    cols = [('100Hz', '학습 형상'), ('100Hz', '처음 보는 형상'), ('30Hz', '처음 보는 형상')]
    rows = [(g.replace('__r00', ''), hz, kind, mo) + stat(rr) for g, r in R.items() for hz, kind in cols for mo in ['레퍼런스', 'CALM']
            if (rr := [x for x in r['results'] if x['policy'] == hz and x['kind'] == kind and x['motion'] == mo])]
    tot = [(hz, kind, mo) + stat(rr) for hz, kind in cols for mo in ['레퍼런스', 'CALM']
           if (rr := [x for r in R.values() for x in r['results'] if x['policy'] == hz and x['kind'] == kind and x['motion'] == mo])]
    groups = list(R); fig, ax = plt.subplots(1, 3, figsize=(18, 0.5 * len(groups) + 3))
    for k, (hz, kind) in enumerate(cols):
        sh = sorted({x['shape'] for r in R.values() for x in r['results'] if x['kind'] == kind}); Mx = np.full((len(groups), len(sh)), np.nan)
        for i, g in enumerate(groups):
            for j, s in enumerate(sh):
                v = [evalpol.passes(x) for x in R[g]['results'] if x['policy'] == hz and x['shape'] == s]
                if v: Mx[i, j] = np.mean(v)
        ax[k].imshow(Mx, vmin=0, vmax=1, cmap='RdYlGn'); ax[k].set_title(f'통과 비율: {hz} · {kind}')
        ax[k].set_xticks(range(len(sh)), sh, rotation=45, ha='right', fontsize=8); ax[k].set_yticks(range(len(groups)), [g.replace('__r00', '') for g in groups])
    fig.tight_layout(); fig.savefig(OUT / 'pass_map.png', dpi=105); plt.close(fig)
    srows = [(g.replace('__r00', ''), hz, x['shape']) + stat(rr) for g, r in R.items() for hz in ['100Hz', '30Hz']
             for x in {x['shape']: x for x in r['results'] if x['policy'] == hz}.values()
             if (rr := [y for y in r['results'] if y['policy'] == hz and y['shape'] == x['shape']])]
    vs = ''.join(video(f'videos/{p.name}', p.stem, tag=p.stem.split('__')[0]) for p in sorted((OUT / 'videos').glob('*.mp4')))
    report(OUT / 'report.html', '7h 단계: 100 Hz 추종 정책 평가 (30 Hz 기준값 비교)', [
        ('통과 지도', img(OUT / 'pass_map.png', '행 = 모션 세트 정책, 열 = 형상')),
        ('전체 요약', table(['정책', '형상', '모션', '통과', '생존', '관절 RMSE 중앙 (°)', '몸통 RMSE 중앙 (cm)'], tot)),
        ('모션 세트별', table(['모션 세트', '정책', '형상', '모션', '통과', '생존', '관절 RMSE 중앙 (°)', '몸통 RMSE 중앙 (cm)'], rows, filter_col=0)),
        ('형상별', table(['모션 세트', '정책', '형상', '통과', '생존', '관절 RMSE 중앙 (°)', '몸통 RMSE 중앙 (cm)'], srows, filter_col=0)),
        (f'영상: 처음 보는 형상 {UNSEEN[0]}, 레퍼런스 모션 (100 Hz / 30 Hz)', f'<div class="grid">{vs}</div>' if vs else '<p>영상 없음</p>')],
        [('100 Hz 체크포인트', '마지막 5 개 중 학습 형상 9 × [레퍼런스 + CALM 1] 통과 수 최대 (동률 = 관절 RMSE 최소)'),
         ('30 Hz 체크포인트', 's7_eval/selected.json'), ('모션', f'세트마다 레퍼런스 + CALM {N_CALM} (seed 0, 30 Hz 도 학습한 CALM 중)'),
         ('처음 보는 형상', ', '.join(UNSEEN) + ' (s8 형상, 격자와 겹치지 않음; 각 주기로 새로 리타겟)'),
         ('평가', f'외란·잡음 없음, 모션 처음부터 최대 {MAXS:.0f} 초'), ('통과 기준', '끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm')])


if __name__ == '__main__':
    main()
