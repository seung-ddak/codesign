"""6c 단계 (2026-11-13 사용자 결정): CALM 모션 하나를 학습 형상 9 개 전부로 학습하는 정책 (6a 의 CALM 판).
목적: s7 에서 CALM 이 실패하는 원인이 모션 (따라가기 어렵거나 품질 문제) 인지 공유 정책 (용량·학습량) 인지 구분.
대상: s7 에서 평가한 CALM 전부 (묶음당 44~45, 합 267). 설정 = 6a 와 같음 (s6_train.base_config, 학습량 s6_train.budget(1) ≈ 5M).
평가: 마지막 체크포인트, 형상 9 개, s7 과 같은 기준 (생존 · 관절 RMSE ≤ 10° · 몸통 RMSE ≤ 10 cm, 최대 10 초). s7 공유 정책 결과와 모션별 비교.
출력: runs/s6c_calm/<묶음>/<모션>/ (plan.json, stages/*, done), _refs/<형상>/00000011/ (9 형상 리타겟), results.json, report.html, videos/"""
import json, os
from pathlib import Path
import numpy as np
import s6_train as s6
from common import RUNS, log, report, table, img, video
OUT = RUNS / 's6c_calm'; S7 = RUNS / 's7_eval'; REFS = OUT / '_refs'
MAXS = 10.0


def targets():
    """[(묶음, 모션)] : s7 results.json 에 나온 CALM 전부."""
    r = json.load(open(S7 / 'results.json'))
    return [(g, c) for g, a in r.items() for c in sorted({x['clip'] for x in a['results'] if x['clip'] != g})]


def prepare_refs(shapes, clips):
    """9 형상 × 모션 리타겟 폴더. s5 references 에 있으면 링크, 없으면 s7 unpaired_refs 링크, 그래도 없으면 새로 리타겟."""
    import pd_replay as pd
    for s, v in shapes.items():
        d = REFS / s / '00000011'; d.mkdir(parents=True, exist_ok=True); todo = []
        for c in clips:
            dst = d / f'{c}.npz'
            if dst.exists(): continue
            for src in (s6.S5 / 'references' / s / '00000011' / f'{c}.npz', S7 / 'unpaired_refs' / s / '00000011' / f'{c}.npz'):
                if src.exists(): dst.symlink_to(src); break
            else: todo.append(c)
        if todo:
            log(f's6c 리타겟 {s} {len(todo)} 개')
            pd.retarget_tt(str(s6.S5 / 'a1_reference'), str(REFS / s), only=todo, choices=['00000011'], uniforms=v['E'], torque_limit=33.5)


def build(g, c, shapes, out=None, steps=None):
    f = (out or OUT) / g / c; f.mkdir(parents=True, exist_ok=True)
    if not (f / 'references').exists(): (f / 'references').symlink_to(REFS)
    if not (f / 'ready').exists(): (f / 'ready').symlink_to(s6.S5 / 'ready')
    cfg = s6.base_config(); cfg.update(resident_designs=len(shapes), train_transitions=steps or s6.budget(1))
    cfg['checkpoint_steps'] = max(1_000_000, cfg['train_transitions'] // 10)
    plan = dict(version=1, kind='ref_ppo', reference=c, group=g, motion_count=1, config=cfg, topology='00000011',
                designs=[dict(design=k, E=v['E']) for k, v in shapes.items()], eval_designs=[],
                assignments={k: [c] for k in shapes}, eval_assignments={k: [] for k in shapes})
    if not (f / 'plan.json').exists(): json.dump(plan, open(f / 'plan.json', 'w'), indent=1)
    return f


def evaluate(f, c, shapes):
    import evalpol
    from s7_eval import ckpts
    ck = ckpts(f)[-1]; out = []
    for s, v in shapes.items():
        r = evalpol.evaluate(ck, v['E'], REFS / s / '00000011', c, multi=True, max_seconds=MAXS)
        out.append(dict(r, shape=s, clip=c, checkpoint=ck, passed=evalpol.passes(r)))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    shapes = json.load(open(s6.S5 / 'assignment.json'))['shapes']; T = targets()
    prepare_refs(shapes, sorted({c for _, c in T}))
    rf = OUT / 'results.json'; res = json.loads(rf.read_text()) if rf.exists() else {}
    for i, (g, c) in enumerate(T):
        key = f'{g}/{c}'
        if key in res: continue
        f = build(g, c, shapes)
        ok = (f / 'done').exists() or s6.train(f)
        if not ok: log(f's6c 학습 실패 {key}'); res[key] = dict(group=g, clip=c, failed=True, results=[])
        else:
            rr = evaluate(f, c, shapes)
            res[key] = dict(group=g, clip=c, results=rr)
            log(f"s6c [{i + 1}/{len(T)}] {key} 통과 {sum(r['passed'] for r in rr)}/{len(rr)} "
                f"관절 중앙 {np.nanmedian([r['joint_rmse_deg'] or np.nan for r in rr]):.1f}°")
        rf.write_text(json.dumps(res, indent=1, default=float))
    videos(res, shapes); summary = make_report(res)
    return summary


def videos(res, shapes):
    """묶음마다 s6c 관절 오차가 가장 작은 모션 2 개 + 가장 큰 모션 2 개, 기준 형상 (가운데) 영상."""
    import evalpol
    from render_tt import render_tt
    center = list(shapes)[len(shapes) // 2]; V = OUT / 'videos'; V.mkdir(exist_ok=True)
    for g in sorted({v['group'] for v in res.values()}):
        items = [(np.nanmedian([r['joint_rmse_deg'] or np.nan for r in v['results']]), v) for v in res.values() if v['group'] == g and v['results']]
        items.sort(key=lambda x: x[0]); pick = [('best', v) for _, v in items[:2]] + [('worst', v) for _, v in items[-2:]]
        for tag, v in pick:
            dst = V / g.replace('__r00', '') / f"{tag}_{v['clip']}.mp4"; dst.parent.mkdir(exist_ok=True)
            if dst.exists(): continue
            try:
                ck = v['results'][0]['checkpoint']
                r = evalpol.evaluate(ck, shapes[center]['E'], REFS / center / '00000011', v['clip'], multi=True, keep_qpos=True, max_seconds=MAXS)
                render_tt(r['qpos'], shapes[center]['E'], dst, fell=not r['survived'],
                          title=f"6c {g.replace('__r00', '')} · {v['clip']} · {center} · {'통과' if evalpol.passes(r) else '실패'} · 관절 {r['joint_rmse_deg'] or 0:.1f}°")
            except Exception as e:
                log(f's6c 영상 실패 {v["clip"]}: {e}')


def make_report(res, note=None):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    import evalpol
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    s7r = json.load(open(S7 / 'results.json'))
    def med(rr): return float(np.nanmedian([r['joint_rmse_deg'] or np.nan for r in rr])) if rr else np.nan
    rows, pts, grp = [], [], {}
    for key, v in sorted(res.items()):
        g, c = v['group'], v['clip']
        a = [r for r in s7r[g]['results'] if r['clip'] == c]; b = v['results']
        pa, pb = sum(map(evalpol.passes, a)), sum(r['passed'] for r in b)
        rows.append((g.replace('__r00', ''), c, a[0]['pair'] if a else '', f'{pa}/{len(a)}', f'{med(a):.1f}', f'{pb}/{len(b)}', f'{med(b):.1f}',
                     '모션 문제' if b and pb / len(b) < .5 else ('공유 정책 문제' if a and pa / len(a) < .5 else '둘 다 통과')))
        pts.append((g, med(a), med(b)))
        d = grp.setdefault(g, dict(n=0, s7p=0, s7n=0, cp=0, cn=0, motion=0, policy=0))
        d['n'] += 1; d['s7p'] += pa; d['s7n'] += len(a); d['cp'] += pb; d['cn'] += len(b)
        d['motion'] += rows[-1][-1] == '모션 문제'; d['policy'] += rows[-1][-1] == '공유 정책 문제'
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for g in sorted(grp):
        p = np.array([(x, y) for gg, x, y in pts if gg == g]); ax.scatter(p[:, 0], p[:, 1], s=14, label=g.replace('__r00', ''))
    ax.plot([0, 40], [0, 40], 'k:', lw=1); ax.axhline(10, color='r', lw=.8); ax.axvline(10, color='r', lw=.8)
    ax.set_xlabel('s7 공유 정책 관절 RMSE 중앙 (°)'); ax.set_ylabel('6c 단독 정책 관절 RMSE 중앙 (°)'); ax.legend(fontsize=8); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'scatter.png', dpi=110); plt.close(fig)
    grows = [(g.replace('__r00', ''), d['n'], f"{d['s7p']}/{d['s7n']}", f"{d['cp']}/{d['cn']}", d['motion'], d['policy']) for g, d in sorted(grp.items())]
    vh = ''.join(video(f'videos/{p.parent.name}/{p.name}', f'{p.parent.name} · {p.stem}', tag=p.parent.name) for p in sorted((OUT / 'videos').glob('*/*.mp4')))
    report(OUT / 'report.html', '6c 단계: CALM 모션 1 개 × 학습 형상 전부 정책 (s7 공유 정책과 비교)', [
        ('묶음별 요약', table(['묶음', '모션 수', 's7 공유 정책 통과', '6c 단독 정책 통과', '모션 문제 (6c 통과 < 50 %)', '공유 정책 문제 (6c 통과, s7 < 50 %)'], grows)),
        ('모션별 관절 오차: 공유 vs 단독', img(OUT / 'scatter.png', '점 = 모션, 빨간선 = 기준 10°. 위쪽 = 혼자 학습해도 안 됨 (모션 문제), 오른쪽 아래 = 혼자 학습하면 됨 (공유 정책 문제)')),
        ('모션별', table(['묶음', '모션', 's7 구분', 's7 통과', 's7 관절 (°)', '6c 통과 (형상 9)', '6c 관절 (°)', '판정'], rows, filter_col=0)),
        ('영상 (기준 형상, 묶음마다 6c 오차 최소 2 · 최대 2)', f'<div class="grid">{vh}</div>' if vh else '<p>영상 없음</p>')],
        [('정책', '모션 1 개 × 학습 형상 9 개, s6_train.base_config (동시 환경 1,024, lr 1e-3, 256-128-64, 30 Hz / 240 Hz), 학습량 약 5M'),
         ('대상', 's7 에서 평가한 CALM 전부 (학습 짝 + 학습하지 않은 짝)'), ('평가', f'마지막 체크포인트, 외란·잡음 없음, 최대 {MAXS:.0f} 초, 형상 9 개'),
         ('통과 기준', '끝까지 생존 · 관절 RMSE ≤ 10° · 몸통 위치 RMSE ≤ 10 cm'),
         ('s7 비교', 's7 공유 정책은 모션마다 1~2 개 형상에서만 평가됨 (학습 짝 = 그 형상이 학습한 모션)')] + ([('주의', note)] if note else []))
    return grp


if __name__ == '__main__':
    main()
