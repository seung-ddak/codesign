"""calm_round_00_00019 (three_calm 모션) 의 발 미끄러짐 점검: check/try3_contact/A1_t3_t3imp 와 같은 방식.
s6 CCW 공용 정책 (100M, 재학습 없음) 을 접촉 설정 A (impratio 1, 학습) · D (타원+10) · E (타원+30) · F (타원+100) 로 실행, 학습 형상·미사용 형상.
가설 (사용자, 11-?): 발 미끄러짐이 줄면 몸통 위치 추종 오차가 좋아진다.
출력: runs/s7_eval/videos/three_calm/impratio/ (report.html, results.json, err_t.png, 영상: A1 · impratio 1 · impratio 100 같은 장면)"""
import json, os, sys
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
import numpy as np
from common import ROOT, RUNS, log, report, table, img, video
sys.path.insert(0, str(ROOT / 'check/slip_check/code')); sys.path.insert(0, str(ROOT / 'check/slide/code'))
import slip_eval as se
from three_calm import M, G, TR, UN, S5, OUT as T3, scene
from s7_eval import ckpts
OUT = T3 / 'impratio'; USE = 'ADEF'
IMP = {'A': 1, 'D': 10, 'E': 30, 'F': 100}


def main():
    OUT.mkdir(parents=True, exist_ok=True); ck = ckpts(RUNS / 's6_train' / G)[-1]; rows, keep = [], {}
    for (s, E), d, kind in [(TR, S5 / 'references' / TR[0] / '00000011', '학습'), (UN, T3 / 'refs' / UN[0] / '00000011', '미사용')]:
        for name, c in [(n, c) for n, c in se.CONFIGS.items() if n[0] in USE]:
            r, q = se.run_ck(ck, E, d, M, c, multi=True, keep=True); r.update(shape=s, kind=kind, config=name); rows.append(r)
            if kind == '학습' and name[0] in 'AF': keep[name[0]] = (q, r)
            log(f"imp {kind} {s} {name}: 생존 {r['survived']} 관절 {r['joint_rmse_deg']:.1f}° 몸통 {r['root_rmse_cm']:.1f} cm 미끄러짐 {r['slip_contact']:.3f}")
    json.dump(rows, open(OUT / 'results.json', 'w'), indent=1)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    fig, ax = plt.subplots(2, 2, figsize=(13, 7), sharex=True)
    for j, kind in enumerate(['학습', '미사용']):
        for r in [r for r in rows if r['kind'] == kind]:
            t = np.arange(len(r['root_err_t'])) * r['dt']; lab = f"impratio {IMP[r['config'][0]]}"
            ax[0, j].plot(t, np.array(r['root_err_t']) * 100, lw=1, label=lab); ax[1, j].plot(t, r['joint_err_t'], lw=1, label=lab)
        ax[0, j].set_title(f"{kind} 형상 {[r['shape'] for r in rows if r['kind'] == kind][0]}"); ax[0, j].set_ylabel('몸통 위치 오차 (cm)'); ax[1, j].set_ylabel('관절 오차 (°)'); ax[1, j].set_xlabel('시간 (s)')
        for a in ax[:, j]: a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'err_t.png', dpi=100); plt.close(fig)
    z = np.load(S5 / 'references' / TR[0] / '00000011' / f'{M}.npz'); dt = float(z['dt'])
    tts = [(keep[k][0], TR[1], f"TT {TR[0]} impratio {IMP[k]}: joint {keep[k][1]['joint_rmse_deg']:.1f} deg, slip {keep[k][1]['slip_contact']:.3f} m/s", not keep[k][1]['survived']) for k in 'AF']
    vp = OUT / f'{M}_A1_imp1_imp100.mp4'
    scene(z['a1_qpos'], dt, tts, vp, f'{M} · CCW policy (100M) · contact impratio 1 vs 100', 'A1 vs TT trained shape (impratio 1, training) vs TT (elliptic, impratio 100, iter 50)')
    trs = [(r['kind'], r['shape'], f"{r['config']} (impratio {IMP[r['config'][0]]})", '생존' if r['survived'] else f"{r['seconds']} 초 넘어짐", f"{r['joint_rmse_deg']:.1f}",
            f"{r['root_rmse_cm']:.1f}", f"{r['slip_center']:.3f}", f"{r['slip_contact']:.3f}") for r in rows]
    report(OUT / 'report.html', f'발 미끄러짐 · impratio 점검: {M}', [
        ('결과', table(['형상', '이름', '접촉 설정', '생존', '관절 RMSE (°)', '몸통 RMSE (cm)', '발 중심 미끄러짐 (m/s)', '접촉점 미끄러짐 (m/s)'], trs)),
        ('시간별 오차', img(OUT / 'err_t.png', '')),
        ('영상 (같은 장면)', video(vp.name, '파랑 A1 · 주황 TT impratio 1 (학습 설정) · 초록 TT impratio 100'))],
        [('정책', f'{G} s6 공용 정책 100M, 재학습 없음 (접촉 설정만 평가 때 바꿈)'), ('모션', f'{M} (CCW 9.1 s + Forward 5.9 s + CW 5.4 s, 20 s)'),
         ('형상', f'학습 {TR[0]} · 미사용 {UN[0]}'), ('접촉 설정', 'A pyramidal impratio 1 반복 5 (학습) · D elliptic 10 반복 50 · E elliptic 30 반복 50 · F elliptic 100 반복 50'),
         ('같은 방식', 'check/try3_contact/A1_t3_t3imp (try3 E0, calm_06117 · calm_00026)')])


if __name__ == '__main__':
    main()
