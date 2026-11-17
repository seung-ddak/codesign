"""보고서 A1_t3_t3imp: try3 PPO 공유 정책 (E0, 재학습 없음) 의 접촉 설정 (impratio 1 / 10 / 30 / 100) 별 몸통 위치 추종 오차·발 미끄러짐.
가설 (사용자): 발 미끄러짐이 줄면 밀림이 줄어 몸통 위치 추종 오차가 좋아진다.
모션: calm_06117 (E0 미학습, 새로 리타겟), calm_00026 (E0 학습). 영상: A1 · try3 (impratio 1) · try3 (impratio 100) 나란히.
출력: check/try3_contact/A1_t3_t3imp/ (report.html, 그림, videos/, results.json)"""
import json, shutil, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(CODE_DIR)); sys.path.insert(0, str(ROOT / 'check/slip_check/code')); sys.path.insert(0, str(ROOT / 'chain/code'))
OUT = HERE / 'A1_t3_t3imp'; (OUT / 'videos').mkdir(parents=True, exist_ok=True)
MOTIONS = ['calm_06117', 'calm_00026']
import slip_eval as se
cfg = {n[0]: (n, c) for n, c in se.CONFIGS.items()}
SETS = ['A', 'D', 'E', 'F']; IMP = {k: cfg[k][1]['impratio'] for k in SETS}
T3 = ROOT / 'legacy/runs/ppo_tracking/try3'; CK = T3 / 'stages/000/files/checkpoints/100270080'; E = [0.5] * 13


SWEEP = list(range(1, 11)); SWEEP_MOTIONS = ['calm_06117']  # calm_00026 은 sweep 안 함 (사용자 결정)


def jitter(q, dt):
    """떨림 지표: 관절 각가속도 (2 차 차분) 의 RMS, rad/s^2"""
    j = np.asarray(q)[:, 7:19]; return float(np.sqrt(np.mean(np.square(np.diff(j, 2, axis=0) / dt ** 2))))


def refdir(M):
    r = T3 / 'references/e0000/00000011'
    return r if (r / f'{M}.npz').exists() else HERE / 'references/E0/00000011'


def main():
    rf = OUT / 'results.json'; res = json.loads(rf.read_text()) if rf.exists() else {}
    for M in MOTIONS:
        for k in SETS:
            key = f'{M}|{k}'
            if key in res: continue
            r, q = se.run_ck(CK, E, refdir(M), M, cfg[k][1], multi=True, keep=True); r.update(shape='E0', config=cfg[k][0], impratio=IMP[k])
            res[key] = r; np.save(OUT / f'qpos_{M}_{k}.npy', q); rf.write_text(json.dumps(res)); print(key, r['slip_contact'], r['root_rmse_cm'], flush=True)
    for M in SWEEP_MOTIONS:  # 추가: impratio 1~10 (타원 원뿔·반복 50/20) 촘촘히
        for i in SWEEP:
            key = f'{M}|s{i}'
            if key in res: continue
            c = dict(cone=1, impratio=i, it=50, ls=20)
            r, q = se.run_ck(CK, E, refdir(M), M, c, multi=True, keep=True); r.update(shape='E0', config=f'elliptic impratio {i}', impratio=i)
            r['jitter'] = jitter(q, r['dt']); res[key] = r; rf.write_text(json.dumps(res)); print(key, r['slip_contact'], r['root_rmse_cm'], r['jitter'], flush=True)
    for M in MOTIONS:
        for k in SETS:
            if 'jitter' not in res[f'{M}|{k}']: res[f'{M}|{k}']['jitter'] = jitter(np.load(OUT / f'qpos_{M}_{k}.npy'), res[f'{M}|{k}']['dt'])
    rf.write_text(json.dumps(res))
    for M in MOTIONS:  # 영상: A1 · impratio 1 · impratio 100 나란히
        dst = OUT / 'videos' / f'{M}_A1_try3_imp1_imp100.mp4'
        if dst.exists(): continue
        sys.argv = [sys.argv[0], M]; import importlib, run as R; importlib.reload(R)
        z = np.load(refdir(M) / f'{M}.npz')
        R.render(np.load(OUT / f'qpos_{M}_A.npy'), np.load(OUT / f'qpos_{M}_F.npy'), z['a1_qpos'], float(z['dt']), dst, res[f'{M}|A'], res[f'{M}|F'],
                 la='TT try3 (middle): impratio 1 (training)', ld='TT try3 (right): elliptic, impratio 100, iter 50',
                 head=f'E0 · {M}   A1 (left) · try3 impratio 1 (middle) · try3 impratio 100 (right)', short=('imp1', 'imp100'), side=True)
    make_report(res)


def make_report(res):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from common import report, table, img, video
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
    cols = {'A': '#555555', 'D': '#2a78d6', 'E': '#1baa6b', 'F': '#e8590c'}
    # 1) 시간에 따른 몸통 위치 오차 (모션별)
    fig, ax = plt.subplots(len(MOTIONS), 2, figsize=(14, 4 * len(MOTIONS)), squeeze=False)
    for i, M in enumerate(MOTIONS):
        for k in SETS:
            r = res[f'{M}|{k}']; t = np.arange(len(r['root_err_t'])) * r['dt']
            ax[i, 0].plot(t, np.array(r['root_err_t']) * 100, color=cols[k], label=f'impratio {IMP[k]}' + (' (학습 설정)' if k == 'A' else ''), lw=1.4)
            ax[i, 1].plot(t, np.array(r['root_xy_err_t']) * 100, color=cols[k], lw=1.4)
        ax[i, 0].set_title(f'{M}: 몸통 위치 오차 (3 차원, cm)'); ax[i, 1].set_title(f'{M}: 몸통 수평 위치 오차 (xy, cm)')
        for a in ax[i]: a.set_xlabel('시간 (s)'); a.grid(alpha=.3)
        ax[i, 0].legend(fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / 'root_err_time.png', dpi=105); plt.close(fig)
    # 2) impratio 대 평균 오차·미끄러짐
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for M, mk in zip(MOTIONS, ['o-', 's--']):
        x = [IMP[k] for k in SETS]
        ax[0].plot(x, [res[f'{M}|{k}']['slip_contact'] for k in SETS], mk, label=M)
        ax[1].plot(x, [res[f'{M}|{k}']['root_rmse_cm'] for k in SETS], mk, label=M)
        ax[2].plot(x, [np.sqrt(np.mean(np.square(res[f'{M}|{k}']['root_xy_err_t']))) * 100 for k in SETS], mk, label=M)
    for a, t in zip(ax, ['접촉점 발 미끄러짐 (m/s)', '몸통 위치 RMSE (cm)', '몸통 수평 위치 RMSE (cm)']):
        a.set_xscale('log'); a.set_xticks([1, 10, 30, 100], ['1', '10', '30', '100']); a.set_xlabel('impratio'); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'impratio_summary.png', dpi=105); plt.close(fig)
    # 3) impratio 1~10 촘촘히: 미끄러짐·몸통 오차·떨림
    fig, ax = plt.subplots(1, 4, figsize=(19, 4))
    for M, mk in zip(SWEEP_MOTIONS, ['o-', 's--']):
        g = [res[f'{M}|s{i}'] for i in SWEEP]
        ax[0].plot(SWEEP, [r['slip_contact'] for r in g], mk, label=M)
        ax[1].plot(SWEEP, [r['root_rmse_cm'] for r in g], mk, label=M)
        ax[2].plot(SWEEP, [r['joint_rmse_deg'] for r in g], mk, label=M)
        ax[3].plot(SWEEP, [r['jitter'] for r in g], mk, label=M)
        for a, v in zip(ax, ['slip_contact', 'root_rmse_cm', 'joint_rmse_deg', 'jitter']):  # 학습 설정 (피라미드, impratio 1) 은 별표
            a.plot([1], [res[f'{M}|A'][v]], '*', ms=12, color=a.lines[-1].get_color())
    for a, t in zip(ax, ['접촉점 발 미끄러짐 (m/s)', '몸통 위치 RMSE (cm)', '관절 RMSE (°)', '떨림: 관절 각가속도 RMS (rad/s²)']):
        a.set_xticks(SWEEP); a.set_xlabel('impratio (타원 원뿔, 반복 50/20)'); a.set_title(t); a.grid(alpha=.3); a.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / 'impratio_sweep_1_10.png', dpi=105); plt.close(fig)
    rows = [(M, f'{IMP[k]}', cfg[k][0], '○' if res[f'{M}|{k}']['survived'] else '×', f"{res[f'{M}|{k}']['slip_contact']:.3f}", f"{res[f'{M}|{k}']['root_rmse_cm']:.2f}",
             f"{np.sqrt(np.mean(np.square(res[f'{M}|{k}']['root_xy_err_t']))) * 100:.2f}", f"{res[f'{M}|{k}']['joint_rmse_deg']:.2f}") for M in MOTIONS for k in SETS]
    vids = ''.join(video(f'videos/{p.name}', p.stem, tag=p.stem.split('_A1')[0]) for p in sorted((OUT / 'videos').glob('*.mp4')))
    report(OUT / 'report.html', 'A1_t3_t3imp: impratio 에 따른 몸통 위치 추종 오차', [
        ('impratio 대 미끄러짐·몸통 오차', img(OUT / 'impratio_summary.png', '가로축 로그. impratio 1 = 학습 설정 (피라미드 원뿔, 반복 5), 나머지 = 타원 원뿔 + 반복 50/20')),
        ('impratio 1~10 촘촘히 (떨림 포함)', img(OUT / 'impratio_sweep_1_10.png', '선 = 타원 원뿔·반복 50/20 에서 impratio 1~10. 별표 = 학습 설정 (피라미드 원뿔, impratio 1, 반복 5). 떨림 = 관절 각가속도 RMS')),
        ('시간에 따른 몸통 위치 오차', img(OUT / 'root_err_time.png', '레퍼런스 몸통 위치와 실제 몸통 위치 거리 (월드 좌표)')),
        ('수치', table(['모션', 'impratio', '설정', '생존', '접촉점 미끄러짐 (m/s)', '몸통 RMSE 3D (cm)', '몸통 RMSE xy (cm)', '관절 RMSE (°)'], rows, filter_col=0)),
        ('영상: A1 (왼쪽) · try3 impratio 1 (가운데) · try3 impratio 100 (오른쪽)', f'<div class="grid">{vids}</div>')],
        [('정책', 'try3 PPO 공유 정책 (CALM 10,000 × 형상 20, 발 미끄러짐 감점 -2.5, 1 억 step), 재학습 없음'), ('형상', 'E0 (e0000, 토큰 전부 0.5)'),
         ('모션', 'calm_06117 (E0 미학습, try3 의 30 Hz A1 레퍼런스에서 E0 로 새로 리타겟), calm_00026 (E0 학습 모션)'),
         ('접촉 설정', 'impratio 1 = 피라미드 원뿔·반복 5/7 (학습), 10·30·100 = 타원 원뿔·반복 50/20'), ('평가', '외란·잡음 없이 모션 처음부터 끝까지 (10 초), 30 Hz 제어'),
         ('몸통 위치 오차', '월드 좌표에서 레퍼런스 몸통 위치와의 거리 (시작 위치 같음, 누적 드리프트 포함)'), ('미끄러짐', '접지한 발의 접촉점 수평 속도 (구름 제외)')])
    print(OUT / 'report.html')


if __name__ == '__main__':
    main()
