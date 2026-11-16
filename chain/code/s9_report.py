"""9 단계 보고서: 학습 곡선 + 예측 길이별 오차 + 실제 rollout vs TT 예측 나란히 영상 (묶음마다 1 개, 가장 긴 검증 에피소드)."""
import json
import numpy as np
from common import RUNS, log, report, table, img, video
OUT = RUNS / 's9_tt'; TT = OUT / 'tt'; V = OUT / 'videos'


def to_qpos(x):
    """S token [T,54] -> qpos [T,19] (xy, z, yaw 는 cos/sin, roll·pitch 는 중력 방향에서)."""
    yaw = np.arctan2(x[:, 4], x[:, 3]); g = x[:, 5:8] / np.linalg.norm(x[:, 5:8], axis=1, keepdims=True)
    p = np.arcsin(np.clip(g[:, 0], -1, 1)); r = np.arctan2(-g[:, 1], -g[:, 2])
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], 1)
    return np.concatenate([x[:, :3], quat, x[:, 14:26]], 1)


def videos():
    import imageio, tempfile
    from render_tt import render_tt
    d = np.load(TT / 'eval_dynamics_ddim_samples.npz'); shapes = json.load(open(RUNS / 's8_collect/shapes.json')); out = []
    for g in ['trotForwardSlow', 'trotBackwardSlow', 'trotLeftSlow', 'trotRightSlow', 'trotCWSlow', 'trotCCWSlow']:
        idx = [i for i, s in enumerate(d['src']) if g in s]
        if not idx: continue
        i = max(idx, key=lambda i: d['mask'][i].sum()); s = d['src'][i].split('/')[0]; m = d['mask'][i]
        dst = V / f'{g}_{s}.mp4'; out.append(dst)
        if dst.exists(): continue
        V.mkdir(parents=True, exist_ok=True); tmp = [V / f'_{k}.mp4' for k in ('true', 'pred')]
        for k, f in zip(('true', 'pred'), tmp):
            render_tt(to_qpos(d[k][i][m]), shapes[s]['E'], f, title=f"{'Real rollout' if k == 'true' else 'TT prediction'} · {g} · {s}", sub='s9 TT (DDIM autoregressive)')
        a, b = imageio.get_reader(str(tmp[0])), imageio.get_reader(str(tmp[1]))
        with imageio.get_writer(str(dst), fps=a.get_meta_data()['fps'], codec='libx264', quality=7, macro_block_size=8) as w:
            for fa, fb in zip(a, b): w.append_data(np.concatenate([fa, fb], 1))
        for f in tmp: f.unlink()
        log(f's9 영상 {dst.name}')
    return out


def curves():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    P = [json.loads(l) for l in open(TT / 'progress.jsonl')]; st = [p['step'] for p in P]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(st, [p['loss'] for p in P], label='학습'); ax[0].plot(st, [p['val_loss'] for p in P], label='검증'); ax[0].set_yscale('log'); ax[0].set_title('손실')
    ax[1].plot(st, [p['joint_kinematic_rmse_deg'] for p in P], label='학습'); ax[1].plot(st, [p['val_joint_kinematic_rmse_deg'] for p in P], label='검증'); ax[1].set_title('관절 RMSE (°)')
    ax[2].plot(st, [p['xy_kinematic_rmse_cm'] for p in P], label='학습'); ax[2].plot(st, [p['val_xy_kinematic_rmse_cm'] for p in P], label='검증'); ax[2].set_title('몸통 xy RMSE (cm)')
    for a in ax: a.set_xlabel('step'); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'curves.png', dpi=100); plt.close(fig)
    E = json.load(open(TT / 'eval_dynamics_ddim.json')); ks = [k for k in E if k.startswith('t')]; t = [int(k[1:]) for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); b = E.get('baseline_hold_s0_all', {})
    for a, f, n in [(ax[0], 'joint_rmse_deg', '관절 RMSE (°)'), (ax[1], 'pos_err_cm', '몸통 위치 오차 (cm)')]:
        a.plot(t, [E[k][f] for k in ks], 'o-', label='TT'); a.axhline(b.get(f, np.nan), color='gray', ls=':', label='처음 자세 유지 (전체 평균)')
        a.set_xscale('log'); a.set_xlabel('예측 step (50 Hz)'); a.set_title(n); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'horizon.png', dpi=100); plt.close(fig)
    return P[-1], E, ks


def main(tt_args=None):
    last, E, ks = curves(); vs = videos(); n = sum(1 for _ in (RUNS / 's8_collect').glob('G_*/*.npz'))
    rows = [(k, E[k].get('n', ''), f"{E[k]['joint_rmse_deg']:.1f}", f"{E[k]['pos_err_cm']:.0f}") for k in ks + ['all', 'baseline_hold_s0_all'] if k in E]
    vh = ''.join(video(f'videos/{p.name}', f'왼쪽 실제 rollout · 오른쪽 TT 예측 · {p.stem}') for p in vs)
    args = tt_args or (OUT / 'command.txt').read_text().split('> ')[-1].split(' --out ')[0]
    report(OUT / 'report.html', '9 단계: 생성 모델 (TT) 학습', [
        ('학습 곡선', img(OUT / 'curves.png', f"마지막 step {last['step']}: 손실 학습 {last['loss']:.3f} / 검증 {last['val_loss']:.3f}")),
        ('예측 길이별 오차 (검증 에피소드, DDIM 이어 붙이기)', img(OUT / 'horizon.png', '') + table(['예측 step', '에피소드', '관절 RMSE (°)', '몸통 위치 오차 (cm)'], rows)),
        ('실제 rollout vs TT 예측 (묶음마다 가장 긴 검증 에피소드)', f'<div class="grid">{vh}</div>' if vh else '<p>영상 없음</p>')],
        [('학습기', 'code/tt_dynamics/train_tt.py'), ('데이터', f'8 단계 rollout {n} 파일 (학습에 쓰지 않은 형상 20 × 묶음 6)'), ('인자', args),
         ('평가', '검증 에피소드 64 개, 처음 상태만 주고 TT 가 끝까지 예측, 실제 rollout 과 비교')])
    return OUT / 'tt'


if __name__ == '__main__':
    main()
