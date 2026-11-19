"""9z 보고서: s8z rollout (100 Hz, 형상 509) 으로 학습한 TT (H 32, split shape). 학습 곡선 + 손실 항목 + 예측 길이별 오차 + 실제 vs TT 예측 영상.
검증 episode 의 형상·모션은 train_tt 와 같은 순서·분할 (shape 단위, seed 0) 을 다시 계산해 찾는다 (eval samples 에 src 만 저장되므로).
영상: 검증 episode (평가 64 개) 중 세트마다 1 개 (가장 긴 것), 실제 = 불투명, TT 예측 = 주황 반투명, 처음 10 s. 영상마다 별도 프로세스 (렌더러 충돌 방지)."""
import json, subprocess, sys
import numpy as np
from common import RUNS, log, report, table, img, video
OUT = RUNS / 's9z_tt_h32'; TT = OUT / 'tt'; V = OUT / 'videos'; S8Z = RUNS / 's8z_collect'
EP = 2000; DT = 0.01; SETS = ['trotForwardSlow', 'trotBackwardSlow', 'trotLeftSlow', 'trotRightSlow', 'trotCWSlow', 'trotCCWSlow']
files = lambda: sorted((S8Z / 'rollouts').glob('*/chunk_*.npz'))


def val_meta():
    """train_tt.py 와 같은 계산: episode 순서 (파일 순, valid ≥ 2), 그룹 = shape, perm(seed 0) 앞 10 % 그룹 = val."""
    shape, motion, setn, L = [], [], [], []
    for f in files():
        with np.load(f) as z:
            v = np.minimum(z['valid'].sum(1), EP); keep = v >= 2
            shape += list(z['shape'][keep]); motion += list(z['motion'][keep]); setn += [f.parent.name] * int(keep.sum()); L += list(v[keep])
    u, inv = np.unique([f'shape:{s}' for s in shape], return_inverse=True); perm = np.random.default_rng(0).permutation(len(u))
    nv = min(len(u) - 1, max(1, int(len(u) * 0.1))); val = np.where(np.isin(inv, perm[:nv]))[0]
    return dict(shape=np.array(shape), motion=np.array(motion), set=np.array(setn), L=np.array(L), val=val, val_shapes=sorted({shape[i] for i in val}))


def to_qpos(x):
    yaw = np.arctan2(x[:, 4], x[:, 3]); g = x[:, 5:8] / np.linalg.norm(x[:, 5:8], axis=1, keepdims=True)
    p = np.arcsin(np.clip(g[:, 0], -1, 1)); r = np.arctan2(-g[:, 1], -g[:, 2])
    cr, sr, cp, sp, cy, sy = np.cos(r / 2), np.sin(r / 2), np.cos(p / 2), np.sin(p / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack([cr * cp * cy + sr * sp * sy, sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy], 1)
    return np.concatenate([x[:, :3], quat, x[:, 14:26]], 1)


def one_video(i, name):
    from render_tt import render_overlay
    d = np.load(TT / 'eval_dynamics_ddim_samples.npz'); plan = json.load(open(S8Z / 'plan.json')); meta = val_meta(); n = meta['val'][i]
    m = d['mask'][i]; T = min(int(m.sum()), 1000)
    qt, qp = to_qpos(d['true'][i][:T])[::2], to_qpos(d['pred'][i][:T])[::2]  # 100 Hz → render_overlay 의 50 Hz
    s = meta['shape'][n]; E = plan['shapes'][s]['E']
    def status(t):
        return f"t = {t * .02:4.1f} s   pos err {np.linalg.norm(qt[min(t, len(qt) - 1), :2] - qp[min(t, len(qp) - 1), :2]) * 100:5.1f} cm"
    render_overlay(qt, qp, E, V / name, title=f"Real (solid) vs TT (orange) · {meta['motion'][n]}", sub=f"val shape {s} · H32 autoregressive 100 Hz", status_fn=status)


def videos(meta, d):
    out = []
    for g in SETS:
        idx = [i for i in range(len(d['mask'])) if meta['set'][meta['val'][i]].startswith(g)]
        if not idx: continue
        i = max(idx, key=lambda i: d['mask'][i].sum()); n = meta['val'][i]; name = f"{g}_{meta['shape'][n]}_{meta['motion'][n]}.mp4"; out.append((name, i, n))
        if not (V / name).exists():
            V.mkdir(parents=True, exist_ok=True); subprocess.run([sys.executable, '-u', __file__, '--video', str(i), name], check=True); log(f's9z 영상 {name}')
    return out


def curves():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'
    P = [json.loads(l) for l in open(TT / 'progress.jsonl')]; st = [p['step'] for p in P]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(st, [p['loss'] for p in P], label='학습', alpha=.6); ax[0].plot(st, [p['val_loss'] for p in P], label='검증 (처음 보는 형상 50)')
    ax[0].set_yscale('log'); ax[0].set_title('손실'); ax[0].legend()
    for k in P[-1].get('val_loss_group_terms', {}):
        ax[1].plot(st, [p['val_loss_group_terms'][k] for p in P], label=k)
    ax[1].set_yscale('log'); ax[1].set_title('검증 손실 항목별'); ax[1].legend(fontsize=8)
    for a in ax: a.set_xlabel('step'); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'curves.png', dpi=100); plt.close(fig)
    E = json.load(open(TT / 'eval_dynamics_ddim.json')); ks = [k for k in E if k[0] == 't' and k[1:].isdigit()]; t = [int(k[1:]) * DT for k in ks]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4)); b = E.get('baseline_hold_s0_all', {})
    for a, f, n in [(ax[0], 'joint_rmse_deg', '관절 RMSE (°)'), (ax[1], 'pos_err_cm', '몸통 위치 오차 (cm)')]:
        a.plot(t, [E[k][f] for k in ks], 'o-', label='TT'); a.axhline(b.get(f, np.nan), color='gray', ls=':', label='처음 자세 유지 (전체 평균)')
        a.set_xscale('log'); a.set_xlabel('예측 시간 (s)'); a.set_title(n); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(OUT / 'horizon.png', dpi=100); plt.close(fig)
    return P, E, ks


def main():
    P, E, ks = curves(); d = np.load(TT / 'eval_dynamics_ddim_samples.npz'); meta = val_meta()
    assert all(int(d['mask'][i].sum()) == meta['L'][meta['val'][i]] for i in range(len(d['mask']))), 'val episode 재계산 불일치'
    vs = videos(meta, d); last = P[-1]; best = min(P, key=lambda p: p['val_loss'])
    rows = [(f"{int(k[1:]) * DT:g} s ({k[1:]} step)", E[k].get('n', ''), f"{E[k]['joint_rmse_deg']:.1f}", f"{E[k]['pos_err_cm']:.1f}") for k in ks]
    rows += [(k, '', f"{E[k]['joint_rmse_deg']:.1f}", f"{E[k]['pos_err_cm']:.1f}") for k in ('all', 'baseline_hold_s0_all') if k in E]
    grp = table(['항목', '마지막 검증 손실', '비율'], [(k, f"{v:.4f}", f"{last['val_loss_group_fractions'][k] * 100:.0f} %") for k, v in last.get('val_loss_group_terms', {}).items()])
    vh = ''.join(video(f'videos/{nm}', f'검증 형상 {meta["shape"][n]} · {meta["motion"][n]} · 길이 {int(d["mask"][i].sum()) / 100:.1f} s (영상은 처음 10 s)') for nm, i, n in vs)
    cmd = (OUT / 'code/train.sh').read_text()
    report(OUT / 'report.html', '9z 단계: TT 학습 (s8z rollout 100 Hz, H 32, 형상 단위 검증)', [
        ('학습 곡선', img(OUT / 'curves.png', f"마지막 step {last['step']}: 학습 {last['loss']:.4f} / 검증 {last['val_loss']:.4f} · 검증 최저 {best['val_loss']:.4f} (step {best['step']})") + grp),
        ('예측 길이별 오차 (검증 episode 64, 처음 상태 + 형상 + 행동 주고 상태 예측, H 32 창 이어 붙이기 = open-loop)', img(OUT / 'horizon.png', '') + table(['예측 시간', 'episode', '관절 RMSE (°)', '몸통 위치 오차 (cm)'], rows)),
        ('실제 rollout vs TT 예측 (세트마다 가장 긴 검증 episode)', f'<div class="grid">{vh}</div>' if vh else '<p>영상 없음</p>')],
        [('학습기', 'code/tt_dynamics/train_tt.py'), ('데이터', 's8z rollout 30,540 episode (형상 509 × 모션 60), 100 Hz, 넘어짐 포함, episode 최대 2000 step'),
         ('분할', f"형상 단위: train 459 / val 50 (학습 형상 중 val: {', '.join(s for s in meta['val_shapes'] if s in json.load(open(S8Z / 'plan.json'))['shapes'] and json.load(open(S8Z / 'plan.json'))['shapes'][s]['seen']) or '없음'})"),
         ('인자', cmd.split('train_tt.py')[1].replace('\\\n', ' ').split('\n')[0]), ('평가', 'dynamics 조건 (E·s0·A 주고 S 예측), DDIM, 창 32 를 31 step 씩 이어 붙여 episode 끝까지')])


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--video': one_video(int(sys.argv[2]), sys.argv[3])
    else: main()
