"""예시 CALM 조이스틱 모션 3 개 (collect/accepted/*.npz) 를 A1 메쉬로 렌더링 + 명령 대 실제 속도 그림 + report.html.
명령: 조이스틱 (vx, vy, wz, 몸통 기준). 실제: 몸통 기준 속도 (30 Hz 차분, 0.3 s 평활)."""
import json, sys
from pathlib import Path
import numpy as np
CODE_DIR = Path(__file__).resolve().parent; HERE = CODE_DIR.parent; ROOT = next(p for p in HERE.parents if (p / 'chain').is_dir() and (p / 'check').is_dir())  # check/<묶음>/<실험> 깊이에 무관
sys.path.insert(0, str(ROOT / 'code/viz')); sys.path.insert(0, str(ROOT / 'chain/code'))
import mujoco
from PIL import Image, ImageDraw
import render_three as r3
from calm_joystick import commands_for
from scipy.spatial.transform import Rotation
from PIL import ImageFont
rm = r3.rm; W, H = 1280, 720; r3.W, r3.H = W, H; VFPS = 30  # 고화질: 1280×720, 30 fps (데이터와 같음), crf 16
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 26); FONT_B = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 28)
OUT = HERE / 'videos'; OUT.mkdir(exist_ok=True)
plan = {m['id']: m for m in json.load(open(HERE / 'collect/plan.json'))['motions']}


def body_vel(root, dt):
    yaw = np.unwrap(Rotation.from_quat(root[:, 3:7]).as_euler('zyx')[:, 0]); v = np.diff(root[:, :2], axis=0) / dt
    c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1]); out = np.c_[c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], np.diff(yaw) / dt]
    k = 9; ker = np.ones(k) / k
    return np.stack([np.convolve(out[:, i], ker, 'same') for i in range(3)], 1)


def seg_label(z, t):
    for a, b, i in zip(z['segment_start'], z['segment_end'], range(len(z['segment_start']))):
        if a <= t <= b: return i
    return None


def render(f, pid, tag='', note=''):
    z = np.load(f); dt = float(z['dt']); root, q = z['root'], z['q']; cmd = commands_for(plan[pid]); segs = plan[pid]['segments']  # 저장된 command 는 변환된 [월드 vx, vy, 목표 yaw] 라 조이스틱 명령을 다시 계산
    m = r3.a1_model(); m.vis.quality.offsamples = 8; p = r3.Panel(m); p.cam.distance = 2.0
    jn = list(z['joint_names']); qa = [m.jnt_qposadr[m.joint(n).id] for n in r3.A1_JOINTS]; src = [jn.index(n) for n in r3.A1_JOINTS]
    vel = body_vel(root, dt); stride = max(1, round(1 / dt / VFPS))

    def frames():
        for t in range(0, len(root), stride):
            qp = np.zeros(m.nq); qp[:3] = root[t, :3]; qp[3:7] = root[t, [6, 3, 4, 5]]; qp[qa] = q[t, src]
            im = Image.fromarray(p.frame(qp, root[:t + 1:2, :2], root[t, :2])); dr = ImageDraw.Draw(im)
            k = min(t, len(cmd) - 1); lab = next((s['label'].replace('__r00', '') for s in segs if s['start'] <= k < s['end']), '')
            dr.rectangle([0, 0, W, 118], fill=(15, 22, 30))
            dr.text((14, 8), f'CALM joystick example {pid}{note}  ({len(root) * dt:.0f} s)   segment: {lab}', font=FONT_B, fill=(255, 255, 255))
            dr.text((14, 46), f'command  vx {cmd[k, 0]:+.2f}  vy {cmd[k, 1]:+.2f} m/s  wz {cmd[k, 2]:+.2f} rad/s', font=FONT, fill=(255, 170, 60))
            dr.text((14, 80), f'actual   vx {vel[k, 0]:+.2f}  vy {vel[k, 1]:+.2f} m/s  wz {vel[k, 2]:+.2f} rad/s', font=FONT, fill=(120, 200, 255))
            dr.rectangle([0, H - 44, W, H], fill=(31, 111, 139)); dr.text((14, H - 38), f't = {t * dt:4.1f} s', font=FONT, fill=(255, 255, 255))
            yield np.asarray(im)
    out = OUT / f'calm_example_{pid}{tag}.mp4'
    try: rm.encode_frames(out, frames(), width=W, height=H, fps=VFPS, crf=16, preset='slow')
    finally: p.r.close()
    return z, vel, out, cmd


def diag_table():
    from common import table
    rows = []
    for d, lab in [('diag_lead2', '수정 전'), ('diag_int03', '수정 후 (차이 제한 0.3)'), ('diag_int01', '차이 제한 0.1')]:
        P = {m['id']: m for m in json.load(open(HERE / d / 'plan.json'))['motions']}
        for i, m in P.items():
            z = np.load(HERE / d / f'accepted/{i:05d}.npz'); v = body_vel(z['root'], float(z['dt']))[60:]
            rows.append((lab, m['segments'][0]['label'], str(m['const']), f'{v[:, 0].mean():+.2f}', f'{v[:, 1].mean():+.2f}', f'{v[:, 2].mean():+.2f}', f'{v[:, 2].std():.2f}'))
    return table(['방식', '명령 이름', '명령 (vx, vy, wz)', '실제 vx', '실제 vy', '실제 wz', 'wz 표준편차'], rows, filter_col=0)


def main():
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = 'Noto Sans CJK TC'; plt.rcParams['axes.unicode_minus'] = False
    files = sorted((HERE / 'collect/accepted').glob('*.npz')); fig, ax = plt.subplots(len(files), 3, figsize=(16, 3.2 * len(files)), squeeze=False)
    for i, f in enumerate(files):
        pid = int(f.stem); z, vel, out, cmd = render(f, pid); dt = float(z['dt']); t = np.arange(len(cmd)) * dt
        for j, (nm, u) in enumerate([('vx', 'm/s'), ('vy', 'm/s'), ('wz', 'rad/s')]):
            a = ax[i, j]; a.plot(t, cmd[:, j], color='#e8590c', label='명령'); a.plot(t[:len(vel)], vel[:, j], color='#2a78d6', label='실제')
            for s in plan[pid]['segments']: a.axvline(s['start'] * dt, color='#999', lw=.6)
            a.set_title(f'예시 {pid}: {nm} ({u})'); a.grid(alpha=.3)
            if j == 0: a.legend(fontsize=8)
        print(out, flush=True)
    ax[-1, 1].set_xlabel('시간 (s)'); fig.tight_layout(); fig.savefig(HERE / 'command_vs_actual.png', dpi=100); plt.close(fig)
    from common import report, table, img, video
    rows = [(pid, f"{m['seconds']} s", ' → '.join(f"{s['label'].replace('__r00', '').replace('trot', '').replace('Slow', '')}({(s['end'] - s['start']) / 30:.1f}s)" for s in m['segments'])) for pid, m in plan.items()]
    vids = ''.join(video(f'videos/calm_example_{pid}.mp4', f'예시 {pid}', tag=f'예시 {pid}') for pid in plan)
    report(HERE / 'report.html', 'CALM 조이스틱 모션 예시 3 개', [
        ('영상 (A1, 주황 = 명령, 파랑 = 실제 속도)', f'<div class="grid">{vids}</div>'),
        ('명령 대 실제 속도 (수정 후: 목표 방향 적분)', img(HERE / 'command_vs_actual.png', '세로선 = 구간 경계. 실제 속도는 몸통 기준, 0.3 초 평활')),
        ('회전 명령 변환 3 가지 비교 (예시 0, 같은 조이스틱 명령)', img(HERE / 'compare_heading.png', '주황 = 명령. 회색 = lead (수정 전), 파랑 = integrate (적분, 지금 기본), 초록 = step (CALM HLC 학습 때처럼 계단형 목표: 구간 시작과 1 초마다 목표 = 현재 방향 + 다음 1 초 wz 합)')
         + table(['방식', 'wz RMSE', 'wz 오차 최대', '방향각 오차 최대 (rad)', '끝 방향각 오차 (rad)', 'vx RMSE', 'vy RMSE'], json.load(open(HERE / 'compare_heading.json')))
         + f'<div class="grid">{video("videos/calm_example_0.mp4", "integrate (적분)", tag="integrate")}{video("videos/calm_example_0_step.mp4", "step (계단형)", tag="step")}</div>'),
        ('전환 순간 토크 대 회전 (수정 후)', img(HERE / 'torque_vs_turn.png', '빨간 띠 = |실제 wz − 명령 wz| > 0.5 rad/s. 세로선 = 구간 경계. 토크 = Isaac Gym 관절 힘 센서 (30 Hz 표본). latent 변화 = CALM 상위 정책 출력의 step 간 1−cos (code/plot_torque.py)')),
        ('수정 전 (목표 방향 = 현재 방향 + wz × 0.5 초)', img(HERE / 'command_vs_actual_lead.png', '같은 계획. 회전 명령보다 빠른 회전 (-2 rad/s 고정), 명령 없는 회전')),
        ('원인 진단: 고정 명령 10 초 (2~10 초 평균)', diag_table()),
        ('구간 순서', table(['예시', '길이', '구간 (레퍼런스 방향, 길이)'], rows))],
        [('생성', 'chain/code/calm_joystick.py make_plan(3, seed 777) → collect_calm_joystick.py (CALM 원본 HLC·LLC, Isaac Gym)'),
         ('레퍼런스', 'trot Slow 6 개 (전·후·좌·우·CW·CCW), 구간 1~3 초, 전환 0.3~1 초 부드럽게, 8 % 정지'),
         ('속도', '레퍼런스 속도 × 0.5~2.5, 30 % 확률로 다른 축 섞기, 손떨림 잡음 (상한의 3 %), 상한 |vx| 1.0 · |vy| 0.8 m/s · |wz| 1.8 rad/s'),
         ('회전 명령 변환 (수정)', 'CALM 상위 정책 (heading task) 은 목표 방향 오차가 0.1 rad 정도만 있어도 약 2 rad/s 로 돈다. 수정 전에는 목표 = 현재 방향 + wz×0.5 초라 오차가 계속 유지되어 항상 최대 속도로 돌았다. 수정 후: 목표 방향을 wz 로 적분하고 실제 방향과의 차이를 0.3 rad 로 제한 (collect_calm_joystick.py HEADING=integrate)'),
         ('통과 검사', '넘어짐 · 비유한 값 · 활동 검사 (3 개 모두 첫 시도에 통과)')])
    print(HERE / 'report.html')


if __name__ == '__main__':
    main()
