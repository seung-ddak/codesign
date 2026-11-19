"""CALM 인코더 z 로 s4 선택 클립 6,000 개와 trot 레퍼런스 6 개를 인코딩해, z 기준 배정을 속도 기준(selected.json)과 비교.
z 계산 = collect_calm_joystick.latent() 와 같은 AMP 관측 60 프레임(30 Hz, 최신→과거) → llc.eval_enc. 창 간격 15 프레임.
배정 = 창마다 코사인 최근접 레퍼런스, 창 수 다수결 (s4 의 '가장 긴 시간' 규칙과 같은 방식). rlgpu python 으로 실행."""
import sys, json
from pathlib import Path
Q = Path('/home/cgr-ugrad-2026/quad_ws/codesign'); sys.path.insert(0, str(Q / 'chain/code'))
from collect_calm_joystick import load_policy  # isaacgym 먼저 import
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
OUT = Q / 'check/z_vs_speed'
WHEEL = Q / 'legacy/runs/menagerie_calm_check/try4/runtime/rl_games-1.1.4-py3-none-any.whl'
TROT = ['trotForwardSlow__r00', 'trotBackwardSlow__r00', 'trotLeftSlow__r00', 'trotRightSlow__r00', 'trotCWSlow__r00', 'trotCCWSlow__r00']
r = load_policy(WHEEL); torch = r['torch']; iq = r['iq']; ns = r['ns']; llc = r['llc']; norm_amp = r['norm_amp']; dev = 'cuda:0'
LEG = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]; FOOT = [1, 0, 3, 2]  # FR,FL,RR,RL -> FL,FR,RL,RR (CALM 순서)

def state(pos, quat, q, feet, dt):
    """quat xyzw. 반환 30 Hz 텐서들."""
    t = np.arange(len(pos)) * dt; t30 = np.arange(0, t[-1], 1 / 30)
    li = lambda a: np.stack([np.interp(t30, t, c) for c in a.reshape(len(a), -1).T], -1).reshape((len(t30),) + a.shape[1:])
    pos, q, feet = li(pos), li(q), li(feet); rot = Slerp(t, R.from_quat(quat))(t30); quat = rot.as_quat()
    v = np.gradient(pos, 1 / 30, axis=0); qv = np.gradient(q, 1 / 30, axis=0)
    w = (rot[1:] * rot[:-1].inv()).as_rotvec() * 30; w = np.r_[w, w[-1:]]
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
    return [T(pos), T(quat), T(q), T(v), T(w), T(qv), T(feet)]

def loop(pos, quat, q, feet, k=4):
    """짧은 trot 레퍼런스(<2 초)를 k 번 이어 붙인다. 다음 사본은 끝 자세의 수평 위치·yaw 로 강체 이동 (마지막 프레임 = 다음 첫 프레임과 겹쳐 제거)."""
    P, Qt, J, F = [pos], [quat], [q], [feet]
    for _ in range(k - 1):
        y0 = R.from_quat(Qt[0][0]).as_euler('zyx')[0]; y1 = R.from_quat(Qt[-1][-1]).as_euler('zyx')[0]
        rz = R.from_euler('z', y1 - y0); o = P[-1][-1] * [1, 1, 0] - rz.apply(pos[0] * [1, 1, 0])
        P.append(rz.apply(pos[1:]) + o); Qt.append((rz * R.from_quat(quat[1:])).as_quat()); J.append(q[1:])
        F.append(rz.apply(feet[1:].reshape(-1, 3)).reshape(-1, 4, 3) + o)
    return np.concatenate(P), np.concatenate(Qt), np.concatenate(J), np.concatenate(F)

def encode(s, stride=15):
    n = len(s[0]); ends = np.arange(59, n, stride)
    if len(ends) == 0: return None
    idx = torch.tensor(np.stack([e - np.arange(60) for e in ends]), device=dev).reshape(-1)
    rp, rq, qp, rv, rw, qv, kp = [x[idx] for x in s]
    amp = ns['build_amp_observations'](rp, rq, rv, rw, qp, qv, kp, True, True, 72, list(range(13)), True, True)
    hip = torch.tensor([[.1805, .047, 0], [.1805, -.047, 0], [-.1805, .047, 0], [-.1805, -.047, 0]], device=dev)
    a = qp.reshape(-1, 4, 3); ch, sh = torch.cos(a[:, :, 0]), torch.sin(a[:, :, 0]); st, ct = torch.sin(a[:, :, 1]), torch.cos(a[:, :, 1])
    side = torch.tensor([.0838, -.0838, .0838, -.0838], device=dev)
    local = torch.stack([-.2 * st, ch * side + .2 * sh * ct, sh * side - .2 * ch * ct], -1) + hip
    knee = rp[:, None] + iq.quat_rotate(rq[:, None].expand(-1, 4, -1).reshape(-1, 4), local.reshape(-1, 3)).reshape(-1, 4, 3)
    amp = torch.cat([amp, (kp[:, :, 2] < .045).float(), (knee[:, :, 2] < .06).float()], -1)
    amp[:, [10, 11]] = 0; amp[:, 85:97] = 0
    return torch.nn.functional.normalize(llc.eval_enc(norm_amp(amp).reshape(len(ends), -1)), dim=-1)

with torch.no_grad():
    REF = Q / 'legacy/artifacts/randomized_reference'; refz = []
    for n in TROT:
        z = np.load(REF / f'{n}.npz')
        refz.append(encode(state(*loop(z['fix_base_pos'], z['fix_base_quat'][:, [1, 2, 3, 0]], z['fix_q'][:, LEG], z['fix_foot_pos'][:, FOOT]), float(z['dt']))))
    refm = torch.nn.functional.normalize(torch.stack([x.mean(0) for x in refz]), dim=-1)
    sel = json.load(open(Q / 'chain/runs/s4_calm/selected.json'))
    rows = []
    for vref, files in sel.items():
        for f in files:
            z = np.load(f)
            zz = encode(state(z['root'][:, :3], z['root'][:, 3:7], z['q'], z['foot_pos'][:, FOOT], 1 / 30))
            sim = zz @ refm.T; votes = np.bincount(sim.argmax(1).cpu().numpy(), minlength=6)
            rows.append(dict(file=f, speed=TROT.index(vref), z=int(votes.argmax()), votes=votes.tolist(),
                             mean_cos=(zz.mean(0) / zz.mean(0).norm() @ refm.T).cpu().numpy().round(4).tolist()))
    refcos = (refm @ refm.T).cpu().numpy()
    within = [float((x @ refm[i]).mean()) for i, x in enumerate(refz)]
json.dump(dict(refs=TROT, ref_cos=refcos.tolist(), ref_within=within, clips=rows), open(OUT / 'assign.json', 'w'))
print('done', len(rows))
