"""조이스틱 방식 CALM 명령 계획 (2026-11-12 사용자 결정).
사람이 조이스틱으로 조종하듯: 1~3 초마다 새 목표 명령 (몸통 기준 vx, vy, 회전 wz) 을 고르고, 0.3~1.0 초에 걸쳐 부드럽게 (smoothstep) 바꾼다.
목표 명령 = trot 레퍼런스 속도 (counts.csv 의 ref_vx·vy·wz) × 0.7~1.3 + 가끔 다른 축 섞기 (전진+회전, 대각선) + 가끔 정지. 손떨림 수준의 작은 OU 잡음.
레퍼런스 선택은 할당량 대비 부족한 레퍼런스에 가중치 (usage 로 균형)."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
FPS = 30
# 2026-11-12 사용자 결정: trot 전·후·좌·우·CW·CCW (Slow) 6 개만 레퍼런스. Fast 는 빼고, 속도 배율 0.5~2.5 로 빠른 속도까지 덮는다.
TROT = ['trotForwardSlow__r00', 'trotBackwardSlow__r00', 'trotLeftSlow__r00', 'trotRightSlow__r00', 'trotCWSlow__r00', 'trotCCWSlow__r00']
SPEED_SCALE = (0.5, 2.5)
REF_CMD = {r['reference']: np.array([float(r['ref_vx']), float(r['ref_vy']), float(r['ref_wz'])])
           for r in csv.DictReader(open(ROOT / 'ref_ppo/_classification/counts.csv'))}
LIMIT = np.array([1.0, 0.8, 1.8])  # |vx|, |vy|, |wz| 상한


def ref_command(ref):
    """레퍼런스의 주 성분만 남긴 명령 (다른 축의 작은 값은 0)."""
    c = REF_CMD[ref].copy(); main = np.argmax(np.abs(c) / np.array([1, 1, 2]))
    out = np.zeros(3); out[main] = c[main]; return out


MAIN_P, MIX_P, STOP_P = 0.75, 0.10, 0.03  # 2026-11-12 사용자 결정 (점유율 높이기): 주 방향 구간 75 %, 다른 축 섞기 10 %, 정지 3 %


def make_joystick(seed, seconds, weights=None, refs=TROT, main=None):
    """main (레퍼런스 이름) 이 있으면 첫 구간과 구간의 MAIN_P 를 그 레퍼런스로 채운다. 없으면 예전 방식 (8 % 정지, 30 % 섞기)."""
    rng = np.random.default_rng(seed); n = int(seconds * FPS)
    w = np.ones(len(refs)) if weights is None else np.asarray(weights, float) + 1e-6
    cmd = np.zeros((n, 3)); cur = np.zeros(3); t = 0; keys = []
    while t < n:
        hold = int(rng.uniform(1.0, 3.0) * FPS); ramp = int(rng.uniform(0.3, 1.0) * FPS)
        u = rng.random(); stop_p, mix_p = (STOP_P, MIX_P) if main else (0.08, 0.30)
        if u < stop_p and t > 0:
            target, label = np.zeros(3), 'stop'
        else:
            if main and (t == 0 or rng.random() < MAIN_P): label = main
            else: k = rng.choice(len(refs), p=w / w.sum()); label = refs[k]
            target = ref_command(label) * rng.uniform(*SPEED_SCALE)
            if u < stop_p + mix_p:  # 다른 축 섞기
                j = rng.integers(3); mix = np.array([0.35, 0.3, 0.6])[j] * rng.uniform(-1, 1)
                if target[j] == 0: target[j] = mix
        target = np.clip(target, -LIMIT, LIMIT)
        end = min(n, t + hold); r = min(ramp, end - t)
        s = np.linspace(0, 1, r, endpoint=False) if r > 0 else np.zeros(0); s = s * s * (3 - 2 * s)
        cmd[t:t + r] = cur + (target - cur) * s[:, None]; cmd[t + r:end] = target
        keys.append(dict(start=int(t), end=int(end), label=label, target=target.round(4).tolist()))
        cur = target; t = end
    ou = np.zeros(3); noise = np.zeros_like(cmd)
    for i in range(n):  # 손떨림: 시간상수 0.5 초, 크기 상한의 3 %
        ou += -ou / (0.5 * FPS) + rng.normal(0, 1, 3) * LIMIT * 0.03 * np.sqrt(2 / (0.5 * FPS)); noise[i] = ou
    cmd = np.clip(cmd + noise, -LIMIT, LIMIT)
    return cmd, keys


def make_plan(count, seed, weights=None, refs=TROT):
    """collect_calm_joystick.py 가 읽는 계획: motions[i] = {id, seed, frames, seconds, joystick_seed, keys, segments}.
    길이: 절반 10 초, 절반 20 초 (기존 10,000 개와 같은 두 길이)."""
    plans = []; w = np.ones(len(refs)) if weights is None else np.asarray(weights, float) + 1e-6
    mains = np.random.default_rng(seed).choice(len(refs), size=count, p=w / w.sum())  # 주 방향은 부족분 가중으로, 나머지 구간은 고르게
    for i in range(count):
        sec = 10 if i % 2 == 0 else 20
        s = seed * 1_000_003 + i; main = refs[int(mains[i])]
        _, keys = make_joystick(s, sec, None, refs, main)
        segs = [dict(index=0, start=k['start'], end=k['end'], label=k['label']) for k in keys]
        sig = hashlib.sha256(json.dumps([s, sec, refs]).encode()).hexdigest()
        plans.append(dict(id=i, seed=int(s), frames=sec * FPS, seconds=sec, initial_pose='standing', joystick=True,
                          refs=list(refs), weights=None, main=main, segments=segs, signature=sig))
    return plans


def commands_for(plan):
    return make_joystick(plan['seed'], plan['seconds'], plan.get('weights'), plan.get('refs', TROT), plan.get('main'))[0]


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
    for j, s in enumerate([1, 2, 3]):
        c, k = make_joystick(s, 20, main=TROT[j])
        for d, name in enumerate(['vx', 'vy', 'wz']): ax[d].plot(np.arange(len(c)) / FPS, c[:, d], label=f'seed {s}'); ax[d].set_ylabel(name)
    ax[0].legend(); ax[2].set_xlabel('s'); fig.tight_layout(); fig.savefig(Path(__file__).with_name('joystick_example.png'), dpi=100)
    print([x['label'] for x in k])
