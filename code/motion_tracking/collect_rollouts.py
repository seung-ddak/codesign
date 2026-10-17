"""Expert rollout 수집: 모션별 expert 를 무작위로 이어 10 초 sequence 를 MJX 에서 실행하고 (E, S, A) 를 저장한다.

DeepMimic composite 방식: 별도 전환 정책 없이 고정된 primitive expert 끼리 넘긴다.
  - 시작: 무작위 모션·무작위 프레임 (RSI)
  - switch_wait (기본 1~3 s) 마다 다른 무작위 모션으로 바꾼다. 들어갈 프레임 = 관절각·접지가 가장 비슷한 프레임,
    B reference 는 로봇 현재 xy·yaw 에 맞춘다 (track_env._switch). 클립이 먼저 끝나면 그 자리에서 바꾼다.
  - 매 step 현재 모션의 expert 가 action 을 낸다. 넘어지면 그 에피소드는 거기서 끝 (valid = 0).

usage (codesign/ 에서):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false ../transformer-transformer/.venv/bin/python \
      code/motion_tracking/collect_rollouts.py runs/pipeline_v1/experts/*__r00 --design e000 --episodes 64
인자 = expert 모션 폴더 (가장 최근 wandb offline-run 의 마지막 checkpoint) 또는 offline-run 폴더.
출력: <out>/<design>/seq_seed<seed>.npz
  qpos [B,T+1,nq], qvel [B,T+1,nv]  MJX 물리 state (root pos·quat·vel + 관절)
  contact [B,T+1,4]                 발 접지 (geom 거리 < CONTACT_DIST)
  action [B,T,12]                   정책 출력 (정규화 값)
  target_q [B,T,12]                 PD 목표 관절각 = action_offset + clip(action) * action_scale
  clip [B,T], ref_frame [B,T]       step t 에 action 을 낸 expert 의 모션 번호 (motions[clip]) 와 reference 프레임
  switched [B,T]                    step t 직후 모션이 바뀌었으면 1
  valid [B,T]                       step t 의 입력 state 가 낙상 이전이면 1
  fell [B], E [13], motions [M], dt
"""

import argparse
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "transformer-transformer"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from pd_replay import build_model
from policy_io import load_run
from track_env import TrackMjxEnv

CONTACT_DIST = 1e-3  # eval_motions.py 와 동일
CODESIGN = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("runs", nargs="+", type=Path)
parser.add_argument("--design", default="e000")
parser.add_argument("--mode", choices=("mixed", "primitive"), default="mixed")
parser.add_argument("--episodes", type=int, default=64, help="병렬 에피소드 수")
parser.add_argument("--seconds", type=float, default=10.0)
parser.add_argument("--switch-wait", type=float, nargs=2, default=(1.0, 3.0), help="s, 모션 바꾸기 간격")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("runs/pipeline_v1/rollouts"))
args = parser.parse_args()
assert args.episodes > 0 and args.seconds > 0
prefix = "seq" if args.mode == "mixed" else "primitive"
path = args.out / args.design / f"{prefix}_seed{args.seed}.npz"
if path.exists() or path.with_name(path.stem + "_pairs.csv").exists():
    raise FileExistsError(path)


def resolve(run):
    if (run / "files/checkpoints").is_dir():
        return run
    runs = [p for p in (run / "wandb").glob("offline-run-*") if (p / "files/checkpoints").is_dir()]
    if not runs:
        raise FileNotFoundError(f"No checkpoints under {run}")
    return max(runs, key=lambda p: p.name)


checkpoint_paths = []
experts = []  # (모션, policy)
for r in args.runs:
    ckpt = sorted(glob.glob(str(resolve(r) / "files/checkpoints/*")), key=lambda p: int(os.path.basename(p)))[-1]
    checkpoint_paths.append(str(Path(ckpt).resolve()))
    cfg, policy = load_run(ckpt)
    (name,) = cfg.eval_env.clips  # expert 1 개 = 모션 1 개
    experts.append((name, policy))
    print(f"{name:22s} <- {ckpt}")
motions = [n for n, _ in experts]
assert len(set(motions)) == len(motions), "같은 모션 expert 가 중복됨"

# env 는 마지막 checkpoint cfg 기준 (eval env: kick·초기 교란 없음, obs noise 는 학습 cfg 그대로). 모든 expert 가 같은 obs 를 쓴다고 가정
ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
ecfg.pop("_target_")
choice = cfg.get("choice")
# 다른 PC 에서 학습한 checkpoint 는 절대 경로가 다르므로 이 저장소 기준으로 바꾼다
ref_root = CODESIGN / Path(ecfg["reference_dir"].split("/codesign/", 1)[-1]).parents[1]
ecfg["reference_dir"] = str(ref_root / args.design / choice)
ecfg.update(clips=motions, clip_boost=None, switch_prob=float(args.mode == "mixed"), switch_wait=list(args.switch_wait), switch_at="time",
            switch_entry="match", max_episode_length=10**6)
with np.load(Path(ecfg["reference_dir"]) / f"{motions[0]}.npz") as z:
    uniforms = jnp.asarray(z["uniforms"], dtype=jnp.float32)
env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=cfg.get("kp"), tau_scale=cfg.get("tau_scale"),
                                                 torque_limit=cfg.get("torque_limit"), uniforms=uniforms), **ecfg)
feet = jnp.asarray(env._foot_ids)
n = env.n_ctrl
branches = [lambda o, k, p=p: p(o, k)[0] for _, p in experts]


def contact(data):
    geom, dist = data.contact.geom, data.contact.dist
    return jnp.stack([((geom == f).any(1) & (dist < CONTACT_DIST)).any() for f in feet])


def obs(s):
    return jnp.concatenate((s.obs, uniforms)) if cfg.get("multi_design") else s.obs


@jax.jit
@jax.vmap
def reset(key, clip):
    s = env.reset(key) if args.mode == "mixed" else env.reset_with_traj(key, clip, env.clip_start[clip])
    return s, contact(s.data)


@jax.jit
@jax.vmap
def step(s, key, alive):
    clip, frame = s.info["clip"], s.info["frame"]
    a = jax.lax.switch(clip, branches, obs(s), key)
    s = env.step(s, a)
    if args.mode == "mixed":
        # 클립 끝에서 reference 가 멈추므로 그 자리에서 다른 모션으로 바꾼다
        at_end = s.info["frame"] >= s.info["clip_end"] - 1
        info = dict(s.info)
        info["rng"], k = jax.random.split(info["rng"])
        p = jnp.where(env.pair_ok[info["clip"]], env.clip_p, 0.0)
        clip_b = jax.random.choice(k, len(motions), p=p / p.sum())
        info = env._switch(s.data, info, at_end, clip_b)
        info, _ = env.tracking_terms(s.data, info)
        s = s.replace(info=info, obs=env._get_obs(s.data, info))
    still = alive & (s.info["done/fellover"] == 0)
    if args.mode == "primitive":
        still &= s.info["frame"] < s.info["clip_end"] - 1
    return s, a, contact(s.data), clip, frame, s.info["clip"], still


T = round(args.seconds / env.dt) if args.mode == "mixed" else int((env.clip_end - env.clip_start).max()) - 1
B = args.episodes
rng = jax.random.PRNGKey(args.seed)
rng, k0 = jax.random.split(rng)
s, c = reset(jax.random.split(k0, B), jnp.arange(B) % len(motions))
rec = {k: [] for k in ("qpos", "qvel", "contact", "action", "clip", "ref_frame", "switched", "next_clip", "valid")}
rec["qpos"].append(s.data.qpos)
rec["qvel"].append(s.data.qvel)
rec["contact"].append(c)
alive = jnp.ones(B, bool)
fell = jnp.zeros(B, bool)
for t in range(T):
    rng, k = jax.random.split(rng)
    s, a, c, clip, frame, next_clip, still = step(s, jax.random.split(k, B), alive)
    if bool((alive & (s.info["state_is_nan"] != 0)).any()):
        raise FloatingPointError("Invalid physics step; rollout was not saved")
    for key, v in (("qpos", s.data.qpos), ("qvel", s.data.qvel), ("contact", c), ("action", a), ("clip", clip),
                   ("ref_frame", frame), ("switched", next_clip != clip), ("next_clip", next_clip), ("valid", alive)):
        rec[key].append(v)
    fell |= alive & (s.info["done/fellover"] != 0)
    alive = still
    if not bool(alive.any()):
        break
out = {k: np.asarray(jnp.stack(v, 1)) for k, v in rec.items()}
out["target_q"] = np.asarray(env.action_offset[:n]) + np.clip(out["action"], *env.clip_actions) * env.action_scale
out.update(fell=np.asarray(fell), E=np.asarray(uniforms), motions=np.array(motions), dt=env.dt, design=args.design, mode=args.mode,
           clip_start=np.asarray(env.clip_start), clip_end=np.asarray(env.clip_end), seed=args.seed,
           checkpoint_paths=np.array(checkpoint_paths), action_offset=np.asarray(env.action_offset[:n]),
           action_scale=np.asarray(env.action_scale), clip_actions=np.asarray(env.clip_actions))
path.parent.mkdir(parents=True, exist_ok=True)
np.savez_compressed(path, **out)
lens = out["valid"].sum(1)
print(f"B={B} T={T} steps={int(lens.sum())} len={lens.mean() * env.dt:.1f}±{lens.std() * env.dt:.1f}s "
      f"fell={out['fell'].mean():.0%} switches/ep={(out['switched'] & out['valid']).sum(1).mean():.1f} -> {path}")

# 전환 pair 별 낙상: 전환 후 FALL_WINDOW 안에 넘어지면 그 전환 탓으로 센다. 그 외 낙상은 모션 구간 낙상
FALL_WINDOW = round(1.0 / env.dt)
pairs, seg_falls = {}, {m: 0 for m in motions}
for b in range(B):
    L, last = int(lens[b]), None
    for t in np.where(out["switched"][b, :L])[0]:
        key = (motions[out["clip"][b, t]], motions[out["next_clip"][b, t]])
        pairs.setdefault(key, [0, 0])[0] += 1
        last = (key, t)
    if out["fell"][b]:
        if last and L - 1 - last[1] <= FALL_WINDOW:
            pairs[last[0]][1] += 1
        else:
            seg_falls[motions[out["clip"][b, L - 1]]] += 1
csv_path = path.with_name(path.stem + "_pairs.csv")
with open(csv_path, "w") as f:
    f.write("from,to,switches,falls_within_1s\n")
    for (a, b_), (k, fl) in sorted(pairs.items(), key=lambda x: -x[1][1] / x[1][0]):
        f.write(f"{a},{b_},{k},{fl}\n")
worst = sorted(((fl / k, a, b_, k) for (a, b_), (k, fl) in pairs.items() if fl), reverse=True)[:10]
print(f"pairs={len(pairs)} switches={sum(k for k, _ in pairs.values())} "
      f"falls: after switch={sum(fl for _, fl in pairs.values())}, in segment={sum(seg_falls.values())} -> {csv_path}")
for r, a, b_, k in worst:
    print(f"  {a} -> {b_}: {r:.0%} of {k}")
