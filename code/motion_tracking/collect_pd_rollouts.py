"""Rollouts without a policy: replay each retargeted reference through the PD actuators in MJX (remind 8-35).

Each episode starts on a reference frame; the next reference joint angles are the PD target at every step.
There is no feedback on balance, so an episode ends when it falls or the reference ends.
Output = <out>/<design>/primitive_seed0.npz in the collect_rollouts format.
A stable (motion, local start frame) group ID keeps matching episodes together across designs.

usage (codesign/ 에서):
  ../transformer-transformer/.venv/bin/python code/motion_tracking/collect_pd_rollouts.py --out runs/pipeline_v1/pd_rollouts
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

from notify_selected_experts import TOKEN_FILE, post

ROOT = Path(__file__).resolve().parents[2]


def episode_group_ids(motions, frames):
    # Numeric IDs stay compatible with rollout_data.source_group_keys.
    return np.array([
        int.from_bytes(hashlib.blake2b(f"pd_reference:{motion}:{int(frame)}".encode(), digest_size=8).digest(), "big")
        for motion, frame in zip(motions, frames)
    ], dtype=np.uint64)


def save_rollout(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            np.savez_compressed(f, **data)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def is_complete_rollout(path):
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"qpos", "qvel", "contact", "action", "target_q", "valid", "clip", "ref_frame",
                        "fell", "E", "motions", "dt", "design", "mode", "group_id", "start_frame"}
            if not required.issubset(archive.files):
                return False
            # Read every member so a truncated/corrupt ZIP cannot pass on metadata alone.
            data = {k: archive[k] for k in archive.files}
        if data["mode"].item() != "pd_reference" or data["valid"].ndim != 2:
            return False
        b, t = data["valid"].shape
        return (b > 0 and t > 0 and data["group_id"].shape == data["start_frame"].shape == (b,)
                and all(data[k].shape[:2] == (b, t + 1) for k in ("qpos", "qvel", "contact"))
                and all(data[k].shape[:2] == (b, t) for k in ("action", "target_q", "clip", "ref_frame")))
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False


def notify_completion(paths, out, issue, elapsed, token):
    lines = [
        "@seung-ddak **PD rollout 데이터 수집 완료**",
        f"소요 시간: {elapsed / 60:.1f}분. PPO 정책 없이 MJX의 PD 액추에이터로 수집했습니다.",
        f"저장 위치: `{out.resolve()}` (로컬 저장 완료이며 데이터 업로드는 하지 않았습니다.)",
        "", "| 형상 | 모션 수 | 에피소드 | 유효 제어 스텝 | 낙상 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            lines.append(f"| {data['design'].item()} | {len(data['motions'])} | {len(data['valid'])} "
                         f"| {int(data['valid'].sum())} | {int(data['fell'].sum())} |")
    lines += ["", "낙상한 에피소드도 포함하며, 수집 완료는 모션 추종 성공을 뜻하지 않습니다."]
    body = "\n".join(lines) + "\n"
    # Keep the comment locally if delivery fails; rerunning skips completed data and retries delivery.
    report = out / "completion_comment.md"
    report.write_text(body)
    response = post(f"/issues/{issue}/comments", {"body": body}, token)
    print(f"GitHub completion notification: {response['html_url']}", flush=True)


def main():
    started = time.monotonic()
    sys.path.insert(0, str(ROOT.parent / "transformer-transformer"))
    import jax
    import jax.numpy as jnp
    from omegaconf import OmegaConf
    from pd_replay import build_model
    from track_env import TrackMjxEnv

    RETARGET = ROOT / "legacy/runs/pipeline_v1/retargeting"
    ALL_DESIGNS = [k for k in json.loads((RETARGET / "designs.json").read_text()) if k.startswith("e") and k != "e003"]
    with open(ROOT / "legacy/runs/pipeline_v1/experts/selection.csv") as f:
        ALL_MOTIONS = [row["motion"] for row in csv.DictReader(f)]

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--designs", nargs="+", default=ALL_DESIGNS)
    parser.add_argument("--motions", nargs="+", default=ALL_MOTIONS)
    parser.add_argument("--starts", type=int, default=8, help="모션당 시작 프레임 수 (전체 구간 균등 간격)")
    parser.add_argument("--min-steps", type=int, default=32, help="시작 후 남아야 하는 reference 길이")
    parser.add_argument("--expert-config", type=Path, default=ROOT / "legacy/runs/pipeline_v1/experts_wide16/walkForwardSlow__r00")
    parser.add_argument("--notify-issue", type=int, help="수집이 모두 끝나면 기존 GitHub 이슈에 멘션 댓글 전송")
    parser.add_argument("--kp", type=float, help="position actuator kp 교체 (remind 8-36 정상상태 처짐 점검). 기본 = expert config / build_model 기본값")
    args = parser.parse_args()
    token = None
    if args.notify_issue is not None:
        if args.notify_issue <= 0:
            parser.error("--notify-issue must be positive")
        if not TOKEN_FILE.is_file() or TOKEN_FILE.stat().st_mode & 0o077:
            parser.error(f"GitHub notification requires a private token file (mode 600): {TOKEN_FILE}")
        token = TOKEN_FILE.read_text().strip()
        if not token:
            parser.error("GitHub notification token is empty")

    cfg = pickle.load(open(sorted(args.expert_config.glob("wandb/offline-run-*/files/config.pkl"))[-1], "rb"))
    ecfg = OmegaConf.to_container(cfg.eval_env, resolve=True)
    ecfg.pop("_target_")
    choice = cfg.get("choice")
    ecfg.update(clips=list(args.motions), clip_boost=None, switch_prob=0.0, max_episode_length=10**6, termination_pose_consecutive_steps=10**6)


    def collect(design):
        ecfg["reference_dir"] = str(RETARGET / design / choice)
        with np.load(RETARGET / design / choice / f"{args.motions[0]}.npz") as z:
            uniforms = np.asarray(z["uniforms"], np.float32)
        kp = cfg.get("kp") if args.kp is None else args.kp
        env = TrackMjxEnv(mj_model_fn=lambda: build_model(choice, kp=kp, tau_scale=cfg.get("tau_scale"),
                                                         torque_limit=cfg.get("torque_limit"), uniforms=uniforms), **ecfg)
        n, starts, ends = env.n_ctrl, np.asarray(env.clip_start), np.asarray(env.clip_end)
        clips, frames = [], []
        for c, (lo, hi) in enumerate(zip(starts, ends)):
            last = hi - 1 - args.min_steps
            assert last >= lo, f"{args.motions[c]}: shorter than --min-steps"
            for f in np.unique(np.linspace(lo, last, args.starts).astype(int)):
                clips.append(c)
                frames.append(f)
        clips, frames = np.array(clips), np.array(frames)
        group_ids = episode_group_ids(np.asarray(args.motions)[clips], frames - starts[clips])
        B, T = len(clips), int((ends[clips] - 1 - frames).max())
        keys = jax.random.split(jax.random.PRNGKey(0), B)
        state = jax.jit(jax.vmap(env.reset_with_traj))(keys, jnp.asarray(clips), jnp.asarray(frames))
        feet, floor = env._foot_ids, int(env.mj_model.geom("floor").id)

        def contact(data):
            geom, dist = data._impl.contact.geom, data._impl.contact.dist
            return jax.vmap(lambda f: ((((geom[:, 0] == f) & (geom[:, 1] == floor)) | ((geom[:, 1] == f) & (geom[:, 0] == floor)))
                                       & (dist <= 1e-3)).any())(feet)

        env.action_reference = "home"
        @jax.jit
        def step(s, target):
            # Scripted PD targets are absolute and use the legacy home-relative API.
            action = (target - env.action_offset[:n]) / env.action_scale
            s = jax.vmap(env.step)(s, action)
            return s, action, jax.vmap(contact)(s.data)

        rec = {k: [] for k in ("qpos", "qvel", "contact", "action", "clip", "ref_frame", "valid")}
        rec["qpos"].append(np.asarray(state.data.qpos))
        rec["qvel"].append(np.asarray(state.data.qvel))
        rec["contact"].append(np.asarray(jax.vmap(contact)(state.data)))
        alive, fell = frames < ends[clips] - 1, np.zeros(B, bool)
        ref_q = np.asarray(env.ref_qpos[:, 7:7 + n])
        for t in range(T):
            frame = np.minimum(frames + t, ends[clips] - 2)
            state, action, touch = step(state, jnp.asarray(ref_q[frame + 1]))
            bad = alive & (np.asarray(state.info["state_is_nan"]) != 0)
            assert not bad.any(), f"{design}: nonfinite physics at step {t}"
            for k, v in (("qpos", state.data.qpos), ("qvel", state.data.qvel), ("contact", touch), ("action", action),
                         ("clip", clips), ("ref_frame", frame), ("valid", alive)):
                rec[k].append(np.asarray(v))
            fell |= alive & (np.asarray(state.info["done/fellover"]) != 0)
            alive = alive & ~fell & (frames + t + 1 < ends[clips] - 1)
            if not alive.any():
                break
        out = {k: np.stack(v, 1) for k, v in rec.items()}
        out["target_q"] = np.asarray(env.action_offset[:n]) + np.clip(out["action"], *env.clip_actions) * env.action_scale
        out.update(group_id=group_ids, start_frame=frames - starts[clips], fell=fell, E=uniforms,
                   motions=np.array(args.motions), dt=env.dt, design=design, mode="pd_reference",
                   kp=np.float32(np.nan if kp is None else kp),
                   contact_definition="foot_floor", clip_start=starts, clip_end=ends, action_offset=np.asarray(env.action_offset[:n]),
                   action_scale=env.action_scale, clip_actions=np.asarray(env.clip_actions), seed=0)
        path = args.out / design / "primitive_seed0.npz"
        save_rollout(path, out)
        steps = out["valid"].sum(1)
        print(f"{design}: fell {fell.sum()}/{B} | steps median {np.median(steps):.0f} / reference {np.median(ends[clips] - 1 - frames):.0f} -> {path}", flush=True)


    args.out.mkdir(parents=True, exist_ok=True)
    for design in args.designs:
        path = args.out / design / "primitive_seed0.npz"
        if is_complete_rollout(path):
            print(f"{design}: exists, skip", flush=True)
        else:
            if path.exists():
                print(f"{design}: incomplete or legacy output; recollect", flush=True)
            collect(design)
    if args.notify_issue is not None:
        paths = [args.out / design / "primitive_seed0.npz" for design in args.designs]
        notify_completion(paths, args.out, args.notify_issue, time.monotonic() - started, token)


if __name__ == "__main__":
    main()
