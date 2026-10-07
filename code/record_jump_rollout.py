"""고정 jump 스케줄 rollout 을 영상과 step 별 trace 로 저장한다 (DR·noise 끔, 결정적 정책)."""

import argparse
import json
import os
import sys
from pathlib import Path

# ── 경로·환경변수 (jax import 전에 설정) ─────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
LAUNCH_DIR = Path.cwd()
TT = ROOT / "transformer-transformer"
CODE = Path(__file__).resolve().parent
STAGE2 = ROOT / "codesign/stage2/code"
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
sys.path[:0] = [str(CODE), str(STAGE2), str(TT)]

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from task_env import JUMP_SETTLE_STEPS, FlatJumpEnv  # noqa: E402
from eval_locomotion_policy import load_policy  # noqa: E402
from record_eval_video import record  # noqa: E402


def main():
    # 1. 인자
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--ckpt", help="checkpoint step directory (default: latest)")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=5.)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    # 2. checkpoint·디자인·환경
    run = (LAUNCH_DIR / args.run).resolve()
    ckpt_name = args.ckpt or sorted(p.name for p in (run / "checkpoints").iterdir())[-1]
    ckpt = run / "checkpoints" / ckpt_name
    designs = [design for group in json.loads((run / "designs.json").read_text()) for design in group]
    choices, seed = designs[args.rank]
    env = FlatJumpEnv(choices, seed, domain_rand=False, add_noise=False)
    policy = load_policy(ckpt)
    steps = round(args.seconds / env.dt)

    # 3. rollout
    @jax.jit
    def rollout(rng):
        state = env.reset(rng)

        def body(state, key):
            command = state.info["command"]
            action, _ = policy(state.obs, key)
            state = env.step(state, action)
            return state, (state.data.qpos, command, action, state.info["last_contacts"], state.done,
                           state.metrics["jump_takeoff_success"], state.metrics["jump_landing_success"])

        return jax.lax.scan(body, state, jax.random.split(rng, steps))[1]

    qpos, command, action, contacts, done, takeoff, success = map(
        np.asarray, jax.device_get(rollout(jax.random.PRNGKey(args.seed))))
    n = int(np.argmax(done > 0)) + 1 if done.any() else steps  # 종료 step 까지만
    qpos, command, action, contacts = qpos[:n], command[:n], action[:n], contacts[:n]
    takeoff, success = takeoff[:n], success[:n]

    # 4. 영상
    out = run / "videos"
    out.mkdir(exist_ok=True)
    video = out / f"jump_rank_{args.rank}.mp4"
    record(env, qpos, command, n, video)

    # 5. trace: 접촉 기준 체공 + env 판정 (점프 인정·착지 성공)
    airborne = ~contacts.any(axis=1)
    jumped = airborne & (np.arange(n) >= JUMP_SETTLE_STEPS)  # 스폰 낙하 제외
    first_air = int(np.argmax(jumped)) if jumped.any() else None
    landed = np.flatnonzero(~airborne & (np.arange(n) > (first_air if first_air is not None else n)))
    first_landing = int(landed[0]) if landed.size else None
    trace = {
        "checkpoint": str(ckpt), "rank": args.rank, "choices": choices,
        "seed": seed, "dt": env.dt, "video": video.name,
        "trigger_steps": np.flatnonzero(command[:, 4] > .5).tolist(),
        "first_airborne_step": first_air,
        "first_landing_step": first_landing,
        "landed_after_airborne": first_landing is not None,
        "takeoff_steps": np.flatnonzero(takeoff > 0).tolist(),  # env 점프 인정
        "success_steps": np.flatnonzero(success > 0).tolist(),  # env 착지 성공
        # 안정 단계 끝 높이 대비 상승 (스폰 낙하 제외)
        "max_base_rise_m": float(qpos[JUMP_SETTLE_STEPS:, 2].max() - qpos[JUMP_SETTLE_STEPS - 1, 2]),
        "max_base_z_m": float(qpos[JUMP_SETTLE_STEPS:, 2].max()),
        "steps": [{"step": i, "jump_trigger": float(command[i, 4]),
                   "base_z": float(qpos[i, 2]), "all_feet_airborne": bool(airborne[i]),
                   "action_max_abs": float(np.abs(action[i]).max())} for i in range(n)],
    }
    trace_path = out / f"jump_rank_{args.rank}_trace.json"
    trace_path.write_text(json.dumps(trace, indent=1))
    print(f"trigger steps: {trace['trigger_steps']}")
    print(f"first airborne: {first_air}, first landing: {first_landing}, "
          f"max rise: {trace['max_base_rise_m']:.3f} m, max z: {trace['max_base_z_m']:.3f} m")
    print(f"takeoff steps: {trace['takeoff_steps']}, success steps: {trace['success_steps']}")
    print(f"wrote {video}\nwrote {trace_path}")


if __name__ == "__main__":
    main()
