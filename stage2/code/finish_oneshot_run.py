"""Evaluate a finished one-shot run, record its best final-policy sample, and write report.html."""

import argparse
import html
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from eval_locomotion_policy import EVAL_COMMANDS, LAUNCH_DIR  # noqa: E402  (chdirs into TT)

import jax  # noqa: E402
import jax.numpy as jp  # noqa: E402
import numpy as np  # noqa: E402

from design_scorer import DesignScorer  # noqa: E402
from locomotion_env import CTRL_DT, LocomotionEnv  # noqa: E402
from make_stage_report import page, table, tiles  # noqa: E402
from record_eval_video import record  # noqa: E402
from train_oneshot_morphology import MorphologyPolicy, valid_choices  # noqa: E402


def sample_final(run, config, count, seed):
    checkpoint = sorted(run.glob("phi_*.pt"))[-1]
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    choices = valid_choices()
    policy = MorphologyPolicy(choices, int(config["latent_size"]), int(config["hidden_size"]))
    policy.load_state_dict(state["policy"])
    torch.manual_seed(seed)
    with torch.no_grad():
        indices, uniforms, _ = policy.sample(torch.randn(count, int(config["latent_size"])))
    designs = [(choices[i].tolist(), u.tolist()) for i, u in zip(indices.tolist(), uniforms.numpy())]
    t5 = (LAUNCH_DIR / config["pi_theta"]).resolve()
    scorer = DesignScorer(json.loads((t5 / "designs.json").read_text()),
                          Path(config["pi_theta_ckpt"]), int(config["slots"]))
    returns = scorer.score(designs, seed=seed)
    rows = [{"choices": c, "uniforms": u, "return": float(r)}
            for (c, u), r in zip(designs, returns)]
    result = {"checkpoint": checkpoint.name, "seed": seed, "designs": rows}
    (run / "final_eval.json").write_text(json.dumps(result, indent=1))
    return rows, scorer.policy


def rollout_video(run, design, policy, seed, segment_s):
    env = LocomotionEnv(design["choices"], uniforms=design["uniforms"],
                        hold_command=True, domain_rand=False)
    segment = int(segment_s / CTRL_DT)
    commands = jp.repeat(jp.array([[vx, vy, wz, 0.] for _, vx, vy, wz in EVAL_COMMANDS]),
                         segment, axis=0)

    @jax.jit
    def rollout(rng):
        def body(state, inputs):
            key, command = inputs
            state = state.replace(info={**state.info, "command": command})
            state = env.step(state, policy(state.obs, key)[0])
            return state, (state.data.qpos, state.done)
        return jax.lax.scan(body, env.reset(rng),
                            (jax.random.split(rng, len(commands)), commands))[1]

    qpos, done = jax.device_get(rollout(jax.random.PRNGKey(seed)))
    commands = np.asarray(commands)
    steps = int(np.argmax(done > 0)) + 1 if done.any() else len(commands)
    videos = run / "videos"
    videos.mkdir(exist_ok=True)
    path = videos / "best_final.mp4"
    record(env, qpos, commands, steps, path)
    meta = {"file": path.name, "checkpoint_return": design["return"],
            "choices": design["choices"], "uniforms": design["uniforms"],
            "steps": steps, "seconds": steps * CTRL_DT,
            "completed_all_commands": steps == len(commands)}
    (videos / "final_videos.json").write_text(json.dumps([meta], indent=1))
    return meta


def report(run, config, history, final_rows, video):
    first = history[0]
    best_initial = max((d for row in history for d in row["designs"]),
                       key=lambda d: d["return"])
    best_final = max(final_rows, key=lambda d: d["return"])
    mean_final = float(np.mean([d["return"] for d in final_rows]))
    top = sorted(final_rows, key=lambda d: -d["return"])[:10]
    fmt_u = lambda values: "[" + ", ".join(f"{v:.2f}" for v in values) + "]"
    checkpoint = sorted(run.glob("phi_*.pt"))[-1].name
    lede = (f"One-shot morphology policy를 {len(history)} update 학습한 뒤 최종 checkpoint "
            f"<code>{html.escape(checkpoint)}</code>에서 "
            f"새 형상 {len(final_rows)}개를 sampling해 frozen πθ로 재평가했습니다.")
    sections = [
        ("요약", tiles([
            ("학습 update", str(len(history)), f"설정 목표 {config['updates']}"),
            ("초기 batch return", f"{first['return_mean']:.2f} / {first['return_max']:.2f}", "평균 / 최고"),
            ("최종 policy 재평가", f"{mean_final:.2f} / {best_final['return']:.2f}", "평균 / 최고"),
            ("최종 topology 다양성", str(len({tuple(d['choices']) for d in final_rows})),
             f"{len(final_rows)}개 sample"),
            ("PPO 진단", f"KL {first['approx_kl']:.2g}",
             f"entropy {first['entropy']:.3f} · grad {first['gradient_norm']:.2f}"),
        ]) + ("<p class='lede'><b>해석:</b> update가 1회뿐이므로 최종 재평가는 기능 확인용입니다. "
               "성능 개선 결론은 여러 update와 동일 budget random baseline 뒤에 내립니다.</p>")),
        ("최종 policy 최고 형상 영상",
         f'<video controls preload="metadata" style="width:100%;max-width:760px;border-radius:10px;background:#000">'
         f'<source src="videos/{html.escape(video["file"])}" type="video/mp4"></video>'
         f'<p class="lede">choice <code>{"".join(map(str, best_final["choices"]))}</code> · '
         f'return {best_final["return"]:.2f} · {video["seconds"]:.1f}s · '
         f'{"7개 명령 완료" if video["completed_all_commands"] else "낙상으로 조기 종료"}</p>'),
        ("최종 sample 상위 10개", table(["순위", "choice", "return", "uniforms"], [
            [i + 1, "".join(map(str, d["choices"])), f'{d["return"]:.2f}', fmt_u(d["uniforms"])]
            for i, d in enumerate(top)])),
        ("초기 batch 최고와 최종 sample 최고", table(["구분", "choice", "return", "uniforms"], [
            ["update 전 batch", "".join(map(str, best_initial["choices"])),
             f'{best_initial["return"]:.2f}', fmt_u(best_initial["uniforms"])],
            ["phi_0001 재평가", "".join(map(str, best_final["choices"])),
             f'{best_final["return"]:.2f}', fmt_u(best_final["uniforms"])]])),
        ("설정", table(["항목", "값"], [[k, v] for k, v in [
            ("updates", config["updates"]), ("batch", config["batch"]),
            ("latent / hidden", f'{config["latent_size"]} / {config["hidden_size"]}'),
            ("PPO epochs", config["ppo_epochs"]), ("learning rate", config["lr"]),
            ("entropy coefficient", config["entropy_coef"]),
            ("frozen pi_theta", config["pi_theta_ckpt"])]])),
    ]
    (run / "report.html").write_text(page(
        "T7 one-shot try1", "codesign · Stage 2 · T7B one-shot morphology RL", lede, sections))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--segment-seconds", type=float, default=3.)
    args = parser.parse_args()
    if args.samples < 1 or args.segment_seconds <= 0:
        parser.error("samples and segment-seconds must be positive")
    run = (LAUNCH_DIR / args.run).resolve()
    config = json.loads((run / "config.json").read_text())
    history = [json.loads(line) for line in (run / "history.jsonl").open()]
    final_rows, policy = sample_final(run, config, args.samples, args.seed)
    best = max(final_rows, key=lambda d: d["return"])
    video = rollout_video(run, best, policy, args.seed, args.segment_seconds)
    report(run, config, history, final_rows, video)
    print(f"wrote {run / 'final_eval.json'}")
    print(f"wrote {run / 'videos/best_final.mp4'}")
    print(f"wrote {run / 'report.html'}")


if __name__ == "__main__":
    main()
