"""v16 Δ-loss A/B (remind 8-49) 의 twin / delta 폴더에 conditions.json·data_audit.json·run_train.py 를 만든다.

    python runs/pipeline_v1/prepare_v16.py        (action_pairs_wide16 16 파일 수집 완료 후)

데이터 (kp30 세계): wide16_action 16 + e_swap/full 16 + action_pairs_wide16 16. 학습 설정 = v12 와 동일 (modes 1/2/4, film, 60k).
twin: --delta-weight 0 (쌍은 모니터만). delta: --delta-weight 1.0 --pair-batch 16. 그 외 전부 동일.
"""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
P1 = ROOT / "runs/pipeline_v1"
DESIGNS = ["e000", "e001", "e002", "e004", "e005", "e006", "e007", "e008", "e009", "e010", "e011", "e012", "e013", "e014", "e015", "e016"]
DATA = ([f"runs/pipeline_v1/wide16_action/rollouts/{d}/primitive_seed0.npz" for d in DESIGNS]
        + [f"runs/pipeline_v1/e_swap/full/{d}/primitive_seed0.npz" for d in DESIGNS]
        + [f"runs/pipeline_v1/action_pairs_wide16/{d}_seed0.npz" for d in DESIGNS])
IMPL = ["code/tt_dynamics/train_tt.py", "code/tt_dynamics/tt_loss.py", "code/tt_dynamics/rollout_data.py", "code/tt_dynamics/tt_model.py"]
COMMON = ["--horizon", "32", "--steps", "60000", "--batch", "32", "--eval-episodes", "8", "--diff-steps", "5", "--norm", "minmax",
          "--clip", "1.0", "--sampler", "ddim", "--lr", "0.0002", "--d-model", "256", "--layers", "6", "--val-frac", "0.1",
          "--seed", "0", "--modes", "1", "2", "4", "--film", "--mode-weights", "0.2", "0.4", "0.4"]
GATE = dict(E_swap_sign_source="held_mode1_same_action_516", wrong_action_velocity_extra_error_min=0.01, wrong_action_ratio_vs_reference_min=2.0,
            E_swap_forward_change_sign_agreement_min=0.7, joint_survival_at_least_reference=True,
            delta_r2_min=0.0, legacy_moving_rmse_max_ratio_vs_reference=1.1)


def sha(p):
    return hashlib.sha256((ROOT / p).read_bytes()).hexdigest()


def audit():
    files = episodes = valid = falls = 0
    for p in DATA:
        with np.load(ROOT / p) as z:
            files += 1
            episodes += len(z["valid"])
            valid += int(z["valid"].sum())
            falls += int(np.asarray(z["fell"]).sum()) if "fell" in z else int((~z["valid"][:, -1]).sum())
    return dict(files=files, episodes=episodes, valid_controls=valid, falls=falls,
                sources=dict(wide16_action=16, e_swap_full=16, action_pairs_wide16=16), kp=30)


def main():
    for p in DATA:
        assert (ROOT / p).exists(), p
    data_audit = audit()
    data_sha = {p: sha(p) for p in DATA}
    impl_sha = {p: sha(p) for p in IMPL}
    runs = dict(
        fast_mode24_v16_twin=dict(extra=[], reference_run="runs/pipeline_v1/fast_mode24_v12_pd_kp200_mode1",
                                  training_reference="runs/pipeline_v1/fast_mode24_v12_pd_kp200_mode1/tt_joint/config.json",
                                  change="twin: v12 설정 + kp30 데이터 (wide16 + e_swap + action_pairs_wide16), --delta-weight 0 (쌍 손실 없음, R²_Δ 모니터만)"),
        fast_mode24_v16_delta=dict(extra=["--delta-weight", "1.0", "--pair-batch", "16"],
                                   reference_run="runs/pipeline_v1/fast_mode24_v16_twin",
                                   training_reference="runs/pipeline_v1/fast_mode24_v16_twin/tt_joint/config.json",
                                   change="delta: twin + 개입 쌍 변화량 손실 λ=1 (E 쌍·A 쌍 각 16/step, Mode 1, 같은 k·noise, 몸통 14 채널 x0_hat 차이 MSE)"))
    for name, spec in runs.items():
        out = P1 / name
        out.mkdir(exist_ok=True)
        conditions = dict(
            training_args=["--data", *DATA, "--out", f"runs/pipeline_v1/{name}/tt_joint", *COMMON, *spec["extra"]],
            training_reference=spec["training_reference"], reference_run=spec["reference_run"],
            baseline_models={"v8": "runs/pipeline_v1/fast_mode24_v8_film/A/tt_joint", "v9": "runs/pipeline_v1/fast_mode24_v9_e_coverage/tt_joint"},
            pair_checkpoint="runs/pipeline_v1/fast_mode24_v9_e_coverage/tt_joint",
            metric_reference="runs/pipeline_v1/fast_mode24_v8_film/A/tt_joint",
            diffusion_seeds=8, main_window_start=30, main_agreement_improvement_threshold=0.1,
            expected_data_files=len(DATA), expected_episodes=data_audit["episodes"], model_modes=[1, 2, 4], mode_weights=[0.2, 0.4, 0.4],
            execution_kp=30, collection_kp=30, kd=0.5,
            single_variable_vs_v12=dict(change=spec["change"], remind="8-49"),
            baseline_comparison="delta vs twin: 같은 데이터·구조·modes·seed·60k step, 쌍 손실 λ 만 다름. twin vs v12: 데이터 세계 (kp30 vs PD kp200) 가 다르므로 직접 비교 안 함",
            evaluation_gate=GATE,
            confounds=["kp30 세계 (PPO expert + e_swap + action pairs); v10~v15 의 PD kp200 결과와 직접 비교하지 않는다.",
                       "E 쌍은 E 와 안착 자세 g(E) 가 함께 바뀐다 (8-41). 순수 E 인과가 아니라 'E + 그 형상의 시작 자세' 효과.",
                       "A 쌍 (action_pairs_wide16) 은 e000~e002 의 기존 action_pairs 와 소스 (wide16 primitive) 가 다르다. 기존 파일은 넣지 않았다.",
                       "Δ-loss 는 x0 공간 차이라 noise 단계 k 가 클수록 (1−ab)/ab 배 증폭 (k=4 에서 1e4 배) → min-SNR-γ=5 상한 (eps 등가 가중 min((1−ab)/ab, 5)). grad-norm clip 1.0 유지."],
            sha256=data_sha, implementation_sha256=impl_sha)
        (out / "conditions.json").write_text(json.dumps(conditions, indent=2, ensure_ascii=False))
        (out / "data_audit.json").write_text(json.dumps(data_audit, indent=2))
        shutil.copy(P1 / "run_train_generic.py", out / "run_train.py")
        print(name, "ok", len(DATA), "files", data_audit)


if __name__ == "__main__":
    main()
