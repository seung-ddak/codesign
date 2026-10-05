"""π_φ 설계 전 사전 조사: guidance_fn 이 몇 번 불리고 어떤 모양의 텐서를 요구하는지 찍는다.

D11 은 "마지막 10 denoising step 에만 residual" 인데, hardware gen 모델의 총 step 수가
10 보다 작으면 그 규정 자체가 무의미해진다. 추측하지 말고 실측한다.

  .venv/bin/python tt_codesign/code/probe_guidance.py
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import os
import pickle

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import hydra
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TT = _TT
os.chdir(TT)

from t2.model.t2 import setup_decoder
from t2.train.augment import AddPositionId
from t2.train.utils import set_torch_configs
from t2.utils.misc import seed_everything


class Probe:
    """guidance_fn 자리에 끼워 호출 순서·모양만 기록한다 (guidance 는 0)."""

    def __init__(self):
        self.calls = 0
        self.shapes = {}
        self.reward_fns = {}

    TERMS = ("dyna_joint", "fixed_joint", "link", "actuator")

    def __call__(self, x_t_plus_one, x_t):
        self.calls += 1
        if self.calls == 1:
            print("\n--- guidance 호출 #1: guidance term 텐서 (= samples[name]) ---")
            for k in self.TERMS:
                v = x_t.get(k)
                print(f"  x_t[{k:12s}] =", tuple(v.shape) if torch.is_tensor(v) else v,
                      getattr(v, "dtype", ""), "requires_grad=",
                      getattr(v, "requires_grad", None))
        return {}


def main():
    set_torch_configs()
    seed_everything(0)
    device = torch.device("cuda")

    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="evaluate_hardware_opt_rl",
            overrides=[
                "hardware_optimizer@eval_fn.hardware_optimizer_fn=guided_diffusion",
                "reward_fns@eval_fn.hardware_optimizer_fn.reward_fns=tracking_only",
                "eval_fn.hardware_optimizer_fn.num_seeds=4",
                "eval_fn.hardware_optimizer_fn.guidance.scale=100.0",
                "eta=1.0",
                "use_flash_attn=false",
                "use_mixed_precision=false",
                "use_torch_compile=false",
                "ckpt_path=checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt",
                "pickle_path=data/july25th2025-huy-20skills-test.pkl",
            ],
        )

    policy_cfg = pickle.load(open(os.path.dirname(cfg.ckpt_path) + "/cfg.pkl", "rb"))
    OmegaConf.resolve(policy_cfg.seq_len)
    cfg.timestep_sampler.num_rollout_steps = int(policy_cfg.seq_len["rollout_steps"])
    OmegaConf.resolve(policy_cfg)

    gen = setup_decoder(
        policy_cfg, ckpt_path=cfg.ckpt_path, device=device, task_name=cfg.task_name,
        num_inference_steps=cfg.num_inference_steps,
        num_repeats_per_step=cfg.num_repeats_per_step,
        clip_samples_in_guidance=cfg.clip_samples_in_guidance,
        use_flash_attn=False, eta=cfg.eta,
        use_mixed_precision=False, use_torch_compile=False,
    )
    print("num_inference_steps(cfg) =", cfg.num_inference_steps)
    for attr in ("num_inference_steps", "eta", "clip_samples_in_guidance"):
        print(f"  decoder.{attr} =", getattr(gen, attr, "N/A"))
    sched = getattr(getattr(gen, "task", gen), "noise_scheduler", None)
    print("  scheduler =", type(sched).__name__ if sched else None,
          "| train timesteps =", getattr(getattr(sched, "config", None),
                                         "num_train_timesteps", "N/A"))

    bpf = hydra.utils.instantiate(policy_cfg.datasets.clean.batch_process_fn)
    add_pos_id = next(a for a in bpf.augmentations if type(a) is AddPositionId)
    trajs = pickle.load(open(cfg.pickle_path, "rb"))
    ts = hydra.utils.instantiate(cfg.timestep_sampler)
    max_traj_len = policy_cfg.model.pos_embs.time.embedding.num_embeddings - 1

    opt = hydra.utils.instantiate(cfg.eval_fn.hardware_optimizer_fn)(
        hardware_generator=gen, seq_len_cfg=dict(policy_cfg.seq_len),
        timestep_sampler=ts, center_traj=cfg.eval_fn.runner.env.center_traj,
        add_pos_id=add_pos_id, device=device, output_cache_dir=None,
        max_traj_len=max_traj_len,
    )
    print("\nreward_fns =", list(opt.reward_fns))
    print("guidance_terms =", opt.guidance.guidance_terms)
    print("batch_size =", opt.batch_size, "num_seeds =", opt.num_seeds)

    # 실제 DSG guidance 의 크기를 재둔다 — π_φ 의 action_scale 을 이 규모에 맞춘다
    real = opt.guidance
    import numpy as np
    norms = {t: [] for t in Probe.TERMS}

    class Wrap:
        def __init__(self, inner):
            self.inner = inner
            self.calls = 0
            self.reward_fns = inner.reward_fns
            self.guidance_terms = inner.guidance_terms

        def __call__(self, a, b):
            self.calls += 1
            g = self.inner(a, b)
            for k, v in g.items():
                norms[k].append(float(v.abs().mean()))
            return g

    opt.guidance = Wrap(real)
    opt._ran_warmup = True
    _d, _v = opt._run_optimization_core([trajs[0]], seed=0)
    print("\n--- DSG guidance |g| 평균 (step 구간별) ---")
    for k, arr in norms.items():
        a = np.array(arr)
        if len(a) == 0:
            continue
        print(f"  {k:12s} 전체 {a.mean():.4e} | 처음10 {a[:10].mean():.4e} "
              f"| 마지막10 {a[-10:].mean():.4e} | max {a.max():.4e}")
    print("DSG predicted value =", _v)

    probe = Probe()
    opt.guidance = probe
    opt._ran_warmup = True          # 워밍업 한 번 더 도는 것 방지
    decoded, value = opt._run_optimization_core([trajs[0]], seed=0)
    print(f"\n총 guidance 호출 {probe.calls} 회 / 생성 {opt.num_seeds // opt.batch_size} 회")
    print(f"→ 생성 1 회당 denoising step {probe.calls / max(1, opt.num_seeds // opt.batch_size):.0f}")
    print("predicted value =", value)


if __name__ == "__main__":
    main()
