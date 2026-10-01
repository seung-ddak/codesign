"""Render a DDPO checkpoint's noise-to-robot morphology with TT's native visualizer."""

import argparse
import os
import pickle
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parents[2]
TT = ROOT.parent / "transformer-transformer"
sys.path[:0] = [str(TT), str(ROOT / "stage2/code")]

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from scripts.visualize_robotoken_diffusion import generate_hardware_visualization  # noqa: E402
from t2.eval.utils import traj2batch  # noqa: E402
from t2.model.t2 import setup_decoder  # noqa: E402
from t2.train.augment import AddPositionId  # noqa: E402


PRIOR = TT / "checkpoints/umi_on_legs_plus_plus/xrh4wk4l/040.pt"
OVERRIDE_KEYS = [
    "link/diaginertia", "dyna_joint/link/id", "fixed_joint/link/id",
    "link/track_link/id", "link/free_link/id", "actuator/dyna_joint/id",
    "dyna_joint/type", "link/contact_dim", "link/mass", "actuator/ctrl_range",
    "actuator/force_range", "actuator/kp", "actuator/kv", "actuator/gear_ratio",
    "actuator/type", "dyna_joint/range",
]


def batch_for(cfg, device):
    with initialize_config_dir(config_dir=str(TT / "config"), version_base="1.3"):
        inference = compose(config_name="evaluate_hardware_opt_rl")
    inference.timestep_sampler.num_rollout_steps = int(cfg.seq_len.rollout_steps)
    sampler = hydra.utils.instantiate(inference.timestep_sampler)
    processor = hydra.utils.instantiate(cfg.datasets.clean.batch_process_fn)
    add_ids = next(x for x in processor.augmentations if isinstance(x, AddPositionId))
    poses = np.tile(np.eye(4, dtype=np.float32),
                    (int(cfg.seq_len.rollout_steps), int(cfg.seq_len.track_link), 1, 1))
    batch = add_ids(traj2batch(poses, cfg.seq_len, sampler, center_traj=False))
    batch["hardware_goal/mask"] = torch.ones_like(batch["target_pose/mask"])
    batch["target_pose/mask"] = torch.ones_like(batch["target_pose/mask"])
    return {key: value[:1].to(device) for key, value in batch.items()}


def decode(checkpoint, seed):
    device = torch.device("cuda")
    with PRIOR.with_name("cfg.pkl").open("rb") as file:
        cfg = pickle.load(file)
    OmegaConf.resolve(cfg.seq_len)
    decoder = setup_decoder(
        cfg, str(PRIOR), device, "hardware_gen", num_inference_steps=100,
        eta=1.0, disable_grad=True, use_flash_attn=False,
        use_mixed_precision=False, use_torch_compile=False, return_intermediates=True,
    )
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        decoder.model.load_state_dict(state["model"])
    batch = batch_for(cfg, device)
    with torch.inference_mode():
        decoded = decoder(batch, seed=seed)
    result = [{key: value[0].cpu() for key, value in {**batch, **step}.items()}
              for step in decoded]
    decoder.unload_model()
    return result


def use_prior_topology(steps, prior_final):
    keys = {
        key for key in prior_final
        if key.endswith(("/mask", "/id", "/type")) or key in OVERRIDE_KEYS
    }
    fixed = {key: prior_final[key] for key in keys}
    return [{**step, **fixed} for step in steps]


def self_check():
    assert PRIOR.is_file()
    assert "dyna_joint/link/id" in OVERRIDE_KEYS
    print("record_ddpo_shape_denoising self-check: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "stage2/runs/T7/try2")
    parser.add_argument("--checkpoint", default="psi_0010.pt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return

    run = args.run.resolve()
    checkpoint = run / args.checkpoint
    output = (args.output or
              run / "denoising/try2_psi0010_shapes_prior_topology.mp4").resolve()
    if not checkpoint.is_file():
        parser.error(f"missing checkpoint: {checkpoint}")
    output.parent.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create({
        "gravcomp": True,
        "render": True,
        "render_height": 720,
        "render_width": 960,
        "video_fps": 12,
        "text_fontsize": 32,
        "text_positions": [[10, 10], [10, 48]],
        "use_pbar": True,
        "last_diffusion_step_frame_repeat": 8,
        "final_frame_hold_count": 16,
        "max_distance": 2.0,
        "colormap_name": "jet",
        "use_original_rgba": True,
    })
    before = set(output.parent.glob("hardware_diffusion_*.mp4"))
    prior = decode(None, args.seed)
    trained = decode(checkpoint, args.seed)
    anchored = use_prior_topology(trained, prior[-1])
    generate_hardware_visualization(
        cfg, anchored, args.seed, output.parent, OVERRIDE_KEYS)
    created = set(output.parent.glob("hardware_diffusion_*.mp4")) - before
    if len(created) != 1:
        raise RuntimeError(f"expected one rendered video, found {len(created)}")
    created.pop().replace(output)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
