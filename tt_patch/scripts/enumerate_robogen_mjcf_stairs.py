"""Dump MuJoCo model attributes for the continuous variations of one discrete
design choice, in the *stair-climbing* scene.

Same as `enumerate_robogen_mjcf.py`, except the staircase is part of the scene.
The training env compiles the identical scene, so the stored per-morphology
arrays line up with the compiled model (see `stair_env.DEFAULT_STAIRS`).
"""

import copy
from typing import Callable

import hydra
import numpy as np
import ray
import zarr
from dm_control import mjcf
from omegaconf import OmegaConf

from t2.io.schema import init_root
from t2.mjx_env.stair_env import DEFAULT_STAIRS, STAIR_RISE_RANGE, build_stair_scene
from t2.utils.ray import wait_with_pbar


def generate_robogen_mjcf(
    choices: list[int],
    seed: int,
    num_uniforms: int,
    robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
    data_attrs: list[str],
):
    rs = np.random.RandomState(seed)
    uniforms = rs.uniform(0, 1, num_uniforms).tolist()
    # terrain curriculum: 설계마다 단높이를 함께 뽑는다 (stair_env.STAIR_RISE_RANGE).
    rise = float(rs.uniform(*STAIR_RISE_RANGE))
    mjcf_model = robogen_from_params(choices=copy.copy(choices), uniforms=uniforms)
    mj_model = mjcf.Physics.from_mjcf_model(
        build_stair_scene(mjcf_model, rise=rise)
    ).model.ptr
    return {attr: getattr(mj_model, attr) for attr in data_attrs}


@hydra.main(
    config_path="../config",
    config_name="generate_robogen_tokens",
    version_base="1.3",
)
def main(cfg):
    hardware_seeds = list(range(cfg.num_hardware))
    ray.init(num_cpus=cfg.num_processes)

    @ray.remote(num_cpus=1)
    def generate_robogen_mjcf_remote(
        choices: list[int],
        seeds: list[int],
        num_uniforms: int,
        robogen_from_params: Callable[[list[int], list[float]], mjcf.RootElement],
        data_attrs: list[str],
    ):
        return [
            generate_robogen_mjcf(
                choices, seed, num_uniforms, robogen_from_params, data_attrs
            )
            for seed in seeds
        ]

    robogen_fn = hydra.utils.instantiate(cfg.robogen_from_params)
    desc = f"robogen-stairs : {cfg.choices}"
    hardware_seed_groups = np.array_split(
        hardware_seeds, min(cfg.num_processes, len(hardware_seeds))
    )
    results = wait_with_pbar(
        {
            desc: [
                generate_robogen_mjcf_remote.remote(
                    choices=cfg.choices,
                    seeds=seeds,
                    num_uniforms=cfg.num_uniforms,
                    robogen_from_params=robogen_fn,
                    data_attrs=cfg.data_attrs,
                )
                for seeds in hardware_seed_groups
            ]
        }
    )[desc]

    root, store = init_root(
        cfg.output_path,
        root_metadata={
            "choices": OmegaConf.to_container(cfg.choices, resolve=True),
            "robogen_fn": OmegaConf.to_container(cfg.robogen_from_params),
            "num_uniforms": cfg.num_uniforms,
            "stairs": dict(DEFAULT_STAIRS),
            "stair_rise_range": list(STAIR_RISE_RANGE),
        },
    )
    data = [entry for group in results for entry in group]
    for attr in cfg.data_attrs:
        arr = np.stack([d[attr] for d in data], axis=0)
        root.create_array(
            attr,
            shape=arr.shape,
            dtype=arr.dtype,
            chunks=(100, *list(arr.shape[1:])),
            shards=(500, *list(arr.shape[1:])),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=9),
        )
        root[attr][:] = arr
    print("wrote", cfg.output_path, {k: root[k].shape for k in cfg.data_attrs})
    store.close()


if __name__ == "__main__":
    main()
