"""(choices, seed) -> 그 형상 **하나만** 담은 모델 필드 zarr — π_θ 학습 입력.

TT 가 expert 를 만든 경로(`scripts/enumerate_robogen_mjcf.py`)와 같은 함수를 쓰되,
seed 를 `range(num_hardware)` 로 훑지 않고 **우리 설계의 seed 만** 넣는다.
결과 store 에 항목이 1 개면 `train_rl_procedural.py` 의
`hardware_idx = randint(0, num_hardware)` 가 항상 그 하나를 고르므로 **DR 없는 형상 고정 학습**이 된다.

  .venv/bin/python tt_codesign/code/enumerate_design_store.py \\
      --choices 00000000 --seed 12345 \\
      --out $CODESIGN_ROOT/assets/design_00000000_12345.zarr
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import argparse
import os
import sys

import numpy as np
import zarr
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

TT = _TT
os.chdir(TT)
sys.path.insert(0, f"{TT}/scripts")

import hydra

from enumerate_robogen_mjcf import generate_robogen_mjcf  # 같은 생성 함수를 그대로 쓴다
from t2.io.schema import init_root


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--choices", required=True, help="8 비트 문자열 (예: 00000000)")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--num-uniforms", type=int, default=13)
    p.add_argument("--extra-seeds", type=int, default=0,
                   help=">0 이면 이웃 설계도 함께 담아 약한 DR 로 학습한다 (기본 0 = 형상 고정)")
    args = p.parse_args()

    choices = [int(c) for c in args.choices]
    assert len(choices) == 8, choices

    with initialize_config_dir(config_dir=f"{TT}/config", version_base="1.3"):
        cfg = compose(
            config_name="generate_robogen_tokens",
            overrides=[f"choices={choices}", f"num_uniforms={args.num_uniforms}",
                       f"output_path={args.out}"],
        )
    robogen_fn = hydra.utils.instantiate(cfg.robogen_from_params)

    seeds = [args.seed] + [args.seed + 1 + i for i in range(args.extra_seeds)]
    print(f"choices {args.choices} | seeds {seeds}")
    results = [
        generate_robogen_mjcf(choices, s, args.num_uniforms, robogen_fn, cfg.data_attrs)
        for s in seeds
    ]

    root, store = init_root(
        args.out,
        root_metadata={
            "choices": choices,
            "robogen_fn": OmegaConf.to_container(cfg.robogen_from_params),
            "num_uniforms": args.num_uniforms,
            "hardware_seeds": seeds,          # 어떤 형상인지 추적용
        },
    )
    for attr in cfg.data_attrs:
        data = np.stack([r[attr] for r in results], axis=0)
        root.create_array(
            attr, shape=data.shape, dtype=data.dtype,
            chunks=(min(100, len(seeds)), *list(data.shape[1:])),
            compressors=zarr.codecs.BloscCodec(cname="zstd", clevel=9),
        )
        root[attr][:] = data
    store.close()
    print(f"saved {args.out}  (설계 {len(seeds)} 개)")


if __name__ == "__main__":
    main()
