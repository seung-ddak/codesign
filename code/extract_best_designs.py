"""생성된 설계를 (choices, seed) 로 복원한다 — π_θ 를 그 형상에 붙이기 위한 1 단계.

평가 파이프라인은 생성된 hardware 토큰을 RoboToken bank 의 **최근접 이웃**으로 스냅해서
`(choices, seed)` 쌍으로 만든다 (`hardware_to_nearest_neighbor_params`). 실제로 시뮬에 올라간
로봇이 바로 그 쌍이므로, 이 둘만 있으면 같은 형상을 언제든 다시 만들 수 있다
(`uniforms = RandomState(seed).uniform(0, 1, 13)`).

실행이 남긴 `hardware_cache/*.pt` 에서 예측 reward 가 높은 순으로 뽑는다.

  .venv/bin/python tt_codesign/code/extract_best_designs.py \\
      --run wandb/offline-run-20260921_074342-rpeb5lrm --top 3
"""

import os as _os
_R = _os.environ.get("CODESIGN_ROOT", _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
_TT = _os.environ.get("CODESIGN_TT", _os.path.join(_os.path.dirname(_R), "transformer-transformer"))

import argparse
import glob
import json
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch

TT = _TT
OUT = _os.path.join(_R, "results")
os.chdir(TT)

from t2.eval.hardware_decoding import hardware_to_nearest_neighbor_params
from t2.eval.utils import extract_masked_hardware


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="wandb offline-run 디렉터리")
    p.add_argument("--top", type=int, default=3)
    p.add_argument("--robotokens", default="umi_on_legs_plus_plus_robotokens")
    p.add_argument("--choice-map", default="choice_to_ckpt_path.json")
    p.add_argument("--out-name", default=None)
    args = p.parse_args()

    cache_dir = os.path.join(args.run, "files", "hardware_cache")
    paths = sorted(glob.glob(os.path.join(cache_dir, "*.pt")))
    assert paths, f"캐시 없음: {cache_dir}"

    items = []
    for pth in paths:
        blob = torch.load(pth, map_location="cpu", weights_only=False)
        items.append((float(blob["predicted_value"]), pth, blob))
    items.sort(key=lambda x: -x[0])
    print(f"캐시 {len(items)} 개 | 예측값 최고 {items[0][0]:.3f} 최저 {items[-1][0]:.3f}")

    choice_to_ckpt = json.load(open(args.choice_map))
    rows = []
    for value, pth, blob in items[: args.top]:
        decoded = {k: (v.numpy() if torch.is_tensor(v) else v)
                   for k, v in blob["decoded"].items()}
        masked, n_link, n_act, n_dyn, n_fix = extract_masked_hardware(decoded)
        choices, seed = hardware_to_nearest_neighbor_params(
            masked, n_link, n_act, n_dyn, n_fix, args.robotokens, local_only=True
        )
        key = "".join(map(str, choices))
        rows.append(dict(
            cache=os.path.basename(pth), predicted_value=value,
            choices=choices, seed=int(seed), choice_key=key,
            expert_ckpt=choice_to_ckpt[key],
            optimize_time=float(blob.get("optimize_time", 0.0)),
        ))
        print(f"  값 {value:.3f} | choices {key} | seed {seed} | expert {choice_to_ckpt[key]}")

    name = args.out_name or f"best_designs_{os.path.basename(args.run)}"
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("saved", f"{OUT}/{name}.json")


if __name__ == "__main__":
    main()
