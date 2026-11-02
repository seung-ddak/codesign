"""Δ-loss (remind 8-49): 쌍 변화량 손실과 쌍 구성의 최소 검증."""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code/tt_dynamics"))
from rollout_data import intervention_pairs  # noqa: E402
from tt_loss import pair_delta_mse  # noqa: E402


def test_pair_delta_mse_zero_when_exact_and_var_positive():
    g = torch.Generator().manual_seed(0)
    x0 = torch.randn(8, 5, 54, generator=g)
    w = torch.ones(8, 5, 54)
    w[:, 0] = 0  # t=0 은 알려진 s0 → 제외
    mse, var = pair_delta_mse(x0, x0, w)
    assert float(mse) == 0.0 and float(var) > 0.5
    # 평균 출력 (Δ_hat = 0) 이면 mse ≈ var (R² ≈ 0)
    mse0, var0 = pair_delta_mse(torch.zeros_like(x0), x0, w)
    assert abs(float(mse0) - float(var0)) / float(var0) < 0.1
    # 기울기가 흐른다
    x_hat = x0.clone().requires_grad_(True)
    m, _ = pair_delta_mse(x_hat * 0.5, x0, w)
    m.backward()
    assert x_hat.grad is not None and float(x_hat.grad.abs().sum()) > 0


def test_intervention_pairs_cross_file_E_and_within_file_A(tmp_path):
    valid = np.ones((3, 10), bool)
    valid[2, 1:] = False  # 길이 1 → load() 와 같이 제외
    for name, sd in (("tgt0", "e000"), ("tgt1", "e008")):
        np.savez(tmp_path / f"{name}.npz", valid=valid, source_design=np.array(["e000", "e000", "e008"]),
                 source_episode=np.array([5, 7, 5]))
    np.savez(tmp_path / "pairs.npz", valid=np.ones((6, 10), bool), group_id=np.repeat([0, 1], 3),
             variant=np.tile(["orig", "swap", "offset"], 2))
    paths = [tmp_path / "tgt0.npz", tmp_path / "tgt1.npz", tmp_path / "pairs.npz"]
    pe, pa = intervention_pairs(paths, 10)
    # 전역 index: tgt0 → 0,1 (row2 제외), tgt1 → 2,3, pairs → 4..9
    assert pe.tolist() == [[0, 2], [1, 3]]
    assert pa.tolist() == [[4, 5], [4, 6], [5, 6], [7, 8], [7, 9], [8, 9]]
