"""Check the added PD diagnostic against the frozen historical sampling algorithm on CPU."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runs/pipeline_v1/fast_mode24_v10_pd_reference"))
from evaluate_native import predict
from tt_model import condition_masks, schedule
from rollout_data import S_DIM


@pytest.mark.parametrize("pin", [False, True])
def test_native_sampling_matches_historical_diagnostic(pin):
    source = ROOT / "runs/pipeline_v1/fast_mode24_v9_e_coverage/measure_given_a.py"
    function = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "gen")
    namespace = dict(torch=torch, condition_masks=condition_masks, S_DIM=S_DIM, dev="cpu", args=SimpleNamespace(pin_mode=2))
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    torch.manual_seed(7)
    s0, e = torch.randn(3, 42), torch.randn(3, 13)
    a = torch.randn(3, 32, 12) if pin else None
    mu = torch.zeros(54)
    beta, abar = schedule(5, "cpu")

    def net(x, e, kx, ke, k, pad):
        return x * .125 + e[:, 0, None, None] * .01, torch.zeros_like(e)

    expected = namespace["gen"](net, s0, e, a, 32, mu, beta, abar, 1., 8)
    actual = predict(net, s0, e, a, mu, beta, abar, 1.)
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
