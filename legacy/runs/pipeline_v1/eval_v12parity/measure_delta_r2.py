"""held-out 개입 쌍에서 Δ 설명 비율 R²_Δ (remind 8-49 기준 3). Mode 1 DDIM 결정적 예측, 8 seed 평균.

    python measure_delta_r2.py <run_dir>   → <run>/evaluation/delta_r2.json

쌍 = checkpoint 의 val 분할 안에서 (둘 다 val, 둘 다 32 step 유효) E 쌍 / A 쌍, 종류별 최대 1024 (seed 0).
Δ = (window 32, 자기 시작 기준) 몸통 채널의 쌍 차이. R² = 1 − Σ(Δ_hat − Δ)² / Σ(Δ − mean)². 평균 출력이면 ≈ 0.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(ROOT / "code/tt_dynamics"))
from e_sensitivity import GROUPS, local_path  # noqa: E402
from rollout_data import S_DIM, intervention_pairs, load  # noqa: E402
from tt_model import condition_masks, make_net, rebase, schedule  # noqa: E402

H, SEEDS, MAX_PAIRS = 32, 8, 1024
REPORT_GROUPS = dict(body=(0, 14), pos=GROUPS["pos"], height=GROUPS["height"], yaw=GROUPS["yaw"], linvel=GROUPS["linvel"])


@torch.no_grad()
def predict_mode1(net, x0n, mask, e, mu, beta, abar, clip, initial_noises=None):
    """Mode 1: s0·A 알려짐 → S 예측 (정규화 공간 x0). evaluate_native.predict 와 같은 DDIM, 조건만 Mode 1."""
    n, dev = len(e), x0n.device
    kx, ke = condition_masks(torch.ones(n, device=dev, dtype=torch.long), mask)
    known = torch.zeros_like(x0n)
    known[:, 0, :S_DIM] = x0n[:, 0, :S_DIM]
    known[..., S_DIM:] = x0n[..., S_DIM:]
    total = 0
    for seed in range(SEEDS):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = (torch.randn(known.shape, device=dev, generator=g) if initial_noises is None else initial_noises[seed].clone())
        for k in reversed(range(len(beta))):
            ab, prev = abar[k], abar[k - 1] if k else torch.ones((), device=dev)
            eps = net(torch.where(kx > 0, known, x), e, kx, ke, torch.full((n,), float(k), device=dev), ~mask)[0]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-clip, clip)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        total = total + torch.where(kx > 0, known, x0)
    return total / SEEDS


def r2(d_hat, d):
    num = float((d_hat - d).square().sum())
    den = float((d - d.mean(0, keepdim=True)).square().sum())
    return 1 - num / den if den > 0 else None


def main():
    ck = torch.load(RUN / "tt_joint/checkpoint.pt", map_location="cpu", weights_only=False)
    cfg = ck["args"]
    assert cfg["horizon"] == H and 1 in cfg["modes"], "Mode 1 학습 checkpoint 필요"
    paths = [local_path(p) for p in cfg["data"]]
    x, e, mask, _, _ = load(paths, 500)
    pairs_E, pairs_A = intervention_pairs(paths, 500)
    val = set(np.asarray(ck["val"]).tolist())
    L = mask.sum(1)
    net = make_net(cfg).cuda()
    net.load_state_dict(ck["net"])
    net.eval()
    mu, sd = [torch.as_tensor(ck[k], device="cuda", dtype=torch.float32) for k in ("mu", "sd")]
    beta, abar = schedule(cfg["diff_steps"], "cuda")
    en = torch.as_tensor((e - .5) / .5, device="cuda", dtype=torch.float32)
    out = dict(_conditions=dict(window=H, seeds=SEEDS, max_pairs=MAX_PAIRS, start=0, conditioning="mode 1 (E, s0, A known)",
                                pair_filter="both in checkpoint val split and both valid >= 32 steps",
                                r2="1 - sum((d_hat-d)^2)/sum((d-mean)^2) over pairs x steps 1..31 x channels; mean-output ~ 0"))
    rng = np.random.default_rng(0)
    for kind, pairs in (("E", pairs_E), ("A", pairs_A)):
        keep = np.array([(i in val and j in val and L[i] >= H and L[j] >= H) for i, j in pairs], bool) if len(pairs) else np.zeros(0, bool)
        pairs = pairs[keep]
        if len(pairs) > MAX_PAIRS:
            pairs = pairs[rng.choice(len(pairs), MAX_PAIRS, replace=False)]
        if len(pairs) == 0:
            out[kind] = dict(pairs=0)
            continue
        idx = np.concatenate([pairs[:, 0], pairs[:, 1]])
        truth = rebase(torch.as_tensor(x[idx, :H], device="cuda"))
        m = torch.ones(len(idx), H, dtype=torch.bool, device="cuda")
        norm = (truth - mu) / sd
        # Explicit noise bank pairs i/j even when pair count is not a multiple of chunk size.
        initial_noises = torch.stack([
            torch.randn((len(pairs), H, norm.shape[-1]), device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(seed)).repeat(2, 1, 1)
            for seed in range(SEEDS)])
        preds = torch.cat([predict_mode1(net, norm[c:c + 128], m[c:c + 128], en[idx[c:c + 128]], mu, beta, abar, cfg["clip"],
                                        initial_noises[:, c:c + 128])
                           for c in range(0, len(idx), 128)]) * sd + mu
        P = len(pairs)
        d_hat = (preds[:P] - preds[P:])[:, 1:]
        d = (truth[:P] - truth[P:])[:, 1:]
        res = dict(pairs=int(P), r2={g: r2(d_hat[..., lo:hi], d[..., lo:hi]) for g, (lo, hi) in REPORT_GROUPS.items()})
        # 보조: 마지막 step 전진 변화 부호 일치 (E 교체 시 예측 전진 변화 방향)
        s_hat, s = torch.sign(d_hat[:, -1, 0]), torch.sign(d[:, -1, 0])
        res["end_forward_change_sign_agree"] = float((s_hat == s).float().mean())
        res["delta_rms_true"] = {g: float(d[..., lo:hi].square().mean().sqrt()) for g, (lo, hi) in REPORT_GROUPS.items()}
        res["delta_rms_pred"] = {g: float(d_hat[..., lo:hi].square().mean().sqrt()) for g, (lo, hi) in REPORT_GROUPS.items()}
        out[kind] = res
        print(kind, json.dumps(res), flush=True)
    (RUN / "evaluation").mkdir(exist_ok=True)
    (RUN / "evaluation/delta_r2.json").write_text(json.dumps(out, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
