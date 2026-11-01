"""Same-A E test. e_sensitivity samples Mode 2, which generates its own A; here the pair's A is pinned at every DDIM step.
Control: does pinning A change S at all? Test: right E vs wrong E error against MJX, whole and moving part."""
import json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "code/tt_dynamics")
import e_sensitivity as es
from rollout_data import S_DIM
from tt_model import EGuidedNet, condition_masks, make_net, rebase, schedule

V8 = Path("runs/pipeline_v1/fast_mode24_v8_film")
import argparse
ap = argparse.ArgumentParser()
ap.add_argument("out", type=Path)
ap.add_argument("models", nargs="?", default="{}")
ap.add_argument("held", nargs="?", default="")
ap.add_argument("--only-requested", action="store_true")
ap.add_argument("--pair-checkpoint", type=Path)
ap.add_argument("--metric-reference", type=Path)
ap.add_argument("--pin-mode", type=int, default=2, choices=(1, 2), help="A 고정 시 조건 mode: 2 = 기존 (A 를 노이즈 수준으로 강제), 1 = Mode 1 (A 를 알려진 조건으로 입력, Mode 1 학습 모델용. remind 8-38)")
args = ap.parse_args()
RUNS = {} if args.only_requested else dict(v7=("runs/pipeline_v1/fast_mode24_v7_eswap/tt_joint", 1), A=(V8 / "A/tt_joint", 1), B_w2=(V8 / "B/tt_joint", 2))
RUNS.update(json.loads(args.models))
assert RUNS
OUT = args.out
HELD = sorted(Path(args.held).glob("n*/primitive_seed0.npz")) if args.held else []
pair_ck = torch.load(args.pair_checkpoint / "checkpoint.pt", map_location="cpu", weights_only=False) if args.pair_checkpoint else None
metric_ck = torch.load(args.metric_reference / "checkpoint.pt", map_location="cpu", weights_only=False) if args.metric_reference else None


dev = "cuda"
GROUPS = {k: es.GROUPS[k] for k in ("pos", "height", "yaw", "tilt", "linvel", "joint")}

@torch.no_grad()
def gen(net, s0, e, a, H, mu, beta, abar, clip, seeds):
    """Mean Mode-2 sample. a=None lets TT choose A; otherwise A is forced (noised to level k) on every step."""
    n = len(e); mask = torch.ones(n, H, dtype=torch.bool, device=dev)
    mode = args.pin_mode if a is not None else 2
    kx, ke = condition_masks(torch.full((n,), mode, device=dev), mask)
    known = torch.zeros(n, H, mu.numel(), device=dev); known[:, 0, :S_DIM] = s0
    if mode == 1:
        known[..., S_DIM:] = a
    total = 0
    for seed in range(seeds):
        g = torch.Generator(device=dev).manual_seed(seed)
        x = torch.randn(known.shape, device=dev, generator=g); z = torch.randn(known.shape, device=dev, generator=g)
        for k in reversed(range(len(beta))):
            ab, prev = abar[k], abar[k - 1] if k else torch.ones((), device=dev)
            if a is not None:
                x = x.clone(); x[..., S_DIM:] = ab.sqrt() * a + (1 - ab).sqrt() * z[..., S_DIM:]
            eps = net(torch.where(kx > 0, known, x), e, kx, ke, torch.full((n,), float(k), device=dev), ~mask)[0]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-clip, clip)
            x = prev.sqrt() * x0 + (1 - prev).sqrt() * (x - ab.sqrt() * x0) / (1 - ab).sqrt()
        total = total + torch.where(kx > 0, known, x0)
    return total / seeds

def moving(x):
    return x - x.mean(1, keepdim=True)

def err(p, t):
    return float((p - t).square().mean().sqrt())

out = {}
for name, (path, w) in RUNS.items():
    ck = torch.load(Path(path) / "checkpoint.pt", map_location=dev, weights_only=False)
    cfg = ck["args"]; net = make_net(cfg).to(dev); net.load_state_dict(ck["net"]); net.eval()
    if w != 1: net = EGuidedNet(net, w)
    H, clip = cfg["horizon"], cfg["clip"]
    mu, sd = [torch.as_tensor(v, device=dev, dtype=torch.float32) for v in (ck["mu"], ck["sd"])]
    beta, abar = schedule(cfg["diff_steps"], dev)
    eps, pairs = es.load_pairs(pair_ck if pair_ck is not None else ck, 0)
    if HELD:
        val_ep = {int(eps[d]["source_episode"][b]) for p in pairs for d, b in p}
        eps = [{k: z[k] for k in ("qpos", "qvel", "contact", "target_q", "valid", "E", "source_design", "source_episode")} for z in map(np.load, HELD)]
        rng = np.random.default_rng(0); pairs = []
        for row in range(len(eps[0]["valid"])):
            if int(eps[0]["source_episode"][row]) in val_ep:
                assert all(d["source_design"][row] == eps[0]["source_design"][row] and d["source_episode"][row] == eps[0]["source_episode"][row] for d in eps)
                alive = [i for i in range(len(eps)) if eps[i]["valid"][row].sum() >= 62]
                if len(alive) > 1:
                    pairs += [tuple((int(i), row) for i in rng.choice(alive, 2, replace=False)) for _ in range(12)]
    res = {}
    for start in (0, 30):
        rows = [(es.window(eps[di], bi, start, H), es.window(eps[dj], bj, start, H), eps[di]["E"], eps[dj]["E"]) for (di, bi), (dj, bj) in pairs]
        rows = [r for r in rows if r[0] is not None and r[1] is not None]
        assert rows, "No matched physical evaluation pairs"
        assert all(np.allclose(r[0][..., S_DIM:], r[1][..., S_DIM:]) for r in rows), "Paired actions differ"
        t_ = lambda c: torch.tensor(np.stack([r[c] for r in rows]), device=dev, dtype=torch.float32)
        xi, xj = rebase(t_(0)), rebase(t_(1)); ei, ej = (t_(2) - .5) / .5, (t_(3) - .5) / .5
        ni, nj = (xi - mu) / sd, (xj - mu) / sd
        run = lambda s0, e, a: torch.cat([gen(net, s0[i:i+128], e[i:i+128], None if a is None else a[i:i+128], H, mu, beta, abar, clip, 8) * sd + mu for i in range(0, len(e), 128)])[:, 1:]
        ai, aj = ni[..., S_DIM:], nj[..., S_DIM:]
        s_i, s_j = ni[:, 0, :S_DIM], nj[:, 0, :S_DIM]
        free_jj = run(s_j, ej, None); pin_jj = run(s_j, ej, aj); pin_ji = run(s_j, ei, aj); other_a = run(s_j, ej, ai.roll(len(ai) // 2, 0))
        ti, tj = xi[:, 1:], xj[:, 1:]
        r = dict(pairs=len(rows))
        for g, (lo, hi) in GROUPS.items():
            s = lambda v: v[..., lo:hi]
            sc = (torch.as_tensor(metric_ck["sd"], device=dev, dtype=sd.dtype) if metric_ck is not None else sd)[lo:hi]
            nrm = lambda v: s(v) / sc
            pc = lambda p: float(np.corrcoef(moving(nrm(p)).flatten().cpu(), moving(nrm(tj)).flatten().cpu())[0, 1])
            r[g] = dict(free_rmse=err(nrm(free_jj), nrm(tj)), pin_rmse=err(nrm(pin_jj), nrm(tj)), wrongE_rmse=err(nrm(pin_ji), nrm(tj)), wrongA_rmse=err(nrm(other_a), nrm(tj)),
                        pin_moving=err(moving(nrm(pin_jj)), moving(nrm(tj))), wrongE_moving=err(moving(nrm(pin_ji)), moving(nrm(tj))), wrongA_moving=err(moving(nrm(other_a)), moving(nrm(tj))),
                        free_track=pc(free_jj), pin_track=pc(pin_jj),
                        e_agree_moving=float(np.corrcoef(moving(nrm(pin_jj) - nrm(pin_ji)).flatten().cpu(), moving(nrm(tj) - nrm(ti)).flatten().cpu())[0, 1]))
        res[f"start{start}"] = r
    out[name] = res
    print(name, json.dumps({g: {k: round(v, 3) for k, v in res["start30"][g].items()} for g in GROUPS}), flush=True)
out["_comparison"] = dict(pair_checkpoint=str(args.pair_checkpoint), metric_reference=str(args.metric_reference), held_files=[str(p) for p in HELD], equal_actions_checked=True)
OUT.write_text(json.dumps(out, indent=1))
print("MEASURE_OK")
