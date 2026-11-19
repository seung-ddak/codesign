"""s8z rollout 전체를 CPU RAM 에 float32 · EP 2000 으로 미리 할당해 올리고 (A 방식), 배치만 GPU 로 보내는 속도를 잰다.
RSS 최고치·시스템 available 최저치를 파일마다 기록. 결과: ../result.json"""
import glob, json, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'code/tt_dynamics'))
from rollout_data import features, X_DIM, S_DIM, E_DIM
EP = 2000; H = 500; B = 32
ROOT = Path(__file__).resolve().parents[3]; OUT = Path(__file__).resolve().parents[1]
files = sorted(glob.glob(str(ROOT / 'chain/runs/s8z_collect/rollouts/*/chunk_*.npz')))

def mem():
    st = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/self/status') if l.startswith(('VmRSS', 'VmHWM'))}
    av = {l.split(':')[0]: int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith(('MemAvailable', 'SwapFree'))}
    return st['VmRSS'] / 2**20, st['VmHWM'] / 2**20, av['MemAvailable'] / 2**20, av['SwapFree'] / 2**20
log = []; min_av = 1e9
def mark(tag):
    global min_av
    r, hwm, av, sw = mem(); min_av = min(min_av, av)
    log.append(dict(tag=tag, rss=round(r, 2), peak=round(hwm, 2), avail=round(av, 2), swap_free=round(sw, 2)))
    print(f'{tag:30s} rss {r:6.2f} peak {hwm:6.2f} avail {av:6.2f} swapfree {sw:5.2f} GB', flush=True)

t0 = time.time(); mark('start')
rows = []
for f in files:
    with np.load(f) as z: rows.append(int((z['valid'].sum(1) >= 2).sum()))
N = sum(rows); print('episodes', N, flush=True)
X = np.zeros((N, EP, X_DIM), np.float32); M = np.zeros((N, EP), bool); E = np.zeros((N, E_DIM), np.float32)
mark('alloc (lazy)')
n = 0; cut = 0
for fi, f in enumerate(files):
    with np.load(f) as z:
        valid, qpos, qvel, con, tq, e = z['valid'], z['qpos'], z['qvel'], z['contact'], z['target_q'], z['E']
        for b in range(len(valid)):
            L0 = int(valid[b].sum()); L = min(L0, EP)
            if L < 2: continue
            cut += L0 > EP
            X[n, :L, :S_DIM] = features(qpos[b, :L], qvel[b, :L], con[b, :L]); X[n, :L, S_DIM:] = tq[b, :L]
            M[n, :L] = True; E[n] = e[b]; n += 1
    del valid, qpos, qvel, con, tq, e
    if fi % 10 == 0 or fi == len(files) - 1: mark(f'file {fi + 1}/{len(files)}')
load_s = time.time() - t0
Xt, Mt = torch.from_numpy(X), torch.from_numpy(M); mark('torch.from_numpy')
dev = 'cuda'; g = torch.Generator().manual_seed(0); times = []
for i in range(200):
    torch.cuda.synchronize(); s = time.perf_counter()
    idx = torch.randint(0, N, (B,), generator=g)
    st = (torch.rand(B, generator=g) * (Mt[idx].sum(1) - 1).clamp(min=1)).long()
    t = st[:, None] + torch.arange(H)
    x = Xt[idx[:, None], t.clamp(max=EP - 1)].to(dev, non_blocking=False); m = Mt[idx[:, None], t.clamp(max=EP - 1)].to(dev)
    torch.cuda.synchronize(); times.append(time.perf_counter() - s)
mark('after 200 batches')
res = dict(episodes=N, EP=EP, cut_over_EP=int(cut), valid_steps=int(M.sum()), X_GB=X.nbytes / 2**30, load_sec=round(load_s, 1),
           batch_ms_median=round(1000 * float(np.median(times[10:])), 2), batch_ms_p95=round(1000 * float(np.percentile(times[10:], 95)), 2),
           gpu_alloc_MB=round(torch.cuda.max_memory_allocated() / 2**20, 1), peak_rss_GB=max(l['peak'] for l in log), min_avail_GB=round(min_av, 2), log=log)
json.dump(res, open(OUT / 'result.json', 'w'), indent=1); print({k: v for k, v in res.items() if k != 'log'})
