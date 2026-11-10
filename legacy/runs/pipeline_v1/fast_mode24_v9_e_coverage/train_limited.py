"""Train v9 with bounded resources; stop only this job if system RAM becomes low."""
import glob
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = Path(__file__).resolve().parent


def available_ram():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Cannot read available RAM')


def train_args():
    patterns = ['wide16_action/rollouts/e*/primitive_seed0.npz',
                'e_swap/full/e*/primitive_seed0.npz',
                'fast_mode24_v9_e_coverage/data_train/n*/primitive_seed0.npz']
    data = []
    for pattern, expected in zip(patterns, [16, 16, 32]):
        paths = sorted(glob.glob(str(ROOT/'runs/pipeline_v1'/pattern)))
        if len(paths) != expected:
            raise ValueError(f'Expected {expected} inputs: {pattern}, got {len(paths)}')
        data.extend(paths)
    assert not any('/data_test/' in path for path in data)
    return ['--data', *data, '--out', str(RUN/'tt_joint'), '--horizon', '32',
            '--steps', '60000', '--batch', '32', '--eval-episodes', '8',
            '--modes', '2', '4', '--diff-steps', '5', '--norm', 'minmax',
            '--clip', '1', '--sampler', 'ddim', '--film']


def main():
    os.chdir(ROOT)
    if '--worker' in sys.argv:
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
        os.nice(10)
        import runpy
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.cuda.set_per_process_memory_fraction(.60, 0)
        sys.path.insert(0, str(ROOT/'code/tt_dynamics'))
        sys.argv = ['train_tt.py', *train_args()]
        print('Resource limits: CPU 2 cores, torch GPU allocator 60%, batch 32', flush=True)
        runpy.run_path(str(ROOT/'code/tt_dynamics/train_tt.py'), run_name='__main__')
        return
    train_args()
    if (RUN/'tt_joint/checkpoint.pt').exists():
        raise FileExistsError('Existing final checkpoint; refusing to overwrite')
    if available_ram() < 8 * 1024**3:
        raise RuntimeError('Less than 8 GiB available RAM before launch')
    env = dict(os.environ, MUJOCO_GL='egl', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
               XLA_PYTHON_CLIENT_PREALLOCATE='false',
               PYTHONPATH=str(ROOT/'code/tt_dynamics'),
               TMPDIR=str(ROOT/'.runtime/tmp'), MPLCONFIGDIR=str(ROOT/'.runtime/mpl'))
    with (RUN/'train_local.log').open('x') as log:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker'],
                                 env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        (RUN/'local_job.json').write_text(json.dumps(dict(pid=child.pid, cpu_cores=2,
            gpu_allocator_fraction=.60, batch=32, steps=60000, status='running'), indent=2))
        low = 0
        with (RUN/'resources_local.jsonl').open('a') as resources:
            while child.poll() is None:
                ram = available_ram()
                gpu = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=10)
                resources.write(json.dumps(dict(time=time.time(), available_ram_gib=ram/1024**3,
                                                gpu=gpu.stdout.strip()))+'\n');resources.flush()
                low = low + 1 if ram < 4 * 1024**3 else 0
                if low >= 2:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    break
                time.sleep(15)
        status = 'completed' if child.returncode == 0 else 'stopped'
        state = json.loads((RUN/'local_job.json').read_text())
        state.update(status=status, returncode=child.returncode)
        (RUN/'local_job.json').write_text(json.dumps(state, indent=2))
        print(status, child.returncode, flush=True)
        if child.returncode:
            raise SystemExit(child.returncode)


if __name__ == '__main__':
    main()
