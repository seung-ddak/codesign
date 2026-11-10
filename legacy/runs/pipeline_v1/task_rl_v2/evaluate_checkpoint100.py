"""Evaluate the prespecified 100-update checkpoint at the v1 matched budget."""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2] / 'code/tt_dynamics'))
from train_task_rl import TaskEnv, ActorCritic, evaluate, aggregate


def main():
    torch.set_num_threads(1)
    for task in ('zigzag', 'circle'):
        run = ROOT / f'{task}_seed0'
        protocol = json.loads((run / 'protocol.json').read_text())
        source = dict(np.load(protocol['source']))
        envs = [TaskEnv(source, i, task, 'dense') for i in range(16)]
        policy = ActorCritic(68, 68, 12, actor_hidden_dims=[64,64], critic_hidden_dims=[64,64],
                             init_noise_std=.3, max_std=.6)
        policy.load_state_dict(torch.load(run / 'policy_update100.pt', weights_only=True))
        rows, trajectories = evaluate(envs, policy)
        result = dict(rl=rows, updates=100, transitions=76800)
        for name, sl in [('all',slice(None)),('train',slice(0,12)),('heldout',slice(12,16))]:
            result[name] = aggregate(rows[sl])
        (run / 'evaluation_update100.json').write_text(json.dumps(result, indent=2))
        np.savez_compressed(run / 'rl_update100_cpu.npz', **trajectories)
        print(task, result['all'], flush=True)


if __name__ == '__main__':
    main()
