"""Training-only source balancing; never admit held-out rows into a minibatch."""
import math
import torch


def source_pools(train, swapped):
    return train[~swapped[train]], train[swapped[train]]


def balanced_sample(pools, batch, fraction, generator=None):
    if not math.isfinite(fraction) or not 0 <= fraction <= 1 or batch < 1:
        raise ValueError('invalid source fraction or batch size')
    count = round(batch * fraction)
    chunks = []
    for pool, n in zip(pools, (batch-count, count)):
        if n and not len(pool):
            raise ValueError('requested source has no training episodes')
        if n:
            chunks.append(pool[torch.randint(len(pool), (n,), device=pool.device, generator=generator)])
    return torch.cat(chunks)
