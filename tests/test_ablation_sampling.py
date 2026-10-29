import sys
from pathlib import Path
import pytest
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from ablation_sampling import balanced_sample, source_pools


def test_balanced_sampler_cannot_draw_validation_rows():
    train = torch.tensor([0, 2, 4, 6])
    swapped = torch.tensor([False, False, False, False, True, True, True, True])
    pools = source_pools(train, swapped)
    generator = torch.Generator().manual_seed(7)
    sampled = balanced_sample(pools, 32, .5, generator)
    assert len(sampled) == 32 and set(sampled.tolist()) <= {0, 2, 4, 6}
    assert int(swapped[sampled].sum()) == 16
    assert torch.equal(sampled, balanced_sample(pools, 32, .5, torch.Generator().manual_seed(7)))


def test_balanced_sampler_rejects_empty_requested_source():
    with pytest.raises(ValueError):
        balanced_sample((torch.tensor([1]), torch.tensor([], dtype=torch.long)), 32, .5)
    with pytest.raises(ValueError):
        balanced_sample((torch.tensor([1]), torch.tensor([2])), 32, float('nan'))
