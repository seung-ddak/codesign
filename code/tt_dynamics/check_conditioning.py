"""Regression: free generation and training must both hide E and s0."""
import torch
from tt_model import CONDITIONING_VERSION, check_free_checkpoint, condition_masks
from rollout_data import E_DIM, S_DIM, X_DIM

valid = torch.ones(4, 32, dtype=torch.bool)
valid[:, -3:] = False
modes = torch.arange(4)
kx, ke = condition_masks(modes, valid)
assert kx.shape == (4, 32, X_DIM) and ke.shape == (4, E_DIM)
assert not kx[:, -3:].any()
for mode in range(4):
    expected = torch.zeros(32, X_DIM)
    if mode != 3: expected[0, :S_DIM] = 1
    if mode == 1: expected[:, S_DIM:] = 1
    torch.testing.assert_close(kx[mode], expected * valid[mode, :, None])
    assert bool(ke[mode].all()) == (mode in (1, 2))
    # Training receives a mixed-mode batch; generation uses one mode for all candidates.
    gx, ge = condition_masks(torch.tensor([mode]), valid[mode:mode+1])
    torch.testing.assert_close(gx[0], kx[mode])
    torch.testing.assert_close(ge[0], ke[mode])
# E-head loss must have a gradient for free mode; the old mask removed it completely.
pred = torch.ones(4, E_DIM, requires_grad=True)
((pred.square()) * (1 - ke)).mean().backward()
assert pred.grad[3].abs().sum() > 0
assert pred.grad[1:3].abs().sum() == 0
for config in ({'modes': [0,1,2,3]}, {'modes': [0,1,2], 'conditioning_version': 2}):
    try: check_free_checkpoint(config)
    except ValueError: pass
    else: raise AssertionError('Invalid free checkpoint accepted')
check_free_checkpoint({'modes': [0,1,2,3], 'conditioning_version': CONDITIONING_VERSION})
# mode 4: s0 의 형상 무관 칸만 앎 (높이 2·관절각 14:26 모름), E·A 모름, 학습·생성 같은 함수
from tt_model import REST_KNOWN, check_rest_checkpoint, rest_s0
k4, e4 = condition_masks(torch.tensor([4]), valid[:1])
assert not k4[0, 0, 2] and not k4[0, 0, 14:26].any() and k4[0, 0, [0, 1, 3, 4, 5, 6, 7]].all() and k4[0, 0, 26:S_DIM].all()
assert not k4[0, 1:].any() and not k4[0, :, S_DIM:].any() and not e4.any()
mixed, _ = condition_masks(torch.tensor([0, 4]), valid[:2])
torch.testing.assert_close(mixed[1], k4[0]); assert mixed[0, 0, :S_DIM].all()
r = rest_s0(); assert r[3] == 1 and r[7] == -1 and r[38:42].eq(1).all() and r[8:14].eq(0).all() and r[26:38].eq(0).all()
try: check_rest_checkpoint({'modes': [0, 1, 2]})
except ValueError: pass
else: raise AssertionError('rest without mode 4 accepted')
print('PASS: all modes, padding, training/generation agreement, free E gradient, legacy checkpoint rejection, mode 4 rest mask')
