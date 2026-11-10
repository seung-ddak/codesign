from pathlib import Path
import sys
import pytest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code/tt_dynamics'))
from tt_loss import diffusion_objective, validate_weights, REQUESTED_WEIGHTS


def inputs():
    return torch.ones(2,3,54,requires_grad=True),torch.zeros(2,3,54),torch.ones(2,3,54),torch.ones(2,13,requires_grad=True),torch.zeros(2,13),torch.ones(2,13)


def test_equal_group_errors_have_requested_weights_despite_different_channel_counts():
    args=inputs();loss,terms=diffusion_objective(*args,REQUESTED_WEIGHTS)
    assert torch.allclose(terms,torch.tensor(REQUESTED_WEIGHTS))
    assert loss.item()==pytest.approx(1.)
    loss.backward()
    assert args[0].grad is not None and args[3].grad is not None


def test_conditioned_action_E_and_padding_are_excluded_without_redistribution():
    px,nx,wx,pe,ne,em=inputs()
    wx[...,42:]=0;wx[:,1:]=0;em[:]=0
    loss,terms=diffusion_objective(px,nx,wx,pe,ne,em,REQUESTED_WEIGHTS)
    assert torch.allclose(terms,torch.tensor([0,.2,.2,.2,.1,0]))
    assert loss.item()==pytest.approx(.7)
    loss.backward()
    assert torch.count_nonzero(px.grad[...,42:])==0
    assert torch.count_nonzero(px.grad[:,1:])==0
    assert torch.count_nonzero(pe.grad)==0


def test_legacy_matches_previous_formula_and_weighted_terms_sum_to_total():
    torch.manual_seed(3)
    px,nx,wx,pe,ne,em=inputs()
    px=torch.randn_like(px);nx=torch.randn_like(nx);pe=torch.randn_like(pe);ne=torch.randn_like(ne)
    wx[:1,:2]=0;em[:1]=0
    loss,terms=diffusion_objective(px,nx,wx,pe,ne,em)
    expected=((px-nx).square()*wx).sum()/wx.sum()+((pe-ne).square()*em).mean()
    assert torch.equal(loss,expected)
    assert torch.allclose(terms.sum(),loss)


def test_masked_E_uses_only_unknown_entries_and_actual_shares_follow_errors():
    px,nx,wx,pe,ne,em=inputs()
    em[0]=0
    with torch.no_grad():
        pe[0]=1000
        px[...,38:42]=2
    loss,terms=diffusion_objective(px,nx,wx,pe,ne,em,REQUESTED_WEIGHTS)
    assert terms[-1].item()==pytest.approx(.1)
    assert terms[-2].item()==pytest.approx(.4)
    assert loss.item()==pytest.approx(1.3)


@pytest.mark.parametrize('weights',[(.2,)*6,(float('nan'),0,0,0,0,1),(-.1,.3,.2,.2,.2,.2),(1,0)])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):validate_weights(weights)


def test_actual_contributions_match_targets_for_unequal_errors_and_keep_gradients():
    px,nx,wx,pe,ne,em=inputs()
    with torch.no_grad():
        px[...,42:]=10
        px[...,14:26]=.05
        px[...,26:38]=3
        px[...,38:42]=100
        pe[:]=7
    loss,terms,raw=diffusion_objective(px,nx,wx,pe,ne,em,REQUESTED_WEIGHTS,balance_actual=True,return_raw=True)
    assert torch.allclose(terms/terms.sum(),torch.tensor(REQUESTED_WEIGHTS),atol=1e-7)
    assert raw[4]>raw[1]*1000
    loss.backward()
    for section in (slice(0,14),slice(14,26),slice(26,38),slice(38,42),slice(42,54)):
        assert px.grad[...,section].abs().sum()>0
    assert pe.grad.abs().sum()>0
    assert torch.isfinite(px.grad).all()


def test_actual_balance_preserves_zero_and_conditioned_groups():
    px,nx,wx,pe,ne,em=inputs()
    wx[...,42:]=0;em[:]=0
    with torch.no_grad():px[...,38:42]=0
    loss,terms=diffusion_objective(px,nx,wx,pe,ne,em,REQUESTED_WEIGHTS,balance_actual=True)
    assert torch.allclose(terms,torch.tensor([0,.2,.2,.2,0,0]))
    loss.backward()
    assert torch.isfinite(px.grad).all()
    assert torch.count_nonzero(px.grad[...,38:])==0


def test_raw_monitor_improves_while_actual_shares_stay_fixed():
    args=inputs()
    a=diffusion_objective(*args,REQUESTED_WEIGHTS,balance_actual=True,return_raw=True)
    px,nx,wx,pe,ne,em=args
    b=diffusion_objective(px*.5,nx,wx,pe*.5,ne,em,REQUESTED_WEIGHTS,balance_actual=True,return_raw=True)
    assert torch.allclose(a[1],b[1])
    assert torch.allclose(b[2],a[2]*.25)
