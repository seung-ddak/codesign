"""Masked diffusion objectives and their actual, additive group contributions."""
import math
import torch

GROUP_NAMES = ('action', 'joint_position', 'joint_velocity', 'body', 'contact', 'E')
# TT channels: body14, joint12, joint velocity12, contact4, action12; separate E13.
GROUP_SLICES = (slice(42,54), slice(14,26), slice(26,38), slice(0,14), slice(38,42))
REQUESTED_WEIGHTS = (.2, .2, .2, .2, .1, .1)


def validate_weights(weights):
    if weights is not None and (len(weights)!=6 or any(not math.isfinite(w) or w<0 for w in weights)
                                or not math.isclose(sum(weights),1.,rel_tol=0.,abs_tol=1e-6)):
        raise ValueError('loss group weights must be 6 finite nonnegative values summing to 1: action, joint_position, joint_velocity, body, contact, E')


def diffusion_objective(px, nx, wx, pe, ne, e_mask, weights=None, balance_actual=False, return_raw=False):
    """Average unknown valid entries within each group; known/absent groups give zero.

    With balance_actual, detached inverse current errors enforce actual scalar shares.
    Otherwise weights specify coefficients, not shares of observed error.
    Active groups keep their weights when other groups are conditioned; no redistribution.
    None reproduces the historical objective (including its E mask normalization).
    """
    ex=(px-nx).square()*wx
    ee=(pe-ne).square()*e_mask
    raw=torch.stack([ex[...,s].sum()/wx[...,s].sum().clamp_min(1) for s in GROUP_SLICES]
                    +[ee.sum()/e_mask.expand_as(ee).sum().clamp_min(1)])
    if balance_actual and weights is None:
        raise ValueError('actual contribution balancing requires group weights')
    if weights is None:
        denominator=wx.sum().clamp_min(1)
        pieces=[ex[...,s].sum()/denominator for s in GROUP_SLICES]+[ee.mean()]
        # Retain the original reduction order for historical reproducibility.
        total=ex.sum()/denominator+ee.mean()
    else:
        values=raw
        if balance_actual:
            # Detach the divisor so equal forward contributions retain learning gradients.
            # Exact-zero/conditioned groups stay zero; the floor bounds inverse-error amplification.
            values=raw/raw.detach().clamp_min(1e-12)
        pieces=[value*weight for value,weight in zip(values,weights)]
        total=torch.stack(pieces).sum()
    result=(total,torch.stack(pieces))
    return (*result,raw) if return_raw else result
