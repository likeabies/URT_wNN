"""Global L2 clipping with CPU float64 accumulation, no gradient tensor dumps."""
import math
import torch


def global_norm64(parameters):
    squared = 0.
    for p in parameters:
        if p.grad is not None:
            g = p.grad.detach().cpu().double()
            if not torch.isfinite(g).all():
                raise RuntimeError('Nonfinite gradient before clipping')
            squared += g.square().sum().item()
    norm = math.sqrt(squared)
    if not math.isfinite(norm):
        raise RuntimeError('Nonfinite float64 global norm')
    return norm


@torch.no_grad()
def clip_global_norm64(parameters, max_norm):
    params = list(parameters)
    if not math.isfinite(max_norm) or max_norm <= 0:
        raise ValueError('max_norm must be positive and finite')
    pre = global_norm64(params)
    coefficient = min(1., max_norm / pre) if pre else 1.
    if coefficient < 1.:
        for p in params:
            if p.grad is not None:
                p.grad.mul_(coefficient)
    post = global_norm64(params)
    if post > max_norm * (1 + 1e-6):
        raise RuntimeError('Post-clip norm exceeds fixed threshold')
    return dict(pre_clip_global_norm=pre, clip_coefficient=coefficient,
                clipped=coefficient < 1., post_clip_global_norm=post)
