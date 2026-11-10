"""Grid50-only Adam: torch 2.7.1 noncapturable real-valued Adam arithmetic.

Only second-moment multiplication order differs. No clipping, decay, AMSGrad,
foreach/fused execution, or fallback. Not used by fixed-T or random-T runners.
"""
import math
import torch


class Grid50SafeAdam(torch.optim.Optimizer):
    implementation = 'grid50_scaled_second_moment_v1'

    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        if lr < 0 or eps < 0 or any(not 0 <= b < 1 for b in betas):
            raise ValueError('Invalid Adam hyperparameters')
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse or p.is_complex():
                    raise ValueError('Grid50SafeAdam supports dense real tensors only')
                state = self.state[p]
                if not state:
                    # Same CPU scalar step convention as noncapturable torch Adam.
                    state['step'] = torch.tensor(0., dtype=torch.float32, device='cpu')
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                step = state['step'].item()
                m, v = state['exp_avg'], state['exp_avg_sq']
                m.lerp_(g, 1 - beta1)
                scaled_grad = g * math.sqrt(1 - beta2)
                v.mul_(beta2).addcmul_(scaled_grad, scaled_grad)
                correction1 = 1 - beta1 ** step
                correction2 = 1 - beta2 ** step
                denom = (v.sqrt() / math.sqrt(correction2)).add_(group['eps'])
                p.addcdiv_(m, denom, value=-group['lr'] / correction1)
        return loss
