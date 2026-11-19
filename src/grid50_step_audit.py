"""Fail-fast diagnostics for the explicitly gated safe-Adam grid50 run only."""
import copy
import json
import os
import random
from pathlib import Path
import numpy as np
import torch


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return copy.deepcopy(value)


def stats(value):
    a = value.detach().cpu().double()
    finite = torch.isfinite(a)
    valid = a[finite]
    return dict(finite=bool(finite.all()), nan=int(torch.isnan(a).sum()),
                inf=int(torch.isinf(a).sum()),
                min=float(valid.min()) if valid.numel() else None,
                max=float(valid.max()) if valid.numel() else None,
                abs_max=float(valid.abs().max()) if valid.numel() else None)


class StepAudit:
    def __init__(self, path, model, optimizer, device):
        self.path, self.model, self.optimizer, self.device = Path(path), model, optimizer, device
        self.context = {}
        self.good = None
        self.last = None

    def begin(self, context, indices):
        self.context = dict(context, indices=indices.tolist())
        # In-memory copy before forward/update; publish only on failure.
        self.good = dict(model=cpu_copy(self.model.state_dict()),
                         optimizer=cpu_copy(self.optimizer.state_dict()),
                         torch_rng=torch.get_rng_state(),
                         mps_rng=torch.mps.get_rng_state() if self.device == 'mps' else None,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                         context=self.context,
                         purpose='forensic pre-batch state; never automatic resume')
        self.before = None
        self.after = None

    def inspect(self):
        tensors = {}
        for name, p in self.model.named_parameters():
            tensors['parameter/' + name] = stats(p)
            if p.grad is not None:
                tensors['gradient/' + name] = stats(p.grad)
            for key, value in self.optimizer.state.get(p, {}).items():
                if torch.is_tensor(value):
                    tensors['optimizer/' + name + '/' + key] = stats(value)
        return tensors

    def fail(self, reason):
        # Idempotent: preserve the first failure, never overwrite evidence.
        target = self.path / 'numeric_failure.json'
        if not target.exists():
            tensors = self.inspect()
            payload = dict(reason=reason, context=self.context, tensors=tensors,
                           offending_tensors=sorted({k for block in [tensors,self.before or {},self.after or {}]
                                                    for k,v in block.items() if not v['finite']}),
                           before=self.before, after=self.after)
            checkpoint = self.path / 'pre_failure_state.pt'
            tmp = checkpoint.with_suffix('.tmp')
            with tmp.open('xb') as f:
                torch.save(dict(last_good=self.good,
                                failing_state=dict(model=cpu_copy(self.model.state_dict()),
                                    optimizer=cpu_copy(self.optimizer.state_dict()),
                                    gradients={n:cpu_copy(p.grad) for n,p in self.model.named_parameters()})), f)
                f.flush(); os.fsync(f.fileno())
            tmp.replace(checkpoint)
            with target.open('x') as f:
                json.dump(payload, f, indent=2, allow_nan=False)
        raise RuntimeError('Safe Adam fail-fast: ' + reason + '; see ' + str(target))

    def before_step(self, loss):
        self.before = self.inspect()
        self.before['loss'] = stats(loss)
        if not torch.isfinite(loss).all().item() or any(not v['finite'] for v in self.before.values()):
            self.fail('Nonfinite loss/gradient/parameter/optimizer before step')
        gradients = {k:v['abs_max'] for k,v in self.before.items() if k.startswith('gradient/')}
        name = max(gradients, key=gradients.get)
        self.last = dict(epoch=self.context['epoch'], batch=self.context['batch'], T=self.context['T'],
                         w2=self.context['w2'], seed=self.context['seed'],
                         gradient_max=gradients[name], gradient_parameter=name.removeprefix('gradient/'),
                         gradient_finite=True, parameter_finite_before=True,
                         exp_avg_finite_before=True, exp_avg_sq_finite_before=True)

    def after_step(self):
        self.after = self.inspect()
        if any(not v['finite'] for v in self.after.values()):
            self.fail('Nonfinite parameter/optimizer after step')
        second = [v['max'] for k,v in self.after.items() if k.endswith('/exp_avg_sq')]
        self.last.update(exp_avg_sq_max=max(second), parameter_finite=True,
                         exp_avg_finite=True, exp_avg_sq_finite=True, nan=0, inf=0)
        with (self.path / 'step_diagnostics.jsonl').open('a') as f:
            f.write(json.dumps(self.last, allow_nan=False) + '\n')
