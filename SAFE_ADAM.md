# Safe Adam used by the local variable-length experiments

```python
from src.grid50_safe_adam import Grid50SafeAdam
optimizer = Grid50SafeAdam(model.parameters(), lr=0.001)
```

This module is the unchanged local implementation used by the grid50 and
sequential-T shared LSTM experiments, not a newly reconstructed optimizer.
The class name, import path, and implementation marker
`grid50_scaled_second_moment_v1` are preserved. Existing local runners
`run_lstm_variable_length.py`, `run_lstm_grid50_gradient_observation.py`, and
`run_lstm_sequential_t.py` already import this module; no import migration is
needed. Those untracked experimental runners and their dependencies are outside
this optimizer-only contribution. Fixed-T official runners are unchanged.

## Arithmetic and supported settings

The sole arithmetic change relative to dense, real, noncapturable Adam is:

```python
scaled_grad = g * math.sqrt(1 - beta2)
v.mul_(beta2).addcmul_(scaled_grad, scaled_grad)
```

This computes the same mathematical second moment
`beta2 * v + (1 - beta2) * g**2`, while scaling before squaring avoids the
unweighted intermediate square overflow observed on MPS float32. Floating-point
rounding can differ. This does not prevent nonfinite gradients or overflow when
the weighted result itself is unrepresentable.

Defaults remain `lr=0.001`, `betas=(0.9, 0.999)`, `eps=1e-8`. The first moment,
bias corrections, epsilon placement, and parameter update are unchanged.
There is no weight decay or AMSGrad in this experimental implementation, and
neither option is accepted by its constructor. Do not treat this as a drop-in
replacement for every PyTorch Adam option. Sparse gradients and complex
parameters are rejected; foreach/fused/capturable paths are not implemented.

## Checkpoints and training integration

The per-parameter state remains `step` (CPU float32 scalar), `exp_avg`, and
`exp_avg_sq`; parameter groups retain `lr`, `betas`, `eps`, and `params`.
The module is byte-for-byte identical to the local source, so existing Safe Adam
optimizer state dictionaries retain their format and restore behavior.

Clipping and finite checks remain caller responsibilities. The observation and
sequential-T runners use: backward, gradient finite check, existing global-norm
clipping, post-clip finite check, Safe Adam step, parameter/optimizer-state finite
check. No checks or clipping are added inside the optimizer. Scientific settings,
results, and checkpoints are not changed by this contribution.

## Lightweight validation

```sh
python -m unittest discover -s tests -p 'test_safe_adam.py' -v
```

Tests cover multi-step CPU/MPS PyTorch Adam parity at ordinary finite gradients,
nondefault hyperparameters, serialized state restoration and continued steps,
a large finite gradient (`7.24e19`) against a float64 second-moment reference,
closure/missing-gradient behavior, and unsupported options. MPS tests skip when
MPS is unavailable. No experiment dataset or training runner is executed.
