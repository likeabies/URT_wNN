"""One immutable shared-length candidate. No evaluation, retries, or tensor dumps."""
import argparse
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from src.grid50_safe_adam import Grid50SafeAdam
from src.data import normalize_ann_series
from src.paper2019_rnn import Paper2019RNNClassifier
from src.train import ann_weighted_cross_entropy

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / 'results/rnn_shared_length_20261029'
FIELDS = ['epoch', 'batch', 'T', 'loss', 'pre_clip_global_grad_norm',
          'post_clip_global_grad_norm', 'clip_coefficient', 'clipped', 'pre_clip_max_abs_grad']


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def balanced_counts(total, cells):
    q, r = divmod(total, cells)
    return [q + (i < r) for i in range(cells)]


def data_for_t(t, n, protocol):
    """Same paper2019 recurrence/normalization; deterministic balanced cell counts."""
    seed = int(np.random.SeedSequence([protocol['data_seed'], t]).generate_state(1)[0])
    rng = np.random.default_rng(seed)
    rhos, betas = protocol['rhos'], protocol['betas']
    counts0 = iter(balanced_counts(n // 2, len(betas)))
    counts1 = iter(balanced_counts(n // 2, (len(rhos) - 1) * len(betas)))
    x = np.empty((n, t), np.float32)
    y = np.empty(n, np.int64)
    offset = 0
    for rho in rhos:
        for beta in betas:
            count = next(counts0 if rho == 1 else counts1)
            errors = rng.normal(size=(count, t + 1))
            innovations = beta * errors[:, :-1] + errors[:, 1:]
            values = np.empty((count, t), np.float64)
            previous = np.zeros(count)
            for i in range(t):
                previous = rho * previous + innovations[:, i]
                values[:, i] = previous
            x[offset:offset + count] = normalize_ann_series(values)
            y[offset:offset + count] = 0 if rho == 1 else 1
            offset += count
    assert offset == n and np.count_nonzero(y == 0) == n // 2
    order = rng.permutation(n)
    return torch.from_numpy(x[order]), torch.from_numpy(y[order])


def epoch_batches(counts, batch_size, ordering, generator):
    permutations = {t: torch.randperm(n, generator=generator) for t, n in sorted(counts.items())}
    lengths = sorted(counts, reverse=ordering == 'descending')
    batches = [(t, start) for t in lengths for start in range(0, counts[t], batch_size)]
    if ordering == 'random':
        order = torch.randperm(len(batches), generator=generator).tolist()
        batches = [batches[i] for i in order]
    return permutations, batches


def finite(tensors):
    tensors = list(tensors)
    return all(bool(torch.isfinite(t).all().item()) for t in tensors)


def grad_stats(parameters):
    # Float64 CPU reduction avoids float32 norm overflow for finite gradients.
    # This does not alter training dtype or add a stabilization operation.
    values = [p.grad.detach().cpu().to(dtype=torch.float64).reshape(-1)
              for p in parameters if p.grad is not None]
    flat = torch.cat(values)
    return float(torch.linalg.vector_norm(flat)), float(flat.abs().max())


class NumericalFailure(Exception):
    pass


def run(candidate_id):
    protocol = read(BASE / 'protocol.json')
    if protocol['safe_adam'] != 'grid50_scaled_second_moment_v1':
        raise RuntimeError('Safe Adam definition has not been confirmed; no training permitted')
    candidate = next(c for c in protocol['candidates'] if c['id'] == candidate_id)
    folder = BASE / candidate_id
    with (folder / 'process.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(folder / 'status.json')
        if state['status'] != 'starting':
            raise RuntimeError('Candidate already attempted; restarting is prohibited')
        state.update(status='running', pid=os.getpid(), phase='data', epoch=0, batch=0, T=None)
        def persist():
            state['updated_at'] = time.time()
            save(folder / 'status.json', state)
        persist()
        try:
            assert torch.backends.mps.is_available(), 'MPS required'
            counts = {int(t): n for t, n in protocol[candidate['family'] + '_counts'].items()}
            assert sum(counts.values()) == 600000
            data = {}
            for t, n in sorted(counts.items()):
                state['T'] = t
                persist()
                data[t] = data_for_t(t, n, protocol)
            random.seed(7)
            np.random.seed(7)
            torch.manual_seed(7)
            torch.mps.manual_seed(7)
            model = Paper2019RNNClassifier().to('mps')
            parameters = list(model.parameters())
            optimizer = Grid50SafeAdam(parameters, lr=.001, betas=(.9, .999), eps=1e-8)
            generator = torch.Generator().manual_seed(7)
            state.update(phase='training', device=str(parameters[0].device))
            persist()
            with (folder / 'batches.csv').open('x', newline='', buffering=1) as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS)
                writer.writeheader()
                for epoch in range(1, 201):
                    tick = time.monotonic()
                    permutations, batches = epoch_batches(counts, candidate['batch_size'], candidate['ordering'], generator)
                    samples = 0
                    for batch, (t, start) in enumerate(batches, 1):
                        state.update(epoch=epoch, batch=batch, T=t)
                        indices = permutations[t][start:start + candidate['batch_size']]
                        x, y = data[t]
                        xb, yb = x[indices].to('mps'), y[indices].to('mps')
                        optimizer.zero_grad(set_to_none=True)
                        row = dict.fromkeys(FIELDS, '')
                        row.update(epoch=epoch, batch=batch, T=t)
                        failure = None
                        try:
                            loss = ann_weighted_cross_entropy(model(xb), yb, w1=1., w2=candidate['w2'])
                            row['loss'] = float(loss.item())
                            if not math.isfinite(row['loss']):
                                raise NumericalFailure('nonfinite_loss')
                            loss.backward()
                            if not finite(p.grad for p in parameters if p.grad is not None):
                                raise NumericalFailure('nonfinite_gradient_after_backward')
                            norm, max_abs = grad_stats(parameters)
                            coefficient = min(1., protocol['max_norm'] / (norm + 1e-6))
                            row.update(pre_clip_global_grad_norm=norm, pre_clip_max_abs_grad=max_abs,
                                       clip_coefficient=coefficient, clipped=int(coefficient < 1.))
                            with torch.no_grad():
                                for param in parameters:
                                    if param.grad is not None:
                                        param.grad.mul_(coefficient)
                            if not finite(p.grad for p in parameters if p.grad is not None):
                                raise NumericalFailure('nonfinite_gradient_after_clipping')
                            row['post_clip_global_grad_norm'] = grad_stats(parameters)[0]
                            optimizer.step()
                            if not finite(parameters):
                                raise NumericalFailure('nonfinite_parameter_after_step')
                            if not finite(v for s in optimizer.state.values() for v in s.values() if torch.is_tensor(v)):
                                raise NumericalFailure('nonfinite_optimizer_state_after_step')
                        except NumericalFailure as exc:
                            failure = exc
                        writer.writerow(row)
                        if failure is not None:
                            raise failure
                        samples += len(indices)
                        if batch == 1 or batch % 25 == 0:
                            persist()
                    assert samples == 600000
                    state.update(completed_epochs=epoch, epoch_seconds=time.monotonic() - tick)
                    persist()
                    print(f'{candidate_id} epoch={epoch} samples={samples} seconds={state["epoch_seconds"]:.3f}', flush=True)
            state.update(status='completed', phase='done', completed_epochs=200)
            persist()
        except NumericalFailure as exc:
            state.update(status='numerical_failure', reason=str(exc))
            persist()
            (folder / 'failure.log').write_text(f'{candidate_id} epoch={state["epoch"]} batch={state["batch"]} T={state["T"]} {exc}\n')
            print((folder / 'failure.log').read_text(), flush=True)
        except BaseException as exc:
            state.update(status='interrupted', reason=f'{type(exc).__name__}: {exc}')
            persist()
            with (folder / 'failure.log').open('x') as stream:
                stream.write(state['reason'] + '\n')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate', required=True)
    run(parser.parse_args().candidate)
