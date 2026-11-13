"""Explicit opt-in preflight only. Never launches production or uses its data."""
import argparse
import copy
import hashlib
import json
import platform
import time
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_lstm_variable_length import SOURCES
from src.grid50_safe_adam import Grid50SafeAdam
from src.variable_length_lstm import VariableLengthLSTM
from src.lstm_binary import make_lstm_dataset as make_ann_dataset
from src.lstm_binary import weighted_cross_entropy as ann_weighted_cross_entropy

CONFIG = Path('configs/lstm_multilength_grid50_equalN_safeadam_20261020.json')
EXTRA_SOURCES = ['src/grid50_safe_adam.py', 'src/grid50_step_audit.py',
                 'scripts/run_lstm_grid50_safeadam_gate.py', 'tests/test_grid50_safeadam_gate.py', __file__, str(CONFIG)]


def source_hashes():
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in SOURCES + EXTRA_SOURCES}


def compare(ref, actual):
    ref, actual = ref.detach().cpu().double(), actual.detach().cpu().double()
    finite = bool(torch.isfinite(ref).all() and torch.isfinite(actual).all())
    error = (ref - actual).abs()
    result = dict(finite=finite, max_abs_error=float(error.max()),
                  max_relative_error=float((error / ref.abs().clamp_min(1e-30)).max()))
    torch.testing.assert_close(actual, ref, rtol=2e-6, atol=2e-7)
    assert finite
    return result


def optimizer_checks():
    rows = []
    for case, gradients in [
        ('normal', [[0., 1e-6, -.01, 1., -10.], [.1, -.2, .3, -.4, .5]] * 5),
        ('large', [[7.243555335899919e19, -7.243555335899919e19]] * 3),
    ]:
        params = [torch.nn.Parameter(torch.ones(len(gradients[0]), device=d))
                  for d in ['cpu', 'cpu', 'mps']]
        opts = [torch.optim.Adam([params[0]], lr=.001, foreach=False, fused=False),
                Grid50SafeAdam([params[1]]), Grid50SafeAdam([params[2]])]
        for step, values in enumerate(gradients, 1):
            for p, opt in zip(params, opts):
                p.grad = torch.tensor(values, dtype=torch.float32, device=p.device)
                opt.step()
            for label, p, opt in zip(['cpu_safe', 'mps_safe'], params[1:], opts[1:]):
                row = dict(case=case, step=step, implementation=label,
                           parameter=compare(params[0], p))
                for key in ['exp_avg', 'exp_avg_sq']:
                    row[key] = compare(opts[0].state[params[0]][key], opt.state[p][key])
                rows.append(row)
    # Preserve exact known scalar reproduction including its nonzero initial v.
    g = torch.tensor([7.243555335899919e19], device='mps')
    v = torch.tensor([1e12], device='mps')
    original = v.clone().mul_(.999).addcmul_(g, g, value=.001)
    scaled = g * (.001 ** .5)
    safe = v.clone().mul_(.999).addcmul_(scaled, scaled)
    ref = v.cpu().mul_(.999).addcmul_(g.cpu(), g.cpu(), value=.001)
    assert torch.isinf(original).all()
    return dict(rows=rows, original_mps_inf=True, scalar_workaround=compare(ref, safe))


def hn_checks():
    rows = []
    for t in [50, 250, 500]:
        raw = make_ann_dataset(t, [1., .95], [.6], 384, 384, 20191007+t)
        torch.manual_seed(7)
        cpu = VariableLengthLSTM()
        mps = VariableLengthLSTM().to('mps')
        mps.load_state_dict(cpu.state_dict())
        a, b = Grid50SafeAdam(cpu.parameters()), Grid50SafeAdam(mps.parameters())
        ids = np.random.default_rng(7).permutation(768)
        for batch in range(3):
            ix = ids[batch*256:(batch+1)*256]
            x = torch.from_numpy(raw['x'][ix]).unsqueeze(-1)
            y = torch.from_numpy(raw['y'][ix])
            a.zero_grad(); b.zero_grad()
            ca, mb = cpu(x), mps(x.to('mps'))
            row = dict(T=t, batch=batch+1, forward=compare(ca, mb), gradients={})
            ann_weighted_cross_entropy(ca, y, 1., .2).backward()
            ann_weighted_cross_entropy(mb, y.to('mps'), 1., .2).backward()
            for (name, p), (_, q) in zip(cpu.named_parameters(), mps.named_parameters()):
                row['gradients'][name] = compare(p.grad, q.grad)
                if name == 'lstm.weight_hh_l0':
                    ga, gb = p.grad.detach().double(), q.grad.detach().cpu().double()
                    mask = (ga.abs()>1e-8) & (gb.abs()>1e-8)
                    row['hh_sign_mismatch_above_1e_minus8_threshold'] = int(((ga.sign()!=gb.sign()) & mask).sum())
                    row['hh_sign_mismatch_all'] = int((ga.sign()!=gb.sign()).sum())
                    row['hh_cosine'] = float(torch.nn.functional.cosine_similarity(ga.flatten(), gb.flatten(), dim=0))
                    assert row['hh_sign_mismatch_above_1e_minus8_threshold']==0 and row['hh_cosine']>.99999
            row['gradient_max'] = max(float(p.grad.abs().max()) for p in mps.parameters())
            a.step(); b.step()
            row['parameter_finite'] = all(bool(torch.isfinite(p).all()) for p in mps.parameters())
            for key in ['exp_avg', 'exp_avg_sq']:
                row[key+'_finite'] = all(bool(torch.isfinite(s[key]).all()) for s in b.state.values())
            row['exp_avg_sq_max'] = max(float(s['exp_avg_sq'].max()) for s in b.state.values())
            assert row['parameter_finite'] and row['exp_avg_finite'] and row['exp_avg_sq_finite']
            rows.append(row)
            mps.load_state_dict(cpu.state_dict())
            b.load_state_dict(copy.deepcopy(a.state_dict()))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True, help='New diagnostic directory, never an existing path')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(status='running', source_hashes=source_hashes(), torch_version=torch.__version__,
                  platform=platform.platform())
    try:
        assert torch.backends.mps.is_available(), 'MPS required'
        suite = unittest.defaultTestLoader.loadTestsFromName('tests.test_grid50_safeadam_gate')
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        report['unit_tests'] = dict(run=result.testsRun, failures=len(result.failures), errors=len(result.errors))
        assert result.wasSuccessful() and result.testsRun == 3, 'Prepared optimizer/gate tests failed'
        report['optimizer'] = optimizer_checks()
        report['hn'] = hn_checks()
        report['status'] = 'passed'
    except Exception as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report['elapsed_seconds'] = time.perf_counter()-start
        with (args.output/'preflight.json').open('x') as f:
            json.dump(report, f, indent=2)
        print(json.dumps(dict(status=report['status'], output=str(args.output), elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__':
    main()
