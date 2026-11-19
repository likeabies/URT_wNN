"""Explicitly launched first-epoch gate; no resume, calibration, or test execution."""
import argparse
import fcntl
import os
import platform
from pathlib import Path

import torch
from scripts.check_grid50_safeadam import CONFIG, source_hashes
from scripts.run_lstm_variable_length import read, require, run, save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    args = parser.parse_args()
    proof = read(args.preflight)
    require(proof['status']=='passed', 'All preflight checks must pass')
    require(proof.get('unit_tests')==dict(run=3,failures=0,errors=0), 'Gate/unit tests must pass')
    require(proof['source_hashes']==source_hashes(), 'Code/config changed after preflight')
    require(proof['torch_version']==torch.__version__ and proof['platform']==platform.platform(),
            'Preflight runtime differs')
    require(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0', 'Explicitly disable MPS fallback')
    require(torch.backends.mps.is_available(), 'MPS required')
    c = read(CONFIG)
    baseline = read('configs/lstm_multilength_grid50_equalN.json')
    changes = {'experiment', 'output_dir', 'optimizer_implementation', 'pause_after_epoch'}
    require({k:v for k,v in c.items() if k not in changes} ==
            {k:v for k,v in baseline.items() if k not in changes}, 'Scientific config changed')
    require(c['pause_after_epoch']==1 and c['epochs']==200 and c['coarse_w2'][0]==.2,
            'Only the first epoch of w2=.2 is authorized by this entry point')
    require(c['optimizer_implementation']=='grid50_scaled_second_moment_v1', 'Wrong optimizer')
    root = Path(c['output_dir'])
    require(root.resolve()!=Path(baseline['output_dir']).resolve(), 'Must preserve failed experiment')
    root.mkdir(parents=True, exist_ok=False)  # Any previous attempt requires explicit user review.
    save(root/'preflight_receipt.json', dict(path=str(args.preflight.resolve()), report=proof))
    with (root/'runner.lock').open('x') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(c, root)
    state = read(root/'run_manifest.json')
    require(state['status']=='awaiting_user_approval', 'Unexpected gate exit')
    print('Epoch 1 checkpoint saved; STOPPED awaiting user approval. No calibration/test or next w2.')


if __name__ == '__main__':
    main()
