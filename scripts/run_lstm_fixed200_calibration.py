"""LSTM-only fixed-200 training-cohort calibration; no validation or final test."""
import argparse
from datetime import datetime, timezone
from functools import partial
import json
import math
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset

from scripts.run_ann_fixed200_calibration import candidates, local_grid, summarize, write_csv
from scripts.run_ann_paper2019 import dataset, evaluate, fingerprint, save
from src.model import Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy, train_one_epoch
from src.utils import set_seed


def load_protocol():
    extension = json.loads(Path('configs/lstm_fixed200_w2_calibration.json').read_text())
    fixed = json.loads(Path(extension['base_config']).read_text())
    paper = json.loads(Path(fixed['base_config']).read_text())
    c = {**paper, **fixed, **extension}
    if (c['epochs'] != 200 or c['fine_points'] != 5 or c['T_values'] != [50, 100, 250]
            or c['model_seeds'] != [7, 17, 27]
            or c['coarse_w2'] != [.2, .25, .3, .35, .4, .45, .5, .55]
            or c['hidden_size'] != 30 or c['num_layers'] != 1 or c['model'] != 'LSTM'):
        raise ValueError('The authorized fixed-200 LSTM protocol must be preserved')
    return c


def train(c, base, stage, t, seed, raw, w2, epochs):
    name = f"{c['family']}_{stage}_T{t}_w2-{w2:.10g}_ms-{seed}"
    path = base / name
    path.mkdir(exist_ok=False)
    cfg = dict(model='LSTM', hidden_size=30, num_layers=1, input_size=1, output_size=2,
               family=c['family'], stage=stage, T=t, model_seed=seed, data_seed=c['data_seed'],
               w1=1., w2=w2, epoch=epochs, epochs=epochs, optimizer='Adam', lr=.001,
               batch_size=256, weight_decay=0., dropout=0., shuffle=True, device='mps',
               validation=False, early_stopping=False, evaluation_cohort='training',
               data_fingerprint=fingerprint(raw), n_train=len(raw['y']))
    save(path / 'config_snapshot.json', cfg)
    criterion = partial(ann_weighted_cross_entropy, w1=1., w2=w2)
    with wandb.init(project='URT_wNN', group=c['family'], name=name, config=cfg, mode='online') as run:
        set_seed(seed)
        model = Paper2019LSTMClassifier().to('mps')
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        loader = DataLoader(TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y'])),
                            batch_size=256, shuffle=True)
        history = []
        start = perf_counter()
        for epoch in range(1, epochs + 1):
            torch.mps.synchronize()
            tick = perf_counter()
            loss, acc = train_one_epoch(model, loader, criterion, optimizer, 'mps')
            torch.mps.synchronize()
            elapsed = perf_counter() - tick
            if not all(math.isfinite(value) for value in (loss, acc)):
                raise RuntimeError('Nonfinite training loss/accuracy')
            if not all(torch.isfinite(p).all().item() for p in model.parameters()):
                raise RuntimeError('Nonfinite LSTM parameters: numerical divergence')
            row = dict(epoch=epoch, train_loss=loss, train_acc=acc, train_seconds=elapsed)
            history.append(row)
            run.log(row)
            save(path / 'history.json', history)
            print(name, row, flush=True)
        # Exactly the ANN training-cohort evaluator and rejection rule. No test data.
        ev = evaluate(model, raw, criterion)
        if not all(math.isfinite(v) for v in ev['overall'].values()) or any(
                not math.isfinite(cell['rejection_rate']) for cell in ev['by_parameter']):
            raise RuntimeError('Nonfinite calibration metrics')
        save(path / f'train_epoch_{epochs}.json', ev)
        checkpoint_path = path / f'epoch_{epochs}.pt'
        torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), epoch=epochs,
                        torch_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state()), checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if checkpoint['epoch'] != epochs or not all(torch.isfinite(v).all().item() for v in checkpoint['model'].values()):
            raise RuntimeError('Invalid final LSTM checkpoint')
        seconds = perf_counter() - start
        run.log(dict(checkpoint_epoch=epochs, checkpoint_train=ev['overall'],
                     type_i_error=ev['overall']['empirical_size'], aggregate_power=ev['overall']['empirical_power'],
                     runtime_seconds=seconds,
                     beta_type_i_error={f"beta={cell['beta']:g}": cell['rejection_rate']
                                        for cell in ev['by_parameter'] if cell['rho'] == 1},
                     rejection_rate={f"rho={cell['rho']:g},beta={cell['beta']:g}": cell['rejection_rate']
                                     for cell in ev['by_parameter']}))
        info = dict(model='LSTM', name=name, path=str(path), T=t, model_seed=seed, w2=w2,
                    evaluations={str(epochs): ev}, seconds=seconds, id=run.id, url=run.url,
                    status='finished', epochs=epochs)
        save(path / 'summary.json', info)
    return info


def execute(c, base, state):
    raw_by_t = {}
    for t in c['T_values']:
        raw, derived_seed = dataset(c, t)
        if len(raw['y']) != 120000 or int((raw['y'] == 0).sum()) != 60000:
            raise RuntimeError('Unexpected training composition')
        for rho in c['rhos']:
            for beta in c['betas']:
                if int(((raw['rho'] == rho) & (raw['beta'] == beta)).sum()) != (5000 if rho == 1 else 1000):
                    raise RuntimeError('Unexpected training cell count')
        raw_by_t[t] = raw
        state['datasets'][str(t)] = dict(data_seed=c['data_seed'], derived_seed=derived_seed,
                                        fingerprint=fingerprint(raw), n=120000, unit_root=60000, stationary=60000)
    save(base / 'run_manifest.json', state)

    def one(stage, t, w, seed):
        info = train(c, base, stage, t, seed, raw_by_t[t], w, 200)
        ev = info['evaluations']['200']
        record = dict(model='LSTM', stage=stage, T=t, w2=w, model_seed=seed, data_seed=c['data_seed'],
                      derived_data_seed=state['datasets'][str(t)]['derived_seed'],
                      data_fingerprint=state['datasets'][str(t)]['fingerprint'], epochs=200,
                      metrics=ev['overall'], beta_type_i_error=[v for v in ev['by_parameter'] if v['rho'] == 1],
                      by_parameter=ev['by_parameter'], runtime_seconds=info['seconds'], path=info['path'],
                      wandb_id=info['id'], wandb_url=info['url'], status='finished')
        save(Path(info['path']) / 'calibration_record.json', record)
        state['runs'].append(record)
        save(base / 'run_manifest.json', state)
        return record

    state['status'] = 'coarse_running'
    save(base / 'run_manifest.json', state)
    for t in c['T_values']:
        for w in c['coarse_w2']:
            for seed in c['model_seeds']:
                one('coarse', t, w, seed)
    coarse = summarize(state['runs'], 'coarse')
    write_csv(base / 'coarse_summary.csv', coarse)
    plans = {str(t): local_grid([p for p in coarse if p['T'] == t], c['target']) for t in c['T_values']}
    for plan in plans.values():
        plan['target_may_be_outside_coarse_range'] = not plan['bracketed']
    state.update(status='fine_running', fine_plans=plans,
                 target_may_be_outside_coarse_range={t: not plan['bracketed'] for t, plan in plans.items()})
    save(base / 'fine_plan.json', plans)
    save(base / 'run_manifest.json', state)
    existing = {(r['T'], r['w2'], r['model_seed']): r for r in state['runs']}
    fine = []
    for t in c['T_values']:
        for w in plans[str(t)]['w2_values']:
            for seed in c['model_seeds']:
                key = (t, w, seed)
                if key in existing:
                    record = dict(existing[key], stage='fine', reused=True, source_stage=existing[key]['stage'])
                else:
                    record = dict(one('fine', t, w, seed), reused=False, source_stage='fine')
                    existing[key] = record
                fine.append(record)
    if not (len(state['runs']) == len(existing) == 99 and len(fine) == 45):
        raise RuntimeError('Unexpected coarse/fine fit count')
    fine_summary = summarize(fine, 'fine')
    write_csv(base / 'fine_summary.csv', fine_summary)
    save(base / 'fine_records.json', fine)
    selected = candidates(fine_summary, fine, c['target'])
    for candidate in selected:
        candidate.update(model='LSTM', target_may_be_outside_coarse_range=not plans[str(candidate['T'])]['bracketed'])
    save(base / 'calibration_candidates.json', selected)
    state.update(status='complete', new_runs_completed=len(state['runs']),
                 fine_reused_results=sum(r['reused'] for r in fine), final_independent_test=False)
    save(base / 'run_manifest.json', state)
    print('Completed 72 coarse + 27 new fine fits; 18 coarse endpoints reused. No independent test.', flush=True)


def smoke(c, base):
    """Three tiny fits only, plus full-batch timing, never statistical calibration."""
    records = []
    speeds = {}
    for t in c['T_values']:
        raw, _ = dataset(c, t, smoke=True)
        # Pad the 240-sample smoke cohort to one full batch only for timing.
        # The actual calibration cohort and all production counts are untouched.
        indices = torch.arange(256).numpy() % len(raw['y'])
        tiny = {key: value[indices] for key, value in raw.items()}
        info = train(c, base, 'smoke', t, 7, tiny, .3, 3)
        model = Paper2019LSTMClassifier().to('mps')
        model.load_state_dict(torch.load(Path(info['path']) / 'epoch_3.pt', map_location='mps', weights_only=True)['model'])
        reloaded = evaluate(model, tiny, partial(ann_weighted_cross_entropy, w1=1., w2=.3))
        if reloaded != info['evaluations']['3']:
            raise RuntimeError('Checkpoint reload changed smoke evaluation')
        history = json.loads((Path(info['path']) / 'history.json').read_text())
        # First epoch includes accelerator warm-up. Extrapolate later full batches.
        speeds[str(t)] = statistics.mean(row['train_seconds'] for row in history[1:])
        records.append(info)
    batches = math.ceil(120000 / 256)
    per_epoch = {t: seconds * batches for t, seconds in speeds.items()}
    coarse_seconds = 24 * 200 * sum(per_epoch.values())
    fine_seconds = 9 * 200 * sum(per_epoch.values())
    estimate = dict(model='LSTM', method='two post-warmup MPS batches of 256 per T; linear extrapolation',
                    seconds_per_batch=speeds, estimated_seconds_per_full_epoch=per_epoch,
                    epochs=200, coarse_fits=72, max_new_fine_fits=27,
                    estimated_coarse_hours=coarse_seconds / 3600,
                    estimated_fine_hours=fine_seconds / 3600,
                    estimated_total_hours=(coarse_seconds + fine_seconds) / 3600,
                    estimated_total_hours_with_20pct_overhead=(coarse_seconds + fine_seconds) * 1.2 / 3600,
                    caveat='Short timing extrapolation, not a guaranteed wall time; thermal/load effects may be substantial.')
    save(base / 'runtime_estimate.json', estimate)
    save(base / 'smoke_summary.json', dict(status='passed', runs=records, runtime=estimate))
    print(json.dumps(estimate, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--runtime-estimate', type=Path)
    args = parser.parse_args()
    c = load_protocol()
    if args.dry_run:
        print(json.dumps(dict(model='LSTM', hidden_size=30, num_layers=1, epochs=200,
                              coarse_runs=72, max_new_fine_runs=27, total_new_runs=99,
                              training_only=True, data_seed=c['data_seed'], model_seeds=c['model_seeds'],
                              coarse_w2=c['coarse_w2'], output_dir=c['output_dir'], wandb_group=c['family']), indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; no CPU fallback')
    if args.smoke:
        stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        c['family'] += f'_smoke_{stamp}'
        base = Path('results/lstm_paper2019_extension') / f'smoke_{stamp}'
    else:
        base = Path(c['output_dir'])
    base.mkdir(parents=True, exist_ok=False)
    save(base / 'protocol.json', c)
    state = dict(status='initializing', family=c['family'], model='LSTM', epochs=200,
                 expected_coarse_runs=72, max_new_fine_runs=27, final_independent_test=False,
                 git_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), datasets={}, runs=[])
    if args.runtime_estimate:
        state['runtime_estimate'] = json.loads(args.runtime_estimate.read_text())
    save(base / 'run_manifest.json', state)
    start = perf_counter()
    try:
        if args.smoke:
            smoke(c, base)
            state['status'] = 'smoke_passed'
        else:
            execute(c, base, state)
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        state['total_runtime_seconds'] = perf_counter() - start
        save(base / 'run_manifest.json', state)


if __name__ == '__main__':
    main()
