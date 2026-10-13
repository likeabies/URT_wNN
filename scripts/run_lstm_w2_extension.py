"""T50/T250 range extension; immutable historical results and unchanged training."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
from time import perf_counter

import numpy as np
import torch

from scripts.run_ann_fixed200_calibration import candidates, local_grid, summarize, write_csv
from scripts.run_ann_paper2019 import dataset, evaluate, fingerprint, save
from scripts.run_lstm_fixed200_calibration import train
from src.model import Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy

CONFIG = Path('configs/lstm_w2_range_extension.json')
SCIENTIFIC_FILES = (
    'scripts/run_lstm_fixed200_calibration.py', 'scripts/run_ann_paper2019.py',
    'scripts/run_ann_fixed200_calibration.py', 'src/data.py', 'src/model.py',
    'src/train.py', 'src/evaluate.py', 'src/utils.py',
)


def key(record):
    return record['T'], round(float(record['w2']), 10), record['model_seed']


def source_inputs():
    extension = json.loads(CONFIG.read_text())
    source = Path(extension['source_dir'])
    historical = json.loads((source / 'run_manifest.json').read_text())
    c = json.loads((source / 'protocol.json').read_text())
    selected = json.loads((source / 'calibration_candidates.json').read_text())
    if historical['status'] != 'complete' or c['epochs'] != 200 or c['model_seeds'] != [7, 17, 27]:
        raise RuntimeError('Source calibration must be complete with the original 200-epoch protocol')
    if extension['T_values'] != [50, 250] or extension['new_coarse_w2'] != [.6,.65,.7,.75,.8,.85,.9,.95,1.]:
        raise ValueError('Only the authorized 54 new coarse fits are allowed')
    if extension['fine_points'] != 9 or extension['target'] != .05:
        raise ValueError('Use nine fine points and target .05')
    for name in SCIENTIFIC_FILES:
        previous = subprocess.check_output(['git', 'show', historical['git_sha'] + ':' + name])
        if Path(name).read_bytes() != previous:
            raise RuntimeError('Scientific code differs from the source calibration: ' + name)
    records = historical['runs']
    if len(records) != 99 or len({key(r) for r in records}) != 99:
        raise RuntimeError('Source must contain 99 unique fits')
    expected = {(t, w, s) for t in [50,100,250] for w in c['coarse_w2'] for s in c['model_seeds']}
    if {key(r) for r in records if r['stage'] == 'coarse'} != expected:
        raise RuntimeError('Source coarse grid is incomplete')
    # Validate every checkpoint before reuse, without evaluating or training it.
    hashes = {}
    for r in records:
        p = Path(r['path'])
        cfg = json.loads((p / 'config_snapshot.json').read_text())
        expected_cfg = dict(model='LSTM', hidden_size=30, num_layers=1, input_size=1, output_size=2,
                            epochs=200, optimizer='Adam', lr=.001, batch_size=256, dropout=0,
                            weight_decay=0, validation=False, early_stopping=False, device='mps',
                            w1=1., w2=r['w2'], model_seed=r['model_seed'], T=r['T'], data_seed=c['data_seed'])
        if any(cfg.get(k) != v for k,v in expected_cfg.items()):
            raise RuntimeError('Source run configuration mismatch: ' + str(p))
        if r['status'] != 'finished' or r['epochs'] != 200:
            raise RuntimeError('Incomplete source run: ' + str(p))
        if cfg['data_fingerprint'] != r['data_fingerprint'] or r['data_fingerprint'] != historical['datasets'][str(r['T'])]['fingerprint']:
            raise RuntimeError('Source dataset identity mismatch')
        ev = json.loads((p / 'train_epoch_200.json').read_text())
        if ev['overall'] != r['metrics'] or ev['by_parameter'] != r['by_parameter']:
            raise RuntimeError('Source evaluation mismatch')
        cp_path = p / 'epoch_200.pt'
        cp = torch.load(cp_path, map_location='cpu', weights_only=True)
        if cp['epoch'] != 200 or not all(torch.isfinite(v).all().item() for v in cp['model'].values()):
            raise RuntimeError('Invalid source checkpoint: ' + str(cp_path))
        # load_state_dict checks actual architecture compatibility as well as finiteness.
        Paper2019LSTMClassifier().load_state_dict(cp['model'])
        hashes[str(cp_path)] = hashlib.sha256(cp_path.read_bytes()).hexdigest()
    frozen = [r for r in selected if r['T'] == 100]
    if len(frozen) != 1 or frozen[0]['w2'] != .425:
        raise RuntimeError('Expected the existing T100 candidate w2=.425')
    c.update(extension)
    return c, historical, selected, hashes


def fine_plan(points, target=.05):
    """Use the exact historical interval selector, changing only the point count."""
    plan = local_grid(points, target)
    plan['w2_values'] = [round(float(v), 10) for v in np.linspace(*plan['interval'], 9)]
    points = sorted(points, key=lambda p: p['w2'])
    plan['crossing_intervals'] = [
        [a['w2'], b['w2']] for a,b in zip(points, points[1:])
        if (a['mean_type_i_error']-target) * (b['mean_type_i_error']-target) <= 0
    ]
    plan['multiple_crossings'] = len(plan['crossing_intervals']) > 1
    plan['selection_rule'] = ('Prefer a crossing containing the closest coarse point; ties use smallest '
                              'endpoint error, summed endpoint error, then lower interval. '
                              'Without a crossing, use the closest point and its best adjacent interval.')
    plan['target_not_bracketed_through_1_00'] = not plan['bracketed']
    return plan


def runtime_estimate(records):
    seconds = {str(t): statistics.mean(r['runtime_seconds'] for r in records if r['T'] == t) for t in [50,250]}
    coarse = 27 * sum(seconds.values())
    fine = 21 * sum(seconds.values())
    return dict(method='Measured complete 200-epoch source LSTM fits; same T/model/device/settings',
                seconds_per_fit=seconds, new_coarse_fits=54, max_new_fine_fits=42,
                estimated_coarse_hours=coarse/3600, estimated_max_fine_hours=fine/3600,
                estimated_max_total_hours=(coarse+fine)/3600,
                estimated_max_hours_with_20pct_overhead=1.2*(coarse+fine)/3600,
                note='Additional historical fine-point reuse can reduce the fine fit count; wall time is not guaranteed.')


def save_plots(base, records, selected):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = summarize(records, 'integrated')
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for ax, t in zip(axes, [50,100,250]):
        points = [r for r in rows if r['T'] == t]
        ax.errorbar([r['w2'] for r in points], [r['mean_type_i_error'] for r in points],
                    yerr=[r['sd_type_i_error'] for r in points], fmt='.-', capsize=3,
                    label='Mean +/- sample SD (3 seeds)')
        old = {r['w2'] for r in records if r['T'] == t and r.get('origin') == 'historical'}
        old_points = [r for r in points if r['w2'] in old]
        ax.scatter([r['w2'] for r in old_points], [r['mean_type_i_error'] for r in old_points],
                   marker='s', facecolors='none', edgecolors='black', label='Historical points', zorder=3)
        ax.axhline(.05, color='red', linestyle='--', label='Target 0.05')
        candidate = next((r for r in selected if r['T'] == t), None)
        if candidate:
            ax.axvline(candidate['w2'], color='green', linestyle=':', label=f"Candidate {candidate['w2']:g}")
        ax.set(title=f"T={t}" + (' (unchanged)' if t == 100 else ''),
               xlabel='w2', ylabel='Training-cohort Type I error')
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(base / 'w2_size.png', dpi=180)
    fig.savefig(base / 'w2_size.pdf')
    plt.close(fig)


def integrated_outputs(base, records, selected):
    if len({key(r) for r in records}) != len(records):
        raise RuntimeError('Duplicate fits in integrated results')
    rows = summarize(records, 'integrated')
    for row in rows:
        row['absolute_error'] = abs(row['mean_type_i_error'] - .05)
    write_csv(base / 'integrated_calibration_summary.csv', rows)
    write_csv(base / 'seed_metrics.csv', [dict(
        T=r['T'], w2=r['w2'], model_seed=r['model_seed'], type_i_error=r['metrics']['empirical_size'],
        power=r['metrics']['empirical_power'], origin=r['origin'], source_stage=r['stage'],
        checkpoint=str(Path(r['path']) / 'epoch_200.pt'), wandb_url=r['wandb_url']
    ) for r in sorted(records, key=key)])
    save(base / 'integrated_records.json', records)
    save(base / 'calibration_candidates.json', selected)
    save_plots(base, records, selected)


def execute(c, base, state, historical, frozen_candidates):
    previous = [dict(r, origin='historical') for r in historical['runs']]
    existing = {key(r): r for r in previous}
    raw_by_t = {}
    for t in c['T_values']:
        raw, derived = dataset(c, t)
        source = historical['datasets'][str(t)]
        if fingerprint(raw) != source['fingerprint'] or derived != source['derived_seed']:
            raise RuntimeError('Reconstructed dataset differs from historical data')
        raw_by_t[t] = raw
        state['datasets'][str(t)] = source

    def persist():
        save(base / 'run_manifest.json', state)

    def one(stage, t, w, seed):
        if (t, w, seed) in existing:
            raise RuntimeError('Attempt to retrain an existing checkpoint')
        info = train(c, base, stage, t, seed, raw_by_t[t], w, 200)
        ev = info['evaluations']['200']
        source = state['datasets'][str(t)]
        record = dict(model='LSTM', stage=stage, T=t, w2=w, model_seed=seed, data_seed=c['data_seed'],
                      derived_data_seed=source['derived_seed'], data_fingerprint=source['fingerprint'],
                      epochs=200, metrics=ev['overall'], by_parameter=ev['by_parameter'],
                      beta_type_i_error=[r for r in ev['by_parameter'] if r['rho'] == 1],
                      runtime_seconds=info['seconds'], path=info['path'], wandb_id=info['id'],
                      wandb_url=info['url'], status='finished', origin='extension')
        save(Path(info['path']) / 'calibration_record.json', record)
        existing[key(record)] = record
        state['runs'].append(record)
        persist()
        return record

    coarse = [r for r in previous if r['stage'] == 'coarse' and r['T'] in c['T_values']]
    state['status'] = 'extension_coarse_running'
    persist()
    # Complete ALL 54 fits, even if an early crossing is observed.
    for t in c['T_values']:
        for w in c['new_coarse_w2']:
            for seed in c['model_seeds']:
                coarse.append(one('extension_coarse', t, w, seed))
            write_csv(base / 'coarse_summary.csv', summarize(coarse, 'combined_coarse'))
    coarse_rows = summarize(coarse, 'combined_coarse')
    plans = {str(t): fine_plan([r for r in coarse_rows if r['T'] == t], c['target']) for t in c['T_values']}
    planned = {(t,w,s) for t in c['T_values'] for w in plans[str(t)]['w2_values'] for s in c['model_seeds']}
    new_fine = planned - set(existing)
    if len(new_fine) > 42:
        raise RuntimeError('Fine fit count exceeds the authorized maximum')
    state.update(status='extension_fine_running', fine_plans=plans, planned_new_fine_fits=len(new_fine),
                 fine_checkpoint_reuses=len(planned)-len(new_fine))
    save(base / 'fine_plan.json', plans)
    persist()
    fine = []
    for t in c['T_values']:
        for w in plans[str(t)]['w2_values']:
            for seed in c['model_seeds']:
                k = (t,w,seed)
                reused = k in existing
                record = existing[k] if reused else one('extension_fine',t,w,seed)
                fine.append(dict(record, reused=reused, source_stage=record['stage'],
                                 checkpoint=str(Path(record['path']) / 'epoch_200.pt')))
            write_csv(base / 'fine_summary.csv', summarize(fine, 'extension_fine'))
    if len(state['runs']) != 54 + len(new_fine) or len(fine) != 54:
        raise RuntimeError('Unexpected extension fit counts')
    save(base / 'fine_records.json', fine)
    selected = candidates(summarize(fine,'extension_fine'),fine,c['target'])
    for row in selected:
        row.update(model='LSTM', target_not_bracketed_through_1_00=not plans[str(row['T'])]['bracketed'],
                   selection_reason='Minimum absolute mean training size error among selected nine-point fine grid; lower w2 breaks ties.')
    selected.extend(r for r in frozen_candidates if r['T'] == 100)
    selected.sort(key=lambda r:r['T'])
    integrated_outputs(base, list(existing.values()), selected)
    state.update(status='complete', new_runs_completed=len(state['runs']), new_coarse_fits=54,
                 new_fine_fits=len(new_fine), final_independent_test=False,
                 unbracketed_T=[int(t) for t,p in plans.items() if not p['bracketed']],
                 T100='Preserved historical w2=.425 candidate; no new training')
    persist()
    print(f"Completed 54 new coarse + {len(new_fine)} new fine fits; no independent test.",flush=True)


def smoke(c, base):
    runs = []
    for t in c['T_values']:
        raw, _ = dataset(c, t, smoke=True)
        r = train(c, base, 'smoke', t, 7, raw, .6, 2)
        model = Paper2019LSTMClassifier().to('mps')
        model.load_state_dict(torch.load(Path(r['path'])/'epoch_2.pt',map_location='mps',weights_only=True)['model'])
        if evaluate(model,raw,partial(ann_weighted_cross_entropy,w1=1.,w2=.6)) != r['evaluations']['2']:
            raise RuntimeError('Smoke checkpoint reload mismatch')
        runs.append(r)
    save(base/'smoke_summary.json',dict(status='passed',runs=runs))


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run',action='store_true')
    mode.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    c, source, selected, hashes = source_inputs()
    estimate = runtime_estimate(source['runs'])
    if args.dry_run:
        print(json.dumps(dict(new_coarse_fits=54,max_new_fine_fits=42,coarse_grid=c['new_coarse_w2'],
                              T_values=c['T_values'],fine_points=9,epochs=200,verified_source_checkpoints=len(hashes),
                              T100_w2=.425,wandb_group=c['family'],output_dir=c['output_dir'],runtime=estimate),indent=2))
        return
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS required; no CPU fallback')
    if args.smoke:
        stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        c['family'] += '_smoke_' + stamp
        base=Path(c['output_dir']).parent / ('extension_smoke_' + stamp)
    else:
        base=Path(c['output_dir'])
    base.mkdir(parents=True,exist_ok=False)
    save(base/'protocol.json',c)
    save(base/'runtime_estimate.json',estimate)
    state=dict(status='initializing',family=c['family'],model='LSTM',epochs=200,
               source_dir=c['source_dir'],source_git_sha=source['git_sha'],source_checkpoints_sha256=hashes,
               git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               expected_new_coarse_fits=54,max_new_fine_fits=42,datasets={},runs=[],final_independent_test=False)
    save(base/'run_manifest.json',state)
    start=perf_counter()
    try:
        if args.smoke:
            smoke(c,base)
            state['status']='smoke_passed'
        else:
            execute(c,base,state,source,selected)
    except Exception as exc:
        state.update(status='failed',error=repr(exc))
        raise
    finally:
        state['total_runtime_seconds']=perf_counter()-start
        save(base/'run_manifest.json',state)


if __name__=='__main__':
    main()
