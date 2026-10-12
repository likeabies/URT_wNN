"""Sequential T250-only extension, reusing the fixed-200 training implementation."""
import argparse
import ast
import json
import math
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scripts.run_ann_paper2019 import dataset, fingerprint, save, train
from scripts.run_ann_fixed200_calibration import summarize, write_csv, candidates

SOURCE = Path('results/ann_paper2019_reproduction/w2_calibration')
BASE = Path('results/ann_paper2019_reproduction/w2_calibration_t250_extension')
FAMILY = 'ann_paper2019_reproduction_w2_calibration_t250_extension'
GRID = [.60, .65, .70, .75, .80, .85, .90, .95, 1.00]
SEEDS = [7, 17, 27]
# These files determine the historical data, preprocessing, architecture and loss.
SCIENTIFIC_FILES = ['scripts/run_ann_paper2019.py', 'src/data.py', 'src/model.py',
                    'src/train.py', 'src/evaluate.py', 'src/utils.py']


def source_inputs():
    m = json.loads((SOURCE/'run_manifest.json').read_text())
    c = json.loads((SOURCE/'protocol.json').read_text())
    assert m['status'] == 'complete' and c['data_seed'] == 20191007
    for name in SCIENTIFIC_FILES:
        historical = subprocess.check_output(['git', 'show', m['git_sha']+':'+name])
        current = Path(name).read_bytes()
        if name == 'src/model.py':
            # Other model classes may be added on main; ANN itself must be identical.
            def ann(source):
                return ast.dump(next(n for n in ast.parse(source).body
                                     if isinstance(n, ast.ClassDef) and n.name == 'ANNClassifier'))
            unchanged = ann(current) == ann(historical)
        else:
            unchanged = current == historical
        if not unchanged:
            raise RuntimeError('Scientific source changed: '+name)
    rows = [r for r in m['runs'] if r['T'] == 250 and r['w2'] == .55]
    assert sorted(r['model_seed'] for r in rows) == SEEDS
    for r in rows:
        p = Path(r['path']); cfg = json.loads((p/'config_snapshot.json').read_text())
        expected = dict(T=250, data_seed=20191007, epochs=200, optimizer='Adam', lr=.001,
                        batch_size=256, dropout=0, weight_decay=0, validation=False,
                        device='mps', w1=1., w2=.55, model_seed=r['model_seed'])
        assert all(cfg[k] == v for k,v in expected.items())
        assert r['status'] == 'finished' and r['epochs'] == 200
        assert r['data_fingerprint'] == cfg['data_fingerprint'] == m['datasets']['250']['fingerprint']
        assert json.loads((p/'train_epoch_200.json').read_text())['overall'] == r['metrics']
        cp = torch.load(p/'epoch_200.pt', map_location='cpu', weights_only=True)
        assert cp['epoch'] == 200 and all(torch.isfinite(v).all().item() for v in cp['model'].values())
    c.update(family=FAMILY, T_values=[250], output_dir=str(BASE))
    return c, m['datasets']['250'], rows, m['git_sha']


def prepare():
    c, info, rows, sha = source_inputs()
    if BASE.exists():
        raise RuntimeError('Extension directory exists; do not overwrite or restart')
    # The prior run persisted a hash, not arrays. Reconstruct identical bytes once.
    raw, derived = dataset(c, 250)
    if fingerprint(raw) != info['fingerprint'] or derived != info['derived_seed']:
        raise RuntimeError('Historical training dataset fingerprint mismatch')
    assert len(raw['y']) == 120000
    BASE.mkdir(parents=True, exist_ok=False)
    np.savez(BASE/'training_dataset.npz', **raw)
    save(BASE/'preflight.json', dict(dataset=info, source_git_sha=sha, reused_baseline_runs=rows,
         method='Deterministic byte-identical reconstruction verified against historical SHA256'))
    save(BASE/'protocol.json', c)
    print('Prepared identical historical T250 training arrays; no ANN training.', flush=True)


def enriched(records, stage):
    rows = summarize(records, stage)
    for row in rows:
        group = [r for r in records if r['w2'] == row['w2']]
        detail = candidates([row], group, .05)[0]
        row['absolute_error'] = detail['absolute_error']
        row['per_seed'] = json.dumps(detail['per_seed'], separators=(',', ':'))
        row['beta_type_i_error'] = json.dumps(detail['beta_type_i_error'], separators=(',', ':'))
    return rows


def execute(c, raw, baseline, base, state):
    def persist(): save(base/'run_manifest.json', state)
    def one(stage, w, seed):
        info = train(c, base, stage, 250, seed, raw, w, 200, lr=.001, batch=256)
        cp = torch.load(Path(info['path'])/'epoch_200.pt', map_location='cpu', weights_only=True)
        assert cp['epoch'] == 200 and all(torch.isfinite(v).all().item() for v in cp['model'].values())
        ev = info['evaluations']['200']
        assert all(math.isfinite(v) for v in ev['overall'].values())
        assert all(math.isfinite(v['rejection_rate']) for v in ev['by_parameter'])
        r = dict(stage=stage,T=250,w2=w,model_seed=seed,data_seed=20191007,epochs=200,
                 data_fingerprint=state['dataset']['fingerprint'],metrics=ev['overall'],
                 beta_type_i_error=[v for v in ev['by_parameter'] if v['rho']==1],
                 by_parameter=ev['by_parameter'],path=info['path'],wandb_id=info['id'],
                 wandb_url=info['url'],runtime_seconds=info['seconds'],status='finished',reused=False)
        save(Path(info['path'])/'calibration_record.json', r)
        state['runs'].append(r); persist()
        return r
    coarse = [dict(r, reused=True, source_stage=r['stage']) for r in baseline]
    state.update(status='extension_coarse_running', baseline_reused=coarse); persist()
    previous = summarize(coarse, 'extension_coarse')[0]
    assert previous['mean_type_i_error'] < .05
    write_csv(base/'extension_coarse_summary.csv', enriched(coarse,'extension_coarse'))
    interval = None
    for w in GRID:
        for seed in SEEDS: coarse.append(one('extension_coarse',w,seed))
        rows = enriched(coarse,'extension_coarse')
        write_csv(base/'extension_coarse_summary.csv', rows)
        current = rows[-1]
        print('coarse',w,'mean size',current['mean_type_i_error'],flush=True)
        if previous['mean_type_i_error'] < .05 <= current['mean_type_i_error']:
            interval = [previous['w2'],w]; break
        previous = current
    save(base/'coarse_records.json',coarse)
    if interval is None:
        state.update(status='complete_unbracketed',reason='target not bracketed through w2=1.00')
        (base/'extension_fine_summary.csv').write_text('stage,T,w2,n,mean_type_i_error,sd_type_i_error,mean_power,sd_power,absolute_error,per_seed,beta_type_i_error\n')
        save(base/'calibration_candidate_t250.json',dict(candidate=None,final=False,reason=state['reason']))
        persist(); return
    grid = [round(float(x),10) for x in np.linspace(*interval,5)]
    state.update(status='extension_fine_running',interval=interval,fine_grid=grid); persist()
    existing = {(r['w2'],r['model_seed']):r for r in coarse}; fine=[]
    for w in grid:
        for seed in SEEDS:
            if (w,seed) in existing:
                r=dict(existing[w,seed],reused=True,source_stage=existing[w,seed]['stage'])
            else: r=one('extension_fine',w,seed)
            fine.append(r)
    rows=enriched(fine,'extension_fine')
    write_csv(base/'extension_fine_summary.csv',rows); save(base/'fine_records.json',fine)
    candidate=candidates(summarize(fine,'extension_fine'),fine,.05)[0]
    candidate.update(label='T250_fine_grid_candidate',final=False)
    save(base/'calibration_candidate_t250.json',candidate)
    assert len(fine)==15 and sum(r['reused'] for r in fine)==6
    assert len({(r['w2'],r['model_seed']) for r in state['runs']})==len(state['runs'])
    state.update(status='complete',new_runs_completed=len(state['runs']),fine_endpoint_reuses=6)
    persist()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true')
    args=parser.parse_args()
    if args.prepare: prepare(); return
    if not torch.backends.mps.is_available(): raise RuntimeError('MPS unavailable')
    c, info, baseline, sha=source_inputs()
    with np.load(BASE/'training_dataset.npz',allow_pickle=False) as f: raw={k:f[k] for k in f.files}
    if fingerprint(raw)!=info['fingerprint']: raise RuntimeError('Cached dataset mismatch')
    state=dict(status='initializing',family=FAMILY,dataset=info,epochs=200,coarse_grid=GRID,
               source_git_sha=sha,git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),runs=[])
    # Exclusive creation prevents duplicate launches; never auto-resume.
    with (BASE/'run_manifest.json').open('x') as f: json.dump(state,f)
    start=perf_counter()
    try: execute(c,raw,baseline,BASE,state)
    except Exception as exc:
        state.update(status='failed',error=repr(exc)); raise
    finally:
        state['total_runtime_seconds']=perf_counter()-start
        save(BASE/'run_manifest.json',state)


if __name__=='__main__': main()
