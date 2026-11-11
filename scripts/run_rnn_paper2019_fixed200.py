"""Resume-safe RNN calibration; test results never enter calibration routines.

Run as a module from the repository root. All mutable state is in the new family.
An exclusive OS lock prevents duplicate workflows. Epoch checkpoints are atomic
and authoritative for optimizer, history and RNG recovery. Training-only,
held-out and exact seedwise linear selections have separate frozen receipts.
"""
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics as st
import subprocess
import time
import traceback
import uuid
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb
from scripts.run_ann_paper2019 import dataset, fingerprint, save
from src.data import make_ann_dataset
from src.evaluate import evaluate_ann
from src.paper2019_rnn import Paper2019RNNClassifier
from src.train import ann_weighted_cross_entropy, train_one_epoch
from src.utils import set_seed

ANN = Path('results/ann_paper2019_reproduction')
CONFIG = Path('configs/rnn_paper2019_fixed200.json')
SOURCES = [CONFIG, Path(__file__), Path('src/paper2019_rnn.py'), Path('src/data.py'),
           Path('src/train.py'), Path('src/evaluate.py'), Path('src/utils.py'),
           Path('scripts/run_ann_paper2019.py')]


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def csv_save(path, rows):
    if not rows:
        return
    with Path(path).with_suffix('.csv.tmp').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    Path(path).with_suffix('.csv.tmp').replace(path)


def immutable(path, value):
    if path.exists():
        require(read(path) == value, f'Frozen record differs: {path}')
    else:
        save(path, value)


def atomic_checkpoint(path, value):
    tmp = path.with_suffix('.pt.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def seed_for(seed, t):
    return int(np.random.SeedSequence([seed, t]).generate_state(1)[0])


def point(rows):
    sizes = [r['size'] for r in rows]
    return dict(w2=rows[0]['w2'], mean_size=st.mean(sizes), sd_size=st.stdev(sizes),
                min_size=min(sizes), max_size=max(sizes), per_seed=rows)


def bracket(points):
    ordered = sorted(points, key=lambda p: p['w2'])
    exact = [p for p in ordered if p['mean_size'] == .05]
    if exact:
        return [exact[0]['w2'], exact[0]['w2']]
    intervals = [(a, b) for a, b in zip(ordered, ordered[1:])
                 if (a['mean_size'] - .05) * (b['mean_size'] - .05) < 0]
    if not intervals:
        return None
    a, b = min(intervals, key=lambda ab: (min(abs(p['mean_size']-.05) for p in ab), ab[0]['w2']))
    return [a['w2'], b['w2']]


def extend(points, step=.05):
    """Infer direction from observed endpoints; never consult test metrics."""
    ordered = sorted(points, key=lambda p: p['w2'])
    a, b = ordered[0], ordered[-1]
    delta = b['mean_size'] - a['mean_size']
    require(abs(delta) > 1e-8, 'Unbracketed target with flat response: direction cannot be inferred')
    up = (b['mean_size'] < .05 and delta > 0) or (b['mean_size'] > .05 and delta < 0)
    result = round(b['w2'] + step if up else a['w2'] - step, 10)
    require(result > 0, 'Cannot extend positive loss weight range downward by the fixed step')
    return result


def regression(points):
    x = np.array([p['w2'] for p in points], dtype=float)
    y = np.array([p['size'] for p in points], dtype=float)
    require(len(x) >= 2 and len(np.unique(x)) == len(x), 'Invalid broad regression points')
    b1, b0 = np.polyfit(x, y, 1)
    require(math.isfinite(b1) and abs(b1) > 1e-12, 'Linear regression has zero/invalid slope')
    w = float((.05-b0)/b1)
    require(math.isfinite(w) and w > 0, f'Exact linear root is not a positive loss weight: {w}')
    residual = float(np.sum((y-(b0+b1*x))**2))
    total = float(np.sum((y-y.mean())**2))
    return dict(b0=float(b0), b1=float(b1), w2_star=w, w2_hex=w.hex(),
                r_squared=1-residual/total if total else None,
                extrapolated=bool(w < min(x) or w > max(x)), points=points,
                degree=1, clipping=False, nearest_grid_replacement=False)


def slopes(history):
    return {f'last_{n}_loss_slope': float(np.polyfit(
        [r['epoch'] for r in history[-n:]], [r['train_loss'] for r in history[-n:]], 1)[0])
        for n in (50, 20) if len(history) >= n}


def evaluate(model, raw, w, device):
    rng = torch.get_rng_state()
    mps = torch.mps.get_rng_state() if device == 'mps' else None
    try:
        ev = evaluate_ann(model, TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y'])),
                          raw, partial(ann_weighted_cross_entropy, w1=1., w2=w),
                          batch_size=256, device=device)
        require(all(math.isfinite(v) for v in ev['overall'].values()), 'Nonfinite evaluation')
        return ev
    finally:
        torch.set_rng_state(rng)
        if mps is not None:
            torch.mps.set_rng_state(mps)


class Workflow:
    def __init__(self, c, base, device, smoke=False):
        self.c, self.base, self.device, self.smoke = c, base, device, smoke
        self.state = read(base/'run_manifest.json') if (base/'run_manifest.json').exists() else dict(
            status='initializing', runs={}, stages={}, datasets={}, errors=[], pid=os.getpid())
        self.state['pid'] = os.getpid()
        self.raw = {}
        self.persist()

    def persist(self):
        self.state['updated_at'] = time.time()
        save(self.base/'run_manifest.json', self.state)

    def data(self, t, cohort):
        key = (t, cohort)
        if key in self.raw:
            return self.raw[key]
        # Only keep one T in memory; arrays are deterministically regenerated.
        self.raw = {k: v for k, v in self.raw.items() if k[0] == t}
        if cohort == 'training':
            raw, derived = dataset(self.c, t, smoke=self.smoke)
            expected = read(ANN/'w2_calibration/run_manifest.json')['datasets'][str(t)]['fingerprint']
        elif cohort == 'validation':
            derived = seed_for(self.c['validation_seed'], t)
            raw = make_ann_dataset(t, [1.], self.c['betas'], 5000, 5000, derived)
            expected = read(ANN/'fixed200_validation_calibration/validation_dataset_fingerprints.json')[str(t)]['validation_fingerprint']
        elif cohort == 'test':
            derived = seed_for(self.c['test_seed'], t)
            raw = make_ann_dataset(t, [1., .95, .9, .8], [.9, .6, .3, 0.], 1000, 1000, derived)
            expected = read(ANN/'independent_test/test_dataset_fingerprints.json')[str(t)]['fingerprint']
        else:
            raise ValueError(cohort)
        fp = fingerprint(raw)
        if not self.smoke:
            require(fp == expected, f'{cohort} fingerprint mismatch T={t}')
            expected_n = {'training': 120000, 'validation': 60000, 'test': 16000}[cohort]
            require(len(raw['y']) == expected_n, 'Dataset size mismatch')
        self.state['datasets'][f'{cohort}_T{t}'] = dict(fingerprint=fp, derived_seed=derived, n=len(raw['y']))
        self.persist()
        self.raw[key] = raw
        return raw

    def fit(self, t, seed, w, purpose='grid', epochs=200, stop_after=None):
        require(w > 0 and math.isfinite(w), 'Invalid class weight')
        require(self.smoke or epochs == 200, 'Production requires exactly 200 epochs')
        identity = f'{purpose}_T{t}_seed{seed}_w{float(w).hex()}'
        key = f'{purpose}_T{t}_seed{seed}_{hashlib.sha256(identity.encode()).hexdigest()[:16]}'
        folder = self.base/'fits'/key
        folder.mkdir(parents=True, exist_ok=True)
        raw = self.data(t, 'training')
        cfg = dict(key=key, identity=identity, T=t, model_seed=seed, w2=w, w1=1.,
                   data_seed=self.c['data_seed'], data_fingerprint=fingerprint(raw), epochs=epochs,
                   architecture='nn.RNN(1,30,1,tanh)->Linear(30,2)', dropout=0,
                   lr=.001, weight_decay=0, batch_size=256, optimizer='Adam', shuffle=True,
                   device=self.device, purpose=purpose)
        immutable(folder/'config.json', cfg)
        cp_path = folder/'checkpoint.pt'
        if (folder/'summary.json').exists():
            summary = read(folder/'summary.json')
            require(cp_path.exists() and sha(cp_path) == summary['checkpoint_sha256'], 'Completed checkpoint changed')
            require(summary['epochs'] == epochs and len(read(folder/'history.json')) == epochs, 'Incomplete finished fit')
            self.state['runs'][key] = summary
            self.persist()
            return summary
        set_seed(seed)
        model = Paper2019RNNClassifier().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=0.)
        history = []
        if cp_path.exists():
            cp = torch.load(cp_path, map_location='cpu', weights_only=False)
            require(cp['identity'] == identity and cp['epoch'] == len(cp['history']), 'Checkpoint identity/history mismatch')
            model.load_state_dict(cp['model'])
            optimizer.load_state_dict(cp['optimizer'])
            history = cp['history']
            torch.set_rng_state(cp['torch_rng'])
            if self.device == 'mps':
                torch.mps.set_rng_state(cp['mps_rng'])
            random.setstate(cp['python_rng'])
            np.random.set_state(cp['numpy_rng'])
        loader = DataLoader(TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y'])),
                            batch_size=256, shuffle=True)
        rid_path = folder/'wandb_id.json'
        rid = read(rid_path)['id'] if rid_path.exists() else uuid.uuid4().hex[:8]
        immutable(rid_path, {'id': rid})
        rng_before_wandb = (torch.get_rng_state(), torch.mps.get_rng_state() if self.device == 'mps' else None, random.getstate(), np.random.get_state())
        mode = 'disabled' if self.smoke else 'online'
        with wandb.init(project=self.c['project'], entity=self.c['entity'], group=self.c['family'],
                        name=key, id=rid, resume='allow', mode=mode, config=cfg) as run:
            torch.set_rng_state(rng_before_wandb[0])
            if self.device == 'mps':
                torch.mps.set_rng_state(rng_before_wandb[1])
            random.setstate(rng_before_wandb[2])
            np.random.set_state(rng_before_wandb[3])
            for epoch in range(len(history)+1, epochs+1):
                tick = time.perf_counter()
                loss, acc = train_one_epoch(model, loader, partial(ann_weighted_cross_entropy, w1=1., w2=w), optimizer, self.device)
                if self.device == 'mps':
                    torch.mps.synchronize()
                require(math.isfinite(loss) and math.isfinite(acc), 'Nonfinite RNN optimization')
                require(all(torch.isfinite(p).all().item() for p in model.parameters()), 'Nonfinite RNN parameters')
                history.append(dict(epoch=epoch, train_loss=loss, train_acc=acc, seconds=time.perf_counter()-tick))
                atomic_checkpoint(cp_path, dict(identity=identity, epoch=epoch, model=model.state_dict(),
                    optimizer=optimizer.state_dict(), history=history, torch_rng=torch.get_rng_state(),
                    mps_rng=torch.mps.get_rng_state() if self.device == 'mps' else None,
                    python_rng=random.getstate(), numpy_rng=np.random.get_state()))
                save(folder/'history.json', history)
                run.log(history[-1])
                self.state.update(status='training', active_run=key, active_epoch=epoch)
                self.persist()
                if epoch == 1 or epoch % 10 == 0:
                    print(key, history[-1], flush=True)
                if stop_after == epoch:
                    return None  # Smoke-only simulated interruption, authoritative checkpoint retained.
            ev = evaluate(model, raw, w, self.device)
            diagnostics = slopes(history)
            run.summary.update(dict(training=ev['overall'], **diagnostics))
            result = dict(key=key, T=t, model_seed=seed, w2=w, purpose=purpose, epochs=epochs,
                status='finished', checkpoint=str(cp_path), checkpoint_sha256=sha(cp_path),
                data_fingerprint=cfg['data_fingerprint'], training=ev, diagnostics=diagnostics,
                wandb_id=rid, wandb_url=run.url if not self.smoke else None,
                seconds=sum(h['seconds'] for h in history))
        save(folder/'summary.json', result)  # Only finished W&B context is committed as successful.
        self.state['runs'][key] = result
        self.persist()
        return result

    def score(self, fit, cohort):
        require(cohort in ('validation', 'test'), 'Unexpected score cohort')
        p = self.base/'evaluations'/f"{cohort}_{fit['key']}.json"
        p.parent.mkdir(exist_ok=True)
        raw = self.data(fit['T'], cohort)
        fp = fingerprint(raw)
        if p.exists():
            value = read(p)
            require(value['checkpoint_sha256'] == fit['checkpoint_sha256'] and value['fingerprint'] == fp, 'Stale evaluation')
            return value['evaluation']
        require(sha(fit['checkpoint']) == fit['checkpoint_sha256'], 'Checkpoint changed before evaluation')
        model = Paper2019RNNClassifier().to(self.device)
        cp = torch.load(fit['checkpoint'], map_location='cpu', weights_only=False)
        require(cp['epoch'] == 200, 'Evaluation requires final fixed-200 model')
        model.load_state_dict(cp['model'])
        ev = evaluate(model, raw, fit['w2'], self.device)
        save(p, dict(checkpoint_sha256=fit['checkpoint_sha256'], fingerprint=fp, evaluation=ev))
        return ev

    def calibrate(self, cohort):
        method = 'training_grid' if cohort == 'training' else 'validation_grid'
        folder = self.base/method
        folder.mkdir(exist_ok=True)
        selected = []
        if method not in self.state['stages']:
            self.state['stages'][method] = dict(status='running', initial_fit_keys=sorted(self.state['runs']))
            self.persist()
        for t in self.c['T_values']:
            path = folder/f'T{t}.json'
            state = read(path) if path.exists() else dict(T=t, cohort=cohort, broad=[], fine=[], points={})
            def one(w, kind):
                k = str(w)
                if k not in state['points']:
                    rows = []
                    for seed in self.c['model_seeds']:
                        fit = self.fit(t, seed, w)
                        ev = fit['training'] if cohort == 'training' else self.score(fit, 'validation')
                        rows.append(dict(w2=w, model_seed=seed, size=ev['overall']['empirical_size'],
                            power=ev['overall']['empirical_power'] if cohort == 'training' else None,
                            fit_key=fit['key'], beta_specific=ev['by_parameter']))
                    state['points'][k] = point(rows)
                if w not in state[kind]:
                    state[kind].append(w)
                save(path, state)
            for w in self.c['coarse_w2']:
                one(w, 'broad')
            while not bracket(list(state['points'].values())):
                w = extend(list(state['points'].values()), self.c['extension_step'])
                require(w not in state['broad'], 'Repeated extension point')
                one(w, 'broad')
            # Freeze the coarse bracket before evaluating any fine interior.
            if 'bracket' not in state:
                state['bracket'] = bracket([state['points'][str(w)] for w in state['broad']])
                save(path, state)
            a, b = state['bracket']
            for w in sorted(set(round(float(v), 10) for v in np.linspace(a, b, 5))):
                one(w, 'fine')
            chosen = min(state['points'].values(), key=lambda p: (abs(p['mean_size']-.05), p['w2']))
            ordered = sorted(state['points'].values(), key=lambda p:p['w2'])
            direction = np.sign(ordered[-1]['mean_size']-ordered[0]['mean_size'])
            reversals = [dict(a=a['w2'], b=b['w2'], change=b['mean_size']-a['mean_size'])
                         for a,b in zip(ordered, ordered[1:]) if (b['mean_size']-a['mean_size'])*direction < 0]
            selection = dict(T=t, method=method, w2=chosen['w2'], mean_size=chosen['mean_size'],
                sd_size=chosen['sd_size'], distance=abs(chosen['mean_size']-.05), bracket=state['bracket'],
                fine_grid=state['fine'], reversals=reversals, per_seed=chosen['per_seed'],
                fit_keys=[p['fit_key'] for p in chosen['per_seed']])
            immutable(folder/f'selection_T{t}.json', selection)
            selected.append(selection)
        immutable(folder/'selections.json', selected)
        initial = set(self.state['stages'][method]['initial_fit_keys'])
        self.state['stages'][method].update(status='selected', selection_sha256=sha(folder/'selections.json'),
            new_unique_fits=len(set(self.state['runs'])-initial),
            broad_results=sum(len(read(folder/f'T{t}.json')['broad'])*3 for t in self.c['T_values']),
            fine_results=sum(len(read(folder/f'T{t}.json')['fine'])*3 for t in self.c['T_values']))
        csv_save(folder/'point_summary.csv', [dict(T=t, w2=p['w2'], mean_size=p['mean_size'],
            sd_size=p['sd_size'], min_size=p['min_size'], max_size=p['max_size'])
            for t in self.c['T_values'] for p in read(folder/f'T{t}.json')['points'].values()])
        self.persist()
        return selected

    def linear(self, cohort):
        grid = 'training_grid' if cohort == 'training' else 'validation_grid'
        method = 'training_linear' if cohort == 'training' else 'validation_linear'
        folder = self.base/method
        folder.mkdir(exist_ok=True)
        selections, errors = [], []
        for t in self.c['T_values']:
            state = read(self.base/grid/f'T{t}.json')
            for seed in self.c['model_seeds']:
                rows = [dict(w2=w, size=next(r['size'] for r in state['points'][str(w)]['per_seed']
                                             if r['model_seed'] == seed)) for w in sorted(state['broad'])]
                try:
                    reg = regression(rows)
                except RuntimeError as exc:
                    errors.append(dict(T=t, model_seed=seed, error=str(exc)))
                    continue
                decision = dict(T=t, model_seed=seed, method=method, **reg)
                immutable(folder/f'regression_T{t}_seed{seed}.json', decision)
                fit = self.fit(t, seed, reg['w2_star'], purpose=method)
                selections.append(dict(T=t, method=method, w2=reg['w2_star'], model_seed=seed,
                                       regression=reg, fit_keys=[fit['key']]))
        immutable(folder/'selections.json', selections)
        save(folder/'invalid_roots.json', errors)
        self.state['errors'].extend(e for e in errors if e not in self.state['errors'])
        self.state['stages'][method] = dict(status='selected' if not errors else 'partial',
                                          selection_sha256=sha(folder/'selections.json'))
        self.persist()
        return selections

    def confirm(self, method, selections):
        # A persisted selection must exist before the test dataset is accessed.
        frozen = self.base/method/'selections.json'
        require(frozen.exists() and read(frozen) == selections, 'Selection is not frozen')
        rows = []
        for choice in selections:
            for key in choice['fit_keys']:
                fit = self.state['runs'][key]
                ev = self.score(fit, 'test')
                rows.append(dict(model='RNN', method=method, T=fit['T'], model_seed=fit['model_seed'],
                    w2=fit['w2'], size=ev['overall']['empirical_size'], power=ev['overall']['empirical_power'],
                    **{f'power_rho_{rho}': st.mean(c['rejection_rate'] for c in ev['by_parameter'] if c['rho']==rho)
                       for rho in (.95,.9,.8)}, by_cell=ev['by_parameter']))
        save(self.base/method/'official_test_by_seed.json', rows)
        csv_save(self.base/method/'official_test_by_seed.csv', [{k:v for k,v in r.items() if k!='by_cell'} for r in rows])
        self.state['stages'][method]['test_confirmed'] = len(rows)
        self.persist()
        return rows

    def execute(self):
        for cohort in ('training', 'validation'):
            grid = f'{cohort}_grid'
            selections = self.calibrate(cohort)
            self.confirm(grid, selections)
            selections = self.linear(cohort)
            self.confirm(f'{cohort}_linear', selections)
        self.report()
        self.state['status'] = 'needs_attention' if self.state['errors'] else 'complete'
        self.persist()

    def report(self):
        rows = []
        for method in ('training_grid','training_linear','validation_grid','validation_linear'):
            p = self.base/method/'official_test_by_seed.json'
            if p.exists():
                for t in self.c['T_values']:
                    group = [r for r in read(p) if r['T']==t]
                    if len(group) != 3:
                        continue
                    rows.append(aggregate('RNN', method, t, group))
        references = self.base/'reference_comparison.json'
        if references.exists():
            rows.extend(read(references)['rows'])
        csv_save(self.base/'model_comparison.csv', rows)
        diagnostics = [dict(T=r['T'], model_seed=r['model_seed'], w2=r['w2'], purpose=r['purpose'],
                            fit_key=r['key'], **r['diagnostics']) for r in self.state['runs'].values()]
        csv_save(self.base/'convergence_diagnostics.csv', diagnostics)
        counts = {p:sum(r['purpose']==p for r in self.state['runs'].values())
                  for p in ('grid','training_linear','validation_linear')}
        save(self.base/'completion_summary.json', dict(unique_fits=len(self.state['runs']), counts=counts,
             stages=self.state['stages'], errors=self.state['errors'], comparison=rows,
             reference_limitations=read(references).get('limitations',[]) if references.exists() else ['No reference results available']))
        lines = ['# RNN fixed-200 결과', '',
                 '선택은 training 또는 독립 validation size만 사용했다. Test는 선택 저장 후 확인에만 사용했다.',
                 'Seed SD는 각 모델 seed의 전체 cell 집계값 간 표본 SD이다. 선형 root는 seed마다 다르다.',
                 f'완료된 고유 학습: {len(self.state["runs"])}; 목적별: {counts}', '',
                 '| 모델 | 방식 | T | w2 (seed 7/17/27) | test size 평균 ± SD | power 평균 ± SD |',
                 '|---|---|---:|---|---|---|']
        for r in rows:
            lines.append(f"| {r['model']} | {r['method']} | {r['T']} | {r['w2_by_seed']} | {r['size_mean']:.6f} ± {r['size_sd']:.6f} | {r['power_mean']:.6f} ± {r['power_sd']:.6f} |")
        lines += ['', '전체 rho별 power는 model_comparison.csv, 모든 학습의 마지막 50/20 epoch loss slope는 convergence_diagnostics.csv에 저장했다.',
                  '이 진단은 epoch 변경·추가 학습·후보 선택에 사용하지 않았다. ANN/LSTM은 재학습하지 않았다.']
        if references.exists():
            lines += ['', *read(references).get('limitations', [])]
        (self.base/'final_report.md').write_text('\n'.join(lines)+'\n')
        receipt = self.base/'wandb_report.json'
        if not receipt.exists() and not self.smoke:
            with wandb.init(project=self.c['project'], entity=self.c['entity'], group=self.c['family'],
                            name=self.c['family']+'_report', job_type='report', mode='online',
                            config=dict(training_performed=False, test_seed=20261008)) as run:
                run.log({'comparison': wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows]),
                         'diagnostics': wandb.Table(columns=list(diagnostics[0]), data=[list(r.values()) for r in diagnostics])})
                run.summary.update(dict(unique_training_fits=len(self.state['runs']), stages=self.state['stages']))
                receipt_value=dict(id=run.id,url=run.url)
            save(receipt,receipt_value)


def aggregate(model, method, t, group):
    group = sorted(group, key=lambda r:r['model_seed'])
    row = dict(model=model, method=method, T=t, w2_by_seed=json.dumps([r['w2'] for r in group]),
               size_mean=st.mean(r['size'] for r in group), size_sd=st.stdev(r['size'] for r in group),
               power_mean=st.mean(r['power'] for r in group), power_sd=st.stdev(r['power'] for r in group))
    row.update({f'power_rho_{rho}':st.mean(r[f'power_rho_{rho}'] for r in group) for rho in (.95,.9,.8)})
    return row


def main():
    global CONFIG
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    CONFIG = args.config
    c = read(CONFIG)
    device = 'cpu' if c.get('preferred_device') == 'cpu' else ('mps' if torch.backends.mps.is_available() else 'cpu')
    base = Path(c['output_dir'])
    if args.smoke:
        base = base.parent/(base.name+'_smoke_'+str(time.time_ns()))
    base.mkdir(parents=True, exist_ok=True)
    with (base/'workflow.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        hashes = {str(p):sha(p) for p in [CONFIG, *SOURCES[1:]]}
        immutable(base/'protocol.json', dict(config=c, source_sha256=hashes, device=device,
            git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            selection='absolute distance of cohort mean size to .05; tie lower w2',
            optimizer_decision='Adam .001, batch256, weight_decay0, exactly200; no epoch gates',
            regression_decision='degree1 OLS using broad/coarse and range extensions only; no clipping',
            training_reuse='grid fits shared by training and validation; exact linear fits separate',
            test_selection=False, smoke=args.smoke))
        wf = Workflow(c,base,device,args.smoke)
        try:
            if args.preflight:
                for t in c['T_values']:
                    for cohort in ('training','validation','test'):
                        wf.data(t,cohort)
                wf.state['status']='preflight_passed'
                wf.persist()
                print('All nine cohort fingerprints verified',flush=True)
            elif args.smoke:
                speeds={}
                for t in c['T_values']:
                    result=wf.fit(t,7,.3,epochs=3,stop_after=1)
                    require(result is None,'Smoke interruption failed')
                    result=wf.fit(t,7,.3,epochs=3)
                    require(wf.fit(t,7,.3,epochs=3)==result,'Completed fit reuse failed')
                    control=wf.fit(t,7,.3,purpose='smoke_uninterrupted',epochs=3)
                    resumed=torch.load(result['checkpoint'],map_location='cpu',weights_only=False)
                    direct=torch.load(control['checkpoint'],map_location='cpu',weights_only=False)
                    require(all(torch.allclose(resumed['model'][k],direct['model'][k],atol=2e-6,rtol=2e-6)
                                for k in resumed['model']), 'Interrupted and uninterrupted models differ')
                    speeds[str(t)]=st.mean(r['seconds'] for r in direct['history'][1:])
                save(base/'smoke_result.json',dict(status='passed',device=device,epoch_resume=True,
                     completed_reuse=True,seconds_per_tiny_epoch=speeds))
                print('SMOKE PASSED',base,speeds,flush=True)
            else:
                wf.execute()
        except Exception as exc:
            wf.state.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            wf.persist()
            raise


if __name__=='__main__':
    main()
