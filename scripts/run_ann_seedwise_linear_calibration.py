"""Seed-wise linear calibration from saved broad grids, followed by fresh fits."""
import csv
import hashlib
import json
import math
import statistics as st
import subprocess
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
import wandb
from torch.utils.data import TensorDataset

from src.data import make_ann_dataset
from src.evaluate import evaluate_ann
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy
from scripts.run_ann_paper2019 import dataset, fingerprint, save, train


ROOT = Path('results/ann_paper2019_reproduction')
TRAIN_MANIFEST = ROOT / 'w2_calibration/run_manifest.json'
EXTENSION_MANIFEST = ROOT / 'w2_calibration_t250_extension/run_manifest.json'
VALIDATION_DIR = ROOT / 'fixed200_validation_calibration'
OFFICIAL_TEST_DIR = VALIDATION_DIR / 'test_seed_20261008'
OUT = ROOT / 'seedwise_linear_calibration'
CONFIG_PATH = Path('configs/ann_seedwise_linear_calibration.json')
GROUP = 'ann_paper2019_seedwise_linear_calibration'
SEEDS = [7, 17, 27]
TS = [50, 100, 250]
TRAIN_SEED = 20191007
VALIDATION_SEED = 20261013
TEST_SEED = 20261008
TARGET = .05
TEST_RHOS = [1., .95, .9, .8]
TEST_BETAS = [.9, .6, .3, 0.]
EXPECTED_TEST_FP = {
    50: 'd1513a67d48f6f689ceabe1f84c383b7e7518a235529f07c9cc5d119f36230e0',
    100: '04aaa1699cc91799f9c3ae32421f58a5d679584d24d1d5f3e62d4071046a64aa',
    250: 'e95d95a0efc66894dcde26bd9d9cda1ca2cff288389605f189cf8ac78b1cd7ce',
}
PROMPT_TEST_FP = {
    50: 'd1513a67af8ca62a4658f002e2ebe1b7e7518a235529f07c9cc5d119f36230e0',
    100: EXPECTED_TEST_FP[100], 250: EXPECTED_TEST_FP[250],
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'Cannot write empty CSV: {path}')
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def regression(xs, ys):
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    require(len(xs) == len(ys) >= 2, 'Regression requires at least two mapped points')
    require(np.isfinite(xs).all() and np.isfinite(ys).all(), 'Nonfinite regression input')
    b1, b0 = np.polyfit(xs, ys, 1)
    require(math.isfinite(float(b0)) and math.isfinite(float(b1)) and b1 != 0,
            'Invalid regression coefficients')
    prediction = b0 + b1 * xs
    total = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 if total == 0 and np.allclose(prediction, ys) else 1 - float(np.sum((ys-prediction)**2))/total
    w2 = (TARGET - float(b0)) / float(b1)
    require(math.isfinite(float(r2)) and math.isfinite(float(w2)), 'Nonfinite regression output')
    return dict(n_points=len(xs), w2_values=xs.tolist(), empirical_sizes=ys.tolist(),
                b0=float(b0), b1=float(b1), r_squared=float(r2), w2_star=float(w2))


def load_regression_inputs():
    train_manifest = read(TRAIN_MANIFEST)
    extension_manifest = read(EXTENSION_MANIFEST)
    val_manifest = read(VALIDATION_DIR / 'run_manifest.json')
    require(train_manifest['status'] == extension_manifest['status'] == 'complete',
            'Existing training calibration is incomplete')
    require(val_manifest['status'] == 'complete', 'Existing validation calibration is incomplete')
    require(train_manifest['epochs'] == extension_manifest['epochs'] == 200,
            'Training calibration epoch mismatch')

    train_runs = {}
    for record in train_manifest['runs']:
        if record.get('stage') == 'coarse':
            train_runs[(record['T'], record['w2'], record['model_seed'])] = record
    for record in extension_manifest['runs']:
        if record.get('stage') == 'extension_coarse':
            key = (record['T'], record['w2'], record['model_seed'])
            require(key not in train_runs, f'Duplicate train coarse point {key}')
            train_runs[key] = record

    # These are exactly the previously executed broad endpoint/bracketing points;
    # all records labelled fine are deliberately excluded.
    validation_runs = {}
    for record in val_manifest['records']:
        if record.get('stage') in ('endpoint', 'bracket_extension'):
            key = (record['T'], record['w2'], record['model_seed'])
            require(key not in validation_runs, f'Duplicate validation broad point {key}')
            validation_runs[key] = record

    fits = []
    for t in TS:
        expected_train = 10 if t == 250 else 8
        expected_val = {50: 2, 100: 3, 250: 3}[t]
        for seed in SEEDS:
            train_points = sorted((w, r['metrics']['empirical_size']) for (tt,w,s),r in train_runs.items()
                                  if tt == t and s == seed)
            val_points = sorted((w, r['validation_size']) for (tt,w,s),r in validation_runs.items()
                                if tt == t and s == seed)
            require(len(train_points) == expected_train,
                    f'Unexpected training coarse count T={t}, seed={seed}: {len(train_points)}')
            require(len(val_points) == expected_val,
                    f'Unexpected validation broad count T={t}, seed={seed}: {len(val_points)}')
            require(all(r['status'] == 'finished' and r['epochs'] == 200 and
                        r['training_seed'] == TRAIN_SEED for (tt,w,s),r in validation_runs.items()
                        if tt == t and s == seed), 'Validation broad provenance mismatch')
            require(all(r['status'] == 'finished' and r['epochs'] == 200 and
                        r['data_seed'] == TRAIN_SEED for (tt,w,s),r in train_runs.items()
                        if tt == t and s == seed), 'Training coarse provenance mismatch')
            train_fit = regression([x for x,_ in train_points], [y for _,y in train_points])
            val_fit = regression([x for x,_ in val_points], [y for _,y in val_points])
            for method, fit in [('train', train_fit), ('validation', val_fit)]:
                fits.append(dict(T=t, model_seed=seed, calibration_method=method, **fit))

    require(len(fits) == 18 and len({(r['T'],r['model_seed'],r['calibration_method']) for r in fits}) == 18,
            'Expected exactly 18 independent regressions')
    return fits, train_manifest, extension_manifest, val_manifest


def load_npz_dataset(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_validation_data(t, expected_fp):
    raw = load_npz_dataset(VALIDATION_DIR / 'datasets' / f'validation_T{t}.npz')
    require(fingerprint(raw) == expected_fp, f'Validation fingerprint mismatch at T={t}')
    require(len(raw['y']) == 60000 and np.all(raw['y'] == 0), 'Validation sample composition mismatch')
    return raw


def make_official_test(t):
    derived = int(np.random.SeedSequence([TEST_SEED, t]).generate_state(1)[0])
    raw = make_ann_dataset(t, TEST_RHOS, TEST_BETAS, 1000, 1000, derived)
    fp = fingerprint(raw)
    require(fp == EXPECTED_TEST_FP[t],
            f'Regenerated test set differs from saved official final test at T={t}: {fp}')
    require(len(raw['y']) == 16000, 'Official test sample size mismatch')
    for rho in TEST_RHOS:
        for beta in TEST_BETAS:
            selected = (raw['rho'] == rho) & (raw['beta'] == beta)
            require(int(selected.sum()) == 1000, 'Official test cell count mismatch')
            require(np.all(raw['y'][selected] == (0 if rho == 1 else 1)), 'Official test label mismatch')
    return raw, derived, fp


def evaluate_checkpoint(t, w2, seed, checkpoint, raw):
    cp = torch.load(checkpoint, map_location='cpu', weights_only=True)
    require(cp['epoch'] == 200 and all(torch.isfinite(v).all().item() for v in cp['model'].values()),
            f'Invalid checkpoint {checkpoint}')
    model = ANNClassifier(t, 20 if t == 50 else 50).to('mps')
    model.load_state_dict(cp['model'], strict=True)
    dataset_obj = TensorDataset(torch.from_numpy(raw['x']), torch.from_numpy(raw['y']))
    rng = torch.get_rng_state()
    mps_rng = torch.mps.get_rng_state()
    try:
        ev = evaluate_ann(model, dataset_obj, raw,
                          partial(ann_weighted_cross_entropy, w1=1., w2=w2),
                          batch_size=256, device='mps')
    finally:
        torch.set_rng_state(rng)
        torch.mps.set_rng_state(mps_rng)
    require(all(math.isfinite(v) for v in ev['overall'].values()), 'Nonfinite evaluation output')
    return ev


def train_and_evaluate(fits, train_manifest, val_manifest, state):
    config = read('configs/ann_paper2019_reproduction.json')
    config['family'] = GROUP
    source_fps = train_manifest['datasets']
    val_fps = val_manifest['datasets']
    test_fps = {}
    training_data = {}
    validation_data = {}
    test_data = {}
    for t in TS:
        raw, _ = dataset(config, t)
        require(fingerprint(raw) == source_fps[str(t)]['fingerprint'],
                f'Training array fingerprint mismatch T={t}')
        training_data[t] = raw
        validation_data[t] = load_validation_data(t, val_fps[str(t)]['validation_fingerprint'])
        test_data[t], derived, fp = make_official_test(t)
        test_fps[str(t)] = dict(test_seed=TEST_SEED, derived_seed=derived, fingerprint=fp,
                                sample_size=len(test_data[t]['y']), expected_official_fingerprint=EXPECTED_TEST_FP[t],
                                prompt_fingerprint=PROMPT_TEST_FP[t],
                                prompt_fingerprint_matches=(PROMPT_TEST_FP[t] == EXPECTED_TEST_FP[t]))
    save(OUT / 'test_dataset_fingerprints.json', test_fps)
    state['status'] = 'training_running'
    save(OUT / 'run_manifest.json', state)
    records = []
    for fit in fits:
        t, seed, method, w2 = fit['T'], fit['model_seed'], fit['calibration_method'], fit['w2_star']
        # New training starts from the default initialization under this same model seed.
        info = train(config, OUT / 'training_runs', f'linear_{method}', t, seed,
                     training_data[t], w2, 200, lr=.001, batch=256)
        checkpoint = str(Path(info['path']) / 'epoch_200.pt')
        train_ev = info['evaluations']['200']
        val_ev = evaluate_checkpoint(t, w2, seed, checkpoint, validation_data[t])
        test_cfg = dict(model='ANN', stage='independent_test', T=t, model_seed=seed,
                        calibration_method=method, w2=w2, test_seed=TEST_SEED,
                        test_fingerprint=test_fps[str(t)]['fingerprint'], training_performed=False,
                        epochs=200, checkpoint_sha256=sha(checkpoint))
        with wandb.init(project='URT_wNN', group=GROUP, job_type='independent-test',
                        name=f'linear_test_T{t}_{method}_seed{seed}', config=test_cfg, mode='online') as run:
            test_ev = evaluate_checkpoint(t, w2, seed, checkpoint, test_data[t])
            run.log({'independent_test_size': test_ev['overall']['empirical_size'],
                     'independent_test_power': test_ev['overall']['empirical_power'],
                     'cells': wandb.Table(columns=['rho','beta','n','rejection_rate'],
                         data=[[v[k] for k in ['rho','beta','n','rejection_rate']] for v in test_ev['by_parameter']])})
            eval_id, eval_url = run.id, run.url
        record = dict(T=t, model_seed=seed, calibration_method=method,
                      regression_point_count=fit['n_points'], regression_w2_values=fit['w2_values'],
                      regression_empirical_sizes=fit['empirical_sizes'], b0=fit['b0'], b1=fit['b1'],
                      regression_R2=fit['r_squared'], w2_star=w2,
                      train_type_i_error=train_ev['overall']['empirical_size'],
                      validation_type_i_error=val_ev['overall']['empirical_size'],
                      independent_test_type_i_error=test_ev['overall']['empirical_size'],
                      independent_test_size_by_beta=[v for v in test_ev['by_parameter'] if v['rho'] == 1],
                      independent_test_power_by_rho=[dict(rho=rho, mean_power=st.mean(
                          v['rejection_rate'] for v in test_ev['by_parameter'] if v['rho'] == rho))
                          for rho in TEST_RHOS if rho != 1],
                      independent_test_power_by_rho_beta=[v for v in test_ev['by_parameter'] if v['rho'] != 1],
                      mean_independent_test_power=test_ev['overall']['empirical_power'],
                      checkpoint_path=checkpoint, checkpoint_sha256=sha(checkpoint),
                      training_result_path=info['path'], train_wandb_id=info['id'], train_wandb_url=info['url'],
                      independent_test_wandb_id=eval_id, independent_test_wandb_url=eval_url,
                      train_metrics=train_ev['overall'], validation_metrics=val_ev['overall'],
                      independent_test_metrics=test_ev['overall'], status='finished')
        save(Path(info['path']) / 'seedwise_calibration_record.json', record)
        records.append(record)
        state['completed_models'] = len(records)
        state['records'] = records
        save(OUT / 'run_manifest.json', state)
        print(f"Finished T={t} seed={seed} method={method} w2={w2:.12g} "
              f"test_size={record['independent_test_type_i_error']:.6f}", flush=True)
    require(len(records) == 18, 'Did not complete exactly 18 models')
    return records, test_fps


def historical_comparisons(records):
    old_train = read(ROOT / 'independent_test/test_manifest.json')
    old_val = read(OFFICIAL_TEST_DIR / 'test_manifest.json')
    require(old_train['status'] == old_val['status'] == 'complete', 'Historical official test incomplete')
    out = []
    for r in records:
        t, seed = r['T'], r['model_seed']
        for label, manifest_dir, basename, selected_w2 in [
            ('train_grid', ROOT/'independent_test', 'evaluation', old_train['selected_w2'][str(t)]),
            ('validation_grid', OFFICIAL_TEST_DIR, 'evaluation', old_val['selected_w2'][str(t)])]:
            payload = read(manifest_dir / f'{basename}_T{t}_seed{seed}.json')
            result = payload['result']
            out.append(dict(T=t, model_seed=seed, calibration_method=r['calibration_method'],
                            comparison_method=label, selected_w2=selected_w2,
                            independent_test_size=result['overall']['empirical_size'],
                            mean_power=result['overall']['empirical_power'],
                            new_w2=r['w2_star'], new_independent_test_size=r['independent_test_type_i_error'],
                            new_mean_power=r['mean_independent_test_power']))
    return out


def finish_tables(records, fits, historical, test_fps):
    fit_map = {(r['T'],r['model_seed'],r['calibration_method']):r for r in fits}
    main = []
    train_table, val_table, aggregates = [], [], []
    for r in records:
        fit = fit_map[(r['T'],r['model_seed'],r['calibration_method'])]
        main.append(dict(T=r['T'], model_seed=r['model_seed'], calibration_method=r['calibration_method'],
                         w2_star=r['w2_star'], R2=r['regression_R2'], train_size=r['train_type_i_error'],
                         validation_size=r['validation_type_i_error'],
                         independent_test_size=r['independent_test_type_i_error'],
                         mean_power=r['mean_independent_test_power']))
        if r['calibration_method'] == 'train':
            train_table.append(dict(T=r['T'],model_seed=r['model_seed'],w2_star=r['w2_star'],
                                    train_size=r['train_type_i_error'],
                                    independent_test_size=r['independent_test_type_i_error'],
                                    mean_power=r['mean_independent_test_power']))
        else:
            val_table.append(dict(T=r['T'],model_seed=r['model_seed'],w2_star=r['w2_star'],
                                  validation_size=r['validation_type_i_error'],
                                  independent_test_size=r['independent_test_type_i_error'],
                                  mean_power=r['mean_independent_test_power']))
    for t in TS:
        for method in ('train','validation'):
            selected = [r for r in records if r['T']==t and r['calibration_method']==method]
            for metric, key in [('w2','w2_star'),('train_size','train_type_i_error'),
                                ('validation_size','validation_type_i_error'),
                                ('independent_test_size','independent_test_type_i_error'),
                                ('mean_power','mean_independent_test_power')]:
                vals = [r[key] for r in selected]
                aggregates.append(dict(T=t,calibration_method=method,metric=metric,mean=st.mean(vals),
                                       sample_sd=st.stdev(vals),model_seed_count=len(vals)))
    comparisons = []
    for r in records:
        for h in historical:
            if (r['T'],r['model_seed'],r['calibration_method']) == (h['T'],h['model_seed'],h['calibration_method']):
                comparisons.append(h)
    write_csv(OUT/'main_results.csv', main)
    write_csv(OUT/'train_based_results.csv', train_table)
    write_csv(OUT/'validation_based_results.csv', val_table)
    write_csv(OUT/'seed_aggregates_by_T.csv', aggregates)
    write_csv(OUT/'grid_comparison.csv', comparisons)
    write_csv(OUT/'regression_results.csv', fits)
    save(OUT/'results_by_model.json', records)
    save(OUT/'historical_grid_comparisons.json', comparisons)
    report = ['# ANN seed-wise linear calibration', '',
              '18개 모델을 model seed별로 독립 보정하고 처음부터 재학습했습니다. Test 결과는 calibration이나 선택에 사용하지 않았습니다.', '',
              '## 개별 모델 결과', '',
              '| T | seed | method | w2* | R² | train size | validation size | independent test size | mean power |',
              '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in main:
        report.append(f"| {r['T']} | {r['model_seed']} | {r['calibration_method']} | {r['w2_star']:.8f} | {r['R2']:.5f} | {100*r['train_size']:.3f}% | {100*r['validation_size']:.3f}% | {100*r['independent_test_size']:.3f}% | {100*r['mean_power']:.3f}% |")
    report += ['', '## 기존 grid 결과와 비교', '',
               '| T | seed | method | grid w2 | linear w2* | grid test size | linear test size | grid power | linear power |',
               '|---:|---:|---|---:|---:|---:|---:|---:|---:|']
    for r in comparisons:
        report.append(f"| {r['T']} | {r['model_seed']} | {r['calibration_method']} vs {r['comparison_method']} | {r['selected_w2']:.6f} | {r['new_w2']:.8f} | {100*r['independent_test_size']:.3f}% | {100*r['new_independent_test_size']:.3f}% | {100*r['mean_power']:.3f}% | {100*r['new_mean_power']:.3f}% |")
    report += ['', '## Seed별 평균과 SD', '',
               '각 T 및 calibration method 안에서 seed 7/17/27의 평균과 sample SD (ddof=1)입니다. Calibration에는 seed 평균을 사용하지 않았습니다.', '',
               '| T | method | metric | mean | sample SD |', '|---:|---|---|---:|---:|']
    for r in aggregates:
        report.append(f"| {r['T']} | {r['calibration_method']} | {r['metric']} | {r['mean']:.8f} | {r['sample_sd']:.8f} |")
    report += ['', '## Dataset fingerprints', '', '| T | seed |', '|---:|---|']
    for t in TS:
        report.append(f"| {t} | `{test_fps[str(t)]['fingerprint']}` |")
    report += ['', 'Official saved final-test fingerprint was used as the reference. The T=50 hash typed in the request differs from the saved official manifest; the saved manifest and all prior official evaluation files agree on the hash shown above.', '']
    (OUT/'final_report.md').write_text('\n'.join(report))
    return main, aggregates, comparisons


def upload_artifact(manifest):
    artifact = wandb.Artifact('ann_seedwise_linear_calibration_results', type='reproduction-supplement')
    artifact.add_dir(str(OUT))
    with wandb.init(project='URT_wNN', group=GROUP, job_type='artifact-upload',
                    name='seedwise-linear-calibration-artifact', mode='online') as run:
        logged = run.log_artifact(artifact)
        logged.wait()
        manifest['artifact'] = dict(name=logged.name, version=logged.version,
                                    url=logged.url, run_id=run.id)
    return manifest


def main():
    protocol_config = read(CONFIG_PATH)
    require(protocol_config['T_values'] == TS and protocol_config['model_seeds'] == SEEDS and
            protocol_config['training_data_seed'] == TRAIN_SEED and
            protocol_config['validation_data_seed'] == VALIDATION_SEED and
            protocol_config['independent_test_seed'] == TEST_SEED and
            protocol_config['expected_new_model_count'] == 18 and
            protocol_config['fine_grid_points_used'] is False and
            protocol_config['clip_w2'] is False and protocol_config['test_based_selection'] is False,
            'Configuration does not match the locked experiment design')
    fits, train_manifest, extension_manifest, val_manifest = load_regression_inputs()
    require(torch.backends.mps.is_available(), 'MPS is required; CPU fallback is forbidden')
    require(not OUT.exists(), f'Output directory already exists; refusing overwrite: {OUT}')
    OUT.mkdir(parents=True)
    (OUT/'training_runs').mkdir()
    save(OUT/'protocol.json', dict(family=GROUP, T_values=TS, model_seeds=SEEDS,
        target=TARGET, train_seed=TRAIN_SEED, validation_seed=VALIDATION_SEED, test_seed=TEST_SEED,
        epochs=200, optimizer='Adam', lr=.001, batch_size=256, architecture='existing ANNClassifier',
        early_stopping=False, calibration='per-T-per-model-seed OLS size = b0 + b1*w2',
        train_source=str(TRAIN_MANIFEST), T250_train_extension_source=str(EXTENSION_MANIFEST),
        validation_source=str(VALIDATION_DIR/'run_manifest.json'), validation_stage_filter=['endpoint','bracket_extension'],
        fine_points_used=False, clipping=False, grid_projection=False, test_based_selection=False,
        independent_test_reference=str(ROOT/'independent_test/test_manifest.json'),
        expected_test_fingerprints={str(t):EXPECTED_TEST_FP[t] for t in TS},
        prompt_T50_fingerprint=PROMPT_TEST_FP[50], prompt_T50_fingerprint_matches_official=False,
        wandb_group=GROUP, no_source_code_training_changes=True))
    state = dict(status='calibration_frozen', started_at=now(), T_values=TS, model_seeds=SEEDS,
                 expected_new_models=18, completed_models=0, training_performed=False,
                 test_seed=TEST_SEED, test_based_selection=False,
                 source_manifests={str(p):sha(p) for p in [TRAIN_MANIFEST,EXTENSION_MANIFEST,
                                   VALIDATION_DIR/'run_manifest.json']},
                 git_sha=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 regression_count=len(fits), runs=[], records=[])
    save(OUT/'regression_results.json', fits)
    write_csv(OUT/'regression_results.csv', fits)
    save(OUT/'frozen_w2_values.json', [dict(T=r['T'],model_seed=r['model_seed'],
                calibration_method=r['calibration_method'],w2_star=r['w2_star']) for r in fits])
    save(OUT/'run_manifest.json',state)
    start = perf_counter()
    try:
        records, test_fps = train_and_evaluate(fits, train_manifest, val_manifest, state)
        historical = historical_comparisons(records)
        main_rows, aggregates, comparisons = finish_tables(records, fits, historical, test_fps)
        state.update(status='complete', finished_at=now(), training_performed=True,
                     completed_models=len(records), independent_test_evaluation_count=len(records),
                     cell_seed_records=sum(len(r['independent_test_size_by_beta']) + len(r['independent_test_power_by_rho_beta']) for r in records),
                     total_runtime_seconds=perf_counter()-start,
                     output_counts=dict(main_rows=len(main_rows),aggregate_rows=len(aggregates),grid_comparisons=len(comparisons)))
        state = upload_artifact(state)
        save(OUT/'run_manifest.json', state)
        receipt = dict(status='uploaded',artifact=state['artifact'],uploaded_at=now(),
                       file_count=sum(1 for p in OUT.rglob('*') if p.is_file()))
        save(OUT/'artifact_receipt.json',receipt)
        print(json.dumps(state,indent=2),flush=True)
    except Exception as exc:
        state.update(status='failed', failed_at=now(), error=repr(exc), total_runtime_seconds=perf_counter()-start)
        save(OUT/'run_manifest.json',state)
        raise


if __name__ == '__main__':
    main()
