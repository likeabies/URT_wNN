"""Audit matched cohorts and export chained/independent-head fixed-cycle results."""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from src.integration_order_lstm import classification, threshold


SCALARS = ['c1', 'c2', 'stage1_size', 'stage1_I1_power', 'stage1_I2_power',
           'stage2_size', 'stage2_I2_power']
COLUMNS = ['T', 'architecture', 'status', 'c1', 'c2', 'Stage1 I0 FPR',
           'Stage1 I1 detection', 'Stage1 I2 detection', 'Stage2 I1 FPR',
           'Stage2 I2 detection', 'I0 recall', 'I1 recall', 'I2 recall',
           'overall accuracy', 'I0 precision', 'I1 precision', 'I2 precision',
           'I1 -> I0 count', 'I1 -> I0 rate']


def read(path):
    return json.loads(path.read_text())


def audit_run(path):
    manifest = read(path / 'run_manifest.json')
    if manifest['status'] == 'failed':
        return manifest, None
    assert manifest['status'] == 'complete', f'{path}: run still active'
    assert manifest['completed_epoch'] == manifest['protocol']['epochs'] == 200
    history = read(path / 'history.json')
    assert [row['epoch'] for row in history] == list(range(1, 201))
    result = read(path / 'final_results.json')
    cal = np.load(path / 'calibration_probabilities.npy')
    cal_d = np.load(path / 'calibration/d.npy')
    assert result['c1'] == threshold(cal[cal_d == 0, 0])['threshold']
    assert result['c2'] == threshold(cal[cal_d == 1, 1])['threshold']
    p = np.load(path / 'test_probabilities.npy')
    d = np.load(path / 'test/d.npy')
    r1, r2 = p[:, 0] > result['c1'], p[:, 1] > result['c2']
    pred = np.where(r1, np.where(r2, 2, 1), 0)
    assert np.array_equal(pred, np.load(path / 'test_predictions.npy'))
    assert result['neural'] == classification(d, pred)
    assert result['adf'] == classification(d, np.load(path / 'adf_predictions.npy'))
    actual = [r1[d == 0].mean(), r1[d == 1].mean(), r1[d == 2].mean(),
              r2[d == 1].mean(), r2[d == 2].mean()]
    assert actual == [result[key] for key in SCALARS[2:]]
    return manifest, result


def copy_evidence(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    names = ['run_manifest.json', 'history.json', 'thresholds.json',
             'loss_weights.json', 'neural_test.json', 'adf_test.json',
             'final_results.json', 'numeric_failure.json', 'failure_traceback.log']
    for name in names:
        if (source / name).exists():
            shutil.copy2(source / name, destination / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--ablation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    base_config = read(args.baseline / 'config_snapshot.json')
    new_config = read(args.ablation / 'config_snapshot.json')
    scientific = dict(new_config)
    scientific.pop('architecture')
    scientific['wandb_group'] = base_config['wandb_group']
    assert scientific == base_config, 'Scientific protocol changed'
    assert read(args.ablation / 'cycle_manifest.json')['status'] == 'finished'
    rows, results, audit = [], {}, []
    for t in base_config['T_values']:
        results[t] = {}
        manifests = {}
        for architecture, root in [('chained', args.baseline), ('independent', args.ablation)]:
            path = root / f'T{t}'
            manifest, result = audit_run(path)
            manifests[architecture] = manifest
            results[t][architecture] = result
            copy_evidence(path, args.output / 'evidence' / architecture / f'T{t}')
            row = dict(T=t, architecture=architecture, status=manifest['status'])
            if result:
                row.update(zip(COLUMNS[3:10], (result[k] for k in SCALARS)))
                row.update(zip(COLUMNS[10:13], result['neural']['recall']))
                row['overall accuracy'] = result['neural']['accuracy']
                row.update(zip(COLUMNS[14:17], result['neural']['precision']))
                row['I1 -> I0 count'] = result['neural']['confusion_matrix'][1][0]
                row['I1 -> I0 rate'] = row['I1 -> I0 count'] / manifest['test_dataset']['class_counts']['1']
            rows.append(row)
        for cohort in ['train_dataset', 'calibration_dataset', 'test_dataset']:
            a, b = manifests['chained'].get(cohort), manifests['independent'].get(cohort)
            if b:
                assert a['fingerprint'] == b['fingerprint'], (t, cohort)
                audit.append(dict(T=t, cohort=cohort, fingerprint=a['fingerprint'], matched=True))
        if results[t]['independent']:
            assert np.array_equal(np.load(args.baseline / f'T{t}/adf_predictions.npy'),
                                  np.load(args.ablation / f'T{t}/adf_predictions.npy'))
    for name, root in [('chained', args.baseline), ('independent', args.ablation)]:
        for filename in ['config_snapshot.json', 'cycle_manifest.json']:
            shutil.copy2(root / filename, args.output / 'evidence' / name / filename)
    with (args.output / 'comparison.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'comparison.json').write_text(json.dumps(dict(rows=rows, results=results, cohort_audit=audit), indent=2) + '\n')
    lines = ['# Independent-head integration-order LSTM ablation', '',
             'Baseline: merged main `9cb02bd` (chained source `37887fd`). Independent: '
             '`Head2(h_diff)`, with the same shared LSTM and L1 + L2 loss.', '',
             'Each completed model was trained from scratch for 200 epochs with standard '
             '`torch.optim.Adam`. All scientific settings match the baseline config. '
             'Train/calibration/test fingerprints are checked per T; thresholds and '
             'metrics are recomputed from saved probabilities. ADF predictions must match exactly.', '',
             'Stage2 rates are unconditional on Stage1. Final decisions use both thresholds '
             'sequentially. Rates below are percentages; confusion matrices contain counts.', '']
    table_columns = COLUMNS[:2] + COLUMNS[3:14]
    lines += ['| ' + ' | '.join(table_columns) + ' |', '| ' + ' | '.join(['---'] * len(table_columns)) + ' |']
    for row in rows:
        def fmt(key):
            value = row.get(key)
            if value is None:
                return 'failed'
            if key in ['T', 'architecture']:
                return str(value)
            return f'{value:.9g}' if key in ['c1', 'c2'] else f'{100 * value:.3f}'
        lines.append('| ' + ' | '.join(fmt(key) for key in table_columns) + ' |')
    lines += ['', '## I(1) → I(0)', '']
    for t in base_config['T_values']:
        a, b = results[t]['chained'], results[t]['independent']
        if b is None:
            lines.append(f'- T={t}: independent run failed; no calibrated comparison is available.')
            continue
        ca, cb = a['neural']['confusion_matrix'][1][0], b['neural']['confusion_matrix'][1][0]
        n = manifests['chained']['test_dataset']['class_counts']['1']
        lines.append(f'- T={t}: {ca}/{n} ({100*ca/n:.3f}%) → {cb}/{n} ({100*cb/n:.3f}%); '
                     f'change {100*(cb-ca)/n:+.3f} percentage points.')
    lines += ['', 'This single-seed ablation isolates removal of the direct head connection. '
              'The shared encoder still receives both losses, so shared-encoder gradient '
              'interference remains possible. Results do not establish a causal mechanism or seed robustness.', '',
              '## Confusion matrices and class metrics', '', 'Rows = true I0/I1/I2; columns = predicted I0/I1/I2.', '']
    for t in base_config['T_values']:
        for architecture, result in results[t].items():
            lines += [f'### T={t} {architecture}', '']
            if result is None:
                lines += ['Failed; see preserved manifest and failure logs.', '']
                continue
            lines += ['```text'] + [' '.join(f'{v:6d}' for v in row) for row in result['neural']['confusion_matrix']] + ['```', '',
                '| Class | Recall (%) | Precision (%) |', '| --- | --- | --- |']
            for i in range(3):
                lines.append(f"| I{i} | {100*result['neural']['recall'][i]:.3f} | {100*result['neural']['precision'][i]:.3f} |")
            lines += ['']
    lines += ['## ADF benchmark', '', '| T | I0 recall (%) | I1 recall (%) | I2 recall (%) | Accuracy (%) |', '| --- | --- | --- | --- | --- |']
    for t in base_config['T_values']:
        adf = results[t]['chained']['adf']
        lines.append('| ' + ' | '.join([str(t)] + [f'{100*v:.3f}' for v in adf['recall'] + [adf['accuracy']]]) + ' |')
    lines += ['', '## W&B', '']
    for architecture, root in [('chained', args.baseline), ('independent', args.ablation)]:
        for t in base_config['T_values']:
            manifest = read(root / f'T{t}/run_manifest.json')
            url = manifest.get('wandb_url')
            lines.append(f'- T={t} {architecture}: [{manifest["status"]}]({url})' if url else f'- T={t} {architecture}: {manifest["status"]}')
    (args.output / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Comparison and audited evidence saved to {args.output}')


if __name__ == '__main__':
    main()
