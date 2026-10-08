"""Lightweight checks: no W&B sessions, full datasets, or calibration runs."""
import json
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.run_ann_paper2019 import polynomial, runtime, select_epoch
from src.data import make_ann_dataset
from src.model import ANNClassifier
from src.train import ann_weighted_cross_entropy

ROOT = Path(__file__).resolve().parents[1]


class ANNWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads((ROOT / 'configs/ann_paper2019_reproduction.json').read_text())

    def test_supplementary_dry_run_counts_and_no_overlap(self):
        jobs = []
        for config, count in [('ann_valratio_1to5_vs_1to1_5x5.yaml', 1800),
                              ('ann_valratio_fine_5x5.yaml', 600)]:
            result = subprocess.run(
                [sys.executable, '-m', 'scripts.run_ann_sweep', '--config',
                 'configs/' + config, '--run-prefix', 'unit_check', '--dry-run'],
                cwd=ROOT, text=True, capture_output=True, check=True)
            names = [line for line in result.stdout.splitlines() if line.startswith('unit_check_T')]
            self.assertEqual(len(names), count)
            self.assertEqual(len(set(names)), count)
            jobs.append(set(names))
        self.assertFalse(jobs[0] & jobs[1])

    def test_convergence_checks_later_checkpoints_for_every_seed(self):
        runs = [{'evaluations': {str(e): {'overall': dict(loss=.4, accuracy=.8,
                 empirical_size=.2, empirical_power=.8)} for e in self.config['checkpoints']}}
                for _ in range(3)]
        self.assertEqual(select_epoch(self.config, runs)[0], 150)
        runs[0]['evaluations']['150']['overall']['empirical_size'] = .23
        self.assertEqual(select_epoch(self.config, runs)[0], 200)
        runs[1]['evaluations']['300']['overall']['empirical_size'] = .23
        self.assertIsNone(select_epoch(self.config, runs)[0])

    def test_polynomial_root_and_no_extrapolation(self):
        x = self.config['w2_values']
        fit = polynomial(self.config, x, [.01 + .1 * w for w in x])
        self.assertTrue(fit['acceptable'])
        self.assertAlmostEqual(fit['chosen_w2'], .4)
        self.assertFalse(polynomial(self.config, x, [.2 + .1 * w for w in x])['acceptable'])

    def test_runtime_counts_reuse_checkpoints(self):
        for seeds in [1, 2, 3]:
            estimate = runtime(self.config, 200, seeds)
            self.assertEqual(estimate['logical_calibration_runs'], 81)
            self.assertEqual(estimate['planned_new_ANN_runs'], 90 + 4 * seeds)

    def test_tiny_data_and_networks(self):
        for t in [50, 100, 250]:
            raw = make_ann_dataset(t, self.config['rhos'], self.config['betas'], 10, 2, 123)
            self.assertEqual(raw['x'].shape, (240, t))
            self.assertEqual(int((raw['y'] == 0).sum()), 120)
            np.testing.assert_allclose(raw['x'].mean(axis=1), 0, atol=1e-6)
            np.testing.assert_allclose(abs(raw['x']).max(axis=1), 1, atol=1e-6)
            model = ANNClassifier(t, 20 if t == 50 else 50)
            logits = model(torch.from_numpy(raw['x'][:8]))
            self.assertEqual(tuple(logits.shape), (8, 2))
            loss = ann_weighted_cross_entropy(logits, torch.from_numpy(raw['y'][:8]))
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_weighted_loss_uses_class_mapping_and_sample_denominator(self):
        logits = torch.tensor([[1., 2.], [3., 1.]])
        targets = torch.tensor([0, 1])
        logp = torch.log_softmax(logits, dim=1)
        expected = -(logp[0, 0] + 3 * logp[1, 1]) / 2
        self.assertTrue(torch.allclose(ann_weighted_cross_entropy(logits, targets, 1, 3), expected))


if __name__ == '__main__':
    unittest.main()
