"""Lightweight LSTM checks: no real calibration, full DGP generation, or W&B."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts import run_ann_fixed200_calibration as ann
from scripts import run_lstm_fixed200_calibration as lstm
from scripts.run_ann_paper2019 import dataset, fingerprint
from src.model import LSTMClassifier, Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy


class LSTMCalibrationTests(unittest.TestCase):
    def test_protocol_and_exact_ann_data_reuse(self):
        c = lstm.load_protocol()
        base = json.loads(Path('configs/ann_paper2019_reproduction.json').read_text())
        self.assertEqual(c['data_seed'], base['data_seed'])
        self.assertEqual(c['epochs'], 200)
        self.assertEqual(len(c['T_values']) * len(c['coarse_w2']) * len(c['model_seeds']), 72)
        self.assertIs(lstm.local_grid, ann.local_grid)
        self.assertIs(lstm.candidates, ann.candidates)
        self.assertIs(lstm.dataset, dataset)
        for t in c['T_values']:
            original, seed = dataset(base, t, smoke=True)
            extension, extension_seed = lstm.dataset(c, t, smoke=True)
            self.assertEqual(seed, extension_seed)
            self.assertEqual(fingerprint(original), fingerprint(extension))

    def test_shapes_backward_labels_and_unchanged_original_lstm(self):
        for t in (50, 100, 250):
            model = Paper2019LSTMClassifier()
            original = LSTMClassifier()
            original.load_state_dict(model.state_dict())
            self.assertEqual(model.lstm.hidden_size, 30)
            self.assertEqual(model.lstm.num_layers, 1)
            self.assertEqual(model.lstm.dropout, 0)
            x = torch.randn(3, t)
            logits = model(x)
            torch.testing.assert_close(logits, original(x.unsqueeze(-1)))
            self.assertEqual(logits.shape, (3, 2))
            labels = torch.tensor([0, 1, 1])
            logp = logits.log_softmax(1)
            expected = -(logp[0, 0] + .3 * logp[1, 1] + .3 * logp[2, 1]) / 3
            loss = ann_weighted_cross_entropy(logits, labels, 1., .3)
            torch.testing.assert_close(loss, expected)
            loss.backward()
            self.assertTrue(all(torch.isfinite(p.grad).all().item() for p in model.parameters()))

    def test_automatic_coarse_fine_endpoint_reuse_and_outside_flags(self):
        c = lstm.load_protocol()
        identities, calls = {}, []

        def fake_dataset(config, t):
            rho, beta = [], []
            for r in c['rhos']:
                for b in c['betas']:
                    n = 5000 if r == 1 else 1000
                    rho.extend([r] * n)
                    beta.extend([b] * n)
            raw = dict(rho=np.array(rho), beta=np.array(beta), y=(np.array(rho) != 1).astype(int))
            identities[t] = id(raw)
            return raw, 123 + t

        def fake_train(config, base, stage, t, seed, raw, w, epochs):
            self.assertEqual(id(raw), identities[t])
            self.assertEqual(epochs, 200)
            calls.append((t, w, seed))
            p = base / f'{t}_{w}_{seed}'
            p.mkdir()
            # One crossing and two no-crossing curves: neither stops fine training.
            size = (.1 + .1*w if t == 50 else .02 + .1*w if t == 100 else .001 + .01*w)
            size += (seed - 17) * .000001
            ev = dict(overall=dict(loss=.3, accuracy=.7, empirical_size=size, empirical_power=.4, n=120000),
                      by_parameter=[dict(rho=1, beta=b, rejection_rate=size, n=5000) for b in c['betas']])
            return dict(path=str(p), id=str(len(calls)), url='mock', seconds=1., evaluations={'200': ev})

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            state = dict(datasets={}, runs=[])
            with patch.object(lstm, 'dataset', side_effect=fake_dataset), \
                 patch.object(lstm, 'fingerprint', return_value='shared'), \
                 patch.object(lstm, 'train', side_effect=fake_train):
                lstm.execute(c, base, state)
            self.assertEqual(len(calls), 99)
            self.assertEqual(len(set(calls)), 99)
            self.assertEqual(state['fine_reused_results'], 18)
            self.assertEqual(state['status'], 'complete')
            self.assertFalse(state['final_independent_test'])
            self.assertEqual(state['target_may_be_outside_coarse_range'], {'50': True, '100': False, '250': True})
            for name in ('coarse_summary.csv', 'fine_summary.csv', 'calibration_candidates.json', 'run_manifest.json'):
                self.assertTrue((base / name).is_file())
            selected = json.loads((base / 'calibration_candidates.json').read_text())
            self.assertEqual(len(selected), 3)
            for candidate in selected:
                self.assertFalse(candidate['final'])
                self.assertEqual(candidate['model'], 'LSTM')
                self.assertEqual(len(candidate['per_seed']), 3)
                self.assertEqual(len(candidate['beta_type_i_error']), 12)
                self.assertGreaterEqual(candidate['w2'], .2)
                self.assertLessEqual(candidate['w2'], .55)


if __name__ == '__main__':
    unittest.main()
