"""Scientific ablation invariants: data/protocol parity and gradient isolation."""
import json
from pathlib import Path
import unittest

import torch

from src.integration_order_lstm import IntegrationOrderLSTM, balanced_weights, loss_parts
from src.integration_order_independent_lstm import IndependentHeadIntegrationOrderLSTM


class IndependentHeadTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = IndependentHeadIntegrationOrderLSTM()
        self.level = torch.randn(6, 10)
        self.diff = torch.randn(6, 9)

    def test_only_head2_input_dimension_changes(self):
        torch.manual_seed(7)
        chained = IntegrationOrderLSTM()
        self.assertEqual(self.model.stage2.in_features, 30)
        self.assertEqual(chained.stage2.in_features, 31)
        self.assertEqual(sum(p.numel() for p in chained.parameters()) -
                         sum(p.numel() for p in self.model.parameters()), 1)
        for key, value in self.model.state_dict().items():
            if not key.startswith('stage2.'):
                torch.testing.assert_close(value, chained.state_dict()[key], rtol=0, atol=0)
        self.assertEqual(self.model.lstm.num_layers, 1)
        self.assertEqual(self.model.lstm.dropout, 0)

    def test_stage2_is_invariant_to_stage1_parameters_and_level(self):
        before = self.model(self.level, self.diff).detach()
        with torch.no_grad():
            self.model.stage1.weight.fill_(100)
            self.model.stage1.bias.fill_(100)
        after = self.model(self.level * 3, self.diff).detach()
        torch.testing.assert_close(before[:, 1], after[:, 1], rtol=0, atol=0)
        self.assertFalse(torch.equal(before[:, 0], after[:, 0]))

    def test_l2_updates_shared_encoder_and_head2_but_not_head1(self):
        orders = torch.tensor([0, 1, 2, 0, 1, 2])
        weights = torch.from_numpy(balanced_weights(orders.numpy()))
        parts = loss_parts(self.model(self.level, self.diff), orders, weights)
        parts[1].backward()
        self.assertGreater(self.model.lstm.weight_ih_l0.grad.abs().sum().item(), 0)
        self.assertGreater(self.model.stage2.weight.grad.abs().sum().item(), 0)
        for parameter in self.model.stage1.parameters():
            self.assertTrue(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0)
        self.model.zero_grad(set_to_none=True)
        loss_parts(self.model(self.level, self.diff), orders, weights).sum().backward()
        self.assertGreater(self.model.stage1.weight.grad.abs().sum().item(), 0)

    def test_scientific_config_unchanged(self):
        root = Path(__file__).resolve().parents[1] / 'configs'
        base = json.loads((root / 'integration_order_lstm_fixed_cycle_20261029.json').read_text())
        ablation = json.loads((root / 'integration_order_lstm_independent_fixed_cycle_20261103.json').read_text())
        ablation.pop('architecture')
        ablation['wandb_group'] = base['wandb_group']
        self.assertEqual(base, ablation)


if __name__ == '__main__':
    unittest.main()
