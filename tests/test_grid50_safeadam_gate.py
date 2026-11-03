"""Prepared tests; no production data, output paths, or W&B calls."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.run_lstm_variable_length import read, train_candidate
from src.grid50_safe_adam import Grid50SafeAdam
from src.grid50_step_audit import StepAudit


class SafeAdamGateTests(unittest.TestCase):
    def test_cpu_reference_multiple_steps(self):
        torch.manual_seed(7)
        p = torch.nn.Parameter(torch.randn(20))
        q = torch.nn.Parameter(p.detach().clone())
        ref = torch.optim.Adam([p], lr=.001, foreach=False, fused=False)
        safe = Grid50SafeAdam([q])
        for _ in range(10):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
            ref.step(); safe.step()
            torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)
            for key in ['exp_avg','exp_avg_sq']:
                torch.testing.assert_close(ref.state[p][key],safe.state[q][key],rtol=2e-6,atol=2e-7)

    def test_failure_preserves_good_state_and_context(self):
        model=torch.nn.Linear(1,1)
        optimizer=Grid50SafeAdam(model.parameters())
        with tempfile.TemporaryDirectory() as tmp:
            audit=StepAudit(tmp,model,optimizer,'cpu')
            audit.begin(dict(epoch=1,batch=1312,T=250,w2=.2,seed=7),np.array([3]))
            original=model.weight.detach().clone()
            model.weight.grad=torch.full_like(model.weight,float('inf'))
            with self.assertRaisesRegex(RuntimeError,'fail-fast'):
                audit.before_step(torch.tensor(1.))
            failure=json.loads((Path(tmp)/'numeric_failure.json').read_text())
            self.assertEqual(failure['context']['batch'],1312)
            self.assertIn('gradient/weight',failure['offending_tensors'])
            # Locally authored forensic file includes NumPy RNG state.
            saved=torch.load(Path(tmp)/'pre_failure_state.pt',weights_only=False)
            torch.testing.assert_close(saved['last_good']['model']['weight'],original)

    def test_epoch_gate_does_not_evaluate_or_mark_fit_complete(self):
        c=read('configs/lstm_multilength_grid50_equalN_safeadam_20261020.json')
        c['batch_size']=3
        raw=dict(x=np.arange(30,dtype=np.float32).reshape(6,5)/30,
                 y=np.array([0,1,0,1,0,1]),lengths=np.array([3,5,4,3,5,4]))
        state=dict(datasets={'train':{'fingerprint':'tr'},'calibration':{'fingerprint':'cal'}},runs=[])
        with tempfile.TemporaryDirectory() as tmp,              patch('scripts.run_lstm_variable_length.wandb.init',return_value=None),              patch('scripts.run_lstm_variable_length.evaluate') as evaluate:
            result=train_candidate(Path(tmp),c,state,raw,raw,.2,'coarse','cpu')
            self.assertEqual(result['status'],'awaiting_user_approval')
            self.assertEqual(state['gate']['epoch'],1)
            self.assertEqual(state['runs'],[])
            self.assertEqual(len(list(Path(tmp).glob('runs/*/epoch_001.pt'))),1)
            self.assertEqual(len(list(Path(tmp).glob('runs/*/epoch_002.pt'))),0)
            evaluate.assert_not_called()


if __name__=='__main__':
    unittest.main()
