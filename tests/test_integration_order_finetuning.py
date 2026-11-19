import unittest
from pathlib import Path
import numpy as np
import torch
from src.integration_order_finetuning import lower_null_threshold, calibrate, margin_loss, boundary_margin, stage1_weights, stage1_only_loss, sample_indices
from src.integration_order_lstm_I3 import IntegrationOrderLSTM, balanced_weights, loss_parts, RHOS, BETAS
from scripts.run_integration_order_finetuning import forward_hidden


class FineTuningTests(unittest.TestCase):
    def test_empirical_threshold_and_ties(self):
        x=np.arange(100,dtype=float)
        self.assertEqual(lower_null_threshold(x)['threshold'],4.)
        self.assertEqual(lower_null_threshold(x)['achieved'],.05)
        self.assertEqual(lower_null_threshold(np.ones(100))['achieved'],1.)
        c=calibrate(np.ones((30,3)),np.repeat([0,1,2],10),.1)
        self.assertIsNone(c['c3']['threshold'])

    def test_existing_corrected_thresholds(self):
        base=Path('results/integration_order_I3_4arch_20261110/original_chained')
        expected={50:(.15881532,.99991512),100:(.10228259,.99998891),250:(.17177728,.99999976)}
        for t,values in expected.items():
            p=base/f'T{t}'
            if not p.exists():self.skipTest('Local baseline unavailable')
            c=calibrate(np.load(p/'calibration_probabilities.npy'),np.load(p/'calibration/d.npy'))
            np.testing.assert_allclose([c['c1']['threshold'],c['c2']['threshold']],values,rtol=0,atol=5e-9)

    def test_sampling_balance_and_budget(self):
        cells=[(d,r,b) for d in range(4) for r in (RHOS if d==0 else [1.]) for b in BETAS]
        raw={k:np.array([c[j] for c in cells]) for j,k in enumerate(['d','rho','beta'])}
        for mode in ['hard_only','hard_replay','hard_margin']:
            idx=sample_indices(raw,mode,RHOS,BETAS,np.random.default_rng(7))
            idx2=sample_indices(raw,mode,RHOS,BETAS,np.random.default_rng(7))
            np.testing.assert_array_equal(idx,idx2);self.assertEqual(len(idx),230400)
            rho=raw['rho'][idx];d=raw['d'][idx]
            if mode=='hard_only':self.assertEqual(set(rho),{.99,1.})
            if mode=='hard_replay':self.assertAlmostEqual(np.isin(rho,[.99,1.]).mean(),.7)
            if mode=='hard_margin':self.assertEqual(set(rho),set(RHOS+[1.]))
            if mode!='hard_margin':
                self.assertEqual(set(d),{0,1})
                self.assertAlmostEqual((d==1).mean(),.5 if mode=='hard_only' else .35)
            w=balanced_weights(d) if mode=='hard_margin' else stage1_weights(d)[None,:]
            for j in range(len(w)):
                y=d>=j+1
                self.assertAlmostEqual(float(w[j,0]*(~y).sum()),len(d)/2,delta=.02)
                self.assertAlmostEqual(float(w[j,1]*y.sum()),len(d)/2,delta=.02)

    def test_margin_finite_gradient_and_satisfied_zero(self):
        z=torch.tensor([[1.,0.],[1.,.1],[-1.,0.]],requires_grad=True)
        loss=margin_loss(z,torch.tensor([0,1,0],dtype=torch.bool))
        self.assertGreater(loss.item(),0);loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(),0)
        far=torch.tensor([[1.,0.],[-1.,0.]],requires_grad=True)
        loss=margin_loss(far,torch.tensor([0,1],dtype=torch.bool));self.assertEqual(loss.item(),0.)
        loss.backward();self.assertEqual(far.grad.abs().sum().item(),0.)

    def test_hidden_hook_does_not_change_classification(self):
        torch.manual_seed(7);m=IntegrationOrderLSTM()
        args=[torch.randn(8,t) for t in (10,9,8)]
        expected=m(*args);actual,z=forward_hidden(m,args)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0);self.assertEqual(z.shape,(8,30))
        d=torch.arange(8)%4;w=torch.tensor(balanced_weights(d.numpy()))
        loss_parts(actual,d,w).sum().backward()
        self.assertTrue(all(p.grad is not None for p in m.parameters()))

    def test_stage1_freezing_preserves_heads_and_adam_state(self):
        torch.manual_seed(7);m=IntegrationOrderLSTM();opt=torch.optim.Adam(m.parameters(),lr=.001)
        d=torch.arange(8)%4;args=[torch.randn(8,t) for t in (10,9,8)]
        loss_parts(m(*args),d,torch.tensor(balanced_weights(d.numpy()))).sum().backward();opt.step()
        before={k:v.clone() for k,v in m.state_dict().items()}
        steps={id(p):opt.state[p]['step'].clone() for p in list(m.stage2.parameters())+list(m.stage3.parameters())}
        m.stage2.requires_grad_(False);m.stage3.requires_grad_(False);opt.zero_grad(set_to_none=True)
        d=torch.arange(8)%2
        loss=stage1_only_loss(m,args[0],d,torch.tensor(stage1_weights(d.numpy())))
        loss.backward();opt.step()
        for name,v in m.state_dict().items():
            if name.startswith(('stage2.','stage3.')):self.assertTrue(torch.equal(v,before[name]))
        for p in list(m.stage2.parameters())+list(m.stage3.parameters()):
            self.assertIsNone(p.grad);self.assertTrue(torch.equal(steps[id(p)],opt.state[p]['step']))
        self.assertFalse(torch.equal(before['stage1.weight'],m.stage1.weight))

    def test_margin_excludes_i2_i3(self):
        z=torch.tensor([[1.,0.],[1.,.1],[1.,0.],[1.,0.]],requires_grad=True)
        d=torch.arange(4)
        loss=boundary_margin(z,d)
        torch.testing.assert_close(loss,margin_loss(z[:2],d[:2]==1))
        loss.backward();self.assertEqual(z.grad[2:].abs().sum().item(),0.)

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_margin_and_adam_state_restore(self):
        p=Path('results/integration_order_I3_4arch_20261110/original_chained/T50/epoch_200.pt')
        cp=torch.load(p,map_location='cpu',weights_only=False)
        m=IntegrationOrderLSTM().to('mps');m.load_state_dict(cp['model'])
        opt=torch.optim.Adam(m.parameters(),lr=.001);opt.load_state_dict(cp['optimizer'])
        d=(torch.arange(8)%4).to('mps');args=[torch.randn(8,t,device='mps') for t in (50,49,48)]
        logits,z=forward_hidden(m,args);w=torch.tensor(balanced_weights(d.cpu().numpy()),device='mps')
        loss=loss_parts(logits,d,w).sum()+.1*boundary_margin(z,d)
        loss.backward();opt.step()
        self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))


if __name__=='__main__':unittest.main()
