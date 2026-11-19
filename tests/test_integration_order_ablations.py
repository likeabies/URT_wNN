import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from src.integration_order_lstm import IntegrationOrderLSTM, loss_parts
from src.integration_order_ablations import make_model, total_loss, ARCHITECTURES
from scripts.run_integration_order_ablations import execute_queue


class AblationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.x=torch.randn(6,10);self.dx=torch.randn(6,9)
        self.d=torch.tensor([0,1,2,0,1,2]);self.w=torch.tensor([[1.5,.75],[.75,1.5]])

    def parts(self,m):return loss_parts(m(self.x,self.dx),self.d,self.w)
    def active(self,module):return any(p.grad is not None and p.grad.abs().sum()>0 for p in module.parameters())
    def absent(self,module):return all(p.grad is None or not p.grad.any() for p in module.parameters())

    def test_detach_values_and_paths(self):
        m=make_model('detach_p1');original=IntegrationOrderLSTM();original.load_state_dict(m.state_dict())
        torch.testing.assert_close(m(self.x,self.dx),original(self.x,self.dx),rtol=0,atol=0)
        x=self.x.clone().requires_grad_();dx=self.dx.clone().requires_grad_()
        loss_parts(m(x,dx),self.d,self.w)[1].backward()
        self.assertTrue(self.absent(m.stage1));self.assertTrue(self.active(m.lstm));self.assertTrue(self.active(m.stage2))
        self.assertTrue(x.grad is None or not x.grad.any());self.assertGreater(dx.grad.abs().sum(),0)
        m.zero_grad(set_to_none=True);self.parts(m)[0].backward()
        self.assertTrue(self.active(m.stage1));self.assertTrue(self.active(m.lstm))

    def test_independent_gradient_isolation(self):
        m=make_model('independent_encoders')
        self.assertFalse({id(p) for p in m.lstm1.parameters()} & {id(p) for p in m.lstm2.parameters()})
        self.parts(m)[1].backward()
        self.assertTrue(self.absent(m.lstm1));self.assertTrue(self.absent(m.stage1))
        self.assertTrue(self.active(m.lstm2));self.assertTrue(self.active(m.stage2))
        m.zero_grad(set_to_none=True);self.parts(m)[0].backward()
        self.assertTrue(self.absent(m.lstm2));self.assertTrue(self.absent(m.stage2))
        self.assertTrue(self.active(m.lstm1));self.assertTrue(self.active(m.stage1))

    def test_reduced_loss_and_chained_gradient(self):
        m=make_model('reduced_l2');parts=self.parts(m)
        torch.testing.assert_close(total_loss(parts,.5),parts[0]+.5*parts[1],rtol=0,atol=0)
        parts[1].backward();self.assertTrue(self.active(m.stage1));self.assertTrue(self.active(m.lstm))
        parts=torch.tensor([2.,6.],requires_grad=True);total_loss(parts,.5).backward()
        torch.testing.assert_close(parts.grad,torch.tensor([1.,.5]),rtol=0,atol=0)

    def test_queue_numerical_continues_infra_stops(self):
        calls=[]
        def run(c,t,path):
            calls.append((c['architecture'],t,c['stage2_weight']))
            return dict(status='failed' if len(calls)==1 else 'complete')
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.run_integration_order_ablations.comparison') as comp:
            execute_queue({'git_sha':'test'},Path(tmp),{},run)
            self.assertEqual([(a,t) for a,t,w in calls],[(a,t) for a in ARCHITECTURES for t in (50,100,250)])
            self.assertEqual([w for a,t,w in calls],[1.]*6+[.5]*3)
            self.assertEqual(json.loads((Path(tmp)/'queue_manifest.json').read_text())['status'],'finished')
            comp.assert_called_once()
        with tempfile.TemporaryDirectory() as tmp:
            def bad(*args):raise OSError('infrastructure test')
            with self.assertRaises(OSError):execute_queue({'git_sha':'test'},Path(tmp),{},bad)
            self.assertEqual(json.loads((Path(tmp)/'queue_manifest.json').read_text())['status'],'blocked')

    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS unavailable')
    def test_mps_tiny_step(self):
        for a in ARCHITECTURES:
            m=make_model(a).to('mps');opt=torch.optim.Adam(m.parameters(),lr=.001)
            parts=loss_parts(m(self.x.to('mps'),self.dx.to('mps')),self.d.to('mps'),self.w.to('mps'))
            total_loss(parts,.5 if a=='reduced_l2' else 1.).backward();opt.step()
            self.assertTrue(all(torch.isfinite(p).all() for p in m.parameters()))


if __name__=='__main__':unittest.main()
