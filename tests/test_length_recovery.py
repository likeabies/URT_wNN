import json,unittest
from pathlib import Path
import numpy as np
import torch
from src.length_recovery import paired_data,short_input,Completion,Encoder,LogitStudent
from src.data import normalize_ann_series
from src.model import Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy
from scripts.run_lstm_rho_aux import control_source

class RecoverySmoke(unittest.TestCase):
    def test_pairs_and_labels(self):
        c=json.loads(Path('configs/ann_paper2019_reproduction.json').read_text())
        p,_=paired_data(c,smoke=True)
        np.testing.assert_array_equal(p['y'],(p['rho']<1).astype(np.int64))
        np.testing.assert_array_equal(p['x'],normalize_ann_series(p['raw']).astype(np.float32))
        for t in [50,100]:
            a=short_input(p['raw'],t);changed=p['raw'].copy();changed[:,t:]+=1e4
            np.testing.assert_array_equal(a,short_input(changed,t))
        np.testing.assert_array_equal(np.array([[2.,1.],[1.,2.]]).argmax(1),[0,1])
    @unittest.skipUnless(torch.backends.mps.is_available(),'MPS required')
    def test_mps_shapes_backward_freeze(self):
        cfg,cp,_=control_source();teacher=Paper2019LSTMClassifier().to('mps');teacher.load_state_dict(cp['model'])
        teacher.requires_grad_(False);original={k:v.clone() for k,v in teacher.state_dict().items()}
        for t in [50,100]:
            x=torch.randn(4,250,device='mps')
            with torch.no_grad():h=teacher.lstm(x.unsqueeze(-1))[1][0][-1];z=teacher(x)
            for model,args,target in [(Completion().to('mps'),(x[:,:t],x[:,t-1:249]),x[:,t:]),
                (Encoder(teacher.lstm.state_dict()).to('mps'),(x[:,:t],),h),
                (LogitStudent(teacher.lstm.state_dict()).to('mps'),(x[:,:t],),z)]:
                pred=model(*args);self.assertEqual(pred.shape,target.shape)
                loss=(pred-target).square().mean();loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
                torch.optim.Adam(model.parameters(),lr=.001).step()
                if isinstance(model,Completion):
                    with torch.no_grad():self.assertEqual(model.generate(x[:,:t],250-t).shape,(4,250-t))
            head=torch.nn.Linear(30,2).to('mps')
            loss=ann_weighted_cross_entropy(head(h),torch.tensor([0,1,0,1],device='mps'),w1=1.,w2=cfg['w2'])
            loss.backward();self.assertTrue(torch.isfinite(head.weight.grad).all())
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in teacher.parameters()))
        self.assertTrue(all(torch.equal(v,original[k]) for k,v in teacher.state_dict().items()))

if __name__=='__main__':unittest.main()
