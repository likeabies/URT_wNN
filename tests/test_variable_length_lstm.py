import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch

from src.model import Paper2019LSTMClassifier
from src.data import normalize_ann_series
from src.variable_length_lstm import VariableLengthLSTM,generate,batches,fingerprint
from scripts.run_lstm_variable_length import read,CONFIG,search_action,get_dataset,restore_records,save,sha
from scripts.run_lstm_variable_length import validate_epoch,atomic_checkpoint,restore_epoch,wb_call
from scripts.run_lstm_variable_length import train_candidate


class VariableLengthTests(unittest.TestCase):
    def test_interrupt_after_atomic_publish_resumes_without_repeating_epoch(self):
        c=read(CONFIG);c.update(epochs=3,batch_size=3,bucket_width=2)
        raw=dict(x=np.arange(30,dtype=np.float32).reshape(6,5)/30,
                 y=np.array([0,1,0,1,0,1]),lengths=np.array([3,5,4,3,5,4]))
        cal={k:v[1::2] for k,v in raw.items()}
        def state():return dict(datasets={'train':{'fingerprint':'tr'},'calibration':{'fingerprint':'cal'}},runs=[])
        def interrupted(*args,**kwargs):
            atomic_checkpoint(*args,**kwargs)
            raise OSError('simulated process interruption after publishing epoch 1')
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.run_lstm_variable_length.wandb.init',return_value=None):
            root=Path(tmp)/'resume';root.mkdir();s=state()
            with patch('scripts.run_lstm_variable_length.atomic_checkpoint',side_effect=interrupted):
                with self.assertRaisesRegex(OSError,'simulated'):
                    train_candidate(root,c,s,raw,cal,.2,'coarse','cpu')
            resumed=train_candidate(root,c,s,raw,cal,.2,'coarse','cpu')
            other=Path(tmp)/'continuous';other.mkdir()
            continuous=train_candidate(other,c,state(),raw,cal,.2,'coarse','cpu')
            a=torch.load(resumed['checkpoint'],weights_only=True);b=torch.load(continuous['checkpoint'],weights_only=True)
            self.assertEqual([r['epoch'] for r in a['history']],[1,2,3])
            for k in a['model']:torch.testing.assert_close(a['model'][k],b['model'][k],rtol=0,atol=0)
            self.assertEqual(len(list(root.glob('runs/*/epoch_001.pt'))),1)
            self.assertEqual(len(list(root.glob('runs/*/epoch_002.pt'))),1)

    def test_validator_reports_duplicate_and_nonfinite_predicates(self):
        row=dict(epoch=29,train_loss=float('nan'),train_acc=.5,train_seconds=1.,batch_count=2)
        with self.assertRaisesRegex(RuntimeError,'unique_coverage.*loss_finite') as ctx:
            validate_epoch(row,np.array([1,2,0]),3,29,True,False)
        self.assertIn('"processed_n": 3',str(ctx.exception))
        self.assertIn('"unique_n": 2',str(ctx.exception))
        self.assertIn('gradients_finite',str(ctx.exception))

    def test_atomic_resume_restores_optimizer_and_rng(self):
        c=read(CONFIG);torch.manual_seed(7)
        a=VariableLengthLSTM();oa=torch.optim.Adam(a.parameters(),lr=.001)
        x=torch.randn(3,8,1);lengths=torch.tensor([8,8,8])
        def step(model,opt):
            opt.zero_grad();model(x,lengths).square().mean().backward();opt.step()
        step(a,oa)
        hist=[dict(epoch=1,expected_n=3,processed_n=3,unique_n=3,missing_n=0,duplicate_n=0,train_loss=.2,train_acc=.5)]
        with tempfile.TemporaryDirectory() as tmp:
            receipt=atomic_checkpoint(Path(tmp)/'epoch_001.pt',a,oa,1,hist,c,'cpu')
            expected=torch.rand(4);step(a,oa)
            b=VariableLengthLSTM();ob=torch.optim.Adam(b.parameters(),lr=.001)
            restored=restore_epoch(receipt,b,ob,c,'cpu')
            self.assertEqual(restored,hist);torch.testing.assert_close(torch.rand(4),expected,rtol=0,atol=0)
            step(b,ob)
            for p,q in zip(a.parameters(),b.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
            with self.assertRaisesRegex(RuntimeError,'overwrite'):
                atomic_checkpoint(Path(tmp)/'epoch_001.pt',b,ob,1,hist,c,'cpu')
            with Path(receipt['checkpoint']).open('ab') as f:f.write(b'corruption')
            with self.assertRaisesRegex(RuntimeError,'checksum'):
                restore_epoch(receipt,b,ob,c,'cpu')

    def test_telemetry_failure_is_local_warning(self):
        def fail():raise RuntimeError('network unavailable')
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(wb_call(tmp,fail))
            self.assertIn('network unavailable',(Path(tmp)/'wandb_errors.jsonl').read_text())

    def test_exact_length_hn_logits_and_gradients(self):
        torch.manual_seed(7)
        model=VariableLengthLSTM();fixed=Paper2019LSTMClassifier()
        fixed.load_state_dict(model.state_dict())
        for t in [3,5,8]:
            x=torch.randn(3,t,1)
            model.zero_grad();fixed.zero_grad()
            torch.testing.assert_close(model(x),fixed(x.squeeze(-1)))
            model(x).sum().backward();fixed(x.squeeze(-1)).sum().backward()
            for p,q in zip(model.parameters(),fixed.parameters()):
                torch.testing.assert_close(p.grad,q.grad)

    def test_grid_counts_and_exact_coverage(self):
        c=read(Path('configs/lstm_multilength_grid50_equalN.json'))
        c.update(train_counts=[5,1])
        with tempfile.TemporaryDirectory() as tmp:
            raw,meta=generate(Path(tmp)/'grid',c,'train')
            self.assertEqual(meta['n'],1200)
            for row in meta['by_T'].values():
                self.assertEqual(row,dict(n=120,unit_root=60,stationary=60))
            orders=[]
            for epoch in [1,2]:
                ids=[]
                for x,y,lengths,idx in batches(raw,17,model_seed=7,epoch=epoch):
                    self.assertEqual(tuple(x.shape),(len(idx),int(lengths[0]),1))
                    self.assertTrue((lengths==x.shape[1]).all())
                    ids.extend(idx.tolist())
                self.assertEqual(sorted(ids),list(range(1200)))
                orders.append(ids)
            self.assertNotEqual(*orders)

    def test_dgp_exact_recurrence_and_unpadded_storage(self):
        c=read(CONFIG);c.update(min_length=2,max_length=5,rhos=[1.,.5],betas=[.6,0.],train_counts=[3,2],generation_chunk=2)
        with tempfile.TemporaryDirectory() as tmp:
            raw,meta=generate(Path(tmp)/'a',c,'train')
            again,other=generate(Path(tmp)/'b',c,'train')
            self.assertEqual(fingerprint(raw),fingerprint(again))
            lr,nr=[np.random.default_rng(s) for s in np.random.SeedSequence(c['data_seed']).spawn(2)]
            offset=0
            for rho in c['rhos']:
                for beta in c['betas']:
                    count=3 if rho==1 else 2;lens=lr.integers(2,6,size=count)
                    for start in range(0,count,2):
                        lengths=lens[start:start+2];errors=nr.normal(size=(len(lengths),int(lengths.max())+1))
                        for i,n in enumerate(lengths):
                            prev=0.;y=[]
                            for t in range(int(n)):
                                prev=rho*prev+(beta*errors[i,t]+errors[i,t+1]);y.append(prev)
                            expected=normalize_ann_series(np.array([y]))[0].astype(np.float32)
                            np.testing.assert_array_equal(raw['x'][raw['offsets'][offset]:raw['offsets'][offset+1]],expected)
                            self.assertEqual(raw['y'][offset],int(rho==1));offset+=1
            ids=np.concatenate([idx for *_,idx in batches(raw,3,2,7,1)])
            self.assertEqual(sorted(ids.tolist()),list(range(meta['n'])))

    def test_adaptive_search_only_pooled_size_and_minimal_grid(self):
        c=read(CONFIG)
        make=lambda w,size,stage='coarse':dict(w2=w,stage=stage,calibration={'pooled_size':size})
        rs=[make(w,w*.1) for w in c['coarse_w2']]
        stage,pending,plan=search_action(rs,c)
        self.assertEqual(stage,'fine');self.assertEqual(pending,[.45,.5,.55])
        rs += [make(w,.05,'fine') for w in pending]
        self.assertEqual(search_action(rs,c)[0],'select')
        self.assertEqual(search_action([make(w,.01) for w in c['coarse_w2']],c)[1],[1.2])
        self.assertEqual(search_action([make(w,.1) for w in c['coarse_w2']],c)[1],[0.])
        self.assertEqual(search_action([make(w,.1) for w in c['coarse_w2']+[0.]],c)[0],'select')

    def test_test_generation_locked_until_selection(self):
        with tempfile.TemporaryDirectory() as tmp,patch('scripts.run_lstm_variable_length.generate') as gen:
            with self.assertRaisesRegex(RuntimeError,'Test forbidden'):
                get_dataset(Path(tmp),read(CONFIG),{'datasets':{}},'test')
            gen.assert_not_called()

    def test_resume_ignores_incomplete_attempt_and_validates_complete(self):
        c=read(CONFIG);c['epochs']=2
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);p=root/'runs'/'done';p.mkdir(parents=True)
            (root/'runs'/'interrupted').mkdir()
            save(root/'runs'/'interrupted'/'history.json',[{'epoch':1}])
            torch.save({'epoch':2,'model':VariableLengthLSTM().state_dict()},p/'epoch_2.pt')
            cfg=dict(c,w2=.4);save(p/'config_snapshot.json',cfg);save(p/'history.json',[{'epoch':1},{'epoch':2}])
            ev={'pooled_size':.05};save(p/'calibration.json',ev)
            r=dict(status='complete',epochs=2,w2=.4,model_seed=7,checkpoint=str(p/'epoch_2.pt'),checkpoint_sha256=sha(p/'epoch_2.pt'),
                   calibration=ev,training_fingerprint='train',calibration_fingerprint='cal')
            save(p/'completed.json',r)
            state={'datasets':{'train':{'fingerprint':'train'},'calibration':{'fingerprint':'cal'}}}
            restore_records(root,c,state)
            self.assertEqual(state['runs'],[r])
            with (p/'epoch_2.pt').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(RuntimeError,'Checkpoint changed'):restore_records(root,c,state)


if __name__=='__main__':unittest.main()
