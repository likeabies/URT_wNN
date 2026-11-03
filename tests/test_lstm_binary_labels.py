import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src import lstm_binary as b
from src.data import make_ann_dataset
from src.model import LSTMClassifier, Paper2019LSTMClassifier
from src.train import ann_weighted_cross_entropy
from src.evaluate import binary_metrics as legacy_metrics
from src.variable_length_lstm import VariableLengthLSTM, load_data, fingerprint, batches


class BinaryLabelTests(unittest.TestCase):
    def test_generator_preserves_values_and_rng(self):
        old=make_ann_dataset(12,[1.,.9],[.6],3,4,7)
        new=b.make_lstm_dataset(12,[1.,.9],[.6],3,4,7)
        for key in ('x','rho','beta'): np.testing.assert_array_equal(old[key],new[key])
        np.testing.assert_array_equal(new['y'],(new['rho']==1).astype(int))
        np.testing.assert_array_equal(old['y'],1-new['y'])
        ar=b.make_binary_dataset(4,3,2,12,7)
        self.assertEqual((ar['y']==0).sum(),4)
        self.assertEqual((ar['y']==1).sum(),5)

    def test_loss_and_gradient_permutation(self):
        for w1,w2 in [(1.,.2),(.7,1.3)]:
            logits=torch.tensor([[.4,-.2],[1.,2.],[-.5,.8]],requires_grad=True)
            labels=torch.tensor([0,1,1])
            old=ann_weighted_cross_entropy(logits,labels,w1,w2)
            new=b.weighted_cross_entropy(logits.flip(1),1-labels,w1,w2)
            torch.testing.assert_close(old,new)
            go=torch.autograd.grad(old,logits,retain_graph=True)[0]
            gn=torch.autograd.grad(new,logits)[0]
            torch.testing.assert_close(go,gn)

    def test_cm_and_class_metrics(self):
        old=np.array([[90,10],[30,70]])
        new=b.metrics_from_cm(old[::-1,::-1])
        self.assertEqual(new['confusion_matrix'],[[70,30],[10,90]])
        self.assertEqual(new['accuracy'],.8)
        self.assertEqual(new['empirical_size'],.1)
        self.assertEqual(new['empirical_power'],.7)
        self.assertEqual(new['recall'],[.7,.9])
        np.testing.assert_allclose(new['precision'],[70/80,90/120])
        np.testing.assert_allclose(new['f1'],[140/180,180/220])
        np.testing.assert_allclose(new['specificity'],[.9,.7])

    def test_checkpoint_storage_and_predictions_including_ties(self):
        torch.manual_seed(7)
        model=Paper2019LSTMClassifier()
        state=copy.deepcopy(model.state_dict())
        loaded=Paper2019LSTMClassifier(); loaded.load_state_dict(state)
        x=torch.randn(9,13)
        with torch.no_grad():
            _,(h,_)=loaded.lstm(x.unsqueeze(-1)); old=loaded.fc(h[-1])
            new=loaded(x)
        torch.testing.assert_close(new,old.flip(1),rtol=0,atol=0)
        torch.testing.assert_close(b.predict(new),1-old.argmax(1),rtol=0,atol=0)
        for k in state: torch.testing.assert_close(state[k],loaded.state_dict()[k],rtol=0,atol=0)
        self.assertEqual(b.predict(torch.zeros(1,2)).item(),1)
        near_tie=torch.tensor([[1e-9,0.]])
        torch.testing.assert_close(b.predict(near_tie,softmax=True),
                                   1-near_tie.flip(1).softmax(1).argmax(1))
        shared=VariableLengthLSTM();shared.load_state_dict(state)
        torch.testing.assert_close(shared(x.unsqueeze(-1)),new)
        # Three-class LSTM behavior is deliberately unchanged.
        multi=LSTMClassifier(num_classes=3)
        _,(h,_)=multi.lstm(x.unsqueeze(-1))
        torch.testing.assert_close(multi(x.unsqueeze(-1)),multi.fc(h[-1]))

    def test_legacy_cache_labels_and_fingerprint(self):
        raw=dict(x=np.ones((3,7),np.float32),y=np.array([0,1,0]),
                 rho=np.array([1.,.9,1.]),beta=np.zeros(3),lengths=np.full(3,7))
        old_fp=fingerprint(raw)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for k,v in raw.items():np.save(p/(k+'.npy'),v)
            (p/'dataset.json').write_text(json.dumps({'fingerprint':old_fp}))
            loaded=load_data(p)
            self.assertEqual(fingerprint(loaded),old_fp)
            np.testing.assert_array_equal(loaded['y'],1-raw['y'])
            np.testing.assert_array_equal(np.load(p/'y.npy'),raw['y'])
            labels=torch.cat([y for _,y,_,_ in batches(loaded,2)]).numpy()
            np.testing.assert_array_equal(labels,1-raw['y'])

    def test_evaluation_scalar_invariance_and_rng(self):
        old=make_ann_dataset(12,[1.,.9],[.6],3,4,7)
        new=b.from_legacy(old);model=Paper2019LSTMClassifier()
        rng=torch.get_rng_state().clone()
        result=b.evaluate(model,new,b.weighted_cross_entropy)
        torch.testing.assert_close(rng,torch.get_rng_state(),rtol=0,atol=0)
        with torch.no_grad():pred=b.predict(model(torch.from_numpy(new['x']))).numpy()
        legacy=legacy_metrics(old['y'],1-pred)
        for k,v in legacy.items():self.assertEqual(result['overall'][k],v)
        self.assertEqual(result['class_metrics']['recall'][0],legacy['empirical_power'])
        self.assertAlmostEqual(result['class_metrics']['recall'][1],1-legacy['empirical_size'])
