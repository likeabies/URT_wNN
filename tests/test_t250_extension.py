import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import torch
from scripts.run_ann_t250_extension import execute


class T250ExtensionTests(unittest.TestCase):
    def exercise(self, sizes):
        calls=[]; raw=object()
        def record(w,seed):
            size=sizes(w)
            return dict(T=250,w2=w,model_seed=seed,stage='coarse',metrics=dict(empirical_size=size,empirical_power=.8),
                        beta_type_i_error=[dict(rho=1,beta=0,rejection_rate=size)],path='old')
        baseline=[record(.55,s) for s in [7,17,27]]
        def fake_train(c,base,stage,t,seed,data,w,epochs,lr,batch):
            self.assertIs(data,raw); self.assertEqual((t,epochs,lr,batch),(250,200,.001,256))
            self.assertGreater(w,.55); calls.append((stage,w,seed)); p=base/f'{w}_{seed}';p.mkdir()
            r=record(w,seed)
            return dict(path=str(p),id='fake',url='fake',seconds=1,
                evaluations={'200':dict(overall=r['metrics'],by_parameter=r['beta_type_i_error'])})
        with tempfile.TemporaryDirectory() as td:
            base=Path(td); state=dict(dataset={'fingerprint':'same'},runs=[])
            with patch('scripts.run_ann_t250_extension.train',side_effect=fake_train), patch('scripts.run_ann_t250_extension.torch.load',return_value={'epoch':200,'model':{'w':torch.ones(1)}}):
                execute({},raw,baseline,base,state)
            for name in ['extension_coarse_summary.csv','extension_fine_summary.csv','calibration_candidate_t250.json','run_manifest.json']:
                self.assertTrue((base/name).is_file())
            result=json.loads((base/'calibration_candidate_t250.json').read_text())
        return calls,state,result

    def test_first_crossing_stops_and_reuses_endpoints(self):
        calls,state,result=self.exercise(lambda w: .04+(w-.55)*.15)
        self.assertEqual(state['interval'],[.60,.65])
        self.assertEqual(state['fine_grid'],[.6,.6125,.625,.6375,.65])
        self.assertEqual(len(calls),15);self.assertEqual(len(set(calls)),15)
        self.assertEqual(state['fine_endpoint_reuses'],6)
        self.assertEqual(result['label'],'T250_fine_grid_candidate')

    def test_first_extension_brackets_with_historical_endpoint(self):
        calls,state,_=self.exercise(lambda w: .04 if w==.55 else .05)
        self.assertEqual(state['interval'],[.55,.6]);self.assertEqual(len(calls),12)

    def test_unbracketed_stops_at_one_without_fine(self):
        calls,state,result=self.exercise(lambda w:.03+(w-.55)*.01)
        self.assertEqual(len(calls),27)
        self.assertTrue(all(stage=='extension_coarse' for stage,_,_ in calls))
        self.assertEqual(state['status'],'complete_unbracketed');self.assertIsNone(result['candidate'])

    def test_reversal_preserved_then_adjacent_crossing(self):
        calls,state,_=self.exercise(lambda w: .04 if w==.55 else .03 if w==.6 else .06)
        self.assertEqual(state['interval'],[.6,.65]);self.assertEqual(len(calls),15)


if __name__=='__main__': unittest.main()
