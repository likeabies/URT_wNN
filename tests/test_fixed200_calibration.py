import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from scripts.run_ann_fixed200_calibration import local_grid, summarize, candidates, execute


class Fixed200Tests(unittest.TestCase):
    def points(self, values):
        return [dict(w2=round(.2+.05*i,2),mean_type_i_error=y) for i,y in enumerate(values)]

    def test_bracket_and_endpoint_reuse(self):
        plan=local_grid(self.points([.02,.03,.043,.057,.07,.08,.09,.1]))
        self.assertEqual(plan['interval'],[.3,.35])
        self.assertEqual(plan['w2_values'],[.3,.3125,.325,.3375,.35])
        self.assertEqual(len(set(plan['w2_values'])-set(p['w2'] for p in self.points([0]*8))),3)

    def test_multiple_crossings_prefers_closest_point(self):
        plan=local_grid(self.points([.02,.06,.03,.049,.06,.08,.09,.1]))
        self.assertEqual(plan['interval'],[.35,.4])

    def test_no_crossing_boundaries_and_interior(self):
        self.assertEqual(local_grid(self.points([.1,.2,.3,.4,.5,.6,.7,.8]))['interval'],[.2,.25])
        self.assertEqual(local_grid(self.points([.01,.02,.03,.035,.04,.042,.044,.046]))['interval'],[.5,.55])
        p=local_grid(self.points([.01,.02,.04,.03,.02,.01,.02,.01]))
        self.assertFalse(p['bracketed']); self.assertEqual(p['interval'],[.3,.35])

    def test_exact_target(self):
        p=local_grid(self.points([.01,.03,.05,.06,.07,.08,.09,.1]))
        self.assertTrue(p['bracketed']); self.assertIn(.3,p['interval'])

    def test_summary_and_candidate_seed_details(self):
        records=[dict(T=50,w2=w,model_seed=s,metrics=dict(empirical_size=y,empirical_power=.7),
                      beta_type_i_error=[dict(beta=0.,rejection_rate=y)])
                 for w,size in [(.3,.04),(.35,.05)] for s,y in zip([7,17,27],[size-.001,size,size+.001])]
        rows=summarize(records,'fine'); result=candidates(rows,records,.05)[0]
        self.assertEqual(result['w2'],.35); self.assertFalse(result['final'])
        self.assertAlmostEqual(result['sd_type_i_error'],.001)
        self.assertEqual(len(result['per_seed']),3)

    def test_whole_workflow_with_mock_training(self):
        # Exercise orchestration and all output files without ANN training or W&B.
        root=Path(__file__).resolve().parents[1]
        cfg=json.loads((root/'configs/ann_fixed200_w2_calibration.json').read_text())
        c=json.loads((root/cfg['base_config']).read_text()); c.update(cfg)
        identities={}; calls=[]
        def fake_dataset(config,t):
            rho=[]; beta=[]
            for r in c['rhos']:
                for b in c['betas']:
                    n=5000 if r==1 else 1000
                    rho.extend([r]*n); beta.extend([b]*n)
            raw=dict(rho=np.array(rho),beta=np.array(beta),y=(np.array(rho)!=1).astype(int))
            identities[t]=id(raw); return raw,123+t
        def fake_train(config,base,stage,t,seed,raw,w,epochs,lr,batch):
            self.assertEqual(id(raw),identities[t]); self.assertEqual((epochs,lr,batch),(200,.001,256))
            calls.append((t,w,seed)); p=base/f'{t}_{w}_{seed}'; p.mkdir()
            size=.02+.1*w+(seed-17)*.00001
            return dict(path=str(p),id=str(len(calls)),url='mock',seconds=1,
                evaluations={'200':dict(overall=dict(loss=.2,accuracy=.8,empirical_size=size,empirical_power=.7,n=120000),
                    by_parameter=[dict(rho=1,beta=b,rejection_rate=size,n=5000) for b in c['betas']])})
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp); state=dict(runs=[],datasets={})
            with patch('scripts.run_ann_fixed200_calibration.dataset',side_effect=fake_dataset), \
                 patch('scripts.run_ann_fixed200_calibration.fingerprint',return_value='fixed'), \
                 patch('scripts.run_ann_fixed200_calibration.train',side_effect=fake_train), \
                 patch('scripts.run_ann_fixed200_calibration.torch.load',return_value=dict(epoch=200,model={'w':torch.ones(1)})):
                execute(c,base,state)
            self.assertEqual(len(calls),99); self.assertEqual(len(set(calls)),99)
            self.assertEqual(state['fine_reused_results'],18)
            self.assertEqual(state['status'],'complete')
            for name in ['coarse_summary.csv','fine_summary.csv','calibration_candidates.json','run_manifest.json']:
                self.assertTrue((base/name).is_file())


if __name__=='__main__': unittest.main()
