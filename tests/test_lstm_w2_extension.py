"""Range-extension scheduling and output checks without production training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import run_lstm_w2_extension as ext
from scripts.run_lstm_fixed200_calibration import train


class RangeExtensionTests(unittest.TestCase):
    def test_interval_selection_and_nine_points(self):
        points = [dict(w2=w, mean_type_i_error=s) for w,s in
                  [(.2,.02),(.25,.06),(.3,.03),(.35,.049),(.4,.06)]]
        plan = ext.fine_plan(points)
        self.assertEqual(plan['interval'], [.35,.4])
        self.assertEqual(len(plan['w2_values']), 9)
        self.assertEqual(plan['w2_values'][::8], [.35,.4])
        self.assertTrue(plan['multiple_crossings'])
        self.assertEqual(len(plan['crossing_intervals']), 3)
        self.assertFalse(plan['target_not_bracketed_through_1_00'])
        self.assertIs(ext.train, train)

    def run_schedule(self, crossing, expected_fine, plot=False):
        c = json.loads(ext.CONFIG.read_text())
        c.update(model_seeds=[7,17,27], data_seed=20191007)
        datasets = {str(t):dict(derived_seed=t+100, fingerprint=str(t)) for t in [50,100,250]}
        old = []
        def record(t,w,s,stage):
            size = .05 + .04*(w-crossing) + (s-17)*.000001
            beta = [dict(rho=1., beta=0., rejection_rate=size, n=5000)]
            return dict(T=t,w2=w,model_seed=s,stage=stage,epochs=200,status='finished',
                        metrics=dict(empirical_size=size,empirical_power=.6),
                        beta_type_i_error=beta,by_parameter=beta,path=f'historical/{t}/{w}/{s}',
                        wandb_url='mock',runtime_seconds=100.)
        for t in [50,100,250]:
            for w in [.2,.25,.3,.35,.4,.45,.5,.55]:
                old.extend(record(t,w,s,'coarse') for s in c['model_seeds'])
            for w in ([.4125,.425,.4375] if t==100 else [.5125,.525,.5375]):
                old.extend(record(t,w,s,'fine') for s in c['model_seeds'])
        original = json.dumps(old, sort_keys=True)
        frozen = [dict(T=100,w2=.425,final=False,original_marker='keep verbatim')]
        raw = {t:object() for t in c['T_values']}
        calls = []
        def fake_train(config,base,stage,t,s,data,w,epochs):
            self.assertIs(data,raw[t])
            self.assertEqual(epochs,200)
            self.assertIn(t,[50,250])
            self.assertNotIn((t,w,s),[ext.key(r) for r in old])
            calls.append((t,w,s))
            p=base/f'{t}_{w}_{s}'
            p.mkdir()
            r=record(t,w,s,stage)
            return dict(path=str(p),id=str(len(calls)),url='mock',seconds=100.,
                        evaluations={'200':dict(overall=r['metrics'],by_parameter=r['by_parameter'])})
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            state=dict(runs=[],datasets={})
            with patch.object(ext,'dataset',side_effect=lambda c,t:(raw[t],t+100)) as gen, \
                 patch.object(ext,'fingerprint',side_effect=lambda x:next(str(t) for t,v in raw.items() if v is x)), \
                 patch.object(ext,'train',side_effect=fake_train), \
                 patch.object(ext,'save_plots',wraps=ext.save_plots if plot else lambda *a:None):
                ext.execute(c,base,state,dict(runs=old,datasets=datasets),frozen)
            self.assertEqual(gen.call_count,2)
            self.assertEqual(len(calls),54+expected_fine)
            self.assertEqual(len(set(calls)),len(calls))
            self.assertEqual(set(calls[:54]),{(t,w,s) for t in c['T_values'] for w in c['new_coarse_w2'] for s in c['model_seeds']})
            self.assertEqual(state['new_fine_fits'],expected_fine)
            self.assertEqual(state['fine_checkpoint_reuses'],54-expected_fine)
            self.assertEqual(state['status'],'complete')
            self.assertFalse(state['final_independent_test'])
            selected=json.loads((base/'calibration_candidates.json').read_text())
            self.assertEqual(next(r for r in selected if r['T']==100),frozen[0])
            integrated=json.loads((base/'integrated_records.json').read_text())
            self.assertEqual(len(integrated),99+54+expected_fine)
            self.assertEqual(len({ext.key(r) for r in integrated}),len(integrated))
            for name in ['coarse_summary.csv','fine_summary.csv','seed_metrics.csv',
                         'integrated_calibration_summary.csv','fine_plan.json','fine_records.json']:
                self.assertGreater((base/name).stat().st_size,0)
            if plot:
                for name in ['w2_size.png','w2_size.pdf']:
                    self.assertGreater((base/name).stat().st_size,1000)
            if crossing>1:
                self.assertEqual(state['unbracketed_T'],[50,250])
                for plan in state['fine_plans'].values():
                    self.assertEqual(plan['interval'],[.95,1.])
                    self.assertLessEqual(max(plan['w2_values']),1.)
            self.assertEqual(json.dumps(old,sort_keys=True),original)

    def test_all_coarse_before_fine_and_endpoint_reuse(self):
        self.run_schedule(.675,42,plot=True)

    def test_historical_fine_checkpoints_reduce_new_fits(self):
        self.run_schedule(.53,24)

    def test_no_crossing_keeps_bounded_historical_fallback(self):
        self.run_schedule(1.1,42)


if __name__=='__main__':
    unittest.main()
