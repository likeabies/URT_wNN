import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts import run_ann_fixed200_validation_calibration as m


class FakeRun:
    id='mock';url='mock-url'
    def __enter__(self):return self
    def __exit__(self,*args):return False
    def log(self,*args):pass


class HeldoutTests(unittest.TestCase):
    def points(self,pairs):return {w:dict(w2=w,mean_size=a) for w,a in pairs}
    def test_endpoint_first_bracket_and_exact_target(self):
        for values in [[(.32,.04),(.37,.06)],[ (.32,.05),(.37,.06)]]:
            a=m.next_action(self.points(values),[.32,.37]);self.assertEqual(a,{'action':'fine','interval':[.32,.37]})

    def test_adaptive_one_point_both_directions_and_negative_slope(self):
        cases=[([.03,.04],.42),([.06,.07],.27),([.07,.06],.42),([.04,.03],.27)]
        for values,expected in cases:
            a=m.next_action(self.points(list(zip([.32,.37],values))),[.32,.37])
            self.assertEqual(a['action'],'extend');self.assertEqual(a['w2'],expected)

    def test_midpoint_then_stop_or_confirm(self):
        points=self.points([(.32,.03),(.37,.0301)])
        a=m.next_action(points,[.32,.37]);self.assertEqual(a['action'],'diagnostic');self.assertEqual(a['w2'],.345)
        points[.345]=dict(w2=.345,mean_size=.03005)
        self.assertEqual(m.next_action(points,[.32,.37],True)['action'],'extend')
        points[.345]['mean_size']=.029
        self.assertEqual(m.next_action(points,[.32,.37],True)['action'],'stop')

    def test_safety_boundary_and_tie(self):
        a=m.next_action(self.points([(.9,.03),(.98,.04)]),[.9,.98]);self.assertEqual(a['w2'],1.)
        self.assertEqual(m.next_action(self.points([(.98,.03),(1.,.04)]),[.98,1.])['action'],'stop')
        p=self.points([(.3,.04),(.4,.04)])
        self.assertEqual(m.choose_candidate(p,[.3,.4])['w2'],.3)

    def test_calibration_reuses_endpoints_and_excludes_validation_from_training(self):
        calls=[];val=object();raw=object()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'training_runs').mkdir()
            old=root/'old.pt';old.write_text('checkpoint')
            index={(50,.32,s):dict(model_seed=s) for s in m.SEEDS}
            def train(c,base,stage,t,seed,data,w,epochs,lr,batch):
                self.assertIs(data,raw);self.assertIsNot(data,val)
                self.assertEqual((t,epochs,lr,batch),(50,200,.001,256));calls.append((stage,w,seed))
                p=base/f'{w}_{seed}';p.mkdir();(p/'epoch_200.pt').write_text('checkpoint')
                return dict(path=str(p),id='mock',url='mock')
            def ev(data,item):
                self.assertIs(data,val)
                size=.04+(item['w2']-.32)*.25
                return {'overall':{'empirical_size':size},'by_parameter':[dict(rho=1,beta=b,n=5000,rejection_rate=size) for b in m.VAL_BETAS]}
            state=dict(records=[],points={'50':[]},bracket_history=[],datasets={})
            with patch.object(m,'OUT',root),patch.object(m,'train',side_effect=train),patch.object(m,'evaluate',side_effect=ev),patch.object(m,'fingerprint',return_value='fp'),patch.object(m,'validate_reuse',return_value=dict(checkpoint=str(old),checkpoint_sha256=m.sha(old))),patch.object(m.wandb,'init',return_value=FakeRun()),patch.object(m.wandb,'Table',return_value='table'):
                c=m.calibrate_t(50,{},raw,val,index,state)
            self.assertEqual(c['fine_grid'],[.32,.3325,.345,.3575,.37])
            self.assertEqual(len(calls),12);self.assertEqual(len(set((w,s) for _,w,s in calls)),12)
            self.assertEqual(c['reused_checkpoint_count'],3);self.assertEqual(c['newly_trained_checkpoint_count'],12)
            self.assertEqual([x[0] for x in calls[:3]],['endpoint']*3)
            self.assertTrue(all(x[0]=='fine' for x in calls[3:]))
            self.assertEqual(len(c['selected_checkpoints']),3)

    def test_new_test_only_after_selection_and_shares_exact_data(self):
        candidates=[dict(T=t,selected_w2=.3,selected_checkpoints=[dict(T=t,w2=.3,model_seed=s,checkpoint='chosen') for s in m.SEEDS]) for t in m.INITIAL]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'validation_candidates.json').write_text(json.dumps(candidates))
            state=dict(status='selection_complete',test_runs=[],datasets={str(t):dict(training_fingerprint='train',validation_fingerprint='val') for t in m.INITIAL})
            identities={};seen=[]
            def generate(t,rhos,betas,n0,n1,seed):
                self.assertEqual((rhos,betas,n0,n1),(m.RHOS,m.BETAS,1000,1000));self.assertEqual(seed,m.derived(20261014,t))
                data={'y':np.zeros(16000)};identities[t]=data;return data
            def ev(raw,item):
                self.assertIs(raw,identities[item['T']]);self.assertEqual(item['checkpoint'],'chosen');seen.append((item['T'],item['model_seed']))
                return dict(overall=dict(empirical_size=.05,empirical_power=.7),by_parameter=[dict(rho=r,beta=b,n=1000,rejection_rate=.05 if r==1 else .7) for r in m.RHOS for b in m.BETAS])
            with patch.object(m,'OUT',root),patch.object(m,'make_ann_dataset',side_effect=generate),patch.object(m,'evaluate',side_effect=ev),patch.object(m,'fingerprint',return_value='test'),patch.object(m.wandb,'init',return_value=FakeRun()),patch.object(m.wandb,'Table',return_value='table'),patch.object(m,'train',side_effect=AssertionError('No test-phase training')):
                rows=m.fresh_test(candidates,state)
                self.assertEqual(len(rows),144);self.assertEqual(len(seen),9)
                state['status']='calibration_running'
                with self.assertRaises(RuntimeError):m.fresh_test(candidates,state)
        self.assertEqual(len({m.TRAIN_SEED,m.VAL_SEED,m.OLD_TEST_SEED,m.NEW_TEST_SEED}),4)


if __name__=='__main__':unittest.main()
