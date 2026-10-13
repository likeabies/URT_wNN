import ast
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scripts import run_ann_independent_test as m


class IndependentANNTests(unittest.TestCase):
    def test_locked_design(self):
        self.assertEqual(m.SELECTED,{50:.375,100:.4375,250:.6375})
        self.assertEqual(m.SEEDS,[7,17,27]);self.assertEqual(m.TEST_SEED,20261008)
        self.assertEqual(m.RHOS,[1,.95,.9,.8]);self.assertEqual(m.BETAS,[.9,.6,.3,0])
        self.assertEqual(m.N,1000);self.assertEqual(len(m.paper_values()),48)

    def test_generation_arguments_counts_and_independent_stream(self):
        # Validate design through a fake generator, without sampling the real test set.
        def fake(t,rhos,betas,n0,n1,seed):
            self.assertEqual((rhos,betas,n0,n1),(m.RHOS,m.BETAS,1000,1000))
            self.assertNotEqual(seed,int(np.random.SeedSequence([20191007,t]).generate_state(1)[0]))
            rho=np.repeat(rhos,4000);beta=np.tile(np.repeat(betas,1000),4)
            x=np.tile(np.array([1.,-1.],dtype=np.float32),(16000,t//2))
            return dict(x=x,y=(rho!=1).astype(int),rho=rho,beta=beta)
        with patch.object(m,'make_ann_dataset',side_effect=fake) as generate:
            for t in m.SELECTED:
                raw,_=m.make_test(t);self.assertEqual(len(raw['y']),16000)
            self.assertEqual(generate.call_count,3)

    def test_all_models_share_same_data(self):
        raw=object();seen=[]
        def evaluate(shared,item):
            self.assertIs(shared,raw);seen.append(item['model_seed']);return item['model_seed']
        result=m.evaluate_checkpoints(raw,[{'model_seed':s} for s in [27,7,17]],evaluate)
        self.assertEqual(result,[7,17,27]);self.assertEqual(seen,[7,17,27])
        with self.assertRaises(RuntimeError):m.evaluate_checkpoints(raw,[{'model_seed':7}],evaluate)

    def test_summary_units_and_sample_sd(self):
        rows=[dict(T=t,rho=rho,beta=beta,seed=s,rejection_rate=rate) for t in m.SELECTED
              for rho in m.RHOS for beta in m.BETAS for s,rate in zip(m.SEEDS,[.01,.02,.03])]
        means,comparison,summary,overall=m.compare(rows)
        self.assertEqual(len(means),48);self.assertAlmostEqual(means[0]['mean'],.02)
        self.assertAlmostEqual(means[0]['SD'],.01)
        self.assertAlmostEqual(comparison[0]['our_mean_percent'],2.)
        self.assertAlmostEqual(comparison[0]['our_SD_percentage_points'],1.)
        self.assertAlmostEqual(comparison[0]['difference_percentage_points'],2-4.9)
        self.assertIsNone(overall['assessment']);self.assertEqual(overall['power_cells_increasing_with_T'],0)
        with self.assertRaises(RuntimeError):m.compare(rows[:-1])

    def test_no_training_or_selection_entrypoints(self):
        source=Path(m.__file__).read_text();tree=ast.parse(source)
        names={n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)}
        attrs={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        self.assertFalse(names & {'train','train_model','train_one_epoch','select_epoch','polynomial','candidates'})
        self.assertFalse(attrs & {'backward','step','train'})
        self.assertEqual(source.count('SELECTED ='),1)


if __name__=='__main__':unittest.main()
