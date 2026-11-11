import tempfile
from pathlib import Path
import unittest

from scripts.run_rnn_paper2019_fixed200 import (Workflow, read, CONFIG, bracket, extend,
                                               regression, slopes)


class SyntheticWorkflow(Workflow):
    def __init__(self, base):
        super().__init__(read(CONFIG), base, 'cpu')
        self.fit_calls = 0

    def fit(self, t, seed, w, purpose='grid', **kwargs):
        self.fit_calls += 1
        key = f'{purpose}-{t}-{seed}-{w.hex()}'
        fit = dict(key=key, T=t, model_seed=seed, w2=w,
                   training={'overall': {'empirical_size': .02+.08*w,
                                         'empirical_power': .4}, 'by_parameter': []})
        self.state['runs'][key] = fit
        return fit

    def score(self, fit, cohort):
        assert cohort == 'validation', 'Test accessed during calibration!'
        return {'overall': {'empirical_size': .018+.1*fit['w2'],
                            'empirical_power': 0}, 'by_parameter': []}


class CalibrationTest(unittest.TestCase):
    def test_direction_and_exact_roots(self):
        def pts(ys):
            return [dict(w2=w, mean_size=y) for w,y in zip([.2,.25],ys)]
        self.assertEqual(extend(pts([.02,.03])), .3)
        self.assertEqual(extend(pts([.07,.08])), .15)
        self.assertEqual(extend(pts([.08,.07])), .3)
        self.assertEqual(bracket(pts([.04,.06])), [.2,.25])
        result=regression([dict(w2=.2,size=.03),dict(w2=.4,size=.07)])
        self.assertAlmostEqual(result['w2_star'], .3)
        self.assertFalse(result['clipping'])
        with self.assertRaises(RuntimeError):
            regression([dict(w2=.2,size=.03),dict(w2=.4,size=.03)])

    def test_selection_test_isolation_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            wf=SyntheticWorkflow(Path(directory))
            train=wf.calibrate('training')
            self.assertTrue(all(abs(s['w2']-.375)<1e-12 for s in train))
            count=wf.fit_calls
            self.assertEqual(wf.calibrate('training'),train)
            self.assertEqual(wf.fit_calls,count)
            val=wf.calibrate('validation')
            self.assertTrue(all(abs(s['w2']-.325)<1e-12 for s in val))
            selected=wf.linear('training')
            self.assertEqual(len(selected),9)
            for choice in selected:
                self.assertEqual(len(choice['regression']['points']),8)
                self.assertAlmostEqual(choice['w2'],.375)

    def test_trajectory_windows(self):
        history=[dict(epoch=i,train_loss=2-.003*i) for i in range(1,201)]
        for value in slopes(history).values():
            self.assertAlmostEqual(value,-.003)


if __name__=='__main__':
    unittest.main()
