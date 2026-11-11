import json
import unittest
from pathlib import Path

from scripts import run_ann_seedwise_linear_calibration as m


class SeedwiseLinearCalibrationTests(unittest.TestCase):
    def test_linear_root_is_unclipped(self):
        result = m.regression([0.2, 0.4, 0.6], [0.01, 0.03, 0.05])
        self.assertAlmostEqual(result['b0'], -0.01)
        self.assertAlmostEqual(result['b1'], 0.1)
        self.assertAlmostEqual(result['r_squared'], 1.0)
        self.assertAlmostEqual(result['w2_star'], 0.6)
        extrapolated = m.regression([0.2, 0.3], [0.01, 0.02])
        self.assertAlmostEqual(extrapolated['w2_star'], 0.6)

    def test_existing_broad_points_map_to_exactly_18_seed_method_fits(self):
        fits, train_manifest, extension_manifest, validation_manifest = m.load_regression_inputs()
        self.assertEqual(len(fits), 18)
        self.assertEqual(len({(r['T'], r['model_seed'], r['calibration_method']) for r in fits}), 18)
        for row in fits:
            self.assertEqual(row['model_seed'] in m.SEEDS, True)
            if row['calibration_method'] == 'train':
                self.assertEqual(row['n_points'], 10 if row['T'] == 250 else 8)
                if row['T'] == 250:
                    self.assertEqual(row['w2_values'][-2:], [0.6, 0.65])
            else:
                expected = {50: 2, 100: 3, 250: 3}[row['T']]
                self.assertEqual(row['n_points'], expected)
        self.assertEqual(train_manifest['status'], 'complete')
        self.assertEqual(extension_manifest['status'], 'complete')
        self.assertEqual(validation_manifest['status'], 'complete')

    def test_validation_mapping_uses_saved_bracketing_stages_only(self):
        manifest = json.loads(m.VALIDATION_DIR.joinpath('run_manifest.json').read_text())
        records = [r for r in manifest['records'] if r['stage'] in ('endpoint', 'bracket_extension')]
        expected = {50: [0.32, 0.37], 100: [0.24, 0.32, 0.4], 250: [0.28, 0.4, 0.52]}
        for t, values in expected.items():
            actual = sorted({r['w2'] for r in records if r['T'] == t})
            self.assertEqual(actual, values)

    def test_official_test_fingerprints_follow_saved_final_manifest(self):
        official = json.loads((m.ROOT/'independent_test/test_manifest.json').read_text())
        self.assertEqual(official['status'], 'complete')
        for t in m.TS:
            self.assertEqual(official['selected_w2'][str(t)], {50: 0.375, 100: 0.4375, 250: 0.6375}[t])
            evaluation = json.loads((m.ROOT/'independent_test'/f'evaluation_T{t}_seed7.json').read_text())
            self.assertEqual(evaluation['config']['test_fingerprint'], m.EXPECTED_TEST_FP[t])
        self.assertNotEqual(m.PROMPT_TEST_FP[50], m.EXPECTED_TEST_FP[50])

    def test_locked_config_and_one_fit_per_cell(self):
        config = json.loads(m.CONFIG_PATH.read_text())
        self.assertEqual(config['expected_new_model_count'], 18)
        self.assertFalse(config['clip_w2'])
        self.assertFalse(config['project_to_existing_grid'])
        self.assertFalse(config['test_based_selection'])
        self.assertEqual(len(m.TS) * len(m.SEEDS) * 2, 18)


if __name__ == '__main__':
    unittest.main()
