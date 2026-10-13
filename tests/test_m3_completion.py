import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pr02.context87.data import ROOT, read_json
from pr02.context87.experiment import METHODS, path
from pr02.context87.m2m3 import check_predictions, load_inputs, thresholds
from pr02.context87.m3_completion import (analyze, grouped_regions, model_diagnostics,
                                         spacing_tables)


class CompletionStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(dict(sample_id=['t1', 't2', 'v1', 'v2', 'unused'],
                                      split=['train', 'train', 'val', 'val', 'test'],
                                      target_log10=[0., 2., 1., 3., 9999.]))
        self.frames = {}
        for subset in ['train', 'val']:
            g = self.data[self.data['split'].eq(subset)]
            for method in METHODS:
                self.frames[(method, subset)] = pd.DataFrame(dict(sample_id=g.sample_id,
                    predicted_value_log10=g.target_log10*.5+.5, prediction_status='ok'))

    def test_baseline_fits_train_only_and_has_known_errors(self):
        baseline, diag, _, ids, mean = model_diagnostics(self.data, self.frames)
        self.assertEqual(mean, 1.)
        self.assertNotIn('unused', set(ids.sample_id))
        row = diag.query('method == "train_mean" and subset == "val"').iloc[0]
        self.assertAlmostEqual(row.r2, -1.)
        self.assertAlmostEqual(row.mae, 1.)
        self.assertAlmostEqual(row.rmse, np.sqrt(2))
        self.assertEqual(row.spearman_status, 'undefined')
        self.assertTrue(np.isnan(row.spearman)); self.assertTrue(np.isnan(row.pearson))
        self.assertEqual(baseline.predicted_value_log10.unique().tolist(), [1.])

    def test_validation_labels_do_not_change_baseline_or_tertiles(self):
        changed = self.data.copy(); changed.loc[changed['split'].ne('train'), 'target_log10'] += 10000
        frames = copy.deepcopy(self.frames)
        self.assertEqual(model_diagnostics(changed, frames)[-1], 1.)
        np.testing.assert_array_equal(thresholds(changed), thresholds(self.data))

    def test_shuffled_predictions_are_aligned_by_id(self):
        changed = {key: value.iloc[::-1] for key, value in self.frames.items()}
        expected = model_diagnostics(self.data, self.frames)[1]
        actual = model_diagnostics(self.data, changed)[1]
        pd.testing.assert_frame_equal(actual, expected)

    def test_failed_prediction_is_not_dropped_or_filled(self):
        frames = copy.deepcopy(self.frames); frame = frames[('thermo', 'val')]
        frame.loc[frame.sample_id.eq('v1'), 'prediction_status'] = 'failed'
        frame.loc[frame.sample_id.eq('v1'), 'predicted_value_log10'] = np.nan
        with self.assertRaisesRegex(ValueError, 'do not drop failures'):
            model_diagnostics(self.data, frames)

    def test_knn_train_score_has_explicit_interpolation_warning(self):
        frames = copy.deepcopy(self.frames)
        frames[('knn', 'train')]['predicted_value_log10'] = [0., 2.]
        _, diagnostics, gaps, _, _ = model_diagnostics(self.data, frames)
        self.assertEqual(diagnostics.query('method == "knn" and subset == "train"').iloc[0].train_score_note, 'self_neighbor_interpolation')
        self.assertEqual(gaps.query('method == "knn"').iloc[0].interpretation_status, 'knn_train_not_generalization_evidence')

    def test_spacer_empty_groups_remain_undefined_and_test_is_excluded(self):
        states = {sid: dict(spacer='A'*17, disc='C'*6) for sid in self.data.sample_id}
        samples, summary, correlations = spacing_tables(self.data, states, thresholds(self.data))
        self.assertEqual(set(samples.subset), {'train', 'val'})
        empty = summary.query('spacer_length == 20')
        self.assertEqual(empty.n.tolist(), [0, 0]); self.assertTrue(empty.median_log10.isna().all())
        self.assertEqual(empty.sample_status.tolist(), ['empty', 'empty'])
        self.assertTrue(correlations.spearman.isna().all())
        self.assertEqual(set(correlations.reason), {'constant_prediction'})

    def test_spacer_quantiles_include_extreme_values(self):
        states = {sid: dict(spacer='A'*17, disc='C'*6) for sid in self.data.sample_id}
        _, summary, _ = spacing_tables(self.data, states, thresholds(self.data))
        g = summary.query('subset == "val" and spacer_length == 17').iloc[0]
        self.assertEqual(g.n, 2); self.assertEqual(g.median_log10, 2.)
        self.assertEqual(g.q25_log10, 1.5); self.assertEqual(g.q75_log10, 2.5)

    def test_region_group_differences_have_correct_direction_and_denominators(self):
        regions = pd.DataFrame([
            dict(sample_id='l', subset='val', annotation_status='canonical_window_assumption', region='-10',
                 target_log10=0., gc_fraction=1., consensus_match_fraction=0., length=6, sequence='GGGGGG'),
            dict(sample_id='h', subset='val', annotation_status='canonical_window_assumption', region='-10',
                 target_log10=3., gc_fraction=0., consensus_match_fraction=1., length=6, sequence='TATAAT'),
            dict(sample_id='x', subset='test', annotation_status='canonical_window_assumption', region='-10',
                 target_log10=3., gc_fraction=1., consensus_match_fraction=0., length=6, sequence='CCCCCC')])
        summary, contrasts, frequency = grouped_regions(regions, [1., 2.])
        self.assertEqual(summary.n.sum(), 2)
        contrast = contrasts.set_index('feature')
        self.assertEqual(contrast.loc['at_fraction', 'high_minus_low'], 1.)
        self.assertEqual(contrast.loc['gc_fraction', 'high_minus_low'], -1.)
        self.assertEqual(contrast.loc['consensus_match_fraction', 'high_minus_low'], 1.)
        high = frequency[frequency.strength_group.eq('high')]
        self.assertTrue(np.allclose(high.groupby('position_1based').fraction.sum(), 1.))
        self.assertEqual(high.query('position_1based == 1 and base == "T"').fraction.iloc[0], 1.)

    def test_existing_run_guard_precedes_expensive_analysis(self):
        (ROOT/'runs').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'runs') as tmp:
            marker = Path(tmp)/'marker'; marker.write_text('preserve')
            with self.assertRaisesRegex(ValueError, 'existing output'):
                analyze({}, Path(tmp))
            self.assertEqual(marker.read_text(), 'preserve')


class CompletionInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_json(ROOT/'project_config.json')
        cls.data, _, _, _ = load_inputs(cls.cfg)
        cls.train = pd.read_csv(path(cls.cfg, 'cnn')/'predictions_train.csv')

    def test_training_predictions_use_training_ids_and_shuffling_is_supported(self):
        frame = check_predictions(self.train.sample(frac=1, random_state=7), self.data, self.cfg, 'cnn', 'train')
        self.assertEqual(len(frame), 8267)
        self.assertEqual(set(frame.sample_id), set(self.data.loc[self.data['split'].eq('train'), 'sample_id']))

    def test_wrong_train_subset_ids_or_labels_are_rejected(self):
        variants = [self.train.iloc[1:].copy()]
        for column, value in [('subset', 'val'), ('true_value_log10', 999.), ('sample_id', 'ecoli87_r000001')]:
            wrong = self.train.copy(); wrong.loc[0, column] = value; variants.append(wrong)
        for frame in variants:
            with self.assertRaises(ValueError): check_predictions(frame, self.data, self.cfg, 'cnn', 'train')

    def test_test_scoring_is_not_supported(self):
        with self.assertRaisesRegex(ValueError, 'Only train/val'):
            check_predictions(self.train, self.data, self.cfg, 'cnn', 'test')


if __name__ == '__main__': unittest.main()
