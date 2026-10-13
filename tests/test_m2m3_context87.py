import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pr02.context87.data import ROOT, coordinate, load, read_json, read_table
from pr02.context87.experiment import fresh, path
from pr02.context87.m2m3 import (check_predictions, error_tables, frame_data, load_inputs,
                               region_features, repo_path, safe_metric, select_cases,
                               strength_groups, thresholds)


class M2M3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = read_json(ROOT/'project_config.json')
        cls.data, cls.transform, cls.frames, cls.states = load_inputs(cls.cfg)

    def test_shuffled_predictions_join_by_id(self):
        original = self.frames['cnn']
        shuffled = original.sample(frac=1, random_state=12)
        result = check_predictions(shuffled, self.data, self.cfg, 'cnn')
        pd.testing.assert_frame_equal(result, original)

    def test_missing_duplicate_unknown_and_wrong_version_ids_rejected(self):
        original = self.frames['ridge']
        variants = [original.iloc[1:].copy(), pd.concat([original, original.iloc[:1]])]
        wrong = original.copy(); wrong.loc[0, 'sample_id'] = 'unknown'; variants.append(wrong)
        for column, value in [('data_version', 'ecoli50_strength_v1'), ('split_id', 'wrong'),
                              ('run_id', 'wrong'), ('subset', 'test'), ('sequence_length', 50)]:
            wrong = original.copy(); wrong.loc[0, column] = value; variants.append(wrong)
        for frame in variants:
            with self.subTest(columns=frame.shape), self.assertRaises(ValueError):
                check_predictions(frame, self.data, self.cfg, 'ridge')

    def test_wrong_truth_and_raw_log_scale_rejected(self):
        for column in ['true_value', 'true_value_log10', 'predicted_value']:
            frame = self.frames['svr'].copy(); frame.loc[0, column] += 1
            with self.subTest(column=column), self.assertRaises(ValueError):
                check_predictions(frame, self.data, self.cfg, 'svr')
        frame = self.frames['thermo'].copy(); frame.loc[0, 'target_scale_model'] = 'tool_raw'
        with self.assertRaises(ValueError): check_predictions(frame, self.data, self.cfg, 'thermo')

    def test_failed_request_preserved_and_common_set_shrinks(self):
        frames = {m: f.copy() for m, f in self.frames.items()}
        sid = frames['thermo'].loc[0, 'sample_id']
        frame = frames['thermo']; frame.loc[0, 'prediction_status'] = 'failed'
        frame.loc[0, ['predicted_value', 'predicted_value_log10']] = np.nan
        frame['error_reason'] = ''; frame.loc[0, 'error_reason'] = 'synthetic failure for regression test'
        frames['thermo'] = check_predictions(frame, self.data, self.cfg, 'thermo')
        tables, _, common = error_tables(self.data, frames)
        self.assertEqual(len(frames['thermo']), 1768)
        self.assertEqual(len(common), 1767); self.assertNotIn(sid, common)
        self.assertEqual(tables['coverage'].query('method == "thermo"').n_failed.iloc[0], 1)
        frame.loc[0, 'predicted_value'] = 0
        with self.assertRaises(ValueError): check_predictions(frame, self.data, self.cfg, 'thermo')

    def test_train_only_thresholds_boundary_and_no_clipping(self):
        changed = self.data.copy()
        changed.loc[changed['split'].ne('train'), 'target_log10'] = 100
        np.testing.assert_array_equal(thresholds(changed), thresholds(self.data))
        a, b = thresholds(self.data)
        self.assertEqual(list(strength_groups(np.array([a, a+1e-8, b, b+1e-8]), [a, b])), ['low', 'mid', 'mid', 'high'])
        expected = (self.data.target_log10-self.transform['min_log10'])/(self.transform['max_log10']-self.transform['min_log10'])
        np.testing.assert_allclose(expected, self.data.target_normalized)
        self.assertGreater(self.data.target_normalized.max(), 1)

    def test_region_evidence_coordinates_and_state_order(self):
        regions = region_features(self.data.iloc[:3], self.states)
        self.assertEqual(set(regions.annotation_status), {'canonical_window_assumption', 'model_inferred'})
        self.assertEqual(len(regions), 36)
        self.assertEqual([coordinate(i) for i in [0, 17, 66, 67, 86]], [-67, -50, -1, 1, 20])
        source = self.data.set_index('sample_id')
        for r in regions.itertuples():
            self.assertEqual(r.sequence, source.loc[r.sample_id, 'sequence'][r.start_0based:r.end_0based_exclusive])
        bad = copy.deepcopy(self.states); sid = self.data.sample_id.iloc[0]
        bad[sid]['hex10_position'] = [54, 60]
        with self.assertRaises(ValueError): region_features(self.data.iloc[:1], bad)

    def test_cases_and_metric_formula_independently(self):
        tables, _, common = error_tables(self.data, self.frames)
        selection, _ = select_cases(tables['prediction_errors'], common, self.cfg['seed'])
        self.assertEqual(len(selection), 80)
        self.assertEqual(selection.selection_rule.value_counts().to_dict(), {'largest_disagreement': 50, 'largest_error': 25, 'random_common': 5})
        second, _ = select_cases(tables['prediction_errors'].sample(frac=1, random_state=8), common, self.cfg['seed'])
        pd.testing.assert_frame_equal(selection, second)
        f = self.frames['cnn']; y = f.true_value_log10.to_numpy(); p = f.predicted_value_log10.to_numpy()
        r2 = 1-((p-y)**2).sum()/((y-y.mean())**2).sum()
        saved = tables['metrics'].query('method == "cnn" and target_scale == "log10" and evaluation_subset == "common"').set_index('metric').value
        self.assertAlmostEqual(saved['r2'], r2, places=12)
        self.assertAlmostEqual(saved['mae'], sum(abs(p-y))/len(y), places=12)
        expected = sorted(zip(f.sample_id, abs(p-y)), key=lambda r: (-r[1], r[0]))[:5]
        actual = selection.query('selection_rule == "largest_error" and selector == "cnn"').sample_id.tolist()
        self.assertEqual(actual, [r[0] for r in expected])

    def test_empty_or_constant_metrics_not_filled_with_zero(self):
        for metric, y, p in [('r2', [], []), ('r2', [1, 1], [1, 2]), ('spearman', [1, 2], [1, 1])]:
            value, status, reason = safe_metric(metric, np.array(y), np.array(p))
            self.assertTrue(np.isnan(value)); self.assertEqual(status, 'undefined'); self.assertTrue(reason)

    def test_paths_and_overwrite_guard(self):
        with self.assertRaises(ValueError): repo_path('../outside')
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as temp:
            directory = Path(temp); (directory/'present.txt').write_text('test')
            with self.assertRaises(ValueError): fresh(directory)


if __name__ == '__main__': unittest.main()
