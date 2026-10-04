"""Checks that protect model comparisons from leakage and silent ID loss."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import DATASET, ROOT, load_data
import common
from eda import label_transform
from error_analysis import attach_calibration, checked_predictions, metric_value
from region_annotation import REGION_TYPES, build_annotations, summarize_regions

INPUTS = ROOT/'analysis_m2m3/inputs/thermo_member_b_89fcc66'
KNN = ROOT/'KNN/results/predictions_knn_ecoli_full_val.csv'


class AnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=load_data()

    def test_transform_uses_train_and_does_not_clip(self):
        meta,scope=label_transform(self.data)
        tr=self.data[self.data['split'].eq('train')]
        self.assertEqual(meta['min_log10'],tr.target_log10.min())
        self.assertEqual(meta['max_log10'],tr.target_log10.max())
        self.assertFalse(meta['clip'])
        self.assertEqual(scope,'analysis_only_pending_team_adoption')
        z=(self.data.target_log10-meta['min_log10'])/(meta['max_log10']-meta['min_log10'])
        self.assertTrue((z>1).any())  # actual held-out maximum is outside train range

    def test_dirty_repository_captures_shared_source_patch(self):
        for content in (b'diff --git a/pr02/comparison.py b/pr02/comparison.py\n', b''):
            with self.subTest(content=content), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp).resolve()
                git_results = iter(['a' * 40, ' M pr02/comparison.py\n', content])
                real_check_output = common.subprocess.check_output
                def side_effect(args, *args_, **kwargs):
                    if args and args[0] == 'git':
                        return next(git_results)
                    return real_check_output(args, *args_, **kwargs)
                with patch.object(common, 'ROOT', root), patch.object(common, 'GIT', 'git'), \
                        patch.object(common.subprocess, 'check_output', side_effect=side_effect) as git:
                    run = common.Run(root / 'run', 'M3', 'test', self.data, [], {})
                    self.assertTrue(run.meta['dirty_worktree'])
                    self.assertEqual((run.output / 'source_patch.diff').read_bytes(), content)
                    self.assertEqual(run.meta['source_patch'], 'run/source_patch.diff')
                    self.assertEqual(git.call_args.args[0], ['git', 'diff', 'HEAD'])

    def test_modified_shared_transform_is_rejected(self):
        meta,_=label_transform(self.data)
        meta['max_log10']+=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'transform.json'
            path.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError,'extrema mismatch'):
                label_transform(self.data,path)

    def test_prediction_alignment_is_by_id_not_row(self):
        p=pd.read_csv(KNN).sample(frac=1,random_state=12)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'shuffled.csv'
            p.to_csv(path,index=False)
            checked=checked_predictions(path,self.data)
            self.assertEqual(set(checked.sample_id),set(p.sample_id))

    def test_missing_requested_id_rejected(self):
        p=pd.read_csv(KNN).iloc[1:]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'missing.csv';p.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'missing requested IDs'):
                checked_predictions(path,self.data)

    def test_unknown_id_and_true_label_rejected(self):
        p=pd.read_csv(KNN)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.csv'
            p.loc[0,'sample_id']='unknown';p.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'unknown or wrong-subset'):
                checked_predictions(path,self.data)
            p=pd.read_csv(KNN);p.loc[0,'true_value']+=1;p.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'true label mismatch'):
                checked_predictions(path,self.data)

    def test_raw_thermo_is_not_numeric_strength(self):
        p=pd.read_csv(INPUTS/'thermo_val.csv')
        p.loc[p.prediction_status.eq('ok'),'predicted_value']=p.loc[p.prediction_status.eq('ok'),'predicted_tx_rate']
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'raw_bad.csv';p.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'Uncalibrated Tx_rate'):
                checked_predictions(path,self.data)

    def test_calibration_verified_and_failed_request_preserved(self):
        raw=checked_predictions(INPUTS/'thermo_val.csv',self.data)
        joined,qc=attach_calibration(raw,INPUTS/'thermo_val_calibrated.csv',INPUTS/'thermo_calibration.json',INPUTS/'thermo_train.csv',self.data)
        self.assertEqual(len(joined),1783)
        self.assertEqual(int(joined.prediction_status.eq('failed').sum()),1)
        f=joined[joined.prediction_status.eq('failed')].iloc[0]
        self.assertEqual(f.sample_id,'ecoli50_r003290')
        self.assertTrue(pd.isna(f.predicted_value_log10))
        self.assertEqual(qc['status'],'pass')

    def test_constant_rank_metric_is_undefined(self):
        value,status,reason=metric_value('spearman',np.arange(5.),np.ones(5))
        self.assertIsNone(value)
        self.assertEqual(status,'undefined')
        self.assertIn('constant',reason)

    def test_calibration_accepts_new_full_request_table(self):
        raw=checked_predictions(INPUTS/'thermo_val.csv',self.data)
        success=pd.read_csv(INPUTS/'thermo_val_calibrated.csv')
        failed=raw[raw.prediction_status.eq('failed')].copy()
        failed['calibration_id']=success.calibration_id.iloc[0]
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'full_calibrated.csv'
            pd.concat([success,failed],ignore_index=True).to_csv(path,index=False)
            joined,qc=attach_calibration(raw,path,INPUTS/'thermo_calibration.json',INPUTS/'thermo_train.csv',self.data)
            self.assertEqual(len(joined),len(raw))
            self.assertEqual(int(joined.prediction_status.eq('failed').sum()),1)

    def test_missing_annotations_leave_coordinates_empty(self):
        annotations=build_annotations(self.data)
        self.assertEqual(len(annotations),len(self.data)*len(REGION_TYPES))
        self.assertTrue(annotations.annotation_status.eq('missing').all())
        self.assertTrue(annotations.strand.eq('unknown').all())
        self.assertTrue(annotations.start_0index.isna().all())
        self.assertTrue(annotations.end_0index_exclusive.isna().all())
        self.assertTrue(annotations.score.isna().all())
        summary=summarize_regions(self.data,annotations)
        self.assertEqual(len(summary),len(REGION_TYPES))
        for row in summary:
            self.assertEqual(row['n_reliable_external_with_coordinates'],0)
            self.assertEqual(row['coverage'],0)
            self.assertEqual(row['relationship_status'],'not_computed_no_reliable_coordinates')
            self.assertNotIn('spearman_gc_vs_log10',row)

    def test_reliable_coordinates_can_report_spearman(self):
        samples=pd.DataFrame({
            'sample_id':['s1','s2','s3','s4'],
            'source_row':[1,2,3,4],
            'sequence':['AAAAAA'+'C'*44,'GGGGGG'+'A'*44,'AAAGGG'+'T'*44,'GGGAAA'+'C'*44],
            'target_log10':[1.0,3.0,2.0,2.2],
            'dataset_id':'course_ecoli50_strength',
            'data_version':'ecoli50_strength_v1',
            'schema_version':'2.0.0',
        })
        annotations=build_annotations(samples)
        mask=annotations.region_type.eq('minus10')
        annotations.loc[mask,'annotation_status']='reliable_external'
        annotations.loc[mask,'strand']='forward'
        annotations.loc[mask,'start_0index']=0
        annotations.loc[mask,'end_0index_exclusive']=6
        annotations.loc[mask,'evidence_ref']='unit-test'
        summary={row['region_type']:row for row in summarize_regions(samples,annotations)}
        self.assertEqual(summary['minus10']['relationship_status'],'computed')
        self.assertEqual(summary['minus10']['n_samples_denominator'],4)
        self.assertEqual(summary['minus10']['coverage'],1)
        self.assertIn('spearman_gc_vs_log10',summary['minus10'])
        self.assertEqual(summary['TSS']['relationship_status'],'not_computed_no_reliable_coordinates')
        self.assertNotIn('spearman_gc_vs_log10',summary['TSS'])

    def test_tool_inferred_boxes_are_not_used_for_strength(self):
        samples=self.data.iloc[:3].copy()
        annotations=build_annotations(samples)
        mask=annotations.region_type.eq('minus35')
        annotations.loc[mask,'annotation_status']='tool_inferred'
        annotations.loc[mask,'strand']='forward'
        annotations.loc[mask,'start_0index']=0
        annotations.loc[mask,'end_0index_exclusive']=6
        annotations.loc[mask,'evidence_ref']='unit-test'
        row=next(item for item in summarize_regions(samples,annotations) if item['region_type']=='minus35')
        self.assertEqual(row['n_tool_inferred_not_used'],3)
        self.assertEqual(row['n_reliable_external_with_coordinates'],0)
        self.assertNotIn('spearman_gc_vs_log10',row)


if __name__=='__main__':
    unittest.main()
