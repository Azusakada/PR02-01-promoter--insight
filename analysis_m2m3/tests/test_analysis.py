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
from map_81bp import LAYOUT, build_mapped_annotations, classify_matches, index_windows, map_interval

INPUTS = ROOT/'analysis_m2m3/inputs/thermo_member_b_89fcc66'
KNN = ROOT/'KNN/results/predictions_knn_ecoli_full_val.csv'


class AnalysisTests(unittest.TestCase):
    def test_normalized_cnn_prediction_must_use_shared_transform(self):
        from pr02.common import read_json
        cfg=read_json(ROOT/'project_config.json')
        path=next(ROOT/p for p in cfg['validation_predictions'] if '/validation/predictions_cnn.csv' in p)
        pred=pd.read_csv(path)
        pred.loc[0,'predicted_value_normalized']+=0.1
        with tempfile.TemporaryDirectory() as tmp:
            bad=Path(tmp)/'wrong_transform.csv'; pred.to_csv(bad,index=False)
            with self.assertRaisesRegex(ValueError,'预测反变换'):
                checked_predictions(bad,self.data)

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

    def _layout_promoter(self):
        sequence = 'A'*25 + 'TTGACA' + 'C'*17 + 'TATAAT' + 'G'*6 + 'A' + 'T'*20
        self.assertEqual(len(sequence), 81)
        self.assertEqual(sequence[LAYOUT['minus35'][0]:LAYOUT['minus35'][1]], 'TTGACA')
        self.assertEqual(sequence[LAYOUT['minus10'][0]:LAYOUT['minus10'][1]], 'TATAAT')
        return sequence

    def test_unique_forward_match_transfers_minus10(self):
        promoter = self._layout_promoter()
        window = promoter[10:60]
        samples = pd.DataFrame({
            'sample_id':['s1'], 'source_row':[1], 'sequence':[window],
            'dataset_id':'course_ecoli50_strength', 'data_version':'ecoli50_strength_v1', 'schema_version':'2.0.0',
        })
        ecoli = pd.DataFrame({'seq_id':[1], 'seq':[promoter], 'label':[1]})
        audit, usable = classify_matches(samples, index_windows(ecoli), {})
        self.assertEqual(audit.match_status.iloc[0], 'position_consistent_positive')
        annotations = build_mapped_annotations(samples, usable)
        minus10 = annotations[annotations.region_type.eq('minus10')].iloc[0]
        self.assertEqual(minus10.annotation_status, 'tool_inferred')
        self.assertEqual(int(minus10.start_0index), 38)
        self.assertEqual(int(minus10.end_0index_exclusive), 44)
        self.assertEqual(minus10.strand, 'forward')
        self.assertEqual(window[38:44], 'TATAAT')
        tss = annotations[annotations.region_type.eq('TSS')].iloc[0]
        self.assertEqual(tss.annotation_status, 'not_applicable')
        self.assertTrue(pd.isna(tss.start_0index))

    def test_reverse_match_puts_minus35_downstream_of_minus10(self):
        promoter = self._layout_promoter()
        original = 10
        window = promoter[original:original+50].translate(str.maketrans('ACGT','TGCA'))[::-1]
        samples = pd.DataFrame({
            'sample_id':['s1'], 'source_row':[1], 'sequence':[window],
            'dataset_id':'course_ecoli50_strength', 'data_version':'ecoli50_strength_v1', 'schema_version':'2.0.0',
        })
        ecoli = pd.DataFrame({'seq_id':[1], 'seq':[promoter], 'label':[1]})
        audit, usable = classify_matches(samples, index_windows(ecoli), {})
        self.assertEqual(audit.strand.iloc[0], 'reverse')
        annotations = build_mapped_annotations(samples, usable)
        minus10 = annotations[annotations.region_type.eq('minus10')].iloc[0]
        minus35 = annotations[annotations.region_type.eq('minus35')].iloc[0]
        self.assertEqual(minus10.strand, 'reverse')
        self.assertGreater(int(minus35.start_0index), int(minus10.start_0index))
        self.assertEqual(map_interval(*LAYOUT['minus10'], original, 'reverse', 81), (int(minus10.start_0index), int(minus10.end_0index_exclusive)))

    def test_ambiguous_or_negative_match_stays_missing(self):
        promoter = self._layout_promoter()
        window = promoter[10:60]
        other = 'G'*5 + window + 'G'*26
        self.assertEqual(len(other), 81)
        samples = pd.DataFrame({
            'sample_id':['unique_neg','ambiguous'], 'source_row':[1,2],
            'sequence':[window, window],
            'dataset_id':'course_ecoli50_strength', 'data_version':'ecoli50_strength_v1', 'schema_version':'2.0.0',
        })
        negative = pd.DataFrame({'seq_id':[9], 'seq':[promoter], 'label':[0]})
        positive = pd.DataFrame({'seq_id':[1,2], 'seq':[promoter, other], 'label':[1,1]})
        audit, usable = classify_matches(samples.iloc[:1], {}, index_windows(negative))
        self.assertEqual(audit.match_status.iloc[0], 'negative_only')
        self.assertEqual(usable, {})
        audit, usable = classify_matches(samples.iloc[1:], index_windows(positive), {})
        self.assertEqual(audit.match_status.iloc[0], 'ambiguous_positive')
        annotations = build_mapped_annotations(samples.iloc[1:], usable)
        self.assertTrue(annotations.annotation_status.eq('missing').all())
        self.assertTrue(annotations.strand.eq('unknown').all())

    def test_different_parent_records_at_same_position_keep_all_sources(self):
        promoter = self._layout_promoter()
        different_parent = 'G' + promoter[1:]
        samples = pd.DataFrame({
            'sample_id':['same_position'], 'source_row':[1], 'sequence':[promoter[10:60]],
            'dataset_id':'course_ecoli50_strength', 'data_version':'ecoli50_strength_v1', 'schema_version':'2.0.0',
        })
        parents = pd.DataFrame({'seq_id':[11,22], 'seq':[promoter,different_parent], 'label':[1,1]})
        audit, usable = classify_matches(samples, index_windows(parents), {})
        row = audit.iloc[0]
        self.assertEqual(row.match_status, 'position_consistent_positive')
        self.assertEqual(row.n_positive_loci, 2)
        self.assertEqual(row.n_positive_positions, 1)
        self.assertEqual(row.n_positive_parents, 2)
        self.assertEqual(json.loads(row.positive_parent_ids), ['11','22'])
        annotation = build_mapped_annotations(samples, usable)
        minus10 = annotation[annotation.region_type.eq('minus10')].iloc[0]
        self.assertIn('seq_id=11,22', minus10.evidence_ref)
        self.assertEqual(minus10.annotation_status, 'tool_inferred')

    def test_inferred_relationships_require_explicit_mode_and_keep_evidence_separate(self):
        samples = self.data.iloc[:5].copy()
        annotations = build_annotations(samples)
        selected = annotations.region_type.eq('minus10')
        annotations.loc[selected, 'annotation_status'] = 'tool_inferred'
        annotations.loc[selected, 'strand'] = 'forward'
        annotations.loc[selected, 'start_0index'] = 0
        annotations.loc[selected, 'end_0index_exclusive'] = 6
        default = next(r for r in summarize_regions(samples, annotations) if r['region_type']=='minus10')
        enabled = next(r for r in summarize_regions(samples, annotations, include_inferred=True) if r['region_type']=='minus10')
        self.assertNotIn('spearman_gc_vs_log10', default)
        self.assertEqual(enabled['n_reliable_external_with_coordinates'], 0)
        self.assertEqual(enabled['n_tool_inferred_with_coordinates'], 5)
        self.assertEqual(enabled['relationship_n'], 5)
        self.assertEqual(enabled['analysis_mode'], 'layout_exploration')

    def test_positive_and_negative_parent_overlap_is_audited_and_excluded(self):
        promoter = self._layout_promoter()
        samples = self.data.iloc[:1].copy()
        samples['sequence'] = promoter[10:60]
        positive = pd.DataFrame({'seq_id':[11], 'seq':[promoter], 'label':[1]})
        negative = pd.DataFrame({'seq_id':[22], 'seq':[promoter], 'label':[0]})
        audit, usable = classify_matches(samples, index_windows(positive), index_windows(negative))
        self.assertEqual(audit.match_status.iloc[0], 'both')
        self.assertEqual(json.loads(audit.positive_parent_ids.iloc[0]), ['11'])
        self.assertEqual(json.loads(audit.negative_parent_ids.iloc[0]), ['22'])
        self.assertFalse(usable)


if __name__=='__main__':
    unittest.main()
