from collections import Counter
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from pr02.common import v
from pr02.kmer import count_matrix, checked_feature, selected_candidates


class KmerIntegrationTests(unittest.TestCase):
    def test_all_supported_counts_match_independent_window_counter(self):
        sequences=['A'*50,'ACGT'*12+'AC']
        for k in [3,4,5]:
            vocab=[''.join(p) for p in itertools.product('ACGT',repeat=k)]
            expected=[]
            for seq in sequences:
                counts=Counter(seq[i:i+k] for i in range(51-k))
                expected.append([counts[mer] for mer in vocab])
            self.assertTrue(np.array_equal(count_matrix(sequences,k),expected))

    def test_invalid_sequence_is_rejected(self):
        with self.assertRaises(v.ContractError): count_matrix(['N'*50],3)

    def test_feature_row_reordering_is_rejected(self):
        samples=[dict(sample_id='a',sequence='A'*50),dict(sample_id='b',sequence='T'*50)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.npz'
            np.savez(path,X=count_matrix([s['sequence'] for s in samples],3),sample_ids=['b','a'],vocabulary=[''.join(p) for p in itertools.product('ACGT',repeat=3)])
            with self.assertRaises(v.ContractError): checked_feature(path,samples,3)

    def test_feature_vocabulary_reordering_is_rejected(self):
        samples=[dict(sample_id='a',sequence='A'*50)]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'features.npz'
            np.savez(path,X=count_matrix(['A'*50],3),sample_ids=['a'],vocabulary=[''.join(p) for p in itertools.product('ACGT',repeat=3)][::-1])
            with self.assertRaises(v.ContractError): checked_feature(path,samples,3)

    def candidates(self):
        cfg=dict(selected=dict(k=3,alpha=100),svr_selected=dict(model='LinearSVR',kernel='linear',C=10))
        ridge=[dict(feature_set='kmer3',k=3,alpha=100,val_r2_log10=.2),dict(feature_set='kmer4',k=4,alpha=1,val_r2_log10=.1),dict(feature_set='kmer3_physchem8',k=3,alpha=10,val_r2_log10=.9)]
        svr=[dict(model='LinearSVR',kernel='linear',C=10,val_r2_log10=.1),dict(model='SVR',kernel='rbf',C=1,val_r2_log10=-.1)]
        return cfg,ridge,svr

    def test_supplementary_physchem_does_not_replace_main_ridge(self):
        cfg,ridge,svr=self.candidates()
        chosen,_=selected_candidates(cfg,ridge,svr)
        self.assertEqual((chosen['k'],chosen['alpha']),(3,100))

    def test_unpublished_svr_selection_is_rejected(self):
        cfg,ridge,svr=self.candidates(); cfg['svr_selected']['C']=1
        with self.assertRaises(v.ContractError): selected_candidates(cfg,ridge,svr)

    def test_new_run_preserves_ridge_and_svr_in_the_pipeline(self):
        from pr02.new_run import create
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); (root/'CNN/configs').mkdir(parents=True)
            cnn=root/'CNN/configs/base.yaml'; cnn.write_text(yaml.safe_dump(dict(run_id='old',output_dir='CNN/runs/old')))
            base=root/'base.json'; base.write_text(json.dumps(dict(cnn_config='CNN/configs/base.yaml',runs=dict(knn='old',thermo='old',comparison='old',ridge='old',svr='old'))))
            with patch('pr02.new_run.ROOT',root): plan=create('fresh',base)
            cfg=json.loads(plan.read_text())
            self.assertEqual(set(cfg['runs']),{'knn','thermo','comparison','ridge','svr'})
            self.assertEqual(len(cfg['validation_predictions']),5)
            self.assertIn('runs/ridge_fresh/predictions_ridge_val.csv',cfg['validation_predictions'])


if __name__=='__main__': unittest.main()
