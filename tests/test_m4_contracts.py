import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from pr02.context87.data import read_json, write_json
from pr02.context87.m4_data_eval import bundle, read_bundle, validate_prediction_rows, bootstrap
from pr02.context87.m4_thermo_mutation import design_mutations
from pr02.context87.research import integrated_gradients


class M4Contracts(unittest.TestCase):
    def test_prediction_alignment_and_failures(self):
        cfg=dict(dataset_id='d',data_version='v',split_id='s')
        truth=[dict(sample_id='a',target_log10='1',strength='10'),dict(sample_id='b',target_log10='2',strength='100')]
        def row(sid,z):return dict(**cfg,sample_id=sid,subset='val',sequence_length=87,target_scale_model='log10',
              true_value=10**z,true_value_log10=z,predicted_value=10**z,predicted_value_log10=z,prediction_status='ok')
        rows=[row('b',2),row('a',1)]
        self.assertEqual([r['sample_id'] for r in validate_prediction_rows(rows,truth,cfg,'val')],['a','b'])
        bad=[dict(r) for r in rows];bad[0]['true_value_log10']=1
        with self.assertRaisesRegex(ValueError,'label'):validate_prediction_rows(bad,truth,cfg,'val')
        bad=[dict(r) for r in rows];bad[0].update(prediction_status='failed',predicted_value_log10=0,error_reason='failed')
        with self.assertRaisesRegex(ValueError,'Failed prediction'):validate_prediction_rows(bad,truth,cfg,'val')
        bad[0]['predicted_value_log10']=''
        validate_prediction_rows(bad,truth,cfg,'val')

    def test_feature_ids_and_checksum(self):
        from pr02.context87.data import ROOT
        cfg=read_json(ROOT/'project_config.json')
        with tempfile.TemporaryDirectory() as folder:
            folder=Path(folder);rows=[dict(sample_id='a'),dict(sample_id='b')]
            bundle(folder,np.array([[1,2],[3,4]]),rows,'test',cfg)
            np.testing.assert_array_equal(read_bundle(folder,['b','a']),[[3,4],[1,2]])
            with self.assertRaisesRegex(ValueError,'Missing'):read_bundle(folder,['c'])
            with (folder/'features.npz').open('ab') as f:f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):read_bundle(folder,['a'])

    def test_integrated_gradients_linear_completeness_and_sign(self):
        class Linear(torch.nn.Module):
            def forward(self,x):return (x*torch.arange(4).reshape(1,4,1)).sum((1,2))
        x=np.zeros((2,4,87),np.float32);x[:,0,:]=1
        values,errors=integrated_gradients(Linear(),x,64)
        np.testing.assert_allclose(values,-1.5,atol=1e-6)
        np.testing.assert_allclose(errors,0,atol=1e-6)

    def test_mutation_counts_distinct_alternatives_and_expectation(self):
        master={'x':dict(sequence='A'*87)};scores={'x':np.arange(87,dtype=float)}
        expected={('x',p,b):(-1 if b=='C' else 1) for p in range(87) for b in 'CGT'}
        rows=design_mutations(master,scores,expected,dict(mutation_direction_threshold_log10=1e-4),'checksum')
        self.assertEqual(sum(r['experiment_type']=='amplitude' for r in rows),30)
        self.assertEqual(sum(r['experiment_type']=='single_direction' for r in rows),15)
        self.assertEqual(len({r['mutation_id'] for r in rows}),45)
        for r in rows:
            self.assertEqual(sum(a!=b for a,b in zip(r['wild_type_sequence'],r['mutant_sequence'])),r['k'])
        self.assertEqual({r['expected_direction'] for r in rows if r['experiment_type']=='single_direction'},{'positive','negative'})

    def test_cluster_bootstrap_does_not_treat_repeated_mutants_as_independent(self):
        x=np.repeat([0.,1.],20);groups=np.repeat(['a','b'],20)
        low,high=bootstrap(x,groups=groups,repeats=200)
        self.assertEqual((low,high),(0.,1.))


if __name__=='__main__':unittest.main()
