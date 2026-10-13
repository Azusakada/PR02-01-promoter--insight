import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pr02.context87.data import ROOT, build, coordinate, load, read_json, read_table, transformed
from pr02.context87.features import encode, kmers, thermal_states
from pr02.context87.experiment import cnn_class, load_cnn, git_revision
from pr02.context87.research import integrated_gradients, target_splits


class Context87Tests(unittest.TestCase):
    def test_zip_checkout_does_not_require_git_metadata(self):
        import subprocess
        with patch('pr02.context87.experiment.subprocess.run',side_effect=subprocess.CalledProcessError(128,['git'])):
            self.assertIsNone(git_revision())

    def test_recovered_labels_split_and_coordinates(self):
        rows,t=load(); old={r['sample_id']:r for r in read_table(ROOT/'data/01_Ecoli_strength/data_v1.tsv')}
        parent={r['sample_id']:r['split'] for r in read_table(ROOT/'data/01_Ecoli_strength/split_manifest.tsv')}
        self.assertEqual(len(rows),11809)
        for row in rows:
            source=old[row['parent_sample_id']]
            self.assertEqual(row['sequence'][17:67],source['sequence'])
            self.assertEqual(float(row['strength']),float(source['strength']))
            self.assertEqual(row['split'],parent[row['parent_sample_id']])
        self.assertEqual([coordinate(i) for i in [0,17,66,67,86]],[-67,-50,-1,1,20])

    def test_transform_does_not_read_heldout_labels(self):
        rows,t=load(); changed=copy.deepcopy(rows)
        for row in changed:
            if row['split']!='train': row['strength']='1e99'
        self.assertEqual(transformed(changed),t)

    def test_reproduction_is_exact_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            output=Path(temp); build(output,verify=False)
            for name in ['samples.tsv','split_manifest.tsv','label_transform.json','data_manifest.json']:
                self.assertEqual((output/name).read_bytes(),(ROOT/'data/03_Ecoli_strength87'/name).read_bytes())
            with self.assertRaises(ValueError): build(output,verify=False)

    def test_lengths_do_not_pad_or_crop_silently(self):
        self.assertEqual(encode(['A'*87],87).shape,(1,4,87))
        with self.assertRaises(ValueError): encode(['A'*61],87)
        self.assertAlmostEqual(kmers(['A'*87],3).sum(),1)
        self.assertAlmostEqual(kmers(['A'*61],3).sum(),1)

    def test_known_tss_evaluates_all_thirty_states(self):
        rows,_=load(); state=thermal_states([rows[0]['sequence']])[0]
        self.assertEqual(state['TSS'],67)
        self.assertEqual(len(state['ITR']),20)
        for key,sequence_key in [('UP_position','UP'),('hex35_position','hex35'),('hex10_position','hex10')]:
            start,end=state[key]; self.assertEqual(rows[0]['sequence'][start:end],state[sequence_key])

    def test_cnn_strict_length_and_explicit_transfer_architecture(self):
        import torch
        cfg=read_json(ROOT/'project_config.json')['cnn']['model']; model=cnn_class()(cfg)
        with self.assertRaises(ValueError): model(torch.zeros(2,4,61))
        target=cnn_class()({**cfg,'sequence_length':61}); target.load_state_dict(model.state_dict())
        self.assertEqual(target(torch.zeros(2,4,61)).shape,(2,))

    def test_ig_completeness_on_linear_model(self):
        import torch
        class Linear(torch.nn.Module):
            def forward(self,x): return x[:,0,:].sum(dim=1)
        scores,error=integrated_gradients(Linear(),encode(['A'*87],87),8)
        self.assertTrue(np.allclose(scores,.75)); self.assertTrue(np.allclose(error,0))

    def test_duplicate_target_sequences_never_cross_splits(self):
        cfg=read_json(ROOT/'project_config.json'); rows=read_table(ROOT/cfg['transfer']['samples'])
        assignment=target_splits(rows,cfg); groups={}
        for row,split in zip(rows,assignment): groups.setdefault(row['sequence'],set()).add(split)
        self.assertEqual(len(rows),591); self.assertEqual(len(groups),580)
        self.assertTrue(all(len(s)==1 for s in groups.values()))


if __name__=='__main__': unittest.main()
