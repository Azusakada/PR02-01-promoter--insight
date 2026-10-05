from pathlib import Path
import tempfile
import unittest

import numpy as np

from pr02.common import ROOT, v
from pr02.data import clean_rows, rebuild


class DataReproductionTests(unittest.TestCase):
    def test_original_arrays_reproduce_every_frozen_sample_and_split(self):
        (ROOT/'work').mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT/'work') as directory:
            output=Path(directory)/'rebuild'
            result=rebuild(output)
            self.assertTrue(all(result['canonical_reproduction'].values()))
            with self.assertRaises(v.ContractError):
                rebuild(output)

    def test_invalid_raw_records_are_rejected_without_reindexing(self):
        for sequence,strength in [('A'*49,'1'),('A'*49+'N','1'),('A'*50,'0'),('A'*50,'nan')]:
            with self.subTest(sequence=sequence,strength=strength),self.assertRaises(v.ContractError):
                clean_rows(np.array([sequence]),np.array([strength]),'v')
        rows=clean_rows(np.array(['a'*50,'t'*50]),np.array(['2.5','3']), 'v')
        self.assertEqual([r['sample_id'] for r in rows],['ecoli50_r000001','ecoli50_r000002'])
        self.assertEqual(rows[0]['sequence'],'A'*50)


if __name__=='__main__': unittest.main()
