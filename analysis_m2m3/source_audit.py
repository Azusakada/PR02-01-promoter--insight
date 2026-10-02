"""Verify source ancestry against immutable author-released NPY arrays."""
from __future__ import annotations

import argparse
import hashlib
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from common import SAMPLES, load_data, relative, require, sha256, write_json

SOURCE_COMMIT = '56c1b331bfbca0665b2c3f6b02345de794fe25df'
BASE = f'https://raw.githubusercontent.com/wangxinglong1990/Promoter_design/{SOURCE_COMMIT}/'
FILES = {'sequence': 'PromoS_and_PromoA_strength_prediction/PromoA_train/promoter.npy',
         'strength': 'PromoS_and_PromoA_strength_prediction/PromoA_train/gene_expression.npy'}


def audit(output):
    require(not output.exists() or not any(output.iterdir()), 'Audit output nonempty')
    data = load_data().sort_values('source_row')
    arrays, inputs = {}, []
    for key, path in FILES.items():
        url = BASE + path
        raw = urllib.request.urlopen(url, timeout=30).read()
        arr = np.load(io.BytesIO(raw), allow_pickle=False)
        require(arr.shape == (len(data),), f'{key}: source shape mismatch')
        arrays[key] = arr
        inputs.append({'source_path': path, 'url': url, 'sha256': hashlib.sha256(raw).hexdigest(), 'shape': list(arr.shape), 'dtype': str(arr.dtype)})
    seq = np.char.upper(arrays['sequence'].astype(str))
    labels = arrays['strength'].astype(float)
    seq_ok = seq == data.sequence.to_numpy()
    label_ok = np.isclose(labels, data.strength.to_numpy(), rtol=0, atol=1e-8)
    require(seq_ok.all() and label_ok.all(), 'Source lineage not confirmed for all sequences and labels')
    output.mkdir(parents=True, exist_ok=True)
    checks = pd.DataFrame({'sample_id': data.sample_id, 'source_row': data.source_row,
                           'source_sequence_uppercase_match': seq_ok, 'source_strength_match': label_ok})
    checks['source_sequence_uppercase_match'] = checks.source_sequence_uppercase_match.map({True:'true',False:'false'})
    checks['source_strength_match'] = checks.source_strength_match.map({True:'true',False:'false'})
    checks.to_csv(output/'source_match.tsv',sep='\t',index=False)
    write_json(output/'source_audit.json', {
        'source_repository': 'https://github.com/wangxinglong1990/Promoter_design',
        'source_commit': SOURCE_COMMIT, 'paper': 'https://doi.org/10.1002/ggn2.202300184',
        'method': 'Source arrays in source_row order; sequence uppercased only; labels parsed as float',
        'target': {'path':relative(SAMPLES), 'sha256':sha256(SAMPLES)}, 'inputs':inputs,
        'n_requested':len(data), 'n_sequence_match':int(seq_ok.sum()), 'n_strength_match':int(label_ok.sum()),
        'source_match_sha256':sha256(output/'source_match.tsv'), 'result':'all_matched',
        'limits': ['Data ancestry verified, not original assay quality', 'No per-sample TSS or motif coordinates in these source NPY files', 'Assay units and label aggregation still require original data documentation'],
    })
    print(relative(output))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    audit(p.parse_args().output)
