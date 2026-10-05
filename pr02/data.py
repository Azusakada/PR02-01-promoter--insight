"""Rebuild the frozen strength dataset from the original course NPY arrays."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .common import (ROOT, DATA, SPLITS, TRANSFORM, artifact, fresh_dir, label,
                     read_json, resolve, v, write_json, write_table)
from api_contract import DatasetBundle, SplitBundle

PROFILE = ROOT/'configs/data_ecoli50_v1.json'
RAW = ROOT/'data/raw_ecoli50'


def clean_rows(sequences, strengths, data_version):
    v.require(sequences.ndim==strengths.ndim==1 and len(sequences)==len(strengths)>0,
              'Raw arrays must be nonempty one-dimensional paired records')
    rows=[]
    for source_row,(sequence,strength) in enumerate(zip(sequences,strengths),1):
        sequence=str(sequence).upper()
        v.require(len(sequence)==50 and set(sequence)<=set('ACGT'),
                  f'Invalid sequence at source row {source_row}; no rows are silently dropped')
        strength=float(strength)
        v.require(math.isfinite(strength) and strength>0,f'Invalid strength at source row {source_row}')
        rows.append(dict(dataset_id='course_ecoli50_strength',sample_id=f'ecoli50_r{source_row:06d}',
                         source_row=source_row,sequence=sequence,sequence_length=50,strength=strength,
                         target_log10=math.log10(strength),annotation_status='missing',
                         schema_version='2.0.0',data_version=data_version))
    return rows


def build_dataset(raw_dir,config_path,output_dir) -> DatasetBundle:
    raw_dir=resolve(raw_dir); cfg=read_json(config_path)
    v.require(cfg['schema_version']=='2.0.0' and cfg['dataset_id']=='course_ecoli50_strength',
              'Unsupported data profile')
    arrays={}; inputs=[]
    for key in ['sequence','strength']:
        path=resolve(raw_dir/cfg['raw_files'][key]['filename'])
        v.require(v.sha256(path)==cfg['raw_files'][key]['sha256'],'Raw source hash mismatch: '+key)
        arrays[key]=np.load(path,allow_pickle=False)
        inputs.append(artifact('raw_'+key,path))
    rows=clean_rows(arrays['sequence'],arrays['strength'],cfg['data_version'])
    v.require(len(rows)==cfg['expected_samples'],'Unexpected raw sample count')
    output=fresh_dir(output_dir); samples=output/'data_v1.tsv'
    write_table(samples,rows,delimiter='\t'); v.load_table('dataset',samples)
    log=output/'cleaning_log.csv'
    write_table(log,[dict(source_row=r['source_row'],sample_id=r['sample_id'],status='retained',
                         sequence_uppercased=str(s)!=r['sequence'],strength_conversion='string_to_float')
                     for r,s in zip(rows,arrays['sequence'])])
    manifest=output/'data_manifest.json'
    write_json(manifest,dict(schema_version='2.0.0',dataset_id=cfg['dataset_id'],data_version=cfg['data_version'],
               n_input=len(rows),n_retained=len(rows),n_removed=0,inputs=inputs,
               operations=['uppercase sequence','parse positive finite strength','compute log10',
                           'stable IDs from original one-based source row; no sorting or filtering'],
               artifacts=[artifact('dataset',samples),artifact('cleaning_log',log),artifact('profile',config_path)]))
    return DatasetBundle(samples,log,manifest)


def build_splits(samples_path,config_path,output_dir) -> SplitBundle:
    cfg=read_json(config_path)
    rows=sorted(v.load_table('dataset',samples_path),key=lambda row:row['source_row'])
    v.require(all(r['data_version']==cfg['data_version'] for r in rows),'Split profile data version mismatch')
    # Confirmed against every frozen ID, not inferred from the split proportions.
    train,heldout=train_test_split(np.arange(len(rows)),test_size=0.30,random_state=cfg['seed'])
    val,test=train_test_split(heldout,test_size=0.50,random_state=cfg['seed'])
    assignment={int(i):sub for sub,indices in [('train',train),('val',val),('test',test)] for i in indices}
    output=resolve(output_dir); output.mkdir(parents=True,exist_ok=True)
    path=output/'split_manifest.tsv'; config=output/'split_config.json'
    v.require(not path.exists() and not config.exists(),'Split outputs already exist; use a fresh destination')
    write_table(path,[dict(dataset_id=r['dataset_id'],split_id=cfg['split_id'],sample_id=r['sample_id'],
                split=assignment[i],group_id=None,split_strategy='random',seed=cfg['seed'],
                schema_version='2.0.0',data_version=r['data_version']) for i,r in enumerate(rows)],delimiter='\t')
    v.bundle_check(samples_path,path)
    write_json(config,dict(schema_version='2.0.0',split_id=cfg['split_id'],seed=cfg['seed'],
               strategy='sklearn two-stage train_test_split: 70/30, then heldout 50/50; source order',
               counts=dict(train=len(train),val=len(val),test=len(test)),
               samples=artifact('dataset',samples_path),splits=artifact('splits',path)))
    return SplitBundle(path,config)


def verify_sources():
    cfg=read_json(PROFILE); arrays={}
    for key in ['sequence','strength']:
        path=RAW/cfg['raw_files'][key]['filename']
        v.require(v.sha256(path)==cfg['raw_files'][key]['sha256'],'Raw source hash mismatch: '+key)
        arrays[key]=np.load(path,allow_pickle=False)
    expected=clean_rows(arrays['sequence'],arrays['strength'],cfg['data_version'])
    actual=sorted(v.load_table('dataset',DATA),key=lambda row:row['source_row'])
    v.require(actual==expected,'Canonical dataset differs from original source records')
    return dict(samples_checked=len(actual),raw_array_hashes_verified=True,all_sequences_and_strengths_match=True)


def rebuild(output,raw_dir=RAW,profile=PROFILE):
    output=resolve(output); cfg=read_json(profile)
    dataset=build_dataset(raw_dir,profile,output)
    split=build_splits(dataset.samples,profile,output)
    transform=label.compute_label_transform(dataset.samples,split.split_manifest,cfg['split_id'])
    transform['inverse']='strength = 10 ** target_log10; target_log10 = min_log10 + normalized * (max_log10 - min_log10)'
    write_json(output/'label_transform.json',transform)
    check=v.bundle_check(dataset.samples,split.split_manifest,transform=output/'label_transform.json')
    same=dict(dataset_bytes=dataset.samples.read_bytes()==DATA.read_bytes(),
              split_bytes=split.split_manifest.read_bytes()==SPLITS.read_bytes(),
              transform_semantics=transform==read_json(TRANSFORM))
    v.require(all(same.values()),'Rebuild differs from this frozen release; publish a new version before use')
    result=dict(status='passed',source_arrays_verified=True,canonical_reproduction=same,check=check)
    write_json(output/'reproduction_check.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--raw-dir',type=Path,default=RAW)
    parser.add_argument('--profile',type=Path,default=PROFILE)
    args=parser.parse_args()
    print(rebuild(args.output,args.raw_dir,args.profile))
