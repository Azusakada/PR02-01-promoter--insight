"""Rebuild the published validation-selected KNN as a reproducible train-only model."""
from __future__ import annotations
import hashlib
import math
import joblib
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from .common import (ROOT, DATA, SPLITS, TRANSFORM, artifact, load_bundle, prediction_base, read_json,
                     read_table, relative, run_dir, v, write_json, write_manifest, write_table)
from .metrics import metric_rows

FEATURES = ['gc_content','at_content','gc_skew','at_skew','melting_temp','bendability','stacking_energy','entropy']
FEATURE_FILE = ROOT/'KNN/input/promoter_physicochemical.csv'
METHOD = 'knn_physchem_full'


def feature_map(samples):
    rows = read_table(FEATURE_FILE)
    v.require(len(rows)==len(samples) and list(rows[0])==FEATURES,'Invalid feature table')
    source_rows = [s['source_row'] for s in samples]
    v.require(set(source_rows)==set(range(1,len(samples)+1)), 'Feature source_row mapping incomplete')
    mapped = {s['sample_id']:np.asarray([float(rows[s['source_row']-1][c]) for c in FEATURES]) for s in samples}
    for sample in samples:
        vector = mapped[sample['sample_id']]
        gc = sum(base in 'GC' for base in sample['sequence'])/50
        v.require(np.isfinite(vector).all() and math.isclose(vector[0],gc,abs_tol=1e-12), 'Feature/sequence GC mapping mismatch')
    return mapped


def run(run_id='knn_physchem_integrated_20261002_v2'):
    samples,groups,transform = load_bundle()
    mapped = feature_map(samples)
    old_config = ROOT/'KNN/results/knn_ecoli_full_config.json'
    search = ROOT/'KNN/results/hyperparameter_search.csv'
    selected = read_json(old_config)
    k = selected['best_k']
    v.require(type(k) is int and 0<k<=len(groups['train']), 'Invalid published k')
    v.require(selected['split_id']==transform['split_id'] and selected['input_sha256']['samples']==v.sha256(DATA), 'Published selection bound to different data')
    output = run_dir(run_id)
    x_train = np.stack([mapped[s['sample_id']] for s in groups['train']])
    y_train = np.asarray([s['target_log10'] for s in groups['train']])
    x_val = np.stack([mapped[s['sample_id']] for s in groups['val']])
    scaler = StandardScaler().fit(x_train)
    model = KNeighborsRegressor(n_neighbors=k,weights='distance',metric='euclidean',n_jobs=4)
    model.fit(scaler.transform(x_train),y_train)
    predictions = model.predict(scaler.transform(x_val))
    checkpoint = output/'knn_model.joblib'
    train_ids = sorted(s['sample_id'] for s in groups['train'])
    meta = dict(model_id=run_id,method_name=METHOD,fit_subsets=['train'],selection_subsets=['val'],
                test_used=False,smoke_test=False,evidence_level='preliminary',feature_columns=FEATURES,
                samples_sha256=v.sha256(DATA),splits_sha256=v.sha256(SPLITS),feature_sha256=v.sha256(FEATURE_FILE),
                target_scale='log10',best_k=k,train_ids_sha256=hashlib.sha256(''.join(s+'\n' for s in train_ids).encode()).hexdigest(),
                selection_source=relative(search),selection_config_sha256=v.sha256(old_config))
    joblib.dump(dict(model=model,scaler=scaler,metadata=meta),checkpoint)
    loaded = joblib.load(checkpoint)
    repeated = loaded['model'].predict(loaded['scaler'].transform(x_val))
    v.require(np.array_equal(predictions,repeated),'KNN reload prediction differs')
    rows = []
    for sample,z in zip(groups['val'],predictions):
        row = prediction_base(sample,METHOD,run_id,transform['split_id'],'val','log10',checkpoint=relative(checkpoint),seed=20260928)
        row.update(prediction_status='ok',error_reason=None,predicted_value_log10=float(z),predicted_value=float(10**z))
        rows.append(row)
    path = output/'predictions_knn_val.csv'
    write_table(path,rows)
    v.bundle_check(DATA,SPLITS,predictions=path)
    write_table(output/'train_ids.tsv',[{'sample_id':s} for s in train_ids],delimiter='\t')
    write_table(output/'feature_sample_ids.tsv',[{'sample_id':s['sample_id'],'source_row':s['source_row']} for s in sorted(samples,key=lambda s:s['source_row'])],delimiter='\t')
    write_table(output/'metrics.csv',metric_rows(rows,'val',comparison_set_id=run_id+'_val_available'))
    write_json(output/'model_manifest.json',dict(meta,checkpoint=relative(checkpoint),checkpoint_sha256=v.sha256(checkpoint)))
    write_json(output/'save_load_check.json',dict(status='passed',n_compared=len(rows),exact_predictions_equal=True,max_abs_difference=float(np.max(np.abs(predictions-repeated)))))
    artifacts = [artifact(kind,output/name) for kind,name in [('predictions','predictions_knn_val.csv'),('metrics','metrics.csv'),
                 ('model_manifest','model_manifest.json'),('checkpoint','knn_model.joblib'),('smoke_report','save_load_check.json'),
                 ('fit_ids','train_ids.tsv'),('feature_index_mapping','feature_sample_ids.tsv')]]
    artifacts.extend([artifact('features',FEATURE_FILE),artifact('selection_history',search),artifact('selection_config',old_config),
                      artifact('dataset',DATA),artifact('splits',SPLITS),artifact('label_transform',TRANSFORM)])
    write_manifest(output,run_id,METHOD,artifacts,parameters=meta,fit=['train'],selection=['val'],seed=20260928)
    print(f'KNN completed: {output}',flush=True)
    return output
