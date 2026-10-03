"""Frozen k-mer inputs and train-only Ridge/SVR with validation-only exports."""
from __future__ import annotations
import io
import hashlib
import itertools
import json
import math
import warnings
import zipfile
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

from .common import (ROOT, DATA, SPLITS, TRANSFORM, artifact, fresh_dir, load_bundle,
                     prediction_base, read_json, read_table, relative, resolve, run_dir,
                     v, write_json, write_manifest, write_table)
from .metrics import metric_rows

FEATURES = ROOT/'Ridge/features'
SELECTION = ROOT/'Ridge/selection'
DELIVERY = ROOT/'history/liqihang_delivery_495ee43.zip'
SOURCE_COMMIT = '495ee435774a1eab31bbfb8b1d2aa8f166873b7f'
SEED = 20260928


def count_matrix(sequences, k):
    """Base-4 windows, independent of the member's dictionary-count builder."""
    v.require(k in (3,4,5), 'Supported k values are 3, 4, 5')
    v.require(bool(sequences) and all(len(s)==50 and set(s)<=set('ACGT') for s in sequences), 'Expected 50bp ACGT')
    codes = np.array([['ACGT'.index(base) for base in seq] for seq in sequences],dtype=np.int16)
    windows = 51-k
    words = np.zeros((len(sequences),windows),dtype=np.int16)
    for offset in range(k): words = words*4+codes[:,offset:offset+windows]
    matrix = np.zeros((len(sequences),4**k),dtype=np.float32)
    np.add.at(matrix,(np.repeat(np.arange(len(sequences)),windows),words.ravel()),1)
    return matrix


def checked_feature(path, samples, k):
    # Legacy NPZ IDs have object dtype. These tracked, hash-bound member artifacts
    # are trusted inputs; newly generated bundles below use Unicode IDs instead.
    with np.load(path,allow_pickle=True) as blob:
        ids = blob['sample_ids'].astype(str).tolist()
        vocabulary = blob['vocabulary'].astype(str).tolist()
        x = blob['X'].astype(np.float64)
    expected_ids = [s['sample_id'] for s in samples]
    expected_vocab = [''.join(parts) for parts in itertools.product('ACGT',repeat=k)]
    v.require(ids==expected_ids and len(set(ids))==len(ids), 'k-mer feature sample IDs/order mismatch')
    v.require(vocabulary==expected_vocab, 'k-mer vocabulary/order mismatch')
    v.require(x.shape==(len(samples),4**k) and np.isfinite(x).all(), 'Invalid k-mer matrix')
    expected = count_matrix([s['sequence'] for s in samples],k)
    v.require(np.array_equal(x,expected), 'k-mer counts do not match frozen sequences')
    return x,ids


def validate_features(directory=FEATURES):
    samples,_,_ = load_bundle()
    directory = resolve(directory)
    result = []
    for k in [3,4,5]:
        x,ids = checked_feature(directory/f'kmer{k}.npz',samples,k)
        v.require((directory/f'kmer{k}_vocabulary.txt').read_text(encoding='utf-8').splitlines()==[''.join(p) for p in itertools.product('ACGT',repeat=k)],'Vocabulary text mismatch')
        result.append(dict(k=k,n_samples=len(ids),n_features=x.shape[1],all_counts_independently_recomputed=True))
    v.require((directory/'sample_ids.txt').read_text(encoding='utf-8').splitlines()==[s['sample_id'] for s in samples], 'Feature ID text mismatch')
    index = read_table(directory/'feature_index.tsv',delimiter='\t')
    v.require([r['sample_id'] for r in index]==[s['sample_id'] for s in samples] and [int(r['row_index']) for r in index]==list(range(len(samples))), 'Feature index mismatch')
    phys = read_table(directory/'physchem8.tsv',delimiter='\t')
    v.require([r['sample_id'] for r in phys]==[s['sample_id'] for s in samples], 'Physchem feature IDs/order mismatch')
    from .knn import feature_map, FEATURES as PHYS_COLUMNS
    canonical = feature_map(samples)
    vals = np.asarray([[float(r[c]) for c in PHYS_COLUMNS] for r in phys])
    v.require(np.allclose(vals,np.asarray([canonical[s['sample_id']] for s in samples]),rtol=1e-9,atol=1e-9), 'Physchem controls differ from canonical features')
    return dict(status='passed',features=result,physchem8_ids_and_values_verified=True)


def check_inputs():
    _,_,transform = load_bundle()
    for name,canonical in [('data_v1.tsv',DATA),('split_manifest.tsv',SPLITS)]:
        v.require(v.sha256(ROOT/'Ridge/data_snapshot'/name)==v.sha256(canonical), 'Member snapshot differs from canonical '+name)
    v.require(read_json(ROOT/'Ridge/data_snapshot/label_transform.json')==transform, 'Member transform differs from shared transform')
    return dict(status='passed',samples_sha256=v.sha256(DATA),splits_sha256=v.sha256(SPLITS),transform_id=transform['transform_id'])


def build_features(output):
    samples,_,_ = load_bundle()
    output = fresh_dir(output)
    ids = np.asarray([s['sample_id'] for s in samples],dtype=str)
    np.save(output/'sample_ids.npy',ids,allow_pickle=False)
    (output/'sample_ids.txt').write_text(''.join(s+'\n' for s in ids),encoding='utf-8',newline='\n')
    entries=[]
    for k in [3,4,5]:
        vocabulary = np.asarray([''.join(p) for p in itertools.product('ACGT',repeat=k)],dtype=str)
        np.savez_compressed(output/f'kmer{k}.npz',X=count_matrix([s['sequence'] for s in samples],k),sample_ids=ids,vocabulary=vocabulary,k=np.asarray([k]))
        (output/f'kmer{k}_vocabulary.txt').write_text(''.join(s+'\n' for s in vocabulary),encoding='utf-8',newline='\n')
        entries.append(artifact('kmer_features',output/f'kmer{k}.npz'))
    write_table(output/'feature_index.tsv',[dict(sample_id=s,row_index=i) for i,s in enumerate(ids)],delimiter='\t')
    from .knn import feature_map, FEATURES as PHYS_COLUMNS
    mapped=feature_map(samples)
    write_table(output/'physchem8.tsv',[dict(sample_id=s['sample_id'],**dict(zip(PHYS_COLUMNS,mapped[s['sample_id']]))) for s in samples],delimiter='\t')
    write_json(output/'feature_manifest.json',dict(data=artifact('dataset',DATA),splits=artifact('splits',SPLITS),n_samples=len(samples),feature_version='kmer_count_acgt_lex_v1',artifacts=entries))
    return validate_features(output)


def selected_candidates(config, ridge_rows, svr_rows):
    main = [r for r in ridge_rows if r['feature_set'] in ['kmer3','kmer4','kmer5']]
    v.require(bool(main) and bool(svr_rows), 'Missing selection history')
    winner=max(main,key=lambda r:float(r['val_r2_log10']))
    selected=config['selected']
    v.require(int(winner['k'])==int(selected['k']) and float(winner['alpha'])==float(selected['alpha']), 'Ridge selection is not the published validation winner')
    svr=max(svr_rows,key=lambda r:float(r['val_r2_log10']))
    chosen=config['svr_selected']
    v.require(svr['model']==chosen['model'] and svr['kernel']==chosen['kernel'] and float(svr['C'])==float(chosen['C']), 'SVR selection is not the published validation winner')
    return selected,chosen


def selection():
    config=read_json(SELECTION/'ridge_config.json')
    _,_,transform=load_bundle()
    v.require(config['final_fit_scope']=='train_only' and config['do_not_refit_train_plus_val'] is True,'Unsupported fit scope')
    for name in ['dataset_id','data_version','split_id']: v.require(config[name]==transform[name], 'Selection identity mismatch')
    v.require(config['input_sha256']['samples']==v.sha256(DATA) and config['input_sha256']['splits']==v.sha256(SPLITS),'Selection data hash mismatch')
    return config,selected_candidates(config,read_table(SELECTION/'ridge_search.csv'),read_table(SELECTION/'svr_search.csv'))


def audit_source_delivery():
    """Reload original member checkpoints and check only val against shared inputs."""
    samples,groups,_=load_bundle()
    config,(ridge,svr)=selection()
    k=int(ridge['k']); x,ids=checked_feature(FEATURES/f'kmer{k}.npz',samples,k)
    positions={sid:i for i,sid in enumerate(ids)}
    xt=x[[positions[s['sample_id']] for s in groups['train']]]
    xv=x[[positions[s['sample_id']] for s in groups['val']]]
    checks=[]
    with zipfile.ZipFile(DELIVERY) as archive:
        import csv
        for kind in ['ridge','svr']:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                saved=joblib.load(io.BytesIO(archive.read(f'Ridge/results/{kind}_model.joblib')))
            rows=list(csv.DictReader(io.StringIO(archive.read(f'Ridge/results/predictions_{kind}.csv').decode('utf-8'))))
            val={r['sample_id']:r for r in rows if r['subset']=='val'}
            v.require(set(val)=={s['sample_id'] for s in groups['val']},'Original val ID mismatch')
            for sample in groups['val']: v.require(math.isclose(float(val[sample['sample_id']]['true_value']),sample['strength'],rel_tol=1e-10),'Original val label mismatch')
            v.require(np.allclose(saved['scaler'].mean_,xt.mean(axis=0),rtol=1e-12,atol=1e-12),'Original scaler not train-only')
            predicted=saved['model'].predict(saved['scaler'].transform(xv))
            expected=np.asarray([float(val[s['sample_id']]['predicted_value_log10']) for s in groups['val']])
            v.require(np.allclose(predicted,expected,rtol=1e-9,atol=1e-9),'Original checkpoint/val prediction mismatch')
            checks.append(dict(method=kind,n_val=len(val),max_abs_difference=float(np.max(np.abs(predicted-expected))),load_warnings=[str(w.message) for w in caught]))
    return dict(status='passed',source_commit=SOURCE_COMMIT,checks=checks,historical_test_results_present=True,test_metrics_recomputed=False)


def run(ridge_run_id,svr_run_id):
    for tag in [ridge_run_id,svr_run_id]:
        path=resolve(ROOT/'runs'/tag)
        v.require(not path.exists() or not any(path.iterdir()),'Refusing to overwrite run: '+str(path))
    v.require(ridge_run_id!=svr_run_id,'Ridge and SVR must use different run IDs')
    inputs=check_inputs(); features=validate_features(); source=audit_source_delivery()
    samples,groups,transform=load_bundle()
    config,(ridge_selected,svr_selected)=selection()
    k=int(ridge_selected['k']); x,ids=checked_feature(FEATURES/f'kmer{k}.npz',samples,k)
    positions={sid:i for i,sid in enumerate(ids)}
    xt=x[[positions[s['sample_id']] for s in groups['train']]]
    xv=x[[positions[s['sample_id']] for s in groups['val']]]
    y=np.asarray([s['target_log10'] for s in groups['train']])
    scaler=StandardScaler().fit(xt)
    plans=[('ridge',ridge_run_id,Ridge(alpha=float(ridge_selected['alpha']),random_state=SEED),dict(k=k,alpha=float(ridge_selected['alpha'])))]
    if svr_selected['model']=='LinearSVR':
        svr=LinearSVR(C=float(svr_selected['C']),random_state=SEED,max_iter=200000,dual=True,tol=1e-4)
    else:
        v.require(svr_selected['model']=='SVR' and svr_selected['kernel']=='rbf','Unsupported SVR')
        svr=SVR(C=float(svr_selected['C']),kernel='rbf',gamma='scale')
    plans.append(('svr',svr_run_id,svr,dict(k=k,C=float(svr_selected['C']),kernel=svr_selected['kernel'])))
    outputs=[]
    for kind,run_id,model,params in plans:
        output=run_dir(run_id); method=f'kmer{k}_{kind}'
        print(f'Fitting {method}: train={len(xt)}, val={len(xv)}, params={params}',flush=True)
        model.fit(scaler.transform(xt),y)
        if isinstance(model,LinearSVR): v.require(model.n_iter_<model.max_iter,'Selected LinearSVR did not converge')
        predicted=model.predict(scaler.transform(xv))
        train_ids=[s['sample_id'] for s in groups['train']]
        checkpoint=output/f'{kind}_model.joblib'
        saved=dict(model=model,scaler=scaler,k=k,method_name=method,run_id=run_id,target_scale='log10',fit_subsets=['train'],train_ids=train_ids,split_id=transform['split_id'],feature_file=relative(FEATURES/f'kmer{k}.npz'),**{key:value for key,value in params.items() if key!='k'})
        joblib.dump(saved,checkpoint)
        meta=dict(schema_version='2.0.0',model_id=run_id,method_name=method,data_version=transform['data_version'],split_id=transform['split_id'],fit_subsets=['train'],selection_subsets=['val'],target_scale='log10',parameters=params,n_fit=len(xt),source_selection_commit=SOURCE_COMMIT,selection_reused=True,test_used_for_fit_or_selection=False,feature_file=artifact('features',FEATURES/f'kmer{k}.npz'),train_ids_sha256=hashlib.sha256(''.join(s+'\n' for s in train_ids).encode()).hexdigest(),n_iter=int(model.n_iter_) if isinstance(model,LinearSVR) else None)
        write_json(output/'model_manifest.json',meta)
        rows=[]
        for sample,z in zip(groups['val'],predicted):
            row=prediction_base(sample,method,run_id,transform['split_id'],'val','log10',checkpoint=relative(checkpoint),seed=SEED)
            row.update(prediction_status='ok',error_reason=None,predicted_value_log10=float(z),predicted_value=float(10**z))
            rows.append(row)
        predfile=output/f'predictions_{kind}_val.csv'
        write_table(predfile,rows); v.bundle_check(DATA,SPLITS,transform=TRANSFORM,predictions=predfile)
        write_table(output/'metrics.csv',metric_rows(rows,'val',comparison_set_id=run_id+'_val_available'))
        write_table(output/'train_ids.tsv',[dict(sample_id=s) for s in train_ids],delimiter='\t')
        reloaded=joblib.load(checkpoint)
        repeated=reloaded['model'].predict(reloaded['scaler'].transform(xv))
        v.require(np.array_equal(predicted,repeated),'Checkpoint save/load changed predictions')
        with zipfile.ZipFile(DELIVERY) as archive:
            import csv
            prior={r['sample_id']:float(r['predicted_value_log10']) for r in csv.DictReader(io.StringIO(archive.read(f'Ridge/results/predictions_{kind}.csv').decode('utf-8'))) if r['subset']=='val'}
        difference=np.abs(predicted-np.asarray([prior[s['sample_id']] for s in groups['val']]))
        write_json(output/'save_load_check.json',dict(status='passed',n_compared=len(rows),max_abs_difference=float(np.max(np.abs(predicted-repeated))),source_prediction_max_abs_difference=float(difference.max()),source_predictions_close=bool(np.allclose(predicted,np.asarray([prior[s['sample_id']] for s in groups['val']]),rtol=1e-9,atol=1e-9))))
        write_json(output/'input_audit.json',dict(inputs=inputs,features=features,source_delivery=source))
        artifacts=[artifact(kindname,output/name) for kindname,name in [('predictions',predfile.name),('metrics','metrics.csv'),('model_manifest','model_manifest.json'),('checkpoint',checkpoint.name),('fit_ids','train_ids.tsv'),('save_load','save_load_check.json'),('input_audit','input_audit.json')]]
        artifacts.extend([artifact('dataset',DATA),artifact('splits',SPLITS),artifact('label_transform',TRANSFORM),artifact('features',FEATURES/f'kmer{k}.npz'),artifact('source_delivery',DELIVERY)])
        artifacts.extend(artifact('selection_history',SELECTION/name) for name in ['ridge_config.json','ridge_search.csv','svr_search.csv'])
        write_manifest(output,run_id,method,artifacts,parameters=meta,fit=['train'],selection=['val'],seed=SEED)
        outputs.append(output)
    return outputs


def verify_models(cfg,frames):
    samples,groups,_=load_bundle()
    validate_features()
    checks=[]
    for kind in ['ridge','svr']:
        directory=ROOT/'runs'/cfg['runs'][kind]
        saved=joblib.load(directory/f'{kind}_model.joblib')
        x,ids=checked_feature(resolve(saved['feature_file']),samples,int(saved['k']))
        positions={sid:i for i,sid in enumerate(ids)}
        xt=x[[positions[s['sample_id']] for s in groups['train']]]
        xv=x[[positions[s['sample_id']] for s in groups['val']]]
        train_ids=[s['sample_id'] for s in groups['train']]
        v.require(saved['train_ids']==train_ids and saved['fit_subsets']==['train'],'Model fit IDs/scope mismatch')
        v.require(np.allclose(saved['scaler'].mean_,xt.mean(axis=0),rtol=1e-12,atol=1e-12),'k-mer scaler not train-only')
        predicted=saved['model'].predict(saved['scaler'].transform(xv))
        rows=next(rows for rows in frames if rows[0]['method_name']==saved['method_name'])
        expected={r['sample_id']:r['predicted_value_log10'] for r in rows}
        values=np.asarray([expected[s['sample_id']] for s in groups['val']])
        v.require(np.allclose(predicted,values,rtol=1e-9,atol=1e-9),'k-mer saved model does not reproduce val')
        if kind=='ridge':
            # Independently solve the centered Ridge normal equations on train.
            z=saved['scaler'].transform(xt); y=np.asarray([s['target_log10'] for s in groups['train']])
            coef=np.linalg.solve(z.T@z+saved['model'].alpha*np.eye(z.shape[1]),z.T@(y-y.mean()))
            v.require(np.allclose(coef,saved['model'].coef_,rtol=1e-8,atol=1e-9),'Ridge coefficients not reproduced from train')
        checks.append((f'{kind}_saved_train_only_model_reload',dict(n_compared=len(values),max_abs_difference=float(np.max(np.abs(predicted-values))),independent_train_ridge_solve=(kind=='ridge'))))
    return checks
