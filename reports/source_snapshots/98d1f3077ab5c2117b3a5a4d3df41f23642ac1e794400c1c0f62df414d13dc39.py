from __future__ import annotations

import copy
import importlib.metadata
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .data import (BASE, ROOT, digest_ids, load, read_json, read_table, records,
                   require, sha, validate, write_json, write_table)
from .features import encode, kmers, physchem, thermal_states, PHYS_NAMES

METHODS=('knn','ridge','svr','cnn','thermo')


def path(cfg,kind):
    run_id=cfg['runs'][kind]
    require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',run_id)),'Invalid run ID')
    return ROOT/'runs'/run_id


def fresh(directory):
    require(not directory.exists() or not any(directory.iterdir()),'Use a new run ID; existing output: '+str(directory))
    directory.mkdir(parents=True,exist_ok=True)
    return directory


def metrics(y,p):
    y=np.asarray(y,float); p=np.asarray(p,float)
    require(len(y)==len(p)>1 and np.isfinite(y).all() and np.isfinite(p).all(),'Invalid predictions')
    rho=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else None
    return dict(n=len(y),r2_log10=float(r2_score(y,p)),mae_log10=float(mean_absolute_error(y,p)),
                rmse_log10=float(np.sqrt(mean_squared_error(y,p))),spearman=rho)


def masks(rows):
    return {subset:np.array([r['split']==subset for r in rows]) for subset in ('train','val','test')}


def input_hashes(cfg):
    return {cfg[k]:sha(ROOT/cfg[k]) for k in ('samples','splits','label_transform')}


def code_hashes():
    files=[*sorted((ROOT/'pr02/context87').glob('*.py')), ROOT/'pr02/__main__.py',
           ROOT/'CNN/pr02_cnn/model.py',ROOT/'thermo/tools/regseq2/promoter_calculator.py',
           ROOT/'thermo/tools/regseq2/free_energy_coeffs.npy',ROOT/'thermo/tools/regseq2/model_intercept.npy']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in files}


def manifest(cfg,kind,directory,details):
    rows,_=load(cfg)
    artifacts={p.relative_to(directory).as_posix():sha(p) for p in sorted(directory.rglob('*'))
               if p.is_file() and p.name!='run_manifest.json'}
    write_json(directory/'run_manifest.json',dict(schema_version='3.0.0',method=kind,run_id=directory.name,
               dataset_id=cfg['dataset_id'],data_version=cfg['data_version'],split_id=cfg['split_id'],
               sequence_length=cfg['sequence_length'],tss_index_0based=cfg['tss_index_0based'],
               input_hashes=input_hashes(cfg),code_hashes=code_hashes(),config=cfg,
               code_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
               environment={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','torch','pandas']},
               train_ids_sha256=digest_ids(r['sample_id'] for r in rows if r['split']=='train'),
               validation_ids_sha256=digest_ids(r['sample_id'] for r in rows if r['split']=='val'),
               fit_subsets=['train'],selection_subsets=['val'] if kind!='thermo' else [],
               test_metrics_enabled=False,evidence_level='preliminary',details=details,artifacts=artifacts))


def predictions(cfg,kind,directory,rows,prediction,transform=None,raw=None):
    for subset in ('train','val'):
        out=[]
        for i,r in enumerate(rows):
            if r['split']!=subset: continue
            z=float(prediction[i]); require(np.isfinite(z),'Nonfinite prediction')
            out.append(dict(dataset_id=r['dataset_id'],data_version=r['data_version'],split_id=r['split_id'],
                            sample_id=r['sample_id'],method_name=kind,run_id=directory.name,subset=subset,
                            true_value=float(r['strength']),predicted_value=10**z,
                            true_value_log10=float(r['target_log10']),predicted_value_log10=z,
                            predicted_tx_rate=float(raw[i]) if raw is not None else '',
                            prediction_status='ok',sequence_length=87,target_scale_model='log10'))
        write_table(directory/f'predictions_{subset}.csv',out)
    write_table(directory/'metrics.csv',[dict(subset=s,**metrics([float(r['target_log10']) for r in rows if r['split']==s],
                             prediction[masks(rows)[s]])) for s in ('train','val')])


def run_ml(cfg,kind):
    require(kind in ('knn','ridge','svr'),'Invalid ML method')
    rows,transform=load(cfg); m=masks(rows); y=np.array([float(r['target_log10']) for r in rows])
    seqs=[r['sequence'] for r in rows]; directory=fresh(path(cfg,kind))
    search=[]; best=None; best_mse=np.inf
    candidates=cfg['search'][kind]
    feature_cache={}
    for params in candidates:
        if kind=='knn':
            key='physchem'; x=feature_cache.setdefault(key,physchem(seqs)) if key not in feature_cache else feature_cache[key]
            estimator=KNeighborsRegressor(n_neighbors=params['k'],weights='distance',n_jobs=cfg['num_threads'])
        else:
            key=params['k']
            if key not in feature_cache: feature_cache[key]=kmers(seqs,key,normalize=True)
            x=feature_cache[key]
            estimator=(Ridge(alpha=params['alpha'],solver='cholesky') if kind=='ridge'
                       else LinearSVR(C=params['C'],epsilon=0.0,dual='auto',max_iter=100000,
                                      tol=1e-5,random_state=cfg['seed']))
        model=make_pipeline(StandardScaler(),estimator)
        model.fit(x[m['train']],y[m['train']]); p=model.predict(x[m['val']])
        mse=float(mean_squared_error(y[m['val']],p)); search.append(dict(**params,val_mse_log10=mse))
        if mse<best_mse: best_mse=mse; best=(model,params.copy(),x)
    require(best is not None,'Empty search grid')
    model,params,x=best
    joblib.dump(dict(model=model,parameters=params,method=kind,input_hashes=input_hashes(cfg),
                     sequence_length=87,feature_policy='kmer frequency per valid window' if kind!='knn' else PHYS_NAMES),directory/'model.joblib')
    write_table(directory/'hyperparameter_search.csv',search); write_json(directory/'selected_parameters.json',params)
    write_table(directory/'fit_ids.tsv',[dict(sample_id=r['sample_id']) for r in rows if r['split']=='train'])
    predictions(cfg,kind,directory,rows,model.predict(x),transform)
    manifest(cfg,kind,directory,dict(parameters=params,feature_columns=PHYS_NAMES if kind=='knn' else 4**params['k'],
                                    scaler_fit_subset='train',selection_metric='validation MSE in log10 space'))
    print(f'{kind} selected {params}: validation RMSE={np.sqrt(best_mse):.4f}',flush=True)
    return directory


def cnn_class():
    sys.path.insert(0,str(ROOT/'CNN'))
    from pr02_cnn.model import PromoterCNN
    return PromoterCNN


def seed_torch(cfg):
    import torch
    torch.set_num_threads(cfg['num_threads']); random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
    torch.use_deterministic_algorithms(True)
    return torch


def train_network(cfg,x,y,train,val,model_cfg=None,initial=None):
    torch=seed_torch(cfg); model_cfg=model_cfg or cfg['cnn']['model']; settings=cfg['cnn']['training']
    model=cnn_class()(model_cfg)
    if initial is not None: model.load_state_dict(initial)
    xt=torch.from_numpy(x); yt=torch.as_tensor(y,dtype=torch.float32)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(xt[train],yt[train]),
            batch_size=settings['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(cfg['seed']))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    best=None; loss_best=np.inf; stale=0; history=[]
    for epoch in range(1,settings['max_epochs']+1):
        model.train(); total=0.0
        for batch,target in loader:
            optimizer.zero_grad(); loss=torch.mean((model(batch)-target)**2); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); total+=float(loss.detach())*len(batch)
        model.eval()
        with torch.no_grad(): val_loss=float(torch.mean((model(xt[val])-yt[val])**2))
        history.append(dict(epoch=epoch,train_mse=total/len(xt[train]),val_mse=val_loss))
        if val_loss<loss_best-settings['min_delta']:
            best=copy.deepcopy(model.state_dict()); loss_best=val_loss; stale=0
        else: stale+=1
        if epoch==1 or epoch%10==0: print(f'cnn epoch={epoch} val_mse={val_loss:.6f}',flush=True)
        if stale>=settings['patience']: break
    require(best is not None,'CNN produced no checkpoint'); model.load_state_dict(best); model.eval()
    return model,history


def cnn_predict(model,x,batch_size=256):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model(torch.from_numpy(x[i:i+batch_size])).numpy() for i in range(0,len(x),batch_size)])


def run_cnn(cfg):
    rows,transform=load(cfg); m=masks(rows); directory=fresh(path(cfg,'cnn'))
    x=encode([r['sequence'] for r in rows],cfg['sequence_length'])
    y=np.array([float(r['target_log10']) for r in rows]); low=transform['min_log10']; span=transform['max_log10']-low
    model,history=train_network(cfg,x,(y-low)/span,m['train'],m['val'])
    import torch
    torch.save(dict(state_dict=model.state_dict(),model_config=cfg['cnn']['model'],transform=transform,
                    input_hashes=input_hashes(cfg),dataset_id=cfg['dataset_id'],data_version=cfg['data_version']),directory/'model.pt')
    write_table(directory/'training_history.csv',history)
    predictions(cfg,'cnn',directory,rows,low+span*cnn_predict(model,x),transform)
    manifest(cfg,'cnn',directory,dict(best_epoch=min(history,key=lambda r:r['val_mse'])['epoch'],
                                     epochs=len(history),model=cfg['cnn']['model'],training=cfg['cnn']['training']))
    return directory


def load_cnn(cfg,length=None):
    import torch
    checkpoint=torch.load(path(cfg,'cnn')/'model.pt',map_location='cpu',weights_only=True)
    require(checkpoint['input_hashes']==input_hashes(cfg),'Checkpoint data identity mismatch')
    require(checkpoint['model_config']==cfg['cnn']['model'],'Checkpoint CNN architecture mismatch')
    architecture=dict(checkpoint['model_config'])
    if length is not None:
        require(architecture['pooling'].startswith('adaptive_'),'Variable-length transfer requires adaptive pooling')
        architecture['sequence_length']=length
    model=cnn_class()(architecture); model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model,checkpoint['transform']


def run_thermo(cfg):
    rows,transform=load(cfg); m=masks(rows); directory=fresh(path(cfg,'thermo'))
    states=thermal_states([r['sequence'] for r in rows],cfg['tss_index_0based'])
    verified=records()
    for actual,source in zip(states,verified):
        require(np.isclose(actual['Tx_rate'],source['promoter_calculator']['Tx_rate'],rtol=1e-12),'Thermodynamic runtime differs from verified source')
    raw=np.array([s['Tx_rate'] for s in states]); x=np.log10(raw); y=np.array([float(r['target_log10']) for r in rows])
    a,b=np.linalg.lstsq(np.column_stack([x[m['train']],np.ones(m['train'].sum())]),y[m['train']],rcond=None)[0]
    write_json(directory/'calibration.json',dict(coef_a=float(a),intercept_b=float(b),fit_subset='train',
               train_ids_sha256=digest_ids(r['sample_id'] for r in rows if r['split']=='train'),n_fit=int(m['train'].sum())))
    with (directory/'states.jsonl').open('w') as f:
        for r,s in zip(rows,states): f.write(json.dumps(dict(sample_id=r['sample_id'],annotation_status='model_inferred',**s))+'\n')
    predictions(cfg,'thermo',directory,rows,a*x+b,transform,raw)
    manifest(cfg,'thermo',directory,dict(tss_mode='fixed_tss',tss_index_0based=67,orientation='experimentally_verified',
               configurations_per_sample=30,spacer_lengths=[15,20],discriminator_lengths=[6,10],
               selection='minimum free energy among 30 states at known TSS',calibration_fit_subset='train',
               coverage=1.0,raw_validation_spearman=metrics(y[m['val']],x[m['val']])['spearman']))
    return directory


def evaluate(cfg):
    rows,_=load(cfg); expected=[r['sample_id'] for r in rows if r['split']=='val']; output=fresh(path(cfg,'comparison'))
    combined=[]; metric_rows=[]
    for kind in METHODS:
        predictions_in=read_table(path(cfg,kind)/'predictions_val.csv')
        require([r['sample_id'] for r in predictions_in]==expected,'Validation cohort/order mismatch: '+kind)
        require(all(r['data_version']==cfg['data_version'] and r['split_id']==cfg['split_id'] and r['method_name']==kind
                    and r['run_id']==cfg['runs'][kind] and r['subset']=='val' for r in predictions_in),'Prediction provenance mismatch')
        require(np.allclose([float(r['true_value']) for r in predictions_in],
                            [float(r['strength']) for r in rows if r['split']=='val']),'Validation label mismatch')
        values=metrics([float(r['true_value_log10']) for r in predictions_in],[float(r['predicted_value_log10']) for r in predictions_in])
        metric_rows.append(dict(method=kind,**values)); combined.extend(predictions_in)
    write_table(output/'metrics.csv',metric_rows); write_table(output/'predictions_val.csv',combined)
    write_table(output/'common_val_ids.tsv',[dict(sample_id=x) for x in expected])
    write_json(output/'comparison_manifest.json',dict(sequence_length=87,shared_context=True,n_common_val=len(expected),
               test_evaluated=False,methods=list(METHODS),comparison_scale='original course strength labels in log10 space',
               context_leakage_note='Inherited random split; nearby genomic contexts may overlap. Historical test is not a fresh blind benchmark.'))
    manifest(cfg,'comparison',output,dict(n_common_val=len(expected),test_evaluated=False))
    return output
