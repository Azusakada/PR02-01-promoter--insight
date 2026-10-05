"""Promote validated immutable runs into the team's current result index."""
from __future__ import annotations
import shutil
from datetime import datetime, timezone, timedelta
from .common import ROOT, DATA, SPLITS, TRANSFORM, artifact, read_json, read_table, relative, resolve, v, write_json, write_table
from .verify import verify


def publish(cfg):
    validation = verify(cfg)
    comparison = ROOT/'runs'/cfg['runs']['comparison']
    current = ROOT/'results'
    current.mkdir(exist_ok=True)
    if (current/'current.json').exists():
        old = read_json(current/'current.json')
        if old['comparison_run_id']!=cfg['runs']['comparison']:
            archive = ROOT/'history'/('current_results_'+old['comparison_run_id'])
            v.require(not archive.exists(),'Previous current-result archive already exists')
            shutil.copytree(current,archive)
    for name in ['metrics.csv','coverage_summary.csv','common_eval_ids.tsv','performance_summary.csv','metrics_unique_context.csv','unique_context_eval_ids.tsv']:
        shutil.copy2(comparison/name,current/name)
    predictions = []
    for source in map(resolve,cfg['validation_predictions']):
        rows = v.load_table('predictions',source)
        destination = current/'predictions'/f"{rows[0]['method_name']}_val.csv"
        destination.parent.mkdir(exist_ok=True)
        shutil.copy2(source,destination)
        predictions.append(dict(method_name=rows[0]['method_name'],run_id=rows[0]['run_id'],
                                n_requested=len(rows),n_success=sum(r['prediction_status']=='ok' for r in rows),
                                original=artifact('predictions',source),current=artifact('predictions',destination)))
    knn = ROOT/'runs'/cfg['runs']['knn']
    for src,dst in [('predictions_knn_val.csv','predictions_knn.csv'),('metrics.csv','metrics.csv'),
                    ('knn_model.joblib','knn_model.joblib'),('model_manifest.json','knn_config.json')]:
        shutil.copy2(knn/src,ROOT/'KNN/results'/dst)
    knn_rows = v.load_table('predictions',knn/'predictions_knn_val.csv')
    n_success = sum(r['prediction_status']=='ok' for r in knn_rows)
    write_table(ROOT/'KNN/results/common_eval_ids.tsv',[{'sample_id':r['sample_id']} for r in knn_rows if r['prediction_status']=='ok'],delimiter='\t')
    write_table(ROOT/'KNN/results/coverage_summary.csv',[dict(method_name='knn_physchem_full',subset='val',n_requested=len(knn_rows),n_success=n_success,n_failed=len(knn_rows)-n_success,coverage=n_success/len(knn_rows))])
    legacy=[artifact('current_knn_alias',ROOT/'KNN/results'/name) for name in ['predictions_knn.csv','metrics.csv','knn_model.joblib','knn_config.json','common_eval_ids.tsv','coverage_summary.csv']]
    if 'ridge' in cfg['runs']:
        coverage=[]; used=[]
        for kind in ['ridge','svr']:
            directory=ROOT/'runs'/cfg['runs'][kind]
            copies=[(f'predictions_{kind}_val.csv',f'predictions_{kind}.csv'),('metrics.csv',f'{kind}_metrics.csv'),
                    (f'{kind}_model.joblib',f'{kind}_model.joblib'),('model_manifest.json',f'{kind}_config.json')]
            for src,dst in copies:
                shutil.copy2(directory/src,ROOT/'Ridge/results'/dst)
                legacy.append(artifact('current_kmer_alias',ROOT/'Ridge/results'/dst))
            rows=v.load_table('predictions',directory/f'predictions_{kind}_val.csv')
            method=rows[0]['method_name']; n=sum(r['prediction_status']=='ok' for r in rows)
            coverage.append(dict(method_name=method,run_id=cfg['runs'][kind],subset='val',n_requested=len(rows),n_success=n,n_failed=len(rows)-n,coverage=n/len(rows)))
            used.extend(dict(comparison_set_id=cfg['runs'][kind]+'_val_available',sample_id=r['sample_id'],dataset_id=r['dataset_id'],split_id=r['split_id'],subset='val',method_name=method,prediction_status=r['prediction_status']) for r in rows if r['prediction_status']=='ok')
        write_table(ROOT/'Ridge/results/coverage_summary.csv',coverage)
        write_table(ROOT/'Ridge/results/common_eval_ids.tsv',used,delimiter='\t')
        legacy.extend(artifact('current_kmer_alias',ROOT/'Ridge/results'/name) for name in ['coverage_summary.csv','common_eval_ids.tsv'])
    shutil.copy2(comparison/'integration_validation.json',ROOT/'reports/integration_validation.json')
    registry = dict(schema_version='2.0.0',published_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    comparison_run_id=cfg['runs']['comparison'],evidence_level='preliminary',test_metrics_enabled=False,
                    data=artifact('dataset',DATA),splits=artifact('splits',SPLITS),label_transform=artifact('label_transform',TRANSFORM),
                    transform_id=validation['transform_id'],predictions=predictions,
                    comparison_manifest=artifact('run_manifest',comparison/'run_manifest.json'),
                    artifacts=[artifact('current_result',current/name) for name in ['metrics.csv','coverage_summary.csv','common_eval_ids.tsv','performance_summary.csv','metrics_unique_context.csv','unique_context_eval_ids.tsv']],
                    validation=artifact('validation',comparison/'integration_validation.json'),
                    legacy_entrypoints=legacy)
    registry['analysis']=[artifact('analysis_'+name,resolve(path)/'run_manifest.json') for name,path in cfg.get('analysis_runs',{}).items()]
    from .data import PROFILE,RAW
    profile=read_json(PROFILE)
    registry['data_sources']=[artifact('data_profile',PROFILE)]+[
        artifact('raw_'+key,RAW/profile['raw_files'][key]['filename']) for key in ['sequence','strength']]
    write_json(current/'current.json',registry)
    return registry
