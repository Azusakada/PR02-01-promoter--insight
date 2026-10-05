"""Independent checks of shared contracts, calibration, models and comparison metrics."""
from __future__ import annotations
import hashlib
import math
import json
import subprocess
import sys
import joblib
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_absolute_error

from .common import ROOT, DATA, SPLITS, TRANSFORM, load_bundle, read_json, read_table, resolve, v, write_json
from .evaluate import align_predictions


def check_prediction_sources(cfg, paths, frames):
    """Reject a valid CSV that belongs to another run or CNN plan."""
    import yaml
    cnn = yaml.safe_load(resolve(cfg['cnn_config']).read_text(encoding='utf-8-sig'))
    expected = {
        'knn_physchem_full': (cfg['runs']['knn'], ROOT/'runs'/cfg['runs']['knn']/'predictions_knn_val.csv'),
        'thermo_regseq2': (cfg['runs']['thermo'], ROOT/'runs'/cfg['runs']['thermo']/'predictions/thermo_val_calibrated.csv'),
        'cnn_1d': (cnn['run_id'], resolve(cnn['output_dir'])/'validation/predictions_cnn.csv'),
    }
    for kind in ['ridge','svr']:
        if kind in cfg['runs']:
            directory=ROOT/'runs'/cfg['runs'][kind]
            manifest=read_json(directory/'model_manifest.json')
            expected[manifest['method_name']]=(cfg['runs'][kind],directory/f'predictions_{kind}_val.csv')
    v.require(len(paths)==len(expected),'Prediction plan must include exactly the configured methods')
    for path,rows in zip(paths,frames):
        method=rows[0]['method_name']
        v.require(method in expected,'Unconfigured prediction method: '+method)
        run_id,source=expected[method]
        v.require(path.resolve()==source.resolve() and rows[0]['run_id']==run_id,
                  'Prediction source does not match configured run: '+method)
    cnn_meta=read_json(resolve(cnn['output_dir'])/'model_manifest.json')
    v.require(cnn_meta['model_id']==cnn['run_id'] and cnn_meta['model_config']==cnn['model']
              and cnn_meta['seed']==cnn['training']['seed'],'CNN configuration does not match saved model')
    return cnn


def check_analysis(directory):
    """Use the standalone numeric verifier without changing published artifacts."""
    completed=subprocess.run([sys.executable,str(ROOT/'analysis_m2m3/verify_run.py'),str(directory)],
                             cwd=ROOT,check=True,capture_output=True,text=True,encoding='utf-8')
    return json.loads(completed.stdout)


def verify(cfg):
    samples,groups,transform = load_bundle()
    paths = [resolve(p) for p in cfg['validation_predictions']]
    frames,common,_ = align_predictions(paths)
    check_prediction_sources(cfg,paths,frames)
    result = dict(status='passed',samples=len(samples),split_counts={s:len(rs) for s,rs in groups.items()},
                  transform_id=transform['transform_id'],common_val_count=len(common),test_evaluated=False,checks=[])
    def record(name,detail=True): result['checks'].append(dict(name=name,status='passed',detail=detail))
    record('dataset_splits_shared_transform')
    from .data import verify_sources
    record('original_course_arrays_and_cleaned_records',verify_sources())
    record('configured_prediction_sources_and_cnn_model')
    from .thermo import INPUTS,check_execution_policy
    record('thermo_input_bundle',v.bundle_check(DATA,SPLITS,calculator_inputs=INPUTS))
    check_execution_policy(v.load_table('calculator_inputs',INPUTS))
    knn_dir = ROOT/'runs'/cfg['runs']['knn']
    thermo_dir = ROOT/'runs'/cfg['runs']['thermo']
    comparison_dir = ROOT/'runs'/cfg['runs']['comparison']
    cnn_path = next(p for p,rows in zip(paths,frames) if rows[0]['method_name']=='cnn_1d')
    cnn_dir = cnn_path.parent.parent
    directories=[knn_dir,thermo_dir,cnn_dir,cnn_dir/'validation',comparison_dir]
    directories.extend(ROOT/'runs'/cfg['runs'][kind] for kind in ['ridge','svr'] if kind in cfg['runs'])
    for directory in directories:
        record('run_manifest:'+directory.name,v.run_check(directory/'run_manifest.json',ROOT))
    for path in paths: record('prediction_bundle:'+path.relative_to(ROOT).as_posix(),v.bundle_check(DATA,SPLITS,transform=TRANSFORM,predictions=path))

    # Calibration is independently reproduced by numpy least squares using successful train requests only.
    raw_train = v.load_table('predictions',thermo_dir/'predictions/thermo_train_raw.csv')
    tr = [r for r in raw_train if r['prediction_status']=='ok']
    xs = np.log10([r['predicted_tx_rate'] for r in tr])
    ys = np.log10([r['true_value'] for r in tr])
    a,b = np.linalg.lstsq(np.column_stack([xs,np.ones(len(xs))]),ys,rcond=None)[0]
    calibration = read_json(thermo_dir/'thermo_calibration.json')
    v.require(np.allclose([a,b],[calibration['coef_a'],calibration['intercept_b']],rtol=1e-10,atol=1e-10),'Calibration not reproduced')
    v.require(calibration['n_fit']==len(tr) and {r['sample_id'] for r in tr}<={s['sample_id'] for s in groups['train']},'Calibration fit IDs wrong')
    fit_hash=hashlib.sha256(''.join(s+'\n' for s in sorted(r['sample_id'] for r in tr)).encode()).hexdigest()
    v.require(calibration['train_ids_sha256']==fit_hash and calibration['fit_subset']=='train',
              'Calibration training identity mismatch')
    for sub in ['train','val','test']:
        raw = v.load_table('predictions',thermo_dir/f'predictions/thermo_{sub}_raw.csv')
        cal = v.load_table('predictions',thermo_dir/f'predictions/thermo_{sub}_calibrated.csv')
        v.bundle_check(DATA,SPLITS,predictions=thermo_dir/f'predictions/thermo_{sub}_calibrated.csv')
        v.require(len(raw)==len(cal)==len(groups[sub]),'Calibration dropped requests')
        rm = {r['sample_id']:r for r in raw}
        for r in cal:
            source = rm[r['sample_id']]
            if source['prediction_status']=='failed':
                v.require(r['prediction_status']=='failed' and r['predicted_value'] is None and r['predicted_tx_rate'] is None,'Failed request lost/filled')
            elif r['prediction_status']=='ok':
                v.require(math.isclose(r['predicted_value_log10'],a*math.log10(source['predicted_tx_rate'])+b,rel_tol=1e-9,abs_tol=1e-9),'Calibration prediction mismatch')
    record('train_only_calibration_and_full_request_coverage',dict(n_fit=len(tr),coef_a=float(a),intercept_b=float(b)))

    from .knn import feature_map
    mapped = feature_map(samples)
    saved = joblib.load(knn_dir/'knn_model.joblib')
    x = np.stack([mapped[s['sample_id']] for s in groups['val']])
    recomputed = saved['model'].predict(saved['scaler'].transform(x))
    knn_frame = next(rs for rs in frames if rs[0]['method_name']=='knn_physchem_full')
    expected = {r['sample_id']:r['predicted_value_log10'] for r in knn_frame}
    expected_knn = np.asarray([expected[s['sample_id']] for s in groups['val']])
    v.require(np.allclose(recomputed,expected_knn,rtol=1e-12,atol=1e-12),'KNN saved model does not reproduce validation')
    train_x = np.stack([mapped[s['sample_id']] for s in groups['train']])
    v.require(np.allclose(saved['scaler'].mean_,train_x.mean(axis=0),rtol=1e-12,atol=1e-12),'Scaler not fitted on train only')
    record('knn_saved_train_only_model_reload',dict(n_compared=len(recomputed),max_abs_difference=float(np.max(np.abs(recomputed-expected_knn)))))

    if 'ridge' in cfg['runs']:
        from .kmer import verify_models
        for name,detail in verify_models(cfg,frames): record(name,detail)

    # CNN checks use the actual frozen checkpoint and all validation sequences.
    sys.path.insert(0,str(ROOT/'CNN'))
    import torch
    from pr02_cnn.common import ModelArtifact
    from pr02_cnn.predict import load_model
    from pr02_cnn.data import one_hot
    model,meta = load_model(ModelArtifact(cnn_dir/'best_checkpoint.pt',cnn_dir/'model_manifest.json',cnn_dir/'cnn_run_config.yaml'),'cpu')
    v.require(meta['samples_sha256']==v.sha256(DATA) and meta['splits_sha256']==v.sha256(SPLITS)
              and meta['transform']==transform,'CNN model is bound to different frozen inputs')
    for sub in ['train','val']:
        ids=sorted(s['sample_id'] for s in groups[sub])
        v.require(meta[sub]['ids']==ids and meta[sub]['n_samples']==len(ids), 'CNN fit/selection IDs mismatch')
    model.eval()
    torch.set_num_threads(4)
    x = torch.stack([one_hot(s['sequence']) for s in groups['val']])
    with torch.no_grad(): normalized = model(x).cpu().numpy().reshape(-1)
    cnn_frame = next(rs for rs in frames if rs[0]['method_name']=='cnn_1d')
    expected = {r['sample_id']:r['predicted_value_normalized'] for r in cnn_frame}
    v.require(np.allclose(normalized,np.asarray([expected[s['sample_id']] for s in groups['val']]),rtol=1e-6,atol=1e-7),'CNN checkpoint reload mismatch')
    record('cnn_shared_transform_and_checkpoint_reload',dict(transform_id=meta['transform']['transform_id'],n_compared=len(normalized)))

    metric_table = v.load_table('metrics',comparison_dir/'metrics.csv')
    common_set = set(common)
    for rows in frames:
        used = [r for r in rows if r['sample_id'] in common_set]
        for scale,field in [('log10','predicted_value_log10'),('raw','predicted_value')]:
            y = np.asarray([math.log10(r['true_value']) if scale=='log10' else r['true_value'] for r in used])
            p = np.asarray([r[field] for r in used])
            refs = dict(r2=float(r2_score(y,p)),mae=float(mean_absolute_error(y,p)),spearman=float(spearmanr(y,p).statistic))
            for name,value in refs.items():
                matched = [m for m in metric_table if m['method_name']==rows[0]['method_name'] and m['target_scale']==scale and m['metric_name']==name]
                v.require(len(matched)==1 and math.isclose(matched[0]['value'],value,rel_tol=1e-10,abs_tol=1e-10),'Unified metric mismatch')
                v.require(matched[0]['n_requested']==len(rows) and matched[0]['n_used']==len(common),'Wrong comparison denominators')
    record('unified_metrics_independently_recomputed_with_sklearn_scipy')
    for name,path in cfg.get('analysis_runs',{}).items():
        directory=resolve(path)
        detail=v.run_check(directory/'run_manifest.json',ROOT)
        analysis_meta=read_json(directory/'run_manifest.json')
        v.require(all(analysis_meta[key]==transform[key] for key in ['dataset_id','data_version','split_id']),
                  'Analysis uses another data version or split: '+name)
        figures=read_json(directory/'figure_manifest.json')
        v.require(all(f['visual_check_status']=='pass' for f in figures),'Analysis figures need manual review before publication: '+str(directory))
        if name=='errors':
            meta=read_json(directory/'run_manifest.json')
            input_paths={entry['path'] for entry in meta['inputs']}
            v.require(set(cfg['validation_predictions'])<=input_paths,'Error analysis uses stale model predictions')
            v.require(all(f['n_samples']==len(common) for f in figures),'Error figure comparison count mismatch')
            error_ids=read_table(directory/'source_tables/common_eval_ids.tsv',delimiter='\t')
            metric_ids={m['comparison_set_id'] for m in metric_table}
            v.require(any(r['comparison_set_id'] in metric_ids for r in error_ids),'Error analysis comparison identity mismatch')
        record('analysis_artifacts_and_visual_review:'+name,detail)
        record('analysis_numeric_verification:'+name,check_analysis(directory))
    current_index = ROOT/'results/current.json'
    if current_index.exists():
        registry = read_json(current_index)
        if registry['comparison_run_id']==cfg['runs']['comparison']:
            v.require({entry['original']['path'] for entry in registry['predictions']}==set(cfg['validation_predictions']),
                      'Current registry and project prediction paths disagree')
            v.require({entry['path'] for entry in registry.get('analysis',[])}==
                      {resolve(path).relative_to(ROOT).as_posix()+'/run_manifest.json'
                       for path in cfg.get('analysis_runs',{}).values()},
                      'Current registry and project analysis paths disagree')
        entries = [registry[key] for key in ['data','splits','label_transform','comparison_manifest','validation']]
        entries.extend(registry['artifacts'])
        entries.extend(registry.get('legacy_entrypoints',[]))
        entries.extend(registry.get('analysis',[]))
        entries.extend(registry.get('data_sources',[]))
        for pred in registry['predictions']:
            entries.extend([pred['original'],pred['current']])
            v.require(resolve(pred['original']['path']).read_bytes()==resolve(pred['current']['path']).read_bytes(),'Current prediction alias differs from immutable run')
        for entry in entries:
            v.require(v.sha256(resolve(entry['path']))==entry['sha256'],'Current registry hash mismatch: '+entry['path'])
    out = comparison_dir/'integration_validation.json'
    # Published validation evidence is hash-bound. A later independent verify
    # must not rewrite it because platform-dependent numerical differences can
    # change diagnostics without changing whether validation passed.
    if not out.exists(): write_json(out,result)
    print(f'Integrated verification passed: {out}',flush=True)
    return result
