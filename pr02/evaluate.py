"""Strict ID alignment, full-request coverage and identical evaluation cohorts."""
from __future__ import annotations
import hashlib
from .common import (DATA, SPLITS, TRANSFORM, artifact, load_bundle, relative, run_dir,
                     v, write_json, write_manifest, write_table)
from .metrics import metric_rows


def align_predictions(paths, subset='val'):
    _, groups, _ = load_bundle()
    frames = []
    for path in paths:
        v.bundle_check(DATA,SPLITS,transform=TRANSFORM,predictions=path)
        rows = v.load_table('predictions',path)
        v.require(rows[0]['subset']==subset,'Mismatched evaluation subset')
        v.require(rows[0]['target_scale_model']!='tool_raw' or rows[0]['calibration_id'] is not None,
                  'Strength comparison requires train-calibrated thermo predictions')
        frames.append(rows)
    methods = [rs[0]['method_name'] for rs in frames]
    v.require(len(methods)==len(set(methods)) and len(methods)>=2,'Need distinct methods')
    common_ids = set.intersection(*[{r['sample_id'] for r in rows if r['prediction_status']=='ok'} for rows in frames])
    v.require(bool(common_ids),'Empty common evaluation cohort')
    return frames,sorted(common_ids),groups[subset]


def run(paths,run_id='comparison_val_integrated_20261002_v2'):
    frames,common,requested = align_predictions(paths)
    output = run_dir(run_id)
    comparison = 'val_common_'+hashlib.sha256(''.join(s+'\n' for s in common).encode()).hexdigest()[:16]
    metrics,coverage,joined = [],[],[]
    common_set = set(common)
    for path,rows in zip(paths,frames):
        method = rows[0]['method_name']
        metrics.extend(metric_rows(rows,'val',comparison_set_id=comparison,common_ids=common_set))
        success = sum(r['prediction_status']=='ok' for r in rows)
        coverage.append(dict(method_name=method,run_id=rows[0]['run_id'],subset='val',n_requested=len(rows),
                             n_success=success,n_failed=len(rows)-success,n_used=len(common),coverage=success/len(rows),
                             comparison_set_id=comparison,predictions_file=relative(path)))
        joined.extend(dict(sample_id=r['sample_id'],method_name=method,true_value=r['true_value'],
                           predicted_value_log10=r['predicted_value_log10'],comparison_set_id=comparison)
                      for r in rows if r['sample_id'] in common_set)
    write_table(output/'metrics.csv',metrics)
    v.load_table('metrics',output/'metrics.csv')
    write_table(output/'coverage_summary.csv',coverage)
    write_table(output/'common_eval_ids.tsv',[{'sample_id':s} for s in common],delimiter='\t')
    write_table(output/'comparison_predictions.csv',joined)
    table = []
    for c in coverage:
        values = {m['metric_name']:m['value'] for m in metrics if m['method_name']==c['method_name'] and m['target_scale']=='log10'}
        table.append(dict(method_name=c['method_name'],log10_r2=values['r2'],spearman=values['spearman'],log10_mae=values['mae'],
                          n_requested=c['n_requested'],n_success=c['n_success'],n_common=c['n_used'],coverage=c['coverage']))
    write_table(output/'performance_summary.csv',table)
    write_json(output/'comparison_config.json',dict(subset='val',target_scale='log10',fit_protocol='train only for every method',
               comparison_set_id=comparison,n_common=len(common),prediction_files=[relative(p) for p in paths],test_metrics_enabled=False))
    artifacts = [artifact(kind,output/name) for kind,name in [('comparison_metrics','metrics.csv'),('coverage','coverage_summary.csv'),
                 ('common_eval_ids','common_eval_ids.tsv'),('comparison_predictions','comparison_predictions.csv'),
                 ('summary','performance_summary.csv'),('config','comparison_config.json')]]
    artifacts.extend(artifact('prediction_input',p) for p in paths)
    artifacts.extend([artifact('dataset',DATA),artifact('splits',SPLITS),artifact('label_transform',TRANSFORM)])
    # A method failing one request does not make the comparison computation fail.
    # The contract requires a partial run when declared prediction artifacts include failures.
    write_manifest(output,run_id,'unified_evaluation',artifacts,parameters=dict(subset='val',comparison_set_id=comparison),
                   status='partial' if any(c['n_failed'] for c in coverage) else 'success')
    print(f'Comparison completed: {output}; common={len(common)}',flush=True)
    return output
