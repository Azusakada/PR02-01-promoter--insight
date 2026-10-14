"""A0 kmer, A1 +CNN, A2 +DNA, A3 +training-attribution-guided features."""
from __future__ import annotations
import joblib
import numpy as np
from .data import ROOT, load, read_json, read_table, write_json, write_table
from .experiment import masks, metrics
from .m4_data_eval import SEEDS, completed, export_predictions, finish, read_bundle, run_root
from .m4_ml_dnalm import fit_head


def run(cfg, protocol):
    root = run_root(protocol); directory = root/'ablation'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); ids = [r['sample_id'] for r in rows]; m = masks(rows)
    y = np.array([float(r['target_log10']) for r in rows])
    kmer = read_bundle(root/'ml/kmer', ids); dna = read_bundle(root/'dna_features', ids)
    result = []; all_search = []
    for seed in SEEDS:
        cnn = read_bundle(root/f'cnn/features_s{seed}', ids); guided = read_bundle(root/f'guided_s{seed}', ids)
        for stage, arrays in [('A0', [kmer]), ('A1', [kmer, cnn]), ('A2', [kmer, cnn, dna]), ('A3', [kmer, cnn, dna, guided])]:
            if stage == 'A0' and seed != SEEDS[0]: continue
            output = directory/(stage if stage == 'A0' else f'{stage}_s{seed}')
            if completed(output):
                result.append(read_json(output/'stage_result.json')); continue
            output.mkdir(parents=True, exist_ok=True); x = np.column_stack(arrays)
            model, search = fit_head(x, y, m['train'], m['val'], protocol['ridge_alpha'])
            joblib.dump(model, output/'model.joblib'); write_table(output/'search.csv', search)
            p = model.predict(x); export_predictions(cfg, 'fusion_'+stage, output, rows, p)
            row = dict(stage=stage, seed=seed if stage != 'A0' else None, n_features=x.shape[1], alpha=float(model[-1].alpha), **metrics(y[m['val']], p[m['val']]))
            write_json(output/'stage_result.json', row); result.append(row)
            finish(output, cfg, dict(stage=stage, seed=row['seed'], fit_subset='train', selection='validation MSE', n_features=x.shape[1]))
    summary = []
    for stage in ['A0', 'A1', 'A2', 'A3']:
        rr = [r for r in result if r['stage'] == stage]; record = dict(stage=stage, n_runs=len(rr), n=rr[0]['n'])
        for metric in ['r2_log10', 'spearman', 'rmse_log10', 'mae_log10']:
            a = [r[metric] for r in rr]; record[metric+'_mean'] = float(np.mean(a)); record[metric+'_sd'] = float(np.std(a, ddof=1)) if len(a) > 1 else None
        summary.append(record)
    for i, row in enumerate(summary):
        row['delta_r2_from_previous'] = row['r2_log10_mean']-summary[i-1]['r2_log10_mean'] if i else None
        row['delta_rmse_from_previous'] = row['rmse_log10_mean']-summary[i-1]['rmse_log10_mean'] if i else None
    write_table(directory/'stage_metrics.csv', result); write_table(directory/'stage_summary.csv', summary)
    finish(directory, cfg, dict(stages=['kmer', '+CNN', '+DNA', '+training-IG-guided-regions'],
        head='StandardScaler(train) + Ridge, alpha 1/10/100 selected on val',
        interpretation='incremental representation comparison, distinct from standalone CNN/DNA predictors; validation reused for selection'),
        [root/'ml/kmer/feature_manifest.json', root/'dna_features/feature_manifest.json']+
        [root/f'guided_s{seed}/feature_recipe.json' for seed in SEEDS])
    return directory
