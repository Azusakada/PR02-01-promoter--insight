"""Traditional models and regression on independently exported frozen DNA features."""
from __future__ import annotations
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .data import ROOT, load, read_json, sha, write_json, write_table
from .experiment import masks
from .features import kmers
from .m4_data_eval import bundle, completed, export_predictions, finish, read_bundle, run_root


def fit_head(x, y, train, val, alphas=(1., 10., 100.)):
    best = None; loss_best = np.inf; search = []
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver='cholesky'))
        model.fit(x[train], y[train]); loss = float(np.mean((model.predict(x[val])-y[val])**2))
        search.append(dict(alpha=alpha, val_mse_log10=loss))
        if loss < loss_best: best = model; loss_best = loss
    return best, search


def reference(directory, kind, cfg, details):
    write_json(directory/'model_ref.json', dict(model_kind=kind, method_name=kind,
        run_id=directory.relative_to(ROOT).as_posix(), checkpoint_path=(directory/'model.joblib').relative_to(ROOT).as_posix(),
        checkpoint_sha256=sha(directory/'model.joblib'), path_base='repository', input_length=87,
        seed=cfg['seed'], predictor_entry='pr02.context87.m4_pipeline:predict_log10', details=details))


def run_traditional(cfg, protocol):
    root = run_root(protocol)/'ml'; rows, _ = load(cfg); m = masks(rows)
    y = np.array([float(r['target_log10']) for r in rows]); seqs = [r['sequence'] for r in rows]
    cache = {}
    for kind in ('ridge', 'svr'):
        directory = root/kind
        if completed(directory): continue
        directory.mkdir(parents=True, exist_ok=True); searches = []; best = None; best_loss = np.inf
        for k in protocol['ml_k']:
            if k not in cache: cache[k] = kmers(seqs, k).astype(np.float32)
            x = cache[k]
            for value in protocol['ridge_alpha' if kind == 'ridge' else 'svr_C']:
                estimator = Ridge(alpha=value, solver='cholesky') if kind == 'ridge' else LinearSVR(
                    C=value, epsilon=0., loss='squared_epsilon_insensitive', dual=False,
                    max_iter=50000, tol=1e-5, random_state=cfg['seed'])
                model = make_pipeline(StandardScaler(), estimator)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x[m['train']], y[m['train']])
                converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
                if not converged: raise RuntimeError('Predeclared LinearSVR did not converge')
                loss = float(np.mean((model.predict(x[m['val']])-y[m['val']])**2))
                params = dict(k=k, **{'alpha' if kind == 'ridge' else 'C': value})
                searches.append(dict(**params, val_mse_log10=loss, converged=converged,
                                     n_iterations=getattr(estimator, 'n_iter_', None)))
                if loss < best_loss: best_loss = loss; best = (model, params, x)
        model, params, x = best
        joblib.dump(dict(model=model, parameters=params, method=kind), directory/'model.joblib')
        write_json(directory/'selected_parameters.json', params); write_table(directory/'hyperparameter_search.csv', searches)
        export_predictions(cfg, kind, directory, rows, model.predict(x))
        write_table(directory/'fit_ids.tsv', [dict(sample_id=r['sample_id']) for r in rows if r['split'] == 'train'])
        reference(directory, kind, cfg, dict(parameters=params, scaler_fit_subset='train', selection='validation MSE'))
        finish(directory, cfg, dict(parameters=params, convergence_checked=True, representation='ACGT lexicographic kmer frequency'))
        print('M4', kind, params, best_loss, flush=True)
    if not (root/'kmer/feature_manifest.json').exists():
        k = read_json(root/'ridge/selected_parameters.json')['k']
        bundle(root/'kmer', cache.get(k, kmers(seqs, k)), rows, 'kmer_frequency', cfg,
               dict(k=k, vocabulary='lexicographic ACGT', normalization='divide by length-k+1'))
    return root


def run_dna_head(cfg, protocol):
    root = run_root(protocol); directory = root/'ml/dnabert2'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); m = masks(rows); y = np.array([float(r['target_log10']) for r in rows])
    x = read_bundle(root/'dna_features', [r['sample_id'] for r in rows])
    model, search = fit_head(x, y, m['train'], m['val'], protocol['ridge_alpha'])
    joblib.dump(dict(model=model, feature_source='dna_features', method='dnabert2_ridge'), directory/'model.joblib')
    write_table(directory/'hyperparameter_search.csv', search)
    write_table(directory/'fit_ids.tsv', [dict(sample_id=r['sample_id']) for r in rows if r['split'] == 'train'])
    export_predictions(cfg, 'dnabert2_ridge', directory, rows, model.predict(x))
    reference(directory, 'dnabert2_ridge', cfg, dict(encoder=protocol['dna_model'], revision=protocol['dna_revision'],
               encoder_frozen=True, scaler_fit_subset='train', feature_manifest=(root/'dna_features/feature_manifest.json').relative_to(ROOT).as_posix()))
    finish(directory, cfg, dict(encoder_frozen=True, selected_alpha=float(model[-1].alpha)), [root/'dna_features/feature_manifest.json'])
    return directory
