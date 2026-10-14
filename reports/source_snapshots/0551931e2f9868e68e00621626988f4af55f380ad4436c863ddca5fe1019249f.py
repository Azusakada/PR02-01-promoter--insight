"""Versioned M4 data, feature and evaluation contracts (no source-test scoring)."""
from __future__ import annotations

import importlib.metadata
import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .data import ROOT, TARGET, digest_ids, load, read_json, read_table, require, sha, write_json, write_table
from .experiment import code_hashes, git_revision, input_hashes, masks, metrics, predictions

SEEDS = (20260928, 20260929, 20260930)
REGIONS = {'UP': (7, 31), '-35': (32, 38), 'spacer': (38, 55), '-10': (55, 61)}


def seeds(protocol):
    return tuple(protocol.get('seeds', SEEDS))


def config_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def validate_protocol(protocol):
    require(protocol.get('source_test_evaluated') is False, 'Source test scoring must remain disabled')
    values = seeds(protocol)
    require(bool(values) and len(values) == len(set(values)) and all(type(s) is int and 0 <= s < 2**32 for s in values), 'Invalid seeds')
    require(type(protocol['bootstrap_repeats']) is int and protocol['bootstrap_repeats'] >= 100, 'Invalid bootstrap repeats')
    require(type(protocol['bootstrap_seed']) is int and 0 <= protocol['bootstrap_seed'] < 2**32, 'Invalid bootstrap seed')
    require(0 < protocol['confidence'] < 1, 'Invalid confidence level')
    require(bool(protocol['ridge_alpha']) and all(a > 0 for a in protocol['ridge_alpha']), 'Invalid Ridge grid')
    require(bool(protocol['top_k']) and all(type(k) is int and 1 <= k <= 87 for k in protocol['top_k']), 'Invalid top-k')
    require(type(protocol['attribution_train_n']) is int and protocol['attribution_train_n'] > 0, 'Invalid attribution train count')
    require(type(protocol['attribution_val_n']) is int and protocol['attribution_val_n'] > 0, 'Invalid attribution val count')
    require(type(protocol['ig_steps']) is int and protocol['ig_steps'] >= 2, 'Invalid IG steps')
    allowed = {'C0', 'C1', 'C2', 'C3'}
    grid = protocol['cnn_configurations']
    require(bool(grid) and len(grid) == len(set(grid)) and set(grid) <= allowed, 'Invalid CNN configurations')


def effective_protocol(directory):
    for parent in (Path(directory), *Path(directory).parents):
        file = parent/'protocol/protocol.json'
        if file.exists(): return read_json(file)
    return None


def run_root(protocol):
    output = ROOT / protocol['output']
    require(output.resolve() != (ROOT/'runs').resolve() and output.resolve().is_relative_to((ROOT/'runs').resolve()), 'M4 output must be a child of repository runs/')
    return output


def finish(directory, cfg, details=None, upstream=()):
    directory = Path(directory)
    protocol = effective_protocol(directory)
    executed = code_hashes()
    for name, digest in executed.items():
        path = ROOT/name
        snapshot = ROOT/'reports/source_snapshots'/(digest+path.suffix)
        if not snapshot.exists():
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, snapshot)
    write_json(directory/'run_manifest.json', dict(
        schema_version='m4_interface_v2', run_id=directory.relative_to(ROOT).as_posix(),
        input_hashes=input_hashes(cfg), code_commit=git_revision(),
        config_sha256=config_digest(cfg), protocol_sha256=config_digest(protocol),
        code_hashes=executed,
        environment={p: importlib.metadata.version(p) for p in ['numpy', 'scipy', 'scikit-learn', 'torch']},
        source_test_evaluated=False, details=details or {},
        upstream={str(Path(p).relative_to(ROOT)): sha(p) for p in upstream},
        artifacts={p.relative_to(directory).as_posix(): sha(p) for p in sorted(directory.rglob('*'))
                   if p.is_file() and p != directory/'run_manifest.json'}))


def verify_sealed_release(root):
    """Historical executable/docs may be archived; numerical outputs must stay in place."""
    file = Path(root)/'release_manifest.json'
    if not file.exists(): return []
    archived = []
    for name, expected in read_json(file)['artifacts'].items():
        path = ROOT/name
        if path.is_file() and sha(path) == expected: continue
        eligible = name.startswith('pr02/') or name == 'README.md'
        snapshot = ROOT/'reports/release_snapshots'/(expected+path.suffix)
        require(eligible and snapshot.is_file() and sha(snapshot) == expected, 'Release artifact mismatch: '+name)
        archived.append(name)
    return archived


def completed(directory, cfg=None, protocol=None, archived_code=False):
    file = Path(directory)/'run_manifest.json'
    if not file.exists():
        return False
    meta = read_json(file)
    for name, value in meta['artifacts'].items():
        path = Path(directory)/name
        require(path.is_file() and sha(path) == value, 'Changed completed artifact: '+name)
    for field, message in [('upstream', 'upstream'), ('input_hashes', 'source input'), ('code_hashes', 'executed code')]:
        for name, value in meta.get(field, {}).items():
            path = ROOT/name
            if field == 'code_hashes' and archived_code and (not path.exists() or sha(path) != value):
                path = ROOT/'reports/source_snapshots'/(value+path.suffix)
            require(path.is_file() and sha(path) == value, 'Changed '+message+'; use a new run: '+name)
    if cfg is not None:
        require(meta.get('config_sha256') == config_digest(cfg), 'Effective config changed; use a new run')
    if protocol is not None:
        require(meta.get('protocol_sha256') == config_digest(protocol), 'Protocol changed; use a new run')
    return True


def initialize(cfg, protocol):
    validate_protocol(protocol)
    directory = run_root(protocol)/'protocol'
    if completed(directory, cfg, protocol):
        require(read_json(directory/'protocol.json') == protocol, 'Protocol changed; use a new output')
        require(read_json(directory/'run_manifest.json')['input_hashes'] == input_hashes(cfg), 'Source inputs changed; use a new run')
        return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg)
    train = [r for r in rows if r['split'] == 'train']
    val = [r for r in rows if r['split'] == 'val']
    old_ids = read_table(ROOT/'runs'/cfg['runs']['explain']/'sample_ids.tsv')
    require({r['sample_id'] for r in old_ids} <= {r['sample_id'] for r in val}, 'Attribution cohort mismatch')
    require(protocol['attribution_train_n'] <= len(train) and protocol['attribution_val_n'] <= len(val), 'Attribution cohort too large')
    rng = np.random.default_rng(seeds(protocol)[0])
    ids = sorted(rng.choice(len(train), protocol['attribution_train_n'], replace=False))
    if len(old_ids) != protocol['attribution_val_n']:
        old_ids = [dict(sample_id=val[i]['sample_id']) for i in sorted(rng.choice(len(val), protocol['attribution_val_n'], replace=False))]
    write_table(directory/'attribution_train_ids.tsv', [dict(sample_id=train[i]['sample_id']) for i in ids])
    write_table(directory/'attribution_val_ids.tsv', old_ids)
    target_split = read_table(ROOT/'runs'/cfg['runs']['transfer']/'target_split.tsv')
    write_table(directory/'target_split.tsv', target_split)
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'source_audit.json', dict(
        source_counts={s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')},
        source_length=87, source_tss_index=67, label_units='unknown original RNA-expression scale',
        target_counts={s: sum(r['split'] == s for r in target_split) for s in ('train', 'val', 'test')},
        target_units='unknown', target_tss='unknown', target_orientation='unknown',
        source_origin='Deep_promoter natural E. coli 50bp labels; U00096.2 recovered context; original TSS map checked',
        target_origin='teacher-provided CSV; bibliography and assay metadata unavailable',
        selection='validation MSE; all new source test scores disabled',
        limitations=['Inherited random source split may share nearby genomic context',
                     'Historical source test and target test were previously reported; neither is a pristine blind benchmark'],
        input_hashes=input_hashes(cfg), target_sha256=sha(TARGET/'samples.tsv')))
    finish(directory, cfg, dict(seeds=list(seeds(protocol)), training_ids_sha256=digest_ids(r['sample_id'] for r in train)))
    return directory


def bundle(directory, x, rows, kind, cfg, details=None):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    x = np.asarray(x, np.float32)
    require(x.ndim == 2 and len(x) == len(rows) and np.isfinite(x).all(), 'Invalid feature matrix')
    ids = np.asarray([r['sample_id'] for r in rows], dtype='U40')
    require(len(set(ids)) == len(ids), 'Duplicate feature IDs')
    np.savez_compressed(directory/'features.npz', X=x, sample_ids=ids)
    write_json(directory/'feature_manifest.json', dict(
        feature_kind=kind, n_rows=len(rows), n_features=x.shape[1], dtype='float32',
        sample_ids_sha256=digest_ids(ids), ordered_ids_sha256=__import__('hashlib').sha256(('\n'.join(ids)+'\n').encode()).hexdigest(),
        feature_sha256=sha(directory/'features.npz'), input_hashes=input_hashes(cfg),
        data_version=cfg['data_version'], split_id=cfg['split_id'], details=details or {}))


def read_bundle(directory, ids):
    directory = Path(directory)
    meta = read_json(directory/'feature_manifest.json')
    require(sha(directory/'features.npz') == meta['feature_sha256'], 'Feature checksum mismatch')
    with np.load(directory/'features.npz', allow_pickle=False) as d:
        x = d['X']; keys = d['sample_ids'].tolist()
    require(len(keys) == len(set(keys)) and len(keys) == len(x), 'Invalid feature IDs')
    lookup = {key: i for i, key in enumerate(keys)}
    require(set(ids) <= set(keys), 'Missing requested feature ID')
    require(x.dtype == np.float32 and x.ndim == 2 and np.isfinite(x).all(), 'Invalid feature values')
    return x[[lookup[sid] for sid in ids]]


def export_predictions(cfg, method, directory, rows, values, raw=None):
    values = np.asarray(values, float)
    predictions(cfg, method, directory, rows, values, raw=raw)
    for subset in ('train', 'val'):
        file = Path(directory)/f'predictions_{subset}.csv'
        out = read_table(file)
        for r in out:
            r['run_id'] = Path(directory).relative_to(ROOT).as_posix()
            r['error_reason'] = ''
        write_table(file, out)


def bootstrap(values, statistic=np.mean, groups=None, repeats=None, protocol=None):
    protocol = protocol or {}
    repeats = repeats if repeats is not None else protocol.get('bootstrap_repeats', 1000)
    confidence = protocol.get('confidence', .95)
    values = np.asarray(values); rng = np.random.default_rng(protocol.get('bootstrap_seed', SEEDS[0]))
    groups = np.arange(len(values)) if groups is None else np.asarray(groups)
    unique = np.unique(groups)
    lookup = [np.flatnonzero(groups == group) for group in unique]
    estimates = []
    for _ in range(repeats):
        sample = np.concatenate([lookup[i] for i in rng.integers(0, len(unique), len(unique))])
        value = statistic(values[sample])
        if value is not None and np.isfinite(value): estimates.append(float(value))
    require(len(estimates) > repeats*.9, 'Too many undefined bootstrap estimates')
    return tuple(float(x) for x in np.quantile(estimates, [(1-confidence)/2, (1+confidence)/2]))


def validate_prediction_rows(export, truth, cfg, subset):
    require(len(export) == len(truth) and len({r['sample_id'] for r in export}) == len(export), 'Prediction coverage/duplicate error')
    lookup = {r['sample_id']: r for r in export}
    require(set(lookup) == {r['sample_id'] for r in truth}, 'Prediction ID mismatch')
    ordered = [lookup[r['sample_id']] for r in truth]
    for actual, source in zip(ordered, truth):
        require(actual['subset'] == subset and actual['data_version'] == cfg['data_version']
                and actual['split_id'] == cfg['split_id'] and actual['dataset_id'] == cfg['dataset_id']
                and int(actual['sequence_length']) == 87 and actual['target_scale_model'] == 'log10', 'Prediction identity mismatch')
        require(np.isclose(float(actual['true_value_log10']), float(source['target_log10']), atol=1e-12)
                and np.isclose(float(actual['true_value']), float(source['strength']), atol=1e-12), 'Prediction label mismatch')
        require(actual['prediction_status'] in ('ok', 'failed'), 'Unknown prediction status')
        if actual['prediction_status'] == 'ok':
            value = float(actual['predicted_value_log10'])
            require(np.isfinite(value) and np.isclose(float(actual['predicted_value']), 10**value), 'Invalid prediction scale')
        else:
            require(actual['predicted_value_log10'] == '' and actual.get('error_reason'), 'Failed prediction must have no numeric value and a reason')
    return ordered


def evaluate(cfg, protocol):
    root = run_root(protocol); directory = root/'evaluation'
    if completed(directory, cfg, protocol): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); train_y = np.array([float(r['target_log10']) for r in rows if r['split'] == 'train'])
    thresholds = np.quantile(train_y, [1/3, 2/3]); selected = read_json(root/'cnn/selected_model.json')
    sources = [('knn', None, ROOT/'runs'/cfg['runs']['knn']),
               ('ridge', None, root/'ml/ridge'), ('linear_svr', None, root/'ml/svr'),
               ('dnabert2_ridge', None, root/'ml/dnabert2'), ('thermo', None, ROOT/'runs'/cfg['runs']['thermo'])]
    sources += [('cnn', seed, ROOT/selected['runs'][str(seed)]) for seed in seeds(protocol)]
    scores = []; groups = []; coverage = []; used_ids = []; exports = []; upstream = []
    for subset in ('train', 'val'):
        truth = [r for r in rows if r['split'] == subset]; y = np.array([float(r['target_log10']) for r in truth])
        prepared = []
        for method, seed, source in sources:
            file = source/f'predictions_{subset}.csv'; upstream.append(file)
            out = validate_prediction_rows(read_table(file), truth, cfg, subset)
            good = np.array([r['prediction_status'] == 'ok' for r in out])
            prepared.append((method, seed, source, out, good))
        common = np.logical_and.reduce([item[-1] for item in prepared])
        used_ids.extend(dict(subset=subset, sample_id=r['sample_id'], common_success=bool(ok)) for r, ok in zip(truth, common))
        for method, seed, source, out, good in prepared:
            coverage.append(dict(method=method, seed=seed, subset=subset, n_requested=len(truth), n_success=int(good.sum()), coverage=float(good.mean())))
            exports.extend(out)
            for cohort, mask in [('all_success', good), ('common', common)]:
                p = np.array([float(r['predicted_value_log10']) if ok else np.nan for r, ok in zip(out, good)])
                scores.append(dict(method=method, seed=seed, run_id=source.relative_to(ROOT).as_posix(), subset=subset, cohort=cohort, **metrics(y[mask], p[mask])))
                if cohort == 'common':
                    for i, label in enumerate(('low', 'middle', 'high')):
                        chosen = mask & (np.digitize(y, thresholds) == i)
                        groups.append(dict(method=method, seed=seed, subset=subset, intensity_group=label, **metrics(y[chosen], p[chosen])))
        scores.append(dict(method='train_mean', seed=None, run_id='training_labels_only', subset=subset, cohort='common',
                           **metrics(y[common], np.full(common.sum(), train_y.mean()))))
    summary = []
    for method in ['knn', 'ridge', 'linear_svr', 'dnabert2_ridge', 'thermo', 'cnn', 'train_mean']:
        rr = [r for r in scores if r['method'] == method and r['subset'] == 'val' and r['cohort'] == 'common']
        row = dict(method=method, n=rr[0]['n'], n_runs=len(rr), seed_policy='all three seeds' if method == 'cnn' else 'deterministic single run')
        for metric in ['r2_log10', 'spearman', 'mae_log10', 'rmse_log10']:
            vv = [r[metric] for r in rr if r[metric] is not None]
            row[metric+'_mean'] = float(np.mean(vv)) if vv else None
            row[metric+'_sd'] = float(np.std(vv, ddof=1)) if len(vv) > 1 else None
        summary.append(row)
    write_table(directory/'metrics.csv', scores); write_table(directory/'model_summary.csv', summary)
    write_table(directory/'group_metrics.csv', groups); write_table(directory/'coverage.csv', coverage)
    write_table(directory/'evaluation_ids.csv', used_ids)
    write_table(directory/'predictions.csv', exports, columns=list(dict.fromkeys(k for r in exports for k in r)))
    finish(directory, cfg, dict(intensity_thresholds_log10=thresholds.tolist(), aggregation='score each run first; sample SD across CNN seeds',
           comparison='validation selected models; descriptive evaluation, not untouched external generalization'), upstream)
    return directory
