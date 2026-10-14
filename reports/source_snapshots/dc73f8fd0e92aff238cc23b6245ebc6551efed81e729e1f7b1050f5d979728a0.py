"""Controlled M4 revision, preserving the original benchmark and presentation."""
from __future__ import annotations

import copy
import joblib
import numpy as np

from .data import ROOT, load, read_json, read_table, require, sha, write_json, write_table
from .experiment import masks, metrics
from .m4_data_eval import (completed, config_digest, finish, read_bundle, run_root,
                           seeds, validate_protocol, verify_sealed_release)
from .m4_ml_dnalm import fit_head


def baseline(protocol):
    path = ROOT/protocol['baseline']
    require(path.resolve().is_relative_to((ROOT/'runs').resolve()), 'Baseline must be inside runs/')
    return path


def initialize(cfg, protocol):
    validate_protocol(protocol)
    require(protocol['interface_version'] == 'm4_refinement_v2', 'Wrong refinement schema')
    require(protocol['output'] != protocol['baseline'], 'Refinement cannot overwrite baseline')
    require(protocol['head_selection'] == 'inherited_validation_descriptive', 'Unsupported head selection policy')
    require(protocol['genomic_block_bp'] >= 87 and protocol['genome_length'] == 4639675, 'Invalid genomic block policy')
    require(0 < protocol['group_dev_fraction'] < 1 and 0 < protocol['group_evaluation_fraction'] < 1 and
            protocol['group_dev_fraction']+protocol['group_evaluation_fraction'] < 1, 'Invalid grouped fractions')
    require(len(protocol['transfer_draw_seeds']) == len(set(protocol['transfer_draw_seeds'])) >= 3, 'Require at least three distinct transfer draws')
    require(all(0 < n <= 100 for n in protocol['few_shot_groups']), 'Invalid few-shot groups')
    root = run_root(protocol); directory = root/'protocol'
    if completed(directory, cfg, protocol): return root
    old = baseline(protocol)
    require((old/'release_manifest.json').is_file(), 'Refinement requires a sealed baseline')
    archived = verify_sealed_release(old)
    original = read_json(old/'protocol/protocol.json')
    require(seeds(protocol) == seeds(original), 'Matched comparisons require the original CNN seeds')
    require(protocol['dna_revision'] == original['dna_revision'], 'DNA encoder revision changed')
    require(protocol['attribution_val_n'] == original['attribution_val_n'], 'Stability requires the original attribution cohort')
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/'protocol.json', protocol)
    write_json(directory/'effective_config.json', cfg)
    write_json(directory/'baseline_ref.json', dict(run=protocol['baseline'],
        release_manifest_sha256=sha(old/'release_manifest.json'), archived_original_source_or_readme=archived,
        reused=['frozen pretrained DNA features', 'three original selected CNN checkpoints and features',
                'training-only guided feature recipes', 'original target grouped split', 'original attribution parent IDs'],
        source_test_evaluated=False, interpretation='Original validation has been used for model selection; not a new blind test'))
    finish(directory, cfg, dict(protocol_sha256=config_digest(protocol)), [old/'release_manifest.json'])
    return root


def aggregate(records, key='method'):
    output = []
    for method in dict.fromkeys(r[key] for r in records):
        rr = [r for r in records if r[key] == method]
        row = {key: method, 'n_runs': len(rr), 'n': rr[0]['n']}
        for metric in ['r2_log10', 'spearman', 'rmse_log10', 'mae_log10']:
            a = [r[metric] for r in rr if r[metric] is not None]
            row[metric+'_mean'] = float(np.mean(a)) if a else None
            row[metric+'_sd'] = float(np.std(a, ddof=1)) if len(a) > 1 else None
        output.append(row)
    return output


def save_predictions(directory, rows, y, prediction, assignments, method, seed=None):
    output = [dict(sample_id=r['sample_id'], subset=str(subset), method=method, seed=seed,
                   observed_log10=float(actual), predicted_log10=float(p))
              for r, actual, p, subset in zip(rows, y, prediction, assignments)
              if subset in ('train', 'dev', 'evaluation', 'val')]
    write_table(directory/'predictions.csv', output)


def regularization(cfg, protocol):
    root = run_root(protocol); old = baseline(protocol); directory = root/'regularization'
    if completed(directory, cfg, protocol): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); ids = [r['sample_id'] for r in rows]; m = masks(rows)
    y = np.array([float(r['target_log10']) for r in rows]); assignment = [r['split'] for r in rows]
    kmer = read_bundle(old/'ml/kmer', ids); dna = read_bundle(old/'dna_features', ids)
    conditions = [('kmer_ridge', None, kmer), ('dnabert2_ridge', None, dna)]
    for seed in seeds(protocol):
        cnn = read_bundle(old/f'cnn/features_s{seed}', ids)
        guided = read_bundle(old/f'guided_s{seed}', ids)
        for stage, parts in [('A1', [kmer, cnn]), ('A2', [kmer, cnn, dna]), ('A3', [kmer, cnn, dna, guided])]:
            conditions.append((stage, seed, np.column_stack(parts)))
    result = []; comparison = []; searches = []
    original = read_table(old/'ablation/stage_metrics.csv')
    dna_old = next(r for r in read_table(old/'evaluation/model_summary.csv') if r['method'] == 'dnabert2_ridge')
    for method, seed, x in conditions:
        name = method if seed is None else f'{method}_s{seed}'; output = directory/name
        if completed(output, cfg, protocol):
            row = read_json(output/'result.json')
        else:
            output.mkdir(parents=True, exist_ok=True)
            head, search = fit_head(x, y, m['train'], m['val'], protocol['ridge_alpha'])
            p = head.predict(x)
            row = dict(method=method, seed=seed, alpha=float(head[-1].alpha), n_features=x.shape[1],
                       alpha_at_grid_boundary=float(head[-1].alpha) in (min(protocol['ridge_alpha']), max(protocol['ridge_alpha'])),
                       **metrics(y[m['val']], p[m['val']]))
            joblib.dump(head, output/'model.joblib'); write_json(output/'result.json', row)
            write_table(output/'search.csv', search)
            write_table(output/'fit_ids.tsv', [dict(sample_id=r['sample_id']) for r in rows if r['split'] == 'train'])
            save_predictions(output, rows, y, p, assignment, method, seed)
            finish(output, cfg, dict(selection='inherited validation MSE; descriptive, selection-biased', frozen_representation=True),
                   [old/'release_manifest.json'])
        result.append(row)
        searches.extend(dict(method=method, seed=seed, **r) for r in read_table(output/'search.csv'))
        previous = float(dna_old['r2_log10_mean']) if method == 'dnabert2_ridge' else float(next(
            r['r2_log10'] for r in original if r['stage'] == ('A0' if method == 'kmer_ridge' else method)
            and (not r['seed'] if seed is None else int(r['seed'] or -1) == seed)))
        comparison.append(dict(method=method, seed=seed, old_r2_log10=previous, refined_r2_log10=row['r2_log10'],
                               delta_r2_log10=row['r2_log10']-previous, selected_alpha=row['alpha']))
        print('Refined head', name, 'alpha', row['alpha'], 'R2', row['r2_log10'], flush=True)
    write_table(directory/'metrics.csv', result); write_table(directory/'summary.csv', aggregate(result))
    write_table(directory/'matched_comparison.csv', comparison); write_table(directory/'search.csv', searches)
    finish(directory, cfg, dict(n_heads=len(result), alpha_grid=protocol['ridge_alpha'],
        control='Same input features, fit samples, scaling, labels and CNN seeds as original; broaden Ridge alpha only',
        limitation='Validation is reused for selection. These improvements do not estimate independent generalization.'),
        [old/'release_manifest.json'])
    return directory


def run(cfg, protocol, stage='all', reviewed=None):
    if stage == 'verify':
        from .m4_refinement_verify import verify
        return verify(cfg, protocol)
    initialize(cfg, protocol)
    if stage in ('all', 'regularization'): regularization(cfg, protocol)
    if stage in ('all', 'grouped'):
        from .m4_grouped_validation import run as grouped
        grouped(cfg, protocol)
    if stage in ('all', 'attribution'):
        from .m4_stability import attribution
        attribution(cfg, protocol)
    if stage in ('all', 'transfer'):
        from .m4_stability import transfer
        transfer(cfg, protocol)
    if stage in ('all', 'report'):
        from .m4_refinement_reporting import build
        build(cfg, protocol)
    if stage == 'acknowledge':
        from .m4_reporting import acknowledge
        return acknowledge(cfg, protocol, reviewed or [])
    if stage == 'publish':
        from .m4_refinement_verify import verify
        result = verify(cfg, protocol)
        write_json(ROOT/'results/context87/refinement_current.json', dict(run_id=protocol['output'],
            protocol_sha256=config_digest(protocol), verification_sha256=sha(run_root(protocol)/'verification.json'),
            report_path=protocol['output']+'/reports/REFINEMENT_REVIEW.md', ppt_updated=False,
            baseline=protocol['baseline']))
        return result
    return run_root(protocol)
