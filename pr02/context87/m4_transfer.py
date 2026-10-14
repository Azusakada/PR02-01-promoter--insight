"""Eight predeclared cross-species conditions on the frozen grouped target split."""
from __future__ import annotations
import joblib
import numpy as np
from scipy.stats import spearmanr
from .data import ROOT, TARGET, read_json, read_table, require, write_json, write_table
from .experiment import cnn_predict, metrics
from .features import encode, kmers
from .m4_cnn import embedding, load_model
from .m4_data_eval import SEEDS, bootstrap, completed, finish, run_root
from .m4_ml_dnalm import fit_head


def run(cfg, protocol):
    root = run_root(protocol); directory = root/'transfer'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_table(TARGET/'samples.tsv'); split = read_table(root/'protocol/target_split.tsv')
    require([r['sample_id'] for r in split] == [r['sample_id'] for r in rows], 'Target split ID mismatch')
    assignment = np.array([r['split'] for r in split]); groups = np.array([r['sequence_group'] for r in rows])
    sets = {s: set(groups[assignment == s]) for s in ('train', 'val', 'test')}
    require(not sets['train'] & sets['val'] and not sets['train'] & sets['test'] and not sets['val'] & sets['test'], 'Target group leakage')
    write_table(directory/'target_split.tsv', split)
    y = np.array([float(r['target_log10']) for r in rows]); test = assignment == 'test'; val = assignment == 'val'
    sequences = [r['sequence'] for r in rows]; saved = joblib.load(root/'ml/ridge/model.joblib')
    kmer = kmers(sequences, saved['parameters']['k']).astype(np.float32)
    selected = read_json(root/'cnn/selected_model.json'); model, transform = load_model(ROOT/selected['representative_run'], length=61)
    xcnn = encode(sequences, 61); frozen = embedding(model, xcnn)
    zero_cnn = transform['min_log10']+(transform['max_log10']-transform['min_log10'])*cnn_predict(model, xcnn)
    conditions = [('zero_shot_ridge', 0, saved['model'].predict(kmer), 'source_label_scale'),
                  ('zero_shot_cnn', 0, zero_cnn, 'source_label_scale')]
    order = sorted(sets['train']); np.random.default_rng(SEEDS[0]).shuffle(order)
    searches = []
    for n in (20, 50, 100):
        fit = np.isin(groups, order[:n]); require(np.all(assignment[fit] == 'train'), 'Target heldout label used in fitting')
        for kind, x in [('kmer', kmer), ('frozen_cnn', frozen)]:
            head, search = fit_head(x, y, fit, val)
            name = f'few_shot_{kind}_{n}_groups'
            joblib.dump(head, directory/(name+'.joblib'))
            write_table(directory/(name+'_fit_ids.tsv'), [dict(sample_id=r['sample_id']) for r, use in zip(rows, fit) if use])
            searches.extend(dict(condition=name, **s) for s in search)
            conditions.append((name, n, head.predict(x), 'target_teacher_label_scale'))
    result = []; predictions = []
    for name, n, p, scale in conditions:
        z = np.column_stack([y[test], p[test]])
        low, high = bootstrap(z, statistic=lambda a: float(spearmanr(a[:, 0], a[:, 1]).statistic), groups=groups[test])
        result.append(dict(condition=name, n_train_groups=n, **metrics(y[test], p[test]), spearman_ci_low=low, spearman_ci_high=high,
                           bootstrap_unit='target_sequence_group', prediction_scale=scale))
        for r, subset, actual, prediction in zip(rows, assignment, y, p):
            if subset == 'train': continue
            predictions.append(dict(sample_id=r['sample_id'], condition=name, subset=subset,
                observed_target_log10=float(actual), predicted_log10=float(prediction), source_run_id=selected['representative_run'] if 'cnn' in name else 'ml/ridge',
                target_split_id='bsub61_grouped_20260928_v1', prediction_status='ok', error_reason='', sequence_length=61))
    write_table(directory/'transfer_metrics.csv', result); write_table(directory/'predictions.csv', predictions)
    write_table(directory/'search.csv', searches)
    write_json(directory/'transfer_manifest.json', dict(source_length=87, target_length=61, target_units='unknown',
        target_tss='unknown', target_orientation='unknown', n_conditions=8, representative_seed=SEEDS[0],
        target_counts={s: int((assignment == s).sum()) for s in ('train', 'val', 'test')},
        few_shot_policy='nested seeded groups, train fit, val alpha selection, frozen source representation',
        scale_interpretation='zero-shot Spearman primary; MAE/RMSE/R2 diagnostic because source/target assay units are not established',
        test_history='Target test already used in earlier project experiments; extension of the same benchmark, not first blind evaluation'))
    finish(directory, cfg, dict(n_conditions=8, source_test_evaluated=False, target_test_evaluated=True),
           [root/'protocol/target_split.tsv', root/'ml/ridge/model.joblib', ROOT/selected['representative_run']/'model.pt'])
    return directory
