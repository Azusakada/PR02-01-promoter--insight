"""IG/ISM explanation of the fixed representative CNN, plus training-only recipes."""
from __future__ import annotations
import json
import numpy as np

from .data import ROOT, coordinate, digest_ids, load, read_json, read_table, write_json, write_table
from .experiment import cnn_predict
from .features import encode, kmers
from .m4_cnn import load_model
from .m4_data_eval import REGIONS, seeds, bundle, completed, finish, run_root
from .research import integrated_gradients, ranks


def log_model(directory):
    model, transform = load_model(directory)
    from pr02_cnn.model import Log10Predictor
    return Log10Predictor(model, transform).eval()


def guided_features(sequences, selected_regions):
    parts = []; names = []
    for name in selected_regions:
        start, end = REGIONS[name]; seqs = [s[start:end] for s in sequences]
        parts.append(kmers(seqs, 3)); names.extend(f'{name}:kmer3:{i}' for i in range(64))
        parts.append(np.array([[sum(b in 'GC' for b in s)/len(s)] for s in seqs])); names.append(name+':GC')
        if name in ('-35', '-10'):
            consensus = 'TTGACA' if name == '-35' else 'TATAAT'
            parts.append(np.array([[sum(a == b for a, b in zip(s, consensus))/6] for s in seqs])); names.append(name+':consensus_match')
    return np.column_stack(parts).astype(np.float32), names


def run(cfg, protocol):
    import torch
    torch.set_num_threads(2)
    root = run_root(protocol); directory = root/'attribution'
    if completed(directory, cfg, protocol): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); master = {r['sample_id']: r for r in rows}
    selected = read_json(root/'cnn/selected_model.json')
    source = ROOT/selected['representative_run']; model = log_model(source)
    val_rows = [master[r['sample_id']] for r in read_table(root/'protocol/attribution_val_ids.tsv')]
    x = encode([r['sequence'] for r in val_rows], 87)
    ig, errors = integrated_gradients(model, x, protocol['ig_steps'])
    inp = torch.from_numpy(x).requires_grad_(True)
    gradient = torch.autograd.grad(model(inp).sum(), inp)[0].numpy()
    expectations = []; ism = []
    for r, score, grad, wild in zip(val_rows, ig, gradient, cnn_predict(model, x)):
        alternatives = []; choices = []
        for pos, ref in enumerate(r['sequence']):
            for alt in 'ACGT':
                if alt == ref: continue
                delta = float(grad['ACGT'.index(alt), pos]-grad['ACGT'.index(ref), pos])
                expectations.append(dict(sample_id=r['sample_id'], position_0based=pos, ref_base=ref, alt_base=alt,
                    expected_delta_log10=delta, expected_direction=direction(delta, protocol), source_model_run_id=selected['representative_run'],
                    rule_version='wild_type_gradient_alt_minus_ref_v1'))
                alternatives.append(r['sequence'][:pos]+alt+r['sequence'][pos+1:]); choices.append((pos, alt))
        delta = cnn_predict(model, encode(alternatives, 87))-wild
        ism.append(np.abs(delta.reshape(87, 3)).mean(1))
    output = []
    for method, matrix in [('integrated_gradients', ig), ('ism_mean_absolute', np.asarray(ism))]:
        for r, scores in zip(val_rows, matrix):
            order = ranks(scores); rank_at = np.empty(87, int); rank_at[order] = np.arange(1, 88)
            for pos, value in enumerate(scores):
                output.append(dict(sample_id=r['sample_id'], method=method, source_model_run_id=selected['representative_run'],
                    position_0based=pos, position_relative_tss=coordinate(pos), nucleotide=r['sequence'][pos],
                    score_log10=float(value), importance_absolute=abs(float(value)), rank_absolute=int(rank_at[pos])))
    write_table(directory/'attributions.tsv', output)
    write_table(directory/'substitution_expectation.csv', expectations)
    write_table(directory/'ig_completeness.csv', [dict(sample_id=r['sample_id'], error_log10=float(e)) for r, e in zip(val_rows, errors)])
    write_json(directory/'interpretation.json', dict(
        source_model=selected['representative_run'], explained_model='pure CNN log10 predictor', n_val=len(val_rows),
        ig_baseline='A/C/G/T=0.25 at every position', ig_steps=protocol['ig_steps'], integration='trapezoid',
        max_absolute_completeness_error_log10=float(np.abs(errors).max()), mean_absolute_completeness_error_log10=float(np.abs(errors).mean()),
        ism_definition='mean absolute log10 output change across the three alternative nucleotides',
        expectation='wild type first derivative alt-channel minus reference-channel; no mutant output used',
        legacy_saliency='legacy research.py saliency is gradient times input difference, not plain absolute gradient'))
    train_rows = [master[r['sample_id']] for r in read_table(root/'protocol/attribution_train_ids.tsv')]
    for seed in seeds(protocol):
        guided = root/f'guided_s{seed}'
        if completed(guided, cfg, protocol): continue
        guided.mkdir(parents=True, exist_ok=True)
        model = log_model(ROOT/selected['runs'][str(seed)])
        train_attr, train_errors = integrated_gradients(model, encode([r['sequence'] for r in train_rows], 87), protocol['ig_steps'])
        ranking = sorted(REGIONS, key=lambda name: (-float(np.abs(train_attr[:, slice(*REGIONS[name])]).mean()), list(REGIONS).index(name)))
        chosen = ranking[:2]
        write_json(guided/'feature_recipe.json', dict(selected_regions=chosen, candidate_regions=list(REGIONS),
            region_mean_absolute={name: float(np.abs(train_attr[:, slice(*REGIONS[name])]).mean()) for name in REGIONS},
            fit_subset='train', n_selection=len(train_rows), selection_ids_sha256=digest_ids(r['sample_id'] for r in train_rows),
            ranking='mean absolute IG per base; ties UP,-35,spacer,-10', seed=seed,
            source_cnn_run=selected['runs'][str(seed)], windows=REGIONS))
        np.savez_compressed(guided/'training_ig.npz', scores=train_attr, sample_ids=np.array([r['sample_id'] for r in train_rows]), completeness_errors=train_errors)
        features, names = guided_features([r['sequence'] for r in rows], chosen)
        bundle(guided, features, rows, 'training_ig_guided_region_kmer', cfg,
               dict(selected_regions=chosen, feature_names=names, selection_subset='train', n_selection=len(train_rows), seed=seed))
        finish(guided, cfg, dict(selected_regions=chosen), [guided/'feature_recipe.json'])
    finish(directory, cfg, dict(n_val=len(val_rows), methods=['IG', 'ISM'], source_model=selected['representative_run']),
           [source/'model.pt', root/'protocol/attribution_val_ids.tsv'])
    return directory


def direction(value, protocol):
    threshold = protocol['mutation_direction_threshold_log10']
    return 'positive' if value > threshold else 'negative' if value < -threshold else 'near_zero'
