"""Length-adjusted biological-window overlap and IG/ISM agreement."""
from __future__ import annotations
import json
import numpy as np
from scipy.stats import spearmanr
from .data import ROOT, read_json, read_table, write_table
from .m4_data_eval import REGIONS, SEEDS, bootstrap, completed, finish, run_root
from .research import ranks


def run(cfg, protocol):
    root = run_root(protocol); directory = root/'explanation_analysis'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    attrs = read_table(root/'attribution/attributions.tsv'); values = {}
    for r in attrs: values.setdefault((r['sample_id'], r['method']), np.zeros(87))[int(r['position_0based'])] = float(r['score_log10'])
    ids = sorted({sid for sid, method in values}); states = {}
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f:
        for line in f:
            state = json.loads(line); states[state['sample_id']] = state
    region_rows = []; overlap = []; consistency = []
    rng = np.random.default_rng(SEEDS[0])
    for sid in ids:
        inferred = {name: tuple(states[sid][key]) for name, key in [('UP', 'UP_position'), ('-35', 'hex35_position'),
                     ('spacer', 'spacer_position'), ('-10', 'hex10_position')]}
        random_orders = np.argsort(rng.random((1000, 87)), axis=1)[:, :10]
        for status, windows in [('canonical_window_assumption', REGIONS), ('model_inferred', inferred)]:
            for name, (start, end) in windows.items():
                region_rows.append(dict(sample_id=sid, region=name, annotation_status=status, start_0based=start, end_0based_exclusive=end))
                for method in ['integrated_gradients', 'ism_mean_absolute']:
                    order = ranks(values[sid, method])
                    for k in protocol['top_k']:
                        hits = int(((order[:k] >= start) & (order[:k] < end)).sum())
                        random_mean = float(((random_orders[:, :k] >= start) & (random_orders[:, :k] < end)).sum(1).mean())
                        overlap.append(dict(sample_id=sid, method=method, annotation_status=status, region=name, top_k=k,
                            n_overlap=hits, overlap_fraction=hits/k, region_length=end-start, expected_random_hits=k*(end-start)/87,
                            expected_random_fraction=(end-start)/87, simulated_random_mean_hits=random_mean,
                            excess_fraction=hits/k-(end-start)/87))
        one = np.abs(values[sid, 'integrated_gradients']); two = np.abs(values[sid, 'ism_mean_absolute'])
        for k in protocol['top_k']:
            a = set(ranks(one)[:k]); b = set(ranks(two)[:k])
            consistency.append(dict(sample_id=sid, top_k=k, spearman=float(spearmanr(one, two).statistic),
                top_k_jaccard=len(a & b)/len(a | b), n_shared=len(a & b), evidence='same fitted CNN, two explanation definitions'))
    summaries = []
    for method in ['integrated_gradients', 'ism_mean_absolute']:
        for status in ['canonical_window_assumption', 'model_inferred']:
            for name in REGIONS:
                for k in protocol['top_k']:
                    rr = [r for r in overlap if r['method'] == method and r['annotation_status'] == status and r['region'] == name and r['top_k'] == k]
                    a = np.array([r['excess_fraction'] for r in rr]); low, high = bootstrap(a)
                    summaries.append(dict(method=method, annotation_status=status, region=name, top_k=k, n_parent=len(rr),
                        mean_overlap_fraction=float(np.mean([r['overlap_fraction'] for r in rr])),
                        mean_random_fraction=float(np.mean([r['expected_random_fraction'] for r in rr])),
                        mean_excess_fraction=float(a.mean()), excess_ci_low=low, excess_ci_high=high,
                        bootstrap_unit='parent_sequence', intervals='descriptive 95% percentile; no multiple-comparison correction'))
    agreement = []
    for k in protocol['top_k']:
        rr = [r for r in consistency if r['top_k'] == k]
        for metric in ['spearman', 'top_k_jaccard']:
            a = np.array([r[metric] for r in rr]); low, high = bootstrap(a)
            agreement.append(dict(top_k=k, metric=metric, n_parent=len(rr), mean=float(a.mean()), median=float(np.median(a)), ci_low=low, ci_high=high))
    cases = sorted([r for r in consistency if r['top_k'] == 5], key=lambda r: (r['top_k_jaccard'], r['sample_id']))
    chosen = cases[:3]+cases[-3:]
    write_table(directory/'regions.csv', region_rows); write_table(directory/'topk_overlap.csv', overlap)
    write_table(directory/'overlap_summary.csv', summaries); write_table(directory/'consistency.csv', consistency)
    write_table(directory/'consistency_summary.csv', agreement); write_table(directory/'consistency_cases.csv', chosen)
    finish(directory, cfg, dict(random_topk_repeats=1000, random_seed=SEEDS[0], n_parent=len(ids),
        case_rule='3 lowest and 3 highest top5 Jaccard; ties sample_id', interpretation='Model explanation and assumed/inferred windows, not experimentally labeled binding sites'),
        [root/'attribution/attributions.tsv', ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl'])
    return directory
