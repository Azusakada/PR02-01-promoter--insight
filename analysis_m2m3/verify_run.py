"""Verify saved evidence; manual visual review must be explicitly acknowledged."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from common import ROOT, load_data, relative, require, sha256, write_json


def check_hashes(entries):
    for item in entries:
        path = (ROOT / item['path']).resolve()
        relative(path)
        require(path.is_file() and sha256(path) == item['sha256'], f"Hash mismatch: {item['path']}")


def verify(output, reviewed=()):
    output = Path(output).resolve()
    relative(output)
    meta = json.loads((output / 'run_manifest.json').read_text())
    check_hashes(meta['inputs'])
    check_hashes(meta['artifacts'])
    figures = json.loads((output / 'figure_manifest.json').read_text())
    for fig in figures:
        check_hashes(fig['source_tables'])
        check_hashes(fig['images'])
    data = load_data()
    results = {'input_and_artifact_hashes': 'pass', 'figure_source_and_image_hashes': 'pass'}
    metrics_path = output / 'source_tables/metrics.csv'
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        ids = pd.read_csv(output / 'source_tables/common_eval_ids.tsv', sep='\t')
        requested = set(data.loc[data['split'].eq('val'), 'sample_id'])
        frames = {}
        for method in metrics.method_name.unique():
            frame = pd.read_csv(output / f'source_tables/{method}_requested_predictions.csv')
            require(frame.sample_id.is_unique and set(frame.sample_id) == requested, f'{method}: request IDs changed')
            truth = frame[['sample_id', 'true_value']].merge(data[['sample_id', 'strength']], validate='one_to_one')
            require(np.allclose(truth.true_value, truth.strength, rtol=0, atol=1e-8), 'Truth mismatch')
            frames[method] = frame.set_index('sample_id')
        successes = {m: set(f.index[f.prediction_status.eq('ok')]) for m, f in frames.items()}
        common = set.intersection(*successes.values())
        for row in metrics.itertuples():
            selected = ids.loc[ids.comparison_set_id.eq(row.comparison_set_id) & ids.method_name.eq(row.method_name), 'sample_id']
            expected = common if row.evaluation_subset == 'val_common' else successes[row.method_name]
            require(selected.is_unique and set(selected) == expected, 'Comparison IDs mismatch')
            require(row.n_requested == len(requested) and row.n_success == len(successes[row.method_name]) and row.n_used == len(selected), 'Metric sample counts mismatch')
            part = frames[row.method_name].loc[selected]
            y = part.true_value.to_numpy(float)
            column = {'raw': 'predicted_value', 'log10': 'predicted_value_log10', 'tool_rank': 'predicted_tx_rate'}[row.target_scale]
            p = part[column].to_numpy(float)
            if row.target_scale == 'log10':
                y = np.log10(y)
            require(row.metric_status == 'ok', 'This verification run requires defined metrics')
            value = {'r2': r2_score, 'mae': mean_absolute_error, 'spearman': lambda a, b: spearmanr(a, b).statistic}[row.metric_name](y, p)
            require(np.isclose(value, row.value, rtol=1e-10, atol=1e-12), 'Recomputed metric mismatch')
        results.update(metric_recomputation='pass', requested_ids=len(requested), common_ids=len(common), metrics_checked=len(metrics))
    else:
        samples = pd.read_csv(output / 'source_tables/eda_samples.tsv', sep='\t')
        require(samples.sample_id.is_unique and set(samples.sample_id) == set(data.sample_id), 'EDA ID mismatch')
        transform = json.loads((output / 'label_transform.analysis.json').read_text())
        train = data.loc[data['split'].eq('train'), 'target_log10']
        require(transform['fit_subset'] == 'train' and transform['clip'] is False, 'Wrong normalization scope')
        require(np.isclose(transform['min_log10'], train.min()) and np.isclose(transform['max_log10'], train.max()), 'Wrong extrema')
        expected = (samples.target_log10 - train.min()) / (train.max() - train.min())
        require(np.allclose(samples.target_normalized, expected), 'EDA normalization mismatch')
        results.update(eda_ids_and_normalization='pass', samples_checked=len(samples))
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if reviewed:
        require(set(reviewed) == {f['figure_id'] for f in figures}, 'Acknowledge all and only manually viewed PNG figure IDs')
        now = datetime.now(timezone.utc).isoformat()
        for fig in figures:
            fig['visual_check_status'] = 'pass'
            fig['visual_check'] = {'checked_at': now, 'method': 'manual inspection of rendered PNG: labels, clipping, scale, sample counts and caption', 'reviewer': 'Codex (not a biological annotation review)'}
        write_json(output / 'figure_manifest.json', figures)
        write_json(output / 'qa_report.json', {'checked_at': now, 'checks': results, 'manually_reviewed_figure_ids': list(reviewed), 'limitations': 'Does not prove model training or biological causality.'})
        meta['artifacts'] = [{'kind': 'source_patch' if p.name == 'source_patch.diff' else p.suffix.lstrip('.'), 'path': relative(p), 'sha256': sha256(p)} for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'run_manifest.json']
        write_json(output / 'run_manifest.json', meta)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--acknowledge-manually-reviewed', nargs='+', default=[])
    args = ap.parse_args()
    verify(args.output, args.acknowledge_manually_reviewed)
