"""M2 descriptive EDA, with optional shared transform and ID-aligned features."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import COLORS, MODULE, ROOT, SAMPLES, SPLITS, Run, load_data, require, sequence_features, sha256, single, write_json


def label_transform(data, path=None):
    ids = sorted(data.loc[data['split'].eq('train'), 'sample_id'])
    a, b = data.loc[data['split'].eq('train'), 'target_log10'].agg(['min', 'max'])
    sig = {'samples': sha256(SAMPLES), 'split_id': single(data, 'split_id'), 'train_ids': ids, 'min_log10': a, 'max_log10': b}
    if path:
        meta = json.loads(Path(path).read_text())
        for col in ['dataset_id', 'data_version', 'schema_version', 'split_id']:
            require(meta.get(col) == single(data, col), f'Transform {col} mismatch')
        require(meta.get('kind') == 'log10_minmax' and meta.get('fit_subset') == 'train' and meta.get('clip') is False, 'Transform must be train-only log10 min-max without clipping')
        require(meta.get('samples_sha256') == sha256(SAMPLES), 'Transform source hash mismatch')
        require(meta.get('train_ids') == ids, 'Transform train ID mismatch')
        require(meta.get('train_ids_sha256') == hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest(), 'Train ID hash mismatch')
        require(np.isclose(meta.get('min_log10'), a, atol=1e-10) and np.isclose(meta.get('max_log10'), b, atol=1e-10), 'Transform extrema mismatch')
        require(bool(meta.get('transform_id')), 'Missing transform_id')
        scope = 'provided_shared_transform'
    else:
        require(a < b, 'Constant train labels')
        meta = {'schema_version': single(data, 'schema_version'), 'dataset_id': single(data, 'dataset_id'),
                'data_version': single(data, 'data_version'), 'split_id': single(data, 'split_id'),
                'transform_id': 'log10mm_' + hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()[:16],
                'kind': 'log10_minmax', 'fit_subset': 'train', 'clip': False, 'train_ids': ids,
                'train_ids_sha256': hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest(),
                'samples_sha256': sha256(SAMPLES), 'min_log10': float(a), 'max_log10': float(b),
                'extra_scope': 'analysis_only_pending_team_adoption'}
        scope = 'analysis_only_pending_team_adoption'
    return meta, scope


def run_eda(output, transform_path=None, features_path=None, feature_index=None):
    data = load_data()
    require(features_path is None or feature_index is not None, 'External feature table requires a row_index/sample_id mapping; do not assume row order')
    features = sequence_features(data)
    inputs = [SAMPLES, SPLITS]
    if features_path:
        external = pd.read_csv(features_path)
        index = pd.read_csv(feature_index, sep='\t')
        require(index.sample_id.is_unique and len(index) == len(external), 'Invalid feature index')
        require(set(index.row_index) == set(range(len(external))), 'Invalid row_index')
        require(set(index.sample_id) == set(data.sample_id), 'Feature ID coverage mismatch')
        if 'data_version' in index:
            require(single(index, 'data_version') == single(data, 'data_version'), 'Feature data_version mismatch')
        ordered = index.sort_values('row_index')
        external.insert(0, 'sample_id', ordered.sample_id.to_numpy())
        require(np.isfinite(external.drop(columns='sample_id').to_numpy(dtype=float)).all(), 'Invalid external features')
        for col in ['gc_content', 'at_content', 'entropy']:
            if col in external:
                check = features[['sample_id', col]].merge(external[['sample_id', col]], on='sample_id', validate='one_to_one')
                require(np.allclose(check[col+'_x'], check[col+'_y'], atol=1e-8), f'Feature mismatch: {col}')
        features = features.merge(external.drop(columns=[c for c in external if c in features and c != 'sample_id']), on='sample_id', validate='one_to_one')
        inputs.extend([features_path, feature_index])
    meta, scope = label_transform(data, transform_path)
    if transform_path:
        inputs.append(transform_path)
    data['target_normalized'] = (data.target_log10 - meta['min_log10']) / (meta['max_log10'] - meta['min_log10'])
    data['transform_id'] = meta['transform_id']
    run = Run(output, 'M2', 'pr02-eda-annotation-m2', data, inputs,
              {'normalization_scope': scope, 'transform_id': meta['transform_id'], 'binning': 'fixed descriptive bins', 'feature_source': 'computed_from_sequence' if features_path is None else 'ID-mapped external bundle', 'inference': 'descriptive correlations only'})
    run.meta['fit_subsets'] = ['train'] if transform_path is None else []
    run.meta['label_transform_id'] = meta['transform_id']
    write_json(run.output / 'label_transform.analysis.json', meta)
    sample_path = run.table('eda_samples.tsv', data.merge(features.drop(columns=['split', 'strength', 'target_log10']), on='sample_id', validate='one_to_one'), '\t')
    targets = data[['dataset_id', 'split_id', 'sample_id', 'target_log10', 'target_normalized', 'transform_id', 'schema_version', 'data_version']]
    run.table('targets.analysis.tsv', targets, '\t')
    summaries = []
    for subset, rows in [('all', data)] + [(s, data[data['split'].eq(s)]) for s in ['train', 'val', 'test']]:
        for col in ['strength', 'target_log10', 'target_normalized']:
            d = rows[col].describe(percentiles=[.01, .1, .25, .5, .75, .9, .99])
            summaries.append({'subset': subset, 'target_scale': col, **d.to_dict()})
    summary_path = run.table('distribution_summary.csv', pd.DataFrame(summaries))
    hist = []
    for scale in ['strength', 'target_log10', 'target_normalized']:
        edges = np.histogram_bin_edges(data[scale], bins=55)
        for subset in ['train', 'val', 'test']:
            counts, _ = np.histogram(data.loc[data['split'].eq(subset), scale], bins=edges)
            hist.extend({'scale': scale, 'subset': subset, 'bin_left': edges[i], 'bin_right': edges[i+1], 'count': int(c)} for i, c in enumerate(counts))
    hist = pd.DataFrame(hist)
    hist_path = run.table('histogram_bins.csv', hist)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, scale, label in zip(axes, ['strength', 'target_log10', 'target_normalized'], ['Raw strength (unit unconfirmed)', 'log10(strength)', 'Train-only normalized log10 strength']):
        h = hist[hist.scale.eq(scale)].groupby(['bin_left', 'bin_right'], as_index=False)['count'].sum()
        ax.bar(h.bin_left, h['count'], width=h.bin_right-h.bin_left, align='edge', color='#386CB0', edgecolor='white', linewidth=.2)
        ax.set(xlabel=label, ylabel='Number of samples', title='All samples: n=11,884' if len(data)==11884 else f'All samples: n={len(data):,}')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 4))
    fig.suptitle('Strength distributions | normalized panel is analysis-only', y=1.02)
    run.figure('m2_strength_distributions', fig, 'Strength distributions', [hist_path, summary_path], 'all', 'raw/log10/normalized', len(data), 'Raw labels have a long right tail. Normalization uses only fixed train extrema, with no clipping; pending shared adoption.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ecdf = []
    for subset in ['train', 'val', 'test']:
        vals = np.sort(data.loc[data['split'].eq(subset), 'target_log10'])
        prob = np.arange(1, len(vals)+1)/len(vals)
        axes[0].step(vals, prob, where='post', color=COLORS[subset], label=f'{subset}: n={len(vals)}')
        ecdf.extend({'subset': subset, 'target_log10': x, 'ecdf': y} for x, y in zip(vals, prob))
        axes[1].hist(vals, bins=np.linspace(data.target_log10.min(), data.target_log10.max(), 45), density=True, histtype='step', lw=1.5, color=COLORS[subset], label=subset)
    for ax in axes:
        ax.set_xlabel('log10(strength)')
        ax.legend(frameon=False)
    axes[0].set(ylabel='Empirical cumulative probability', title='Fixed split distributions')
    axes[1].set(ylabel='Density', title='Log10 distributions')
    ecdf_path = run.table('split_ecdf.csv', pd.DataFrame(ecdf))
    run.figure('m2_split_distributions', fig, 'Fixed split distributions', [ecdf_path, sample_path], 'train/val/test', 'log10', len(data), 'Descriptive split QC; similar marginals do not prove absence of sequence-related leakage.')
    feature_cols = [c for c in features if c not in ['sample_id', 'split', 'strength', 'target_log10']]
    correlations = []
    for col in feature_cols:
        for subset in ['all', 'train', 'val', 'test']:
            rows = features if subset == 'all' else features[features['split'].eq(subset)]
            correlations.append({'feature': col, 'subset': subset, 'n': len(rows), 'pearson_log10': rows[col].corr(rows.target_log10), 'spearman_log10': rows[col].corr(rows.target_log10, method='spearman')})
    corr = pd.DataFrame(correlations)
    corr_path = run.table('feature_correlations.csv', corr)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hexbin(features.gc_content, features.target_log10, gridsize=35, mincnt=1, cmap='Blues', bins='log')
    axes[0].set(xlabel='GC fraction (computed from sequence)', ylabel='log10(strength)', title=f'GC vs strength | n={len(data):,}')
    selected = corr[corr['subset'].eq('all')]
    axes[1].barh(selected.feature, selected.spearman_log10, color='#386CB0')
    axes[1].axvline(0, color='gray', lw=.8)
    axes[1].set(xlabel='Spearman rho with log10(strength)', title='Descriptive feature correlations')
    run.figure('m2_feature_relationships', fig, 'Sequence features vs strength', [sample_path, corr_path], 'all', 'log10', len(data), 'Sequence-derived features, not independent measurements. GC and AT are redundant; correlations do not establish biological causality.')
    binned = features.assign(gc_bin=pd.cut(features.gc_content, bins=np.linspace(0, 1, 11), include_lowest=True)).groupby('gc_bin', observed=True).agg(n=('sample_id','size'), gc_mean=('gc_content','mean'), median_log10=('target_log10','median'), q25=('target_log10',lambda x:x.quantile(.25)), q75=('target_log10',lambda x:x.quantile(.75))).reset_index()
    binned_path = run.table('gc_binned_summary.csv', binned)
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios':[2,1]})
    axes[0].errorbar(binned.gc_mean, binned.median_log10, yerr=np.stack([binned.median_log10-binned.q25, binned.q75-binned.median_log10]), fmt='o-', color='#386CB0', capsize=3)
    axes[0].set(ylabel='Median log10 strength (IQR)', title='GC bins: central trend and sample support')
    axes[1].bar(binned.gc_mean, binned.n, width=.07, color='#45A37A')
    axes[1].set(xlabel='Mean GC fraction within bin', ylabel='Sample count')
    run.figure('m2_gc_binned', fig, 'GC binned strength', [binned_path], 'all', 'log10', len(data), 'Bars show bin sizes; error bars are interquartile ranges, not confidence intervals.')
    positional = pd.DataFrame([{'position_1index': pos+1, 'base': base, 'fraction': data.sequence.str[pos].eq(base).mean(), 'n': len(data)} for pos in range(50) for base in 'ACGT'])
    pos_path = run.table('position_base_frequencies.csv', positional)
    matrix = positional.pivot(index='base', columns='position_1index', values='fraction').reindex(list('ACGT'))
    fig, ax = plt.subplots(figsize=(12, 2.6))
    im = ax.imshow(matrix.to_numpy(), aspect='auto', origin='upper', cmap='YlGnBu', vmin=0, vmax=1, extent=(.5,50.5,3.5,-.5))
    ax.set(xlabel='Local sequence position (1-based)', title='Observed base frequencies | no biological region coordinates assigned')
    ax.set_yticks(range(4), list('ACGT'))
    fig.colorbar(im, ax=ax, label='Fraction')
    run.figure('m2_base_frequencies', fig, 'Local base frequencies', [pos_path], 'all', 'sequence', len(data), 'Positions are local 1..50; no TSS, -10/-35 or UP region is inferred from this heatmap.')
    gc_rho = corr.loc[corr.feature.eq('gc_content') & corr['subset'].eq('all'), 'spearman_log10'].iloc[0]
    status = data.annotation_status.value_counts().to_dict()
    text = f'''# M2 数据探索结果

输入：{single(data, 'data_version')}；固定划分 {single(data, 'split_id')}；样本 {len(data):,} 条，每条 50 bp。

## 实际观察

- strength 最小值 {data.strength.min():.2f}，中位数 {data.strength.median():.3f}，99% 分位数 {data.strength.quantile(.99):.4f}，最大值 {data.strength.max():.2f}；原始标尺右尾很长。
- log10 变换是确定性派生，原始标签始终保留。归一化使用固定 train 的 min={meta['min_log10']:.10f}、max={meta['max_log10']:.10f}；不裁剪，val/test 可以超出 [0,1]。
- 归一化产物范围：{scope}。当前分析用变换保存在本 run，未写入共享 data/ 或替代李宇飞的正式 transform。
- GC 与 log10 strength 的 Spearman={gc_rho:.4f}，是弱负相关。该观察不能证明 GC 因果决定强度。
- 注释状态计数：{status}。missing 表示没有可靠证据，不表示元件不存在；−10/−35 关系分析当前不可计算。
- GC/AT 和各碱基比例存在确定性关系，不应当作多条独立生物学证据；本轮特征直接从序列计算，避免按未声明行序拼接 KNN 8 维表。

## 图与源表

每张图的源表、尺度、样本数、参数、hash、注释等级见 figure_manifest.json。source_tables/eda_samples.tsv 保留 sample_id、split、三种尺度和特征；distribution_summary.csv 给出各子集分位数。

## 限制与交接

主表未给出实验单位、来源论文、逐样本 TSS/方向/盒区坐标。本轮展示全部样本的分布用于 EDA/QC，没有用 test 选择模型或调整分析方案。模型误差分析仅在 val 开展。−10/−35 注释来源和待确认事项见 analysis_m2m3/reports/annotation_source_review.md。
'''
    (run.output / 'eda_summary.md').write_text(text, encoding='utf-8')
    run.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--transform', type=Path)
    ap.add_argument('--features', type=Path)
    ap.add_argument('--feature-index', type=Path)
    args = ap.parse_args()
    run_eda(args.output, args.transform, args.features, args.feature_index)


if __name__ == '__main__':
    main()
