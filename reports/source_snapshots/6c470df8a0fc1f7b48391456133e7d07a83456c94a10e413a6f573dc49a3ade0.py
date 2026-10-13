"""Supplementary M2/M3 analysis of frozen v3 inputs; never trains a model."""
from __future__ import annotations

import importlib.metadata
import io
import itertools
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .analysis import REGIONS
from .data import ROOT, coordinate, digest_ids, load, read_json, require, sha, validate, write_json
from .experiment import METHODS, code_hashes, fresh, git_revision, input_hashes, path
from .features import PHYS_NAMES, physchem

INDEX = ROOT / 'results/context87/m2m3_current.json'
DEFAULT_OUTPUT = Path('runs/ecoli87_m2m3_20261013_v1')
CONSENSUS = {'-35': 'TTGACA', '-10': 'TATAAT'}
LABELS = {'knn': 'KNN', 'ridge': 'Ridge', 'svr': 'SVR', 'cnn': 'CNN', 'thermo': '热力学'}
GROUPS = ['low', 'mid', 'high']
COLORS = ['#386cb0', '#45a37a', '#e18b35', '#8a6fb0', '#c95959']


def repo_path(value):
    p = Path(value)
    p = (ROOT / p).resolve() if not p.is_absolute() else p.resolve()
    require(p.is_relative_to(ROOT), 'Path must remain inside repository')
    return p


def frame_data(cfg):
    rows, transform = load(cfg)
    data = pd.DataFrame(rows)
    for column in ['strength', 'target_log10', 'source_row', 'tss_1based', 'sequence_length', 'tss_index_0based']:
        data[column] = pd.to_numeric(data[column], errors='raise')
    data = data.sort_values('sample_id').reset_index(drop=True)
    data['target_normalized'] = (data.target_log10-transform['min_log10'])/(transform['max_log10']-transform['min_log10'])
    return data, transform


def thresholds(data):
    values = data.loc[data['split'].eq('train'), 'target_log10']
    require(len(values) > 0, 'No training labels')
    bounds = values.quantile([1/3, 2/3]).to_numpy()
    require(bounds[0] < bounds[1], 'Training tertiles coincide')
    return bounds


def strength_groups(values, bounds):
    return np.select([values <= bounds[0], values <= bounds[1]], ['low', 'mid'], default='high')


def check_predictions(frame, data, cfg, method):
    """Join by ID, require a row for each request, and preserve failed rows."""
    needed = {'dataset_id', 'data_version', 'split_id', 'sample_id', 'method_name', 'run_id', 'subset',
              'true_value', 'true_value_log10', 'predicted_value', 'predicted_value_log10',
              'prediction_status', 'sequence_length', 'target_scale_model'}
    require(needed <= set(frame), f'{method}: missing columns {needed-set(frame)}')
    frame = frame.copy()
    requested = set(data.loc[data['split'].eq('val'), 'sample_id'])
    require(frame.sample_id.notna().all() and frame.sample_id.is_unique, f'{method}: duplicate/missing ID')
    require(set(frame.sample_id) == requested, f'{method}: validation request ID mismatch')
    for key in ['dataset_id', 'data_version', 'split_id']:
        require(frame[key].eq(cfg[key]).all(), f'{method}: wrong {key}')
    require(frame.method_name.eq(method).all() and frame.run_id.eq(cfg['runs'][method]).all(), f'{method}: wrong model identity')
    require(frame.subset.eq('val').all() and frame.target_scale_model.eq('log10').all(), f'{method}: wrong subset/scale')
    require(pd.to_numeric(frame.sequence_length).eq(87).all(), f'{method}: wrong sequence length')
    for key in ['true_value', 'true_value_log10', 'predicted_value', 'predicted_value_log10']:
        frame[key] = pd.to_numeric(frame[key], errors='raise')
    truth = data.set_index('sample_id').loc[frame.sample_id]
    require(np.allclose(frame.true_value, truth.strength, rtol=0, atol=1e-8) and
            np.allclose(frame.true_value_log10, truth.target_log10, rtol=0, atol=1e-10), f'{method}: wrong truth')
    require(frame.prediction_status.isin(['ok', 'failed']).all(), f'{method}: unknown status')
    ok = frame.prediction_status.eq('ok')
    successful = frame.loc[ok]
    require(np.isfinite(successful[['predicted_value', 'predicted_value_log10']]).all().all()
            and successful.predicted_value.gt(0).all(), f'{method}: invalid successful prediction')
    with np.errstate(over='ignore'):
        require(np.allclose(10**successful.predicted_value_log10, successful.predicted_value, rtol=1e-8, atol=0), f'{method}: raw/log10 mismatch')
    require(frame.loc[~ok, ['predicted_value', 'predicted_value_log10']].isna().all().all(), f'{method}: failed outputs must be null')
    if (~ok).any():
        require('error_reason' in frame and frame.loc[~ok, 'error_reason'].fillna('').str.strip().ne('').all(), f'{method}: failure reason missing')
    return frame.sort_values('sample_id').reset_index(drop=True)


def inputs(cfg):
    files = [ROOT/cfg[k] for k in ['samples', 'splits', 'label_transform']]
    files += [path(cfg, method)/'predictions_val.csv' for method in METHODS]
    files += [path(cfg, 'thermo')/'states.jsonl', ROOT/'data/03_Ecoli_strength87/source_manifest.json',
              ROOT/'data/03_Ecoli_strength87/data_manifest.json']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in files}


def load_inputs(cfg):
    data, transform = frame_data(cfg)
    frames = {m: check_predictions(pd.read_csv(path(cfg, m)/'predictions_val.csv'), data, cfg, m) for m in METHODS}
    with (path(cfg, 'thermo')/'states.jsonl').open(encoding='utf-8') as f:
        states = [json.loads(line) for line in f]
    require(len(states) == len(data) and len({s['sample_id'] for s in states}) == len(data)
            and {s['sample_id'] for s in states} == set(data.sample_id), 'Thermodynamic state ID mismatch')
    return data, transform, frames, {s['sample_id']: s for s in states}


def region_features(data, states):
    out = []
    keys = {'UP': ('UP_position', 'UP'), '-35': ('hex35_position', 'hex35'),
            'spacer': ('spacer_position', 'spacer'), '-10': ('hex10_position', 'hex10'),
            'discriminator': ('disc_position', 'disc')}
    for row in data.itertuples(index=False):
        state = states[row.sample_id]
        require(state['TSS'] == 67 and state['annotation_status'] == 'model_inferred', 'Wrong thermodynamic coordinate/evidence level')
        windows = {name: tuple(state[key]) for name, (key, _) in keys.items()}
        windows['ITR'] = (67, 87)
        require(windows['UP'][1]+1 == windows['-35'][0] and windows['-35'][1] == windows['spacer'][0]
                and windows['spacer'][1] == windows['-10'][0] and windows['-10'][1] == windows['discriminator'][0]
                and windows['discriminator'][1] == 67, 'Discontinuous tool regions')
        for name, (key, seqkey) in keys.items():
            start, end = windows[name]
            require(type(start) is int and type(end) is int and 0 <= start < end <= 87
                    and row.sequence[start:end] == state[seqkey], 'Tool region slice mismatch')
        require(len(state['UP']) == 24 and len(state['hex35']) == len(state['hex10']) == 6
                and 15 <= len(state['spacer']) <= 20 and 6 <= len(state['disc']) <= 10
                and state['ITR'] == row.sequence[67:87], 'Tool state length mismatch')
        for policy, intervals in [('canonical_window_assumption', REGIONS), ('model_inferred', windows)]:
            for name, (start, end) in intervals.items():
                sequence = row.sequence[start:end]
                consensus = CONSENSUS.get(name)
                out.append(dict(sample_id=row.sample_id, subset=row.split, annotation_status=policy, region=name,
                                start_0based=start, end_0based_exclusive=end, start_relative_tss=coordinate(start),
                                end_relative_tss=coordinate(end-1), sequence=sequence, length=end-start,
                                gc_fraction=sum(b in 'GC' for b in sequence)/len(sequence),
                                consensus_match_fraction=sum(a == b for a, b in zip(sequence, consensus))/6 if consensus else np.nan,
                                target_log10=row.target_log10))
    return pd.DataFrame(out)


def safe_metric(name, y, p):
    if len(y) == 0 or (name in ['r2', 'spearman'] and len(y) < 2):
        return np.nan, 'undefined', 'insufficient_samples'
    if name in ['r2', 'spearman'] and np.ptp(y) == 0:
        return np.nan, 'undefined', 'constant_truth'
    if name == 'spearman' and np.ptp(p) == 0:
        return np.nan, 'undefined', 'constant_prediction'
    value = {'r2': r2_score, 'mae': mean_absolute_error, 'rmse': lambda a, b: np.sqrt(mean_squared_error(a, b)),
             'spearman': lambda a, b: spearmanr(a, b).statistic}[name](y, p)
    return float(value), 'ok', ''


def error_tables(data, frames):
    bounds = thresholds(data)
    successes = {m: set(f.loc[f.prediction_status.eq('ok'), 'sample_id']) for m, f in frames.items()}
    common = set.intersection(*successes.values())
    metrics, used, groups, errors, coverage = [], [], [], [], []
    for method in METHODS:
        frame = frames[method]
        coverage.append(dict(method=method, n_requested=len(frame), n_success=len(successes[method]),
                             n_failed=len(frame)-len(successes[method]), n_common=len(common)))
        e = frame.loc[frame.prediction_status.eq('ok')].copy()
        e['residual_log10'] = e.predicted_value_log10-e.true_value_log10
        e['absolute_error_log10'] = e.residual_log10.abs()
        e['strength_group'] = strength_groups(e.true_value_log10, bounds)
        e['in_common_set'] = e.sample_id.isin(common)
        errors.append(e)
        for set_name, ids in [('all_success', successes[method]), ('common', common)]:
            part = e[e.sample_id.isin(ids)]
            used.extend(dict(method=method, evaluation_subset=set_name, sample_id=sid) for sid in sorted(ids))
            for scale, ycol, pcol in [('raw', 'true_value', 'predicted_value'), ('log10', 'true_value_log10', 'predicted_value_log10')]:
                for metric in ['r2', 'mae', 'rmse', 'spearman']:
                    value, status, reason = safe_metric(metric, part[ycol].to_numpy(), part[pcol].to_numpy())
                    metrics.append(dict(method=method, model_run_id=frame.run_id.iloc[0], evaluation_subset=set_name,
                                        target_scale=scale, metric=metric, value=value, metric_status=status, reason=reason,
                                        n_requested=len(frame), n_success=len(successes[method]), n_used=len(part)))
            for group in GROUPS:
                g = part[part.strength_group.eq(group)]
                groups.append(dict(method=method, evaluation_subset=set_name, strength_group=group, n=len(g),
                                   mae_log10=g.absolute_error_log10.mean(), bias_log10=g.residual_log10.mean()))
    return {'metrics': pd.DataFrame(metrics), 'evaluation_ids': pd.DataFrame(used, columns=['method', 'evaluation_subset', 'sample_id']),
            'group_errors': pd.DataFrame(groups), 'prediction_errors': pd.concat(errors, ignore_index=True),
            'coverage': pd.DataFrame(coverage)}, bounds, sorted(common)


def select_cases(errors, common, seed):
    choices, disagreements = [], []
    for method in METHODS:
        top = errors[errors.method_name.eq(method)].sort_values(['absolute_error_log10', 'sample_id'], ascending=[False, True]).head(5)
        choices.extend(dict(selection_rule='largest_error', selector=method, rank=i+1, sample_id=sid) for i, sid in enumerate(top.sample_id))
    matrix = errors[errors.sample_id.isin(common)].pivot(index='sample_id', columns='method_name', values='predicted_value_log10').reindex(columns=METHODS)
    for a, b in itertools.combinations(METHODS, 2):
        delta = (matrix[a]-matrix[b]).abs().rename('absolute_disagreement_log10').reset_index()
        delta['method_a'], delta['method_b'] = a, b
        disagreements.append(delta)
        top = delta.sort_values(['absolute_disagreement_log10', 'sample_id'], ascending=[False, True]).head(5)
        choices.extend(dict(selection_rule='largest_disagreement', selector=f'{a}:{b}', rank=i+1, sample_id=sid) for i, sid in enumerate(top.sample_id))
    rng = np.random.default_rng(seed)
    random_ids = rng.choice(sorted(common), size=min(5, len(common)), replace=False)
    choices.extend(dict(selection_rule='random_common', selector=str(seed), rank=i+1, sample_id=sid) for i, sid in enumerate(random_ids))
    return pd.DataFrame(choices, columns=['selection_rule', 'selector', 'rank', 'sample_id']), pd.concat(disagreements, ignore_index=True)


def numeric_tables(cfg):
    data, transform, frames, states = load_inputs(cfg)
    features = pd.DataFrame(physchem(data.sequence), columns=PHYS_NAMES)
    features.insert(0, 'sample_id', data.sample_id)
    samples = data.merge(features, on='sample_id', validate='one_to_one')
    tables = {'samples': samples, 'requested_predictions': pd.concat(frames.values(), ignore_index=True)}
    distributions, correlations, base_frequency = [], [], []
    for subset in ['all', 'train', 'val', 'test']:
        part = samples if subset == 'all' else samples[samples['split'].eq(subset)]
        for scale in ['strength', 'target_log10', 'target_normalized']:
            distributions.append(dict(subset=subset, scale=scale, **part[scale].describe(percentiles=[.01, .25, .5, .75, .99]).to_dict()))
        for feature in PHYS_NAMES:
            value, status, reason = safe_metric('spearman', part.target_log10.to_numpy(), part[feature].to_numpy())
            correlations.append(dict(subset=subset, feature=feature, n=len(part), spearman=value, metric_status=status, reason=reason))
    for pos in range(87):
        base_frequency.extend(dict(position_0based=pos, position_relative_tss=coordinate(pos), base=b,
                                   fraction=data.sequence.str[pos].eq(b).mean(), n=len(data)) for b in 'ACGT')
    tables.update(distributions=pd.DataFrame(distributions), feature_correlations=pd.DataFrame(correlations),
                  base_frequency=pd.DataFrame(base_frequency), region_features=region_features(data, states))
    region_summary, frequencies = [], []
    for (policy, region), part in tables['region_features'].groupby(['annotation_status', 'region'], sort=True):
        for feature in ['gc_fraction', 'consensus_match_fraction']:
            valid = part[part[feature].notna()]
            if valid.empty:
                continue
            rho, status, reason = safe_metric('spearman', valid.target_log10.to_numpy(), valid[feature].to_numpy())
            region_summary.append(dict(annotation_status=policy, region=region, feature=feature, n=len(valid),
                                       mean_feature=valid[feature].mean(), spearman=rho, metric_status=status, reason=reason))
        if region in CONSENSUS:
            for i in range(6):
                frequencies.extend(dict(annotation_status=policy, region=region, motif_position_1based=i+1, base=b,
                                        fraction=part.sequence.str[i].eq(b).mean(), n=len(part)) for b in 'ACGT')
    tables.update(region_summary=pd.DataFrame(region_summary), region_base_frequency=pd.DataFrame(frequencies))
    motif_rows = tables['region_features'][tables['region_features'].region.isin(CONSENSUS)]
    tables['region_similarity_groups'] = motif_rows.groupby(
        ['annotation_status', 'region', 'consensus_match_fraction'], as_index=False
    ).agg(n=('sample_id', 'size'), median_log10=('target_log10', 'median'))
    tables['inferred_spacing'] = pd.DataFrame([dict(sample_id=sid, spacer_length=len(s['spacer']),
                                                  discriminator_length=len(s['disc']), annotation_status='model_inferred') for sid, s in sorted(states.items())])
    error, bounds, common = error_tables(data, frames)
    tables.update(error)
    choices, disagreements = select_cases(error['prediction_errors'], common, cfg['seed'])
    tables['case_selection'], tables['model_disagreements'] = choices, disagreements
    unique = samples[samples.sample_id.isin(choices.sample_id)].copy()
    for method, frame in frames.items():
        detail = frame[['sample_id', 'prediction_status', 'predicted_value', 'predicted_value_log10']].copy()
        detail['residual_log10'] = detail.predicted_value_log10-detail.sample_id.map(data.set_index('sample_id').target_log10)
        detail['absolute_error_log10'] = detail.residual_log10.abs()
        detail = detail.rename(columns={c: f'{method}_{c}' for c in detail if c != 'sample_id'})
        unique = unique.merge(detail, on='sample_id', validate='one_to_one')
    for policy in ['canonical_window_assumption', 'model_inferred']:
        part = tables['region_features'][tables['region_features'].annotation_status.eq(policy)]
        for region in CONSENSUS:
            x = part[part.region.eq(region)][['sample_id', 'sequence', 'gc_fraction', 'consensus_match_fraction']]
            x = x.rename(columns={c: f'{policy}_{region}_{c}' for c in x if c != 'sample_id'})
            unique = unique.merge(x, on='sample_id', validate='one_to_one')
    tables['case_review'] = choices.merge(unique, on='sample_id', validate='many_to_one')
    return tables, {'thresholds_log10': bounds.tolist(), 'common_val_ids': common, 'seed': cfg['seed'],
                    'normalization': transform, 'n_all': len(data), 'n_val': int(data['split'].eq('val').sum())}


def chinese_font():
    candidates = ['Noto Sans CJK SC', 'Source Han Sans SC', 'Heiti SC', 'SimHei', 'Microsoft YaHei', 'WenQuanYi Zen Hei']
    for name in candidates:
        try:
            file = fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
        except ValueError:
            continue
        plt.rcParams.update({'font.family': [name, 'DejaVu Sans'], 'axes.unicode_minus': False, 'font.size': 10,
                             'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'path'})
        return {'family': name, 'font_file_name': Path(file).name, 'sha256': sha(file)}
    raise ValueError('Chinese font required: install Noto Sans CJK SC or another documented CJK font')


def make_figures(output, tables, parameters):
    font = chinese_font()
    folder = output/'figures'
    folder.mkdir()
    figures = []
    samples = tables['samples']
    n, common_n = parameters['n_all'], len(parameters['common_val_ids'])

    def save(name, fig, title, sources, subset, scale, count, caption, annotation='not_applicable'):
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, .06, 1, .94))
        fig.text(.02, .015, f'n={count:,}；{caption}', fontsize=9)
        images = []
        for extension in ['png', 'svg']:
            file = folder/f'{name}.{extension}'
            fig.savefig(file, dpi=180, bbox_inches='tight')
            images.append({'path': file.relative_to(output).as_posix(), 'sha256': sha(file)})
        plt.close(fig)
        figures.append(dict(figure_id=name, title=title, subset=subset, scale=scale, n_samples=count,
                            annotation_status=annotation, caption=caption, visual_review='pending', images=images,
                            source_tables=[{'path': f'source_tables/{s}.csv', 'sha256': sha(output/f'source_tables/{s}.csv')} for s in sources]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, column, title in zip(axes, ['strength', 'target_log10', 'target_normalized'], ['原始强度（单位未确认）', 'log10 强度', '训练集参数归一化强度']):
        ax.hist(samples[column], bins=55, color=COLORS[0]); ax.set(xlabel=title, ylabel='样本数')
    save('01_strength_distribution', fig, '强度分布：原始值右尾较长', ['samples', 'distributions'], 'all', 'raw/log10/normalized', n, '仅训练集拟合归一化，不裁剪范围')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for subset, color in zip(['train', 'val', 'test'], COLORS):
        y = np.sort(samples.loc[samples['split'].eq(subset), 'target_log10'])
        axes[0].step(y, np.arange(1, len(y)+1)/len(y), where='post', label=f'{subset} n={len(y)}', color=color)
    axes[0].set(xlabel='log10 强度', ylabel='累计比例'); axes[0].legend()
    axes[1].boxplot([samples.loc[samples['split'].eq(s), 'target_log10'] for s in ['train', 'val', 'test']], tick_labels=['train', 'val', 'test'], showfliers=True)
    axes[1].set(ylabel='log10 强度')
    save('02_split_distribution', fig, '继承划分的强度分布', ['samples'], 'train/val/test', 'log10', n, '仅描述标签分布；未计算 test 模型成绩')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    image = axes[0].hexbin(samples.gc_fraction, samples.target_log10, gridsize=40, bins='log', mincnt=1, cmap='Blues')
    fig.colorbar(image, ax=axes[0], label='六边形样本数（对数色阶）'); axes[0].set(xlabel='整体 GC 比例', ylabel='log10 强度')
    corr = tables['feature_correlations'].query('subset == "all"').set_index('feature').reindex(PHYS_NAMES)
    axes[1].barh(['GC比例', 'AT比例', 'GC偏斜', 'AT偏斜', 'Tm近似', '二核苷酸持久性', '二核苷酸堆积', '碱基熵'], corr.spearman, color=COLORS[1])
    axes[1].axvline(0, color='gray', lw=.7); axes[1].set(xlabel='与 log10 强度的 Spearman')
    save('03_feature_relationships', fig, '整体序列特征与强度的描述性关系', ['samples', 'feature_correlations'], 'all', 'log10', n, '序列派生特征，GC与AT等存在冗余；不是因果证据')
    freq = tables['base_frequency'].pivot(index='base', columns='position_0based', values='fraction').reindex(list('ACGT'))
    fig, ax = plt.subplots(figsize=(13, 3.4)); image = ax.imshow(freq, aspect='auto', vmin=0, vmax=1, cmap='YlGnBu')
    ticks = [0, 17, 32, 55, 67, 76, 86]
    ax.set_xticks(ticks, [str(coordinate(i)) if i < 67 else f'+{coordinate(i)}' for i in ticks]); ax.set_yticks(range(4), list('ACGT'))
    ax.axvline(66.5, color='#c95959', lw=1); ax.set(xlabel='相对 TSS 坐标（没有0位）', ylabel='碱基')
    fig.colorbar(image, ax=ax, label='频率')
    save('04_tss_base_frequency', fig, '经核验 TSS 周围的碱基频率', ['base_frequency'], 'all', 'sequence', n, 'TSS对应+1；盒区没有逐条实验注释', 'experimental_tss_verified')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, region in enumerate(['-35', '-10']):
        frequencies = tables['region_base_frequency']
        for i, policy in enumerate(['canonical_window_assumption', 'model_inferred']):
            part = frequencies[(frequencies.region == region) & (frequencies.annotation_status == policy)]
            at = part[part.base.isin(['A', 'T'])].groupby('motif_position_1based').fraction.sum()
            axes[0, col].plot(at.index, at, 'o-', label=['固定窗口假设', '工具推定'][i], color=COLORS[i])
            traits = tables['region_similarity_groups']
            binned = traits[(traits.region == region) & (traits.annotation_status == policy)].sort_values('consensus_match_fraction')
            axes[1, col].plot(binned.consensus_match_fraction, binned.median_log10, 'o-', label=['固定窗口假设', '工具推定'][i], color=COLORS[i])
            last = binned.iloc[-1]
            axes[1, col].annotate(f'n={int(last.n)}', (last.consensus_match_fraction, last.median_log10),
                                  xytext=(-8, 7 if i == 1 else -14), textcoords='offset points',
                                  ha='right', fontsize=8, color=COLORS[i])
        axes[0, col].set(title=f'{region} 六碱基窗口', xlabel='窗口内位置', ylabel='A/T 频率', ylim=(0, 1)); axes[0, col].legend()
        axes[1, col].set(xlabel=f'与 {CONSENSUS[region]} 的相同字母比例', ylabel='组内 log10 强度中位数'); axes[1, col].legend()
    save('05_region_relationships', fig, '区域特征分析：固定窗口与工具推定分别统计', ['region_base_frequency', 'region_features', 'region_summary', 'region_similarity_groups'], 'all', 'sequence/log10', n, '高共识组可能很小，标注末组n；各组n见源表；不证明机制', 'canonical_window_assumption/model_inferred')
    common_metrics = tables['metrics'].query('evaluation_subset == "common" and target_scale == "log10"')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6))
    for ax, metric, title in zip(axes, ['r2', 'mae', 'spearman'], ['R2（log10）', 'MAE（log10）', 'Spearman']):
        values = common_metrics[common_metrics.metric.eq(metric)].set_index('method').reindex(METHODS).value
        ax.bar([LABELS[m] for m in METHODS], values, color=COLORS); ax.set(title=title); ax.axhline(0, color='gray', lw=.6)
    save('06_model_comparison', fig, '五方法共同验证集成绩', ['metrics', 'coverage', 'evaluation_ids'], 'val_common', 'log10', common_n, '五方法输入均为87bp；不据单次验证值宣称显著优越')
    errors = tables['prediction_errors']
    for number, kind in [('07', 'scatter'), ('08', 'residual')]:
        fig, axes = plt.subplots(2, 3, figsize=(13, 8))
        for ax, method in zip(axes.flat, METHODS):
            e = errors[errors.method_name.eq(method) & errors.in_common_set]
            y = e.true_value_log10
            ax.scatter(y, e.predicted_value_log10 if kind == 'scatter' else e.residual_log10, s=8, alpha=.35, color=COLORS[METHODS.index(method)])
            if len(e) and kind == 'scatter':
                limits = [min(y.min(), e.predicted_value_log10.min()), max(y.max(), e.predicted_value_log10.max())]
                ax.plot(limits, limits, 'k--', lw=.8)
            else:
                ax.axhline(0, color='gray', lw=.8)
            ax.set(title=f'{LABELS[method]} n={len(e)}', xlabel='观测 log10 强度', ylabel='预测 log10 强度' if kind == 'scatter' else '预测减观测（log10）')
        axes.flat[-1].axis('off')
        save(f'{number}_{kind}', fig, '五方法预测与观测' if kind == 'scatter' else '五方法残差：正值高估，负值低估', ['prediction_errors', 'evaluation_ids'], 'val_common', 'log10', common_n, '保留共同成功样本，不选择性删除大误差')
    groups = tables['group_errors'].query('evaluation_subset == "common"')
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8)); x = np.arange(3)
    for ax, column, title in zip(axes, ['mae_log10', 'bias_log10'], ['分组绝对误差', '分组平均偏差（预测减观测）']):
        for i, method in enumerate(METHODS):
            g = groups[groups.method.eq(method)].set_index('strength_group').reindex(GROUPS)
            ax.bar(x+(i-2)*.15, g[column], width=.14, label=LABELS[method], color=COLORS[i])
        counts = groups[groups.method.eq(METHODS[0])].set_index('strength_group').reindex(GROUPS).n
        ax.set_xticks(x, [f'{label}\nn={count}' for label, count in zip(['低强度', '中强度', '高强度'], counts)])
        ax.set(title=title, ylabel='log10'); ax.axhline(0, color='gray', lw=.7)
    axes[1].legend(fontsize=9)
    save('09_strength_group_errors', fig, '训练集阈值固定的低中高强度误差', ['group_errors', 'prediction_errors'], 'val_common', 'log10', common_n, '阈值仅取train三分位点；不能按预测值重新分组')
    delta = tables['model_disagreements']; matrix = np.zeros((5, 5))
    for row in delta.groupby(['method_a', 'method_b']).absolute_disagreement_log10.mean().reset_index().itertuples():
        i, j = METHODS.index(row.method_a), METHODS.index(row.method_b); matrix[i, j] = matrix[j, i] = row.absolute_disagreement_log10
    fig, ax = plt.subplots(figsize=(7, 6)); image = ax.imshow(matrix, cmap='Blues')
    ax.set_xticks(range(5), [LABELS[m] for m in METHODS]); ax.set_yticks(range(5), [LABELS[m] for m in METHODS])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{matrix[i,j]:.3f}', ha='center', va='center', color='white' if matrix[i,j] > matrix.max()*.5 else 'black')
    fig.colorbar(image, ax=ax, label='平均绝对预测差（log10）')
    save('10_model_disagreement', fig, '五方法预测分歧', ['model_disagreements', 'evaluation_ids'], 'val_common', 'log10', common_n, '模型分歧不等同于模型误差；最大分歧案例另存表')
    return figures, font


def report(tables, params, figures):
    def number(value):
        return '不可计算' if pd.isna(value) else f'{value:.4f}'
    metrics = tables['metrics'].query('evaluation_subset == "common" and target_scale == "log10"').pivot(index='method', columns='metric', values='value')
    summary = '| 方法 | R2(log10) | MAE(log10) | RMSE(log10) | Spearman |\n|---|---:|---:|---:|---:|\n'
    for method in METHODS:
        summary += f'| {LABELS[method]} | ' + ' | '.join(number(metrics.loc[method, k]) for k in ['r2', 'mae', 'rmse', 'spearman']) + ' |\n'
    samples = tables['samples']; rho = tables['feature_correlations'].query('subset == "all" and feature == "gc_fraction"').spearman.iloc[0]
    counts = samples['split'].value_counts().to_dict()
    regional = '| 窗口来源 | 区域 | 平均A/T比例 | GC与强度Spearman | 共识评分与强度Spearman |\n|---|---|---:|---:|---:|\n'
    for policy, label in [('canonical_window_assumption', '固定窗口假设'), ('model_inferred', '工具推定')]:
        for region in ['-35', '-10']:
            part = tables['region_summary'].query('annotation_status == @policy and region == @region').set_index('feature')
            regional += f'| {label} | {region} | {1-part.loc["gc_fraction", "mean_feature"]:.2%} | {number(part.loc["gc_fraction", "spearman"])} | {number(part.loc["consensus_match_fraction", "spearman"])} |\n'
    group_table = '| 方法 | 强度组 | n | MAE(log10) | 平均偏差(log10) |\n|---|---|---:|---:|---:|\n'
    for r in tables['group_errors'].query('evaluation_subset == "common"').itertuples():
        group_table += f'| {LABELS[r.method]} | {dict(low="低", mid="中", high="高")[r.strength_group]} | {r.n} | {number(r.mae_log10)} | {number(r.bias_log10)} |\n'
    spacing = tables['inferred_spacing'].spacer_length.value_counts().sort_index().to_dict()
    text = f'''# E. coli 87 bp：M2 数据与区域分析、M3 误差与案例

## 当前发现

原始强度中位数为 {samples.strength.median():.3f}，最大值为 {samples.strength.max():.2f}，右尾较长。整体 GC 与 log10 强度的 Spearman 为 {rho:.4f}。这是序列派生特征的描述性关系，不说明因果。

五方法在同一批 {len(params['common_val_ids']):,} 条共同成功验证样本上比较。当前冻结模型的数值如下，验证集也参与过队友选参，不把单次排名解释为稳定或显著优越。

{summary}
## M2：数据、归一化与区域来源

本次全部 {len(samples):,} 条样本为方向已核验的 87 bp，划分计数为 {counts}。原强度未改变，87 bp 中 [17:67] 保留原 50 bp。历史参考 U00096.2 的恢复上下文逐条对照原论文 TSS 表；75 条多位置记录由上游排除并保存，分析未再删样本。

来源登记在 data/03_Ecoli_strength87/source_manifest.json，原作者库为 [Deep_promoter](https://github.com/HaochenW/Deep_promoter)，TSS依据为 [Thomason等论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC4288677/) 的附件。标签实验单位与条件仍未确认，不写成荧光或RNA/DNA比值。

归一化读取正式 label_transform，不重新拟合：训练范围 {params['normalization']['min_log10']:.10f} 到 {params['normalization']['max_log10']:.10f}；clip=false。全体归一化范围为 {samples.target_normalized.min():.4f} 到 {samples.target_normalized.max():.4f}，超出[0,1]时保留。

TSS索引为67，对应生物坐标+1；相对坐标为−67…−1、+1…+20，没有0位。可靠实验来源核验的是TSS，不是每条−10/−35/UP盒区。区域表分别保存 canonical_window_assumption 和 model_inferred，六碱基共识匹配仅用 TTGACA/TATAAT 的相同字母比例，不称独立实测元件。工具间隔来自实际最小自由能状态；固定窗口的17bp间隔只是布局假设。

区域统计源表包含每条序列、窗口、坐标、GC、共识评分、均值和Spearman。盒区推定参与了热力学模型，因此其特征与热力学预测的关系不能作为独立验证。固定/推定两种范围的观察分别解释，不沿用旧81bp小子集结论。

{regional}
每行n={len(samples):,}。各共识匹配分组的n和中位数在region_similarity_groups.csv；高共识组可能很小，不将曲线末端当作稳定趋势。模型推定 spacer 长度的实际样本计数为 {spacing}；这些长度来自工具选择，不能称为实验测得的间隔。

## M3：分组误差与案例

低、中、高强度分界为训练集 log10 三分位点 {params['thresholds_log10'][0]:.10f}、{params['thresholds_log10'][1]:.10f}。低组≤第一阈值，中组>第一阈值且≤第二阈值，高组>第二阈值。分组表保存n、MAE、平均残差；正残差为高估，负残差为低估。

指标同时报告各方法全部成功样本及共同成功样本，raw/log10分开；缺失或常量导致无定义时保留undefined及原因，不填0。完整请求表保留失败行和原因；不能把Tx_rate直接作为原始strength预测。本轮未重新训练、校准或挑选模型，不评价E.coli test。

{group_table}

案例选择为每方法最大绝对log10误差5条、每对方法最大log10预测差5条，以及种子 {params['seed']} 的共同样本随机5条。同分按sample_id排序；重复入选保留规则身份。共 {len(tables['case_review'])} 个选择记录、{tables['case_review'].sample_id.nunique()} 个不同样本。case_review.csv保留父ID、源行、87bp、TSS/方向、整体与区域特征、五方法预测和误差。最大误差来自每方法成功集合，分歧与随机来自共同集合；案例不代表全部数据，也不编造未实验核验的生物学原因。

## 图与数值源表

'''
    for item in figures:
        text += f"### {item['title']}\n\n![{item['title']}]({item['images'][0]['path']})\n\n{item['caption']}。n={item['n_samples']:,}，子集={item['subset']}，尺度={item['scale']}。源表：" + '、'.join(f"[{Path(s['path']).name}]({s['path']})" for s in item['source_tables']) + '。\n\n'
    text += '## 解释边界\n\n继承随机划分可能含邻近基因组上下文重叠，历史项目也曾报告test成绩。本轮验证分析不是全新盲测或外部泛化；没有额外显著性检验，区域相关性不代表实验机制，预测和案例不代表湿实验结果。\n'
    return text


def update_artifacts(output, meta):
    meta['artifacts'] = {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name != 'run_manifest.json'}
    write_json(output/'run_manifest.json', meta)


def independent_numeric_check(cfg, output):
    """Second calculation: scalar sums/rank Pearson, not the producer's metric helper."""
    samples = pd.read_csv(ROOT/cfg['samples'], sep='\t').set_index('sample_id')
    predictions = {m: pd.read_csv(path(cfg, m)/'predictions_val.csv').set_index('sample_id') for m in METHODS}
    ids = pd.read_csv(output/'source_tables/evaluation_ids.csv')
    stored = pd.read_csv(output/'source_tables/metrics.csv')
    for row in stored.itertuples():
        used = ids.loc[ids.method.eq(row.method) & ids.evaluation_subset.eq(row.evaluation_subset), 'sample_id'].tolist()
        frame = predictions[row.method].loc[used]
        y = frame['true_value_log10' if row.target_scale == 'log10' else 'true_value'].to_numpy(float)
        p = frame['predicted_value_log10' if row.target_scale == 'log10' else 'predicted_value'].to_numpy(float)
        require(len(used) == row.n_used, 'Independent metric n differs')
        if row.metric_status == 'undefined':
            require(np.isnan(row.value) and (len(y) == 0 or
                    (row.metric in ['r2', 'spearman'] and (len(y) < 2 or np.ptp(y) == 0)) or
                    (row.metric == 'spearman' and np.ptp(p) == 0)), 'Incorrect undefined metric')
            continue
        residuals = [float(b-a) for a, b in zip(y, p)]
        if row.metric == 'mae': value = sum(abs(r) for r in residuals)/len(y)
        elif row.metric == 'rmse': value = (sum(r*r for r in residuals)/len(y))**.5
        elif row.metric == 'r2':
            mean = sum(float(v) for v in y)/len(y)
            value = 1-sum(r*r for r in residuals)/sum((float(v)-mean)**2 for v in y)
        else: value = pd.Series(y).rank(method='average').corr(pd.Series(p).rank(method='average'))
        require(np.isclose(value, row.value, rtol=1e-10, atol=1e-12), 'Independent metric differs')
    bounds = np.quantile(samples.loc[samples['split'].eq('train'), 'target_log10'], [1/3, 2/3])
    groups = pd.read_csv(output/'source_tables/group_errors.csv')
    for row in groups.itertuples():
        used = ids.loc[ids.method.eq(row.method) & ids.evaluation_subset.eq(row.evaluation_subset), 'sample_id'].tolist()
        part = predictions[row.method].loc[used]
        y = part.true_value_log10
        mask = (y <= bounds[0]) if row.strength_group == 'low' else ((y > bounds[1]) if row.strength_group == 'high' else ((y > bounds[0]) & (y <= bounds[1])))
        residuals = [float(r.predicted_value_log10-r.true_value_log10) for r in part.loc[mask].itertuples()]
        require(len(residuals) == row.n, 'Independent group count differs')
        for column, values in [('mae_log10', [abs(r) for r in residuals]), ('bias_log10', residuals)]:
            actual = getattr(row, column)
            require(np.isnan(actual) if not values else np.isclose(actual, sum(values)/len(values), rtol=1e-10, atol=1e-12), 'Independent group metric differs')
    regions = pd.read_csv(output/'source_tables/region_features.csv')
    for row in regions.itertuples():
        sequence = samples.loc[row.sample_id, 'sequence'][row.start_0based:row.end_0based_exclusive]
        require(sequence == row.sequence and row.length == len(sequence), 'Independent region slice differs')
        require(np.isclose(row.gc_fraction, (sequence.count('G')+sequence.count('C'))/len(sequence)), 'Independent region GC differs')
        if row.region in CONSENSUS:
            match = sum(a == b for a, b in zip(sequence, CONSENSUS[row.region]))/6
            require(np.isclose(row.consensus_match_fraction, match), 'Independent motif score differs')
    summary = pd.read_csv(output/'source_tables/region_summary.csv')
    for row in summary.itertuples():
        part = regions[regions.annotation_status.eq(row.annotation_status) & regions.region.eq(row.region)]
        values = part[row.feature]
        require(row.n == len(values) and np.isclose(row.mean_feature, sum(values)/len(values)), 'Independent regional mean/n differs')
        rho = values.rank().corr(part.target_log10.rank())
        require((row.metric_status == 'undefined' and np.isnan(row.spearman)) or np.isclose(row.spearman, rho, atol=1e-12), 'Independent regional correlation differs')
    selected = pd.read_csv(output/'source_tables/case_selection.csv')
    for row in selected.itertuples():
        if row.selection_rule == 'largest_error':
            frame = predictions[row.selector]
            candidates = [(sid, abs(float(r.predicted_value_log10-r.true_value_log10))) for sid, r in frame.iterrows() if r.prediction_status == 'ok']
        elif row.selection_rule == 'largest_disagreement':
            a, b = row.selector.split(':')
            common = ids.loc[ids.method.eq(a) & ids.evaluation_subset.eq('common'), 'sample_id']
            candidates = [(sid, abs(float(predictions[a].loc[sid, 'predicted_value_log10']-predictions[b].loc[sid, 'predicted_value_log10']))) for sid in common]
        else:
            common = sorted(ids.loc[ids.method.eq('cnn') & ids.evaluation_subset.eq('common'), 'sample_id'])
            random_ids = np.random.default_rng(cfg['seed']).choice(common, size=min(5, len(common)), replace=False)
            require(row.sample_id == random_ids[row.rank-1], 'Independent random case differs')
            continue
        expected = sorted(candidates, key=lambda x: (-x[1], x[0]))[:5]
        require(row.sample_id == expected[row.rank-1][0], 'Independent extreme case differs')
    return {'metrics': len(stored), 'groups': len(groups), 'region_rows': len(regions), 'region_summaries': len(summary), 'case_selections': len(selected)}


def analyze(cfg, output=DEFAULT_OUTPUT):
    output = repo_path(output)
    require(output.is_relative_to(ROOT/'runs') and output != ROOT/'runs', 'Analysis output must be a named run under runs/')
    # Validate all inputs before allocating the final run directory.
    proof = validate(cfg)
    before = inputs(cfg)
    tables, parameters = numeric_tables(cfg)
    require(inputs(cfg) == before, 'Inputs changed during analysis')
    fresh(output); (output/'source_tables').mkdir()
    for name, frame in tables.items():
        frame.to_csv(output/f'source_tables/{name}.csv', index=False, na_rep='', lineterminator='\n')
    figures, font = make_figures(output, tables, parameters)
    for item in figures:
        item.update(run_id=output.name, dataset_id=cfg['dataset_id'], data_version=cfg['data_version'], split_id=cfg['split_id'])
    write_json(output/'figure_manifest.json', {'figures': figures})
    write_json(output/'config.json', cfg)
    write_json(output/'parameters.json', parameters)
    write_json(output/'source_validation.json', proof)
    (output/'analysis_report.md').write_text(report(tables, parameters, figures), encoding='utf-8')
    (output/'presentation_summary.md').write_text('# 汇报简稿\n\n'+report(tables, parameters, []).split('## 图与数值源表')[0].split('## 当前发现')[1], encoding='utf-8')
    meta = dict(schema_version='3.0.0', task_id='pr02-ecoli87-m2m3-analysis', run_id=output.name,
                dataset_id=cfg['dataset_id'], data_version=cfg['data_version'], split_id=cfg['split_id'],
                command=[sys.executable, *sys.argv], created_at=datetime.now(timezone.utc).isoformat(),
                code_commit=git_revision(), code_hashes=code_hashes(), input_hashes=before, parameters=parameters,
                config=cfg, fit_subsets=[], selection_subsets=[], evaluation_subsets=['val'], test_metrics_enabled=False,
                environment={'python': platform.python_version(), 'platform': platform.platform(), 'font': font,
                             'packages': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib', 'torch', 'openpyxl']}},
                execution_status='partial' if tables['coverage'].n_failed.sum() else 'success', evidence_level='preliminary')
    result = subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True)
    meta['dirty_worktree'] = bool(result.stdout)
    if meta['dirty_worktree']:
        patch = subprocess.run(['git', 'diff', 'HEAD', '--binary'], cwd=ROOT, capture_output=True, check=True).stdout
        (output/'source_patch.diff').write_bytes(patch)
    update_artifacts(output, meta)
    return output


def verify(cfg, output=DEFAULT_OUTPUT, visual=False):
    """Read-only: recompute all tables from frozen inputs and check saved evidence."""
    output = repo_path(output); meta = read_json(output/'run_manifest.json')
    validate(cfg)
    require(meta['schema_version'] == '3.0.0' and meta['config'] == cfg and meta['input_hashes'] == inputs(cfg), 'Analysis config/input binding changed')
    require(meta['fit_subsets'] == meta['selection_subsets'] == [] and meta['evaluation_subsets'] == ['val']
            and meta['test_metrics_enabled'] is False, 'Analysis changed fit/test policy')
    for name, expected in meta['code_hashes'].items():
        current = ROOT/name
        if sha(current) != expected:
            snapshot = ROOT/'reports/source_snapshots'/(expected+current.suffix)
            require(snapshot.is_file() and sha(snapshot) == expected, 'Missing executed source snapshot: '+name)
    for name, expected in meta['artifacts'].items():
        file = (output/name).resolve(); require(file.is_relative_to(output) and sha(file) == expected, 'Artifact changed: '+name)
    tables, parameters = numeric_tables(cfg)
    require(read_json(output/'parameters.json') == parameters, 'Analysis thresholds/IDs changed')
    for name, expected in tables.items():
        actual = pd.read_csv(output/f'source_tables/{name}.csv', keep_default_na=True)
        serialized = pd.read_csv(io.StringIO(expected.to_csv(index=False, na_rep='', lineterminator='\n')))
        pd.testing.assert_frame_equal(actual, serialized, check_exact=False, rtol=1e-10, atol=1e-12)
    figures = read_json(output/'figure_manifest.json')['figures']
    require(len(figures) == 10 and len({f['figure_id'] for f in figures}) == 10, 'Wrong figure coverage')
    for figure in figures:
        require(figure['data_version'] == cfg['data_version'] and figure['split_id'] == cfg['split_id'] and figure['run_id'] == output.name, 'Wrong figure provenance')
        for record in figure['source_tables']+figure['images']:
            file = (output/record['path']).resolve()
            require(file.is_relative_to(output) and sha(file) == record['sha256'], 'Figure source/image hash changed')
        if visual:
            require(figure['visual_review'] == 'reviewed', 'View and explicitly acknowledge every figure')
    independent = independent_numeric_check(cfg, output)
    return dict(status='passed',run_id=output.name,tables_recomputed=len(tables),figures_checked=len(figures),independent_checks=independent,
                n_all=parameters['n_all'],n_val=parameters['n_val'],n_common=len(parameters['common_val_ids']),
                cases=len(tables['case_review']),test_evaluated=False,visual_review_required=visual)


def acknowledge(cfg, output, reviewed):
    output = repo_path(output)
    verify(cfg, output)
    record = read_json(output/'figure_manifest.json')
    require(set(reviewed) == {f['figure_id'] for f in record['figures']} and len(reviewed) == 10, 'Name all and only actually viewed figures')
    for figure in record['figures']:
        figure['visual_review'] = 'reviewed'
        figure['visual_check'] = {'checked_at': datetime.now(timezone.utc).isoformat(),
                                  'method': 'Codex visual inspection of rendered PNG: Chinese glyphs, clipping, labels, scale, n and caption'}
    write_json(output/'figure_manifest.json', record)
    update_artifacts(output, read_json(output/'run_manifest.json'))


def publish(cfg, output):
    output = repo_path(output); proof = verify(cfg, output, visual=True)
    write_json(INDEX, dict(schema_version='3.0.0', dataset_id=cfg['dataset_id'],data_version=cfg['data_version'],split_id=cfg['split_id'],
                           run_id=output.name,run_path=output.relative_to(ROOT).as_posix(),
                           run_manifest_sha256=sha(output/'run_manifest.json'),verification=proof))
    return INDEX


def verify_published(cfg, visual=False):
    if not INDEX.exists():
        return None
    index = read_json(INDEX)
    # New experimental configs may use the same data, but different frozen models.
    directory = repo_path(index['run_path']); meta = read_json(directory/'run_manifest.json')
    require(index['run_id'] == directory.name and sha(directory/'run_manifest.json') == index['run_manifest_sha256'], 'Supplementary analysis index changed')
    if meta['config'] != cfg:
        return None
    require(all(index[k] == cfg[k] for k in ['dataset_id', 'data_version', 'split_id'])
            and index['run_id'] == directory.name and sha(directory/'run_manifest.json') == index['run_manifest_sha256'], 'Supplementary analysis index changed')
    return verify(cfg, directory, visual=visual)
