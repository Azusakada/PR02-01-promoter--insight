"""Close M3 analysis gaps using frozen predictions; no model training or test scoring."""
from __future__ import annotations

import importlib.metadata
import io
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import ROOT, digest_ids, read_json, require, sha, validate, write_json
from .experiment import METHODS, code_hashes, fresh, git_revision, path
from .m2m3 import (CONSENSUS, GROUPS, LABELS, check_predictions, chinese_font, inputs,
                   load_inputs, region_features, repo_path, safe_metric, strength_groups,
                   thresholds, update_artifacts)

DEFAULT_OUTPUT = Path('runs/ecoli87_m3_completion_20261013_v1')
INDEX = ROOT/'results/context87/m3_completion_current.json'
POLICIES = ['canonical_window_assumption', 'model_inferred']
REGIONS = ['UP', '-35', 'spacer', '-10']
POLICY_LABELS = {'canonical_window_assumption': '固定窗口假设', 'model_inferred': '热力学工具推定'}
GROUP_LABELS = {'low': '低强度', 'mid': '中强度', 'high': '高强度'}
FIGURE_IDS = ['01_spacing_strength', '02_strength_group_regions', '03_box_base_differences',
              '04_model_diagnostics', '05_cnn_learning_curve']


def completion_inputs(cfg):
    result = inputs(cfg)
    for method in METHODS:
        for name in ['predictions_train.csv', 'run_manifest.json']:
            p = path(cfg, method)/name
            result[p.relative_to(ROOT).as_posix()] = sha(p)
    p = path(cfg, 'cnn')/'training_history.csv'
    result[p.relative_to(ROOT).as_posix()] = sha(p)
    return result


def grouped_regions(regions, bounds):
    """Separate train/val, with group cutoffs fitted once on training labels."""
    part = regions[regions.subset.isin(['train', 'val']) & regions.region.isin(REGIONS)].copy()
    part['strength_group'] = strength_groups(part.target_log10, bounds)
    part['at_fraction'] = 1-part.gc_fraction
    summary, contrasts, bases = [], [], []
    for (subset, policy, region), frame in part.groupby(['subset', 'annotation_status', 'region'], sort=True):
        for group in GROUPS:
            g = frame[frame.strength_group.eq(group)]
            summary.append(dict(subset=subset, annotation_status=policy, region=region, strength_group=group,
                                n=len(g), mean_gc=g.gc_fraction.mean(), median_gc=g.gc_fraction.median(),
                                mean_at=g.at_fraction.mean(), mean_consensus=g.consensus_match_fraction.mean(),
                                median_length=g.length.median(), median_log10=g.target_log10.median()))
            if region in CONSENSUS:
                for position in range(6):
                    for base in 'ACGT':
                        bases.append(dict(subset=subset, annotation_status=policy, region=region, strength_group=group,
                                          position_1based=position+1, base=base, n=len(g),
                                          fraction=g.sequence.str[position].eq(base).mean() if len(g) else np.nan))
        low, high = frame[frame.strength_group.eq('low')], frame[frame.strength_group.eq('high')]
        features = ['gc_fraction', 'at_fraction']+(['consensus_match_fraction'] if region in CONSENSUS else [])
        if region == 'spacer':
            features.append('length')
        for feature in features:
            contrasts.append(dict(subset=subset, annotation_status=policy, region=region, feature=feature,
                                  n_low=len(low), n_high=len(high), low_mean=low[feature].mean(), high_mean=high[feature].mean(),
                                  high_minus_low=high[feature].mean()-low[feature].mean(), statistic='difference_of_means'))
    return pd.DataFrame(summary), pd.DataFrame(contrasts), pd.DataFrame(bases)


def spacing_tables(data, states, bounds):
    rows = []
    for row in data[data['split'].isin(['train', 'val'])].itertuples(index=False):
        state = states[row.sample_id]
        rows.append(dict(sample_id=row.sample_id, subset=row.split, spacer_length=len(state['spacer']),
                         discriminator_length=len(state['disc']), target_log10=row.target_log10,
                         annotation_status='model_inferred'))
    samples = pd.DataFrame(rows)
    samples['strength_group'] = strength_groups(samples.target_log10, bounds)
    summary, correlations = [], []
    for subset in ['train', 'val']:
        part = samples[samples.subset.eq(subset)]
        rho, status, reason = safe_metric('spearman', part.target_log10.to_numpy(), part.spacer_length.to_numpy())
        correlations.append(dict(subset=subset, n=len(part), annotation_status='model_inferred', spearman=rho,
                                 metric_status=status, reason=reason))
        for length in range(15, 21):
            g = part[part.spacer_length.eq(length)]
            summary.append(dict(subset=subset, spacer_length=length, n=len(g), annotation_status='model_inferred',
                                median_log10=g.target_log10.median(), q25_log10=g.target_log10.quantile(.25),
                                q75_log10=g.target_log10.quantile(.75), mean_log10=g.target_log10.mean(),
                                low_fraction=g.strength_group.eq('low').mean() if len(g) else np.nan,
                                high_fraction=g.strength_group.eq('high').mean() if len(g) else np.nan,
                                sample_status='empty' if len(g) == 0 else ('sparse_n_lt_30' if len(g) < 30 else 'observed')))
    return samples, pd.DataFrame(summary), pd.DataFrame(correlations)


def model_diagnostics(data, frames):
    """Train mean in log10 space is the squared-error constant baseline."""
    train_mean = float(data.loc[data['split'].eq('train'), 'target_log10'].mean())
    baseline, diagnostics, evaluation_ids = [], [], []
    for subset in ['train', 'val']:
        part = data[data['split'].eq(subset)]
        ids = part.sample_id.tolist()
        y = part.target_log10.to_numpy(float)
        constant = np.full(len(y), train_mean)
        baseline.extend(dict(sample_id=sid, subset=subset, true_value_log10=float(a), predicted_value_log10=train_mean,
                             method_name='train_mean', fit_subset='train', target_scale_model='log10') for sid, a in zip(ids, y))
        evaluation_ids.extend(dict(sample_id=sid, subset=subset) for sid in ids)
        constant_rmse = float(np.sqrt(np.mean((constant-y)**2)))
        predictions = {'train_mean': constant}
        for method in METHODS:
            frame = frames[(method, subset)].set_index('sample_id').loc[ids]
            require(frame.prediction_status.eq('ok').all(), 'M3 diagnostics require complete successful predictions; do not drop failures')
            predictions[method] = frame.predicted_value_log10.to_numpy(float)
        for method, p in predictions.items():
            record = dict(method=method, subset=subset, n=len(y), target_scale='log10',
                          evaluation_ids_sha256=digest_ids(ids), baseline_fit_subset='train', baseline_constant_log10=train_mean,
                          train_score_note='self_neighbor_interpolation' if method == 'knn' and subset == 'train' else 'in_sample' if subset == 'train' else 'heldout_validation')
            for metric in ['r2', 'mae', 'rmse', 'spearman']:
                value, status, reason = safe_metric(metric, y, p)
                record[metric], record[metric+'_status'], record[metric+'_reason'] = value, status, reason
            predicted_sd = float(np.std(p)) if np.ptp(p) > 0 else 0.
            record.update(true_sd_log10=float(np.std(y)), predicted_sd_log10=predicted_sd,
                          predicted_to_true_sd=predicted_sd/float(np.std(y)), bias_log10=float(np.mean(p-y)),
                          pearson=float(np.corrcoef(y, p)[0, 1]) if np.ptp(p) > 0 else np.nan,
                          rmse_improvement_vs_mean_pct=100*(1-record['rmse']/constant_rmse))
            diagnostics.append(record)
    diagnostics = pd.DataFrame(diagnostics)
    gaps = []
    for method in METHODS:
        train = diagnostics[(diagnostics.method == method) & (diagnostics.subset == 'train')].iloc[0]
        val = diagnostics[(diagnostics.method == method) & (diagnostics.subset == 'val')].iloc[0]
        gaps.append(dict(method=method, train_r2=train.r2, val_r2=val.r2, train_rmse=train.rmse, val_rmse=val.rmse,
                         r2_train_minus_val=train.r2-val.r2, rmse_val_minus_train=val.rmse-train.rmse,
                         interpretation_status='knn_train_not_generalization_evidence' if method == 'knn' else 'descriptive_gap_only'))
    return pd.DataFrame(baseline), diagnostics, pd.DataFrame(gaps), pd.DataFrame(evaluation_ids), train_mean


def numeric_tables(cfg):
    data, transform, val_frames, states = load_inputs(cfg)
    bounds = thresholds(data)
    frames = {(m, 'val'): f for m, f in val_frames.items()}
    for method in METHODS:
        frames[(method, 'train')] = check_predictions(pd.read_csv(path(cfg, method)/'predictions_train.csv'), data, cfg, method, 'train')
    regions = region_features(data, states)
    regional, contrast, base = grouped_regions(regions, bounds)
    spacing, spacing_summary, spacing_rho = spacing_tables(data, states, bounds)
    baseline, diagnostics, gaps, ids, mean = model_diagnostics(data, frames)
    history = pd.read_csv(path(cfg, 'cnn')/'training_history.csv')
    require(list(history.columns) == ['epoch', 'train_mse', 'val_mse'] and len(history) > 0
            and history.epoch.tolist() == list(range(1, len(history)+1)), 'Invalid CNN history')
    require(np.isfinite(history[['train_mse', 'val_mse']]).all().all()
            and history[['train_mse', 'val_mse']].ge(0).all().all(), 'Invalid CNN losses')
    settings = cfg['cnn']['training']; best_loss = np.inf; selected_epoch = None
    for row in history.itertuples():
        if row.val_mse < best_loss-settings['min_delta']:
            best_loss, selected_epoch = row.val_mse, int(row.epoch)
    metadata = read_json(path(cfg, 'cnn')/'run_manifest.json')
    require(selected_epoch == metadata['details']['best_epoch'], 'CNN selected epoch disagrees with saved run')
    span = transform['max_log10']-transform['min_log10']
    history['train_mse_log10'] = history.train_mse*span**2
    history['val_mse_log10'] = history.val_mse*span**2
    history['selected_checkpoint'] = history.epoch.eq(selected_epoch)
    membership = data[data['split'].isin(['train', 'val'])][['sample_id', 'split', 'target_log10']].copy()
    membership['strength_group'] = strength_groups(membership.target_log10, bounds)
    tables = dict(group_membership=membership, spacing_samples=spacing, spacing_summary=spacing_summary,
                  spacing_correlations=spacing_rho, region_group_summary=regional, region_group_contrasts=contrast,
                  region_group_base_frequency=base, mean_baseline_predictions=baseline, model_diagnostics=diagnostics,
                  train_validation_gaps=gaps, evaluation_ids=ids, cnn_learning_curve=history)
    parameters = dict(thresholds_log10=bounds.tolist(), threshold_fit_subset='train', baseline_fit_subset='train',
                      baseline_constant_log10=mean, subsets=['train', 'val'], n_train=int(data['split'].eq('train').sum()),
                      n_val=int(data['split'].eq('val').sum()), best_cnn_epoch=selected_epoch, cnn_epochs=len(history),
                      sparse_group_flag='n < 30 (descriptive warning only)', test_evaluated=False)
    return tables, parameters


def make_figures(output, tables, params):
    font = chinese_font(); folder = output/'figures'; folder.mkdir(); figures = []

    def save(name, fig, title, sources, subset, caption, annotation='not_applicable'):
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, .10, 1, .94))
        fig.text(.025, .025, caption, fontsize=9)
        images = []
        for suffix in ['png', 'svg']:
            file = folder/f'{name}.{suffix}'; fig.savefig(file, dpi=180, bbox_inches='tight')
            images.append(dict(path=file.relative_to(output).as_posix(), sha256=sha(file)))
        plt.close(fig)
        figures.append(dict(figure_id=name, title=title, subset=subset, annotation_status=annotation,
                            visual_review='pending', caption=caption, images=images,
                            source_tables=[dict(path=f'source_tables/{s}.csv', sha256=sha(output/f'source_tables/{s}.csv')) for s in sources]))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    spacing = tables['spacing_samples']
    for ax, subset in zip(axes, ['train', 'val']):
        part = spacing[spacing.subset.eq(subset)]
        boxes, positions = [], []
        for length in range(15, 21):
            values = part.loc[part.spacer_length.eq(length), 'target_log10'].to_numpy()
            if len(values): boxes.append(values); positions.append(length)
        ax.boxplot(boxes, positions=positions, widths=.55, showfliers=False)
        ax.set_xticks(range(15, 21), [f'{k} bp\nn={int(part.spacer_length.eq(k).sum())}' for k in range(15, 21)])
        ax.set(xlim=(14.4, 20.6), xlabel='模型推定 spacer 长度', ylabel='观测 log10 强度',
               title=f'{"训练集" if subset == "train" else "验证集"} n={len(part):,}')
    save(FIGURE_IDS[0], fig, 'spacer 长度与强度：分别检查训练集和验证集', ['spacing_samples', 'spacing_summary', 'spacing_correlations'],
         'train/val', '箱线为中位数、四分位数及 1.5 IQR 须；隐藏离群点但统计保留全部样本。\n长度由热力学择优推定；n<30 的组仅作描述；不能解释为独立 spacer 效应。', 'model_inferred')

    summary = tables['region_group_summary']; contrasts = tables['region_group_contrasts']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, policy in zip(axes, POLICIES):
        part = summary[summary.subset.eq('val') & summary.annotation_status.eq(policy)]
        x = np.arange(len(REGIONS))
        for offset, group, color in [(-.18, 'low', '#386cb0'), (.18, 'high', '#c95959')]:
            values = part[part.strength_group.eq(group)].set_index('region').reindex(REGIONS)
            ax.bar(x+offset, values.mean_at, width=.34, color=color, label=f'{GROUP_LABELS[group]} n={int(values.n.iloc[0])}')
        ax.set_xticks(x, REGIONS); ax.set(ylim=(0, 1), ylabel='区域平均 A/T 比例', title=POLICY_LABELS[policy]); ax.legend()
    save(FIGURE_IDS[1], fig, '强弱组的区域对照：UP、−35、spacer、−10', ['region_group_summary', 'region_group_contrasts', 'group_membership'],
         'val', f'验证集 n={params["n_val"]:,}；只显示低/高组；阈值取训练集三分位点，中组也保存在源表。\n同时保留固定与推定窗口，差异仅为描述性关系。', 'canonical_window_assumption/model_inferred')

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    frequencies = tables['region_group_base_frequency']
    for row, policy in enumerate(POLICIES):
        for col, region in enumerate(['-35', '-10']):
            part = frequencies[frequencies.subset.eq('val') & frequencies.annotation_status.eq(policy) & frequencies.region.eq(region)]
            matrices = [part[part.strength_group.eq(group)].pivot(index='base', columns='position_1based', values='fraction').reindex(list('ACGT')).to_numpy() for group in ['low', 'high']]
            ax = axes[row, col]; im = ax.imshow(matrices[1]-matrices[0], cmap='RdBu_r', vmin=-.15, vmax=.15, aspect='auto')
            ax.set_yticks(range(4), list('ACGT')); ax.set_xticks(range(6), [f'{i+1}\n{b}' for i, b in enumerate(CONSENSUS[region])])
            c = contrasts[(contrasts.subset == 'val') & (contrasts.annotation_status == policy) & (contrasts.region == region) & (contrasts.feature == 'consensus_match_fraction')].iloc[0]
            ax.set(title=f'{POLICY_LABELS[policy]} {region}；共识均分差={c.high_minus_low:+.3f}', xlabel='盒内位置 / 共识碱基')
            fig.colorbar(im, ax=ax, label='高组减低组的碱基频率')
    counts = tables['group_membership'].query('split == "val"').strength_group.value_counts()
    save(FIGURE_IDS[2], fig, '−35 / −10 强弱组碱基频率差异', ['region_group_base_frequency', 'region_group_contrasts', 'group_membership'],
         'val', f'低组 n={int(counts.get("low", 0))}，高组 n={int(counts.get("high", 0))}；所有图统一色阶。\n共识均分为相同碱基数/6，不是结合能；盒区不是逐条实验注释。', 'canonical_window_assumption/model_inferred')

    diag = tables['model_diagnostics'].query('subset == "val"').set_index('method')
    order = ['train_mean', *METHODS]; labels = ['训练均值', *[LABELS[m] for m in METHODS]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bars = axes[0].bar(labels, diag.loc[order, 'r2'], color=['#999999']+['#386cb0']*5)
    for bar, value in zip(bars, diag.loc[order, 'r2']): axes[0].text(bar.get_x()+bar.get_width()/2, value+.004, f'{value:.3f}', ha='center', fontsize=9)
    axes[0].axhline(0, color='gray', lw=.7); axes[0].set(ylabel='验证 R²（log10）', ylim=(-.015, .145))
    axes[1].bar([LABELS[m] for m in METHODS], diag.loc[list(METHODS), 'predicted_to_true_sd'], color='#45a37a')
    axes[1].axhline(1, color='gray', ls='--'); axes[1].set(ylabel='预测标准差 / 真实标准差', ylim=(0, 1.08))
    save(FIGURE_IDS[3], fig, '五模型与训练均值基线：验证成绩和预测范围', ['model_diagnostics', 'mean_baseline_predictions', 'evaluation_ids'],
         'val', f'共同验证集 n={params["n_val"]:,}；常数仅由训练标签的 log10 均值确定。\n常数预测的 Spearman 未定义；范围收缩是观察结果，不等于已经证明数据噪声上限。')

    curve = tables['cnn_learning_curve']; fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(curve.epoch, curve.train_mse_log10, label='训练批次损失（含 dropout）', color='#386cb0')
    ax.plot(curve.epoch, curve.val_mse_log10, label='验证损失（eval）', color='#c95959')
    ax.axvline(params['best_cnn_epoch'], color='gray', ls='--', label=f'选定轮次 {params["best_cnn_epoch"]}')
    ax.set(xlabel='训练轮次', ylabel='MSE（log10 强度）'); ax.legend()
    save(FIGURE_IDS[4], fig, 'CNN 学习曲线：验证最优之后的训练变化', ['cnn_learning_curve'], 'train/val',
         f'训练 n={params["n_train"]:,}，验证 n={params["n_val"]:,}；恢复已有日志尺度，没有新增训练。\n训练曲线含 dropout，不能直接当作选定 checkpoint 在 eval 模式下的训练集成绩。')
    return figures, font


def markdown_table(frame, columns, labels=None):
    labels = labels or columns
    lines = ['| '+' | '.join(labels)+' |', '| '+' | '.join(['---']*len(columns))+' |']
    for row in frame[columns].itertuples(index=False, name=None):
        values = ['未定义' if pd.isna(v) else f'{v:.4f}' if isinstance(v, (float, np.floating)) else str(v) for v in row]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def report(tables, params, figures):
    diag = tables['model_diagnostics']; val = diag.query('subset == "val"'); cnn = val[val.method.eq('cnn')].iloc[0]
    spacing = tables['spacing_summary'].query('subset == "val"')
    rho = tables['spacing_correlations'].set_index('subset').spearman
    contrasts = tables['region_group_contrasts'].query('subset == "val"').copy()
    contrasts['窗口'] = contrasts.annotation_status.map(POLICY_LABELS)
    at = contrasts.query('feature == "at_fraction"')
    consensus = contrasts.query('feature == "consensus_match_fraction"')
    inferred_at = at[at.annotation_status.eq('model_inferred')].set_index('region').high_minus_low
    figure_text = '\n\n'.join(f'### {f["title"]}\n\n![{f["title"]}](figures/{f["figure_id"]}.png)\n\n{f["caption"].replace(chr(10), " ")}' for f in figures)
    return f'''# E. coli 87 bp：M3 补齐与冻结基线

本次补齐 spacer 长度与强度关系、强弱组区域对照、五模型低分诊断。它与 [已有 M2/M3 分析](../ecoli87_m2m3_20261013_v1/analysis_report.md) 合起来构成当前 M3 交付。没有重训、改标签、改划分或新增 M4 实验；汇报材料不在本次范围。

## 1. spacer 长度与强度

验证集按工具推定长度的统计如下；训练集完整统计在 spacing_summary.csv。spacer 长度与 log10 强度的 Spearman：训练 {rho['train']:.4f}，验证 {rho['val']:.4f}，目前没有观察到强的单调关系。长度20没有被选中的样本仍记录 n=0、统计为空，不能填成0强度。

{markdown_table(spacing, ['spacer_length', 'n', 'median_log10', 'q25_log10', 'q75_log10', 'sample_status'], ['spacer(bp)', 'n', '强度中位数(log10)', 'Q25', 'Q75', '样本状态'])}

这些长度由固定 TSS 下最小自由能状态选择而来，同时改变盒区位置和 discriminator 长度，不能解释成实验 spacer 注释或独立的间距因果效应。n<30 仅用于提示小样本组；四分位数是样本分布，不是置信区间。固定窗口的 spacer 恒为17bp，不能用它估计长度相关性。

## 2. 强弱组的区域比较

低/中/高阈值仅由训练标签三分位点确定：{params['thresholds_log10'][0]:.10f}、{params['thresholds_log10'][1]:.10f}。按观测标签分组；训练与验证分开，未使用预测值选组。区域为 UP、−35、spacer、−10，两种窗口分别报告。下表是验证集高组减低组的平均 A/T 差异（比例单位）：

{markdown_table(at, ['窗口', 'region', 'n_low', 'n_high', 'low_mean', 'high_mean', 'high_minus_low'], ['窗口', '区域', '低组n', '高组n', '低组A/T', '高组A/T', '高减低'])}

在验证集的推定窗口中，高组相对低组的平均 A/T 差为：UP {inferred_at['UP']:+.4f}、−35 {inferred_at['-35']:+.4f}、spacer {inferred_at['spacer']:+.4f}、−10 {inferred_at['-10']:+.4f}。前三区域有 A/T 组成差异，−10 的组间差很小；固定窗口也分别报告以检查坐标假设影响。这里没有把均值差当作显著性检验。

−35/−10 共识匹配均分的对照如下；评分为相同字母数/6，不把单一共识得分等同于结合能或启动子活性：

{markdown_table(consensus, ['窗口', 'region', 'low_mean', 'high_mean', 'high_minus_low'], ['窗口', '区域', '低组共识均分', '高组共识均分', '高减低'])}

region_group_summary.csv 保存低、中、高各组的 n、GC/AT、共识和长度统计；region_group_contrasts.csv 保存差异；region_group_base_frequency.csv 保存盒内逐位 A/C/G/T 频率。group_membership.csv 保留每条样本的分组。TSS 有来源证据，盒区仍是固定窗口假设或模型推定；仅作描述性比较，没有以这些统计证明生物学机制。

## 3. 五模型预测效果诊断

常数基线使用训练集 log10 标签均值 {params['baseline_constant_log10']:.10f}，训练/验证分别导出预测。五模型与基线在同一 {params['n_val']:,} 条验证 ID 上比较，指标均为 log10 尺度。

{markdown_table(val, ['method', 'r2', 'spearman', 'mae', 'rmse', 'predicted_to_true_sd', 'rmse_improvement_vs_mean_pct'], ['方法', 'R²', 'Spearman', 'MAE', 'RMSE', '预测/真实标准差', 'RMSE较均值改善%'])}

CNN 的验证 Pearson={cnn.pearson:.4f}，RMSE 相对训练均值改善 {cnn.rmse_improvement_vs_mean_pct:.2f}%，预测标准差为真实值的 {100*cnn.predicted_to_true_sd:.2f}%。结合已有分组残差，高估低组、低估高组和范围收缩说明当前预测信号较弱，不证明数据无法预测或已经达到性能上限。

{markdown_table(tables['train_validation_gaps'], ['method', 'train_r2', 'val_r2', 'train_rmse', 'val_rmse', 'interpretation_status'], ['方法', '训练R²', '验证R²', '训练RMSE', '验证RMSE', '解释标记'])}

KNN 使用距离权重，训练样本以自身为零距离邻居，训练 R²=1 是插值性质，不能拿来判定模型很好或证明严重过拟合。其训练值从一般学习能力判断中排除。CNN 在第 {params['best_cnn_epoch']} 轮选定 checkpoint；日志共 {params['cnn_epochs']} 轮，后续训练损失下降而验证损失未改善。日志训练损失含 dropout；表中训练指标来自选定 checkpoint 的 eval 预测，两者不混用。

## 4. 数据与模型解释边界

[Deep_promoter 原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC7337522/) 在相同来源的天然 dRNA-seq 表达数据上报告 CNN 测试 Pearson=0.25，并讨论表达还受到其他调控因素影响。本次 CNN 验证 Pearson/Spearman、数据去重、长度、架构和划分与其不同，不能当作严格复现或直接宣称优于原论文。原论文第二轮 SVR 使用另一个合成启动子报告实验数据集，其 Pearson=0.57 不能用作这批天然表达标签的基准。

因此标签与短序列任务的匹配程度是需要继续验证的解释；具体标签单位、条件归属和不可约噪声仍未确定。[Promoter Calculator](https://www.nature.com/articles/s41467-022-32829-5) 的转录起始率与体内表达测量之间也可能受其他过程影响。KNN 的整体特征、无位置 k-mer，以及 CNN 的位置池化都有表达限制，但本次没有用新的受控模型对照量化各因素贡献，不能据此断言谁是主因。

## 5. M3 验收与后续范围

已完成：经核验数据与来源、五方法公共验证比较、−10/−35/UP/spacer 探索、spacer 与强度关系、强弱组区域对照、分组误差、极端与随机案例、训练均值对照和学习曲线诊断。原有图表、80条案例选择及其独立核验保留在已有 M2/M3 运行，本运行补齐上述三项。所有结论为当前继承随机划分上的初步分析。

本次只分析训练/验证样本，未生成 E. coli test 预测或成绩。阈值和常数仅取 train；标签、五模型 checkpoint/参数、val 预测、热力学状态和 M4/跨物种产物保持其冻结字节。邻近上下文可能重叠，历史 test 曾使用，不能称为全新盲测。M4 模型优化、归因/突变后续解释、消融与选做实验由后续安排；M3 完成不等同于 M4 完成或模型达到高准确率。

## 图与复算源表

{figure_text}
'''


def analyze(cfg, output=DEFAULT_OUTPUT):
    output = repo_path(output)
    require(output.is_relative_to(ROOT/'runs') and output != ROOT/'runs', 'Use a named run under runs/')
    require(not output.exists() or not any(output.iterdir()), 'Use a new run ID; existing output: '+str(output))
    validate(cfg); before = completion_inputs(cfg); tables, params = numeric_tables(cfg)
    require(completion_inputs(cfg) == before, 'Frozen inputs changed during analysis')
    fresh(output); (output/'source_tables').mkdir()
    for name, frame in tables.items(): frame.to_csv(output/f'source_tables/{name}.csv', index=False, na_rep='', lineterminator='\n')
    figures, font = make_figures(output, tables, params)
    for figure in figures:
        figure.update(run_id=output.name, data_version=cfg['data_version'], split_id=cfg['split_id'])
    write_json(output/'figure_manifest.json', dict(figures=figures, font=font))
    write_json(output/'parameters.json', params)
    (output/'analysis_report.md').write_text(report(tables, params, figures), encoding='utf-8')
    metadata = dict(schema_version='3.0.0', task_id='pr02-ecoli87-m3-completion', run_id=output.name,
                    dataset_id=cfg['dataset_id'], data_version=cfg['data_version'], split_id=cfg['split_id'],
                    created_at=datetime.now(timezone.utc).isoformat(), command=[sys.executable, *sys.argv],
                    code_commit=git_revision(), code_hashes=code_hashes(), input_hashes=before, config=cfg,
                    fit_subsets=[], selection_subsets=[], evaluation_subsets=['train', 'val'], test_metrics_enabled=False,
                    parameters=params, evidence_level='preliminary', font=font,
                    environment=dict(python=platform.python_version(), platform=platform.platform(),
                                     packages={p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scipy', 'scikit-learn', 'matplotlib']}),
                    dirty_worktree=bool(subprocess.run(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT, capture_output=True, text=True).stdout))
    update_artifacts(output, metadata)
    return output


def independent_check(output):
    """Recheck scalar errors, membership and base-frequency denominators without sklearn metrics."""
    tables = {p.stem: pd.read_csv(p) for p in (output/'source_tables').glob('*.csv')}
    params = read_json(output/'parameters.json'); members = tables['group_membership']
    train = members[members['split'].eq('train')].target_log10.to_numpy()
    require(np.allclose(np.quantile(train, [1/3, 2/3]), params['thresholds_log10'], atol=1e-12), 'Independent thresholds differ')
    require(np.isclose(sum(float(y) for y in train)/len(train), params['baseline_constant_log10'], atol=1e-12), 'Independent baseline differs')
    for row in members.itertuples():
        a, b = params['thresholds_log10']; expected = 'low' if row.target_log10 <= a else 'mid' if row.target_log10 <= b else 'high'
        require(row.strength_group == expected and row.split in ['train', 'val'], 'Independent membership differs')
    spacing = tables['spacing_samples']; summary = tables['spacing_summary']
    for row in summary.itertuples():
        values = sorted(spacing.loc[spacing.subset.eq(row.subset) & spacing.spacer_length.eq(row.spacer_length), 'target_log10'])
        require(len(values) == row.n, 'Independent spacing n differs')
        for column, q in [('median_log10', .5), ('q25_log10', .25), ('q75_log10', .75)]:
            actual = getattr(row, column)
            require(np.isclose(actual, np.quantile(values, q), atol=1e-12) if values else np.isnan(actual), 'Independent spacing quantile differs')
    frequencies = tables['region_group_base_frequency']
    totals = frequencies.groupby(['subset', 'annotation_status', 'region', 'strength_group', 'position_1based']).fraction.sum()
    require(np.allclose(totals, 1, atol=1e-12), 'Base frequency denominator differs')
    for row in tables['region_group_contrasts'].itertuples():
        require(np.isclose(row.high_mean-row.low_mean, row.high_minus_low, atol=1e-12), 'Independent contrast direction differs')
    cfg = read_json(output/'run_manifest.json')['config']
    for row in tables['model_diagnostics'].itertuples():
        truth = members[members['split'].eq(row.subset)].set_index('sample_id').target_log10.sort_index()
        if row.method == 'train_mean':
            frame = tables['mean_baseline_predictions'].query('subset == @row.subset').set_index('sample_id').sort_index()
        else:
            frame = pd.read_csv(path(cfg, row.method)/f'predictions_{row.subset}.csv').set_index('sample_id').sort_index()
        require(list(frame.index) == list(truth.index) and len(frame) == row.n, 'Independent diagnostic IDs differ')
        y = [float(v) for v in truth]; p = [float(v) for v in frame.predicted_value_log10]
        residuals = [b-a for a, b in zip(y, p)]; mean = sum(y)/len(y)
        mse = sum(r*r for r in residuals)/len(y); variance = sum((v-mean)**2 for v in y)/len(y)
        pmean = sum(p)/len(p); pvariance = sum((v-pmean)**2 for v in p)/len(p)
        base_mse = sum((v-params['baseline_constant_log10'])**2 for v in y)/len(y)
        expected = dict(r2=1-mse/variance, mae=sum(abs(r) for r in residuals)/len(y), rmse=mse**.5,
                        true_sd_log10=variance**.5, predicted_sd_log10=pvariance**.5,
                        predicted_to_true_sd=(pvariance/variance)**.5, bias_log10=sum(residuals)/len(y),
                        rmse_improvement_vs_mean_pct=100*(1-(mse/base_mse)**.5))
        for column, value in expected.items():
            require(np.isclose(getattr(row, column), value, rtol=1e-10, atol=1e-12), 'Independent diagnostic differs: '+column)
        if row.method == 'train_mean':
            require(np.isnan(row.spearman) and row.spearman_status == 'undefined' and np.isnan(row.pearson), 'Constant correlation must be undefined')
        else:
            rho = pd.Series(y).rank().corr(pd.Series(p).rank()); pearson = pd.Series(y).corr(pd.Series(p))
            require(np.isclose(row.spearman, rho, atol=1e-12) and np.isclose(row.pearson, pearson, atol=1e-12), 'Independent diagnostic correlation differs')
    return dict(model_diagnostics=len(tables['model_diagnostics']), group_memberships=len(members), spacing_summaries=len(summary), region_contrasts=len(tables['region_group_contrasts']),
                base_frequency_distributions=len(totals))


def verify(cfg, output=DEFAULT_OUTPUT, visual=False):
    output = repo_path(output); meta = read_json(output/'run_manifest.json')
    require(meta['task_id'] == 'pr02-ecoli87-m3-completion' and meta['config'] == cfg and meta['input_hashes'] == completion_inputs(cfg), 'M3 config/input binding changed')
    require(meta['fit_subsets'] == meta['selection_subsets'] == [] and meta['evaluation_subsets'] == ['train', 'val']
            and meta['test_metrics_enabled'] is False, 'M3 fit/test policy changed')
    for name, expected in meta['code_hashes'].items():
        if sha(ROOT/name) != expected:
            snapshot = ROOT/'reports/source_snapshots'/(expected+Path(name).suffix)
            require(snapshot.is_file() and sha(snapshot) == expected, 'Missing executed source snapshot: '+name)
    for name, expected in meta['artifacts'].items():
        p = (output/name).resolve(); require(p.is_relative_to(output) and sha(p) == expected, 'M3 artifact changed: '+name)
    tables, params = numeric_tables(cfg)
    require(params == read_json(output/'parameters.json') == meta['parameters'], 'M3 parameters changed')
    require({p.stem for p in (output/'source_tables').glob('*.csv')} == set(tables), 'M3 source table coverage differs')
    for name, expected in tables.items():
        serialized = pd.read_csv(io.StringIO(expected.to_csv(index=False, na_rep='', lineterminator='\n')))
        actual = pd.read_csv(output/f'source_tables/{name}.csv')
        pd.testing.assert_frame_equal(actual, serialized, check_exact=False, rtol=1e-10, atol=1e-12)
    figures = read_json(output/'figure_manifest.json')['figures']
    require([f['figure_id'] for f in figures] == FIGURE_IDS, 'M3 figure coverage differs')
    for figure in figures:
        require(figure['run_id'] == output.name and figure['data_version'] == cfg['data_version'] and figure['split_id'] == cfg['split_id'], 'M3 figure identity differs')
        for entry in figure['source_tables']+figure['images']:
            p = (output/entry['path']).resolve(); require(p.is_relative_to(output) and sha(p) == entry['sha256'], 'M3 figure/source changed')
        if visual: require(figure['visual_review'] == 'reviewed', 'Inspect and acknowledge M3 figures first')
    independent = independent_check(output)
    return dict(status='passed', run_id=output.name, tables_recomputed=len(tables), figures_checked=len(figures),
                independent_checks=independent, n_train=params['n_train'], n_val=params['n_val'],
                test_evaluated=False, visual_review_required=visual)


def acknowledge(cfg, output, reviewed):
    output = repo_path(output); verify(cfg, output)
    require(len(reviewed) == len(FIGURE_IDS) and set(reviewed) == set(FIGURE_IDS), 'Name all and only actually viewed M3 figures')
    record = read_json(output/'figure_manifest.json')
    for figure in record['figures']:
        figure.update(visual_review='reviewed', visual_check=dict(checked_at=datetime.now(timezone.utc).isoformat(),
                       method='Codex visual inspection of PNG: glyphs, clipping, labels, units, denominators and captions'))
    write_json(output/'figure_manifest.json', record); update_artifacts(output, read_json(output/'run_manifest.json'))


def publish(cfg, output):
    output = repo_path(output); proof = verify(cfg, output, visual=True)
    write_json(INDEX, dict(schema_version='3.0.0', dataset_id=cfg['dataset_id'], data_version=cfg['data_version'],
                           split_id=cfg['split_id'], run_id=output.name, run_path=output.relative_to(ROOT).as_posix(),
                           run_manifest_sha256=sha(output/'run_manifest.json'), verification=proof))
    return INDEX


def verify_published(cfg, visual=False):
    if not INDEX.exists(): return None
    index = read_json(INDEX); output = repo_path(index['run_path']); meta = read_json(output/'run_manifest.json')
    require(index['run_id'] == output.name and sha(output/'run_manifest.json') == index['run_manifest_sha256'], 'M3 published index changed')
    if meta['config'] != cfg: return None
    require(all(index[k] == cfg[k] for k in ['dataset_id', 'data_version', 'split_id']), 'M3 index dataset differs')
    return verify(cfg, output, visual=visual)
