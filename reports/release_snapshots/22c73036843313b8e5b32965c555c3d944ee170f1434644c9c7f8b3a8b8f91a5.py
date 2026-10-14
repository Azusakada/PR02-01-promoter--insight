"""Figures and source-linked narrative generated from completed M4 tables."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .data import ROOT, read_json, read_table, sha, write_json, write_table
from .m4_data_eval import SEEDS, completed, finish, run_root

COLORS = ['#386A91', '#E39B3D', '#679C87', '#A6687B']


def figures(cfg, protocol):
    root = run_root(protocol); directory = root/'figures'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True); records = []
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False,
                         'figure.facecolor': 'white', 'savefig.facecolor': 'white', 'svg.fonttype': 'none'})
    def save(name, fig, sources, caption):
        fig.tight_layout(pad=1.5)
        for suffix in ['png', 'svg']: fig.savefig(directory/(name+'.'+suffix), dpi=180)
        plt.close(fig)
        records.append(dict(figure_id=name, png=name+'.png', svg=name+'.svg', caption=caption,
            source_tables={str((root/s).relative_to(ROOT)): sha(root/s) for s in sources},
            sha256=sha(directory/(name+'.png')), visual_review='pending'))
    rows = read_table(root/'evaluation/model_summary.csv')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6)); labels = [r['method'].replace('dnabert2_ridge', 'DNA+Ridge').replace('linear_svr', 'LinSVR') for r in rows]
    for ax, metric, title in zip(axes, ['r2_log10', 'spearman'], ['R² on log10 strength', 'Spearman correlation']):
        a = [float(r[metric+'_mean']) if r[metric+'_mean'] else np.nan for r in rows]
        err = [float(r[metric+'_sd']) if r[metric+'_sd'] else 0 for r in rows]
        ax.bar(np.arange(len(rows)), a, color=COLORS[0], yerr=err, capsize=4)
        ax.set_xticks(np.arange(len(rows)), labels, rotation=35, ha='right'); ax.axhline(0, color='#888888', lw=.7)
        ax.set_title(title); ax.set_ylabel('Validation score')
    fig.suptitle('Source validation n=1,768; CNN mean ± sample SD across 3 seeds')
    save('01_model_comparison', fig, ['evaluation/model_summary.csv'], '各模型使用同一1768条验证序列；CNN显示三个种子的均值及样本标准差。')
    search = read_table(root/'cnn/search.csv'); fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for i, condition in enumerate(['C0', 'C1', 'C2', 'C3']):
        vals = [np.sqrt(float(r['val_mse_log10'])) for r in search if r['condition'] == condition]
        axes[0].scatter(np.full(3, i), vals, color=COLORS[i], s=40)
        axes[0].plot([i-.18, i+.18], [np.mean(vals)]*2, color='black', lw=2)
    axes[0].set_xticks(range(4), ['C0', 'C1', 'C2', 'C3']); axes[0].set(ylabel='Validation RMSE (log10)', title='All predeclared CNN configurations')
    selected = read_json(root/'cnn/selected_model.json')
    for i, seed in enumerate(SEEDS):
        history = read_table(ROOT/selected['runs'][str(seed)]/'training_history.csv')
        axes[1].plot([int(r['epoch']) for r in history], [float(r['val_mse']) for r in history], color=COLORS[i], label=str(seed))
    axes[1].set(xlabel='Epoch', ylabel='Validation MSE (train min-max scale)', title='Selected configuration: '+selected['condition']); axes[1].legend(fontsize=9)
    save('02_cnn_stability', fig, ['cnn/search.csv']+[str((ROOT/selected['runs'][str(seed)]/'training_history.csv').relative_to(root)) for seed in SEEDS], '四配置均按预定三种子比较；代表模型固定采用20260928，未按最佳单种子挑选。')
    overlap = read_table(root/'explanation_analysis/overlap_summary.csv'); fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, (method, status) in zip(axes.flat, [(m, s) for m in ['integrated_gradients', 'ism_mean_absolute'] for s in ['canonical_window_assumption', 'model_inferred']]):
        rr = [r for r in overlap if r['method'] == method and r['annotation_status'] == status and r['top_k'] == '5']
        a = np.array([float(r['mean_excess_fraction']) for r in rr]); lows = a-[float(r['excess_ci_low']) for r in rr]; highs = np.array([float(r['excess_ci_high']) for r in rr])-a
        ax.bar([r['region'] for r in rr], a, color=COLORS[0], yerr=np.vstack([lows, highs]), capsize=4); ax.axhline(0, color='gray', lw=.8)
        ax.set(title=('IG' if method == 'integrated_gradients' else 'ISM')+' / '+('fixed windows' if status.startswith('canonical') else 'PC-inferred windows'), ylabel='Top-5 fraction minus length/87')
    fig.suptitle('64 parent sequences; 95% parent bootstrap intervals, descriptive')
    save('03_topk_overlap', fig, ['explanation_analysis/overlap_summary.csv'], 'Top-5与区域的重合比例扣除区域长度/87。固定窗口与热力学推定窗口分别展示；区间未做多重比较校正。')
    consistency = read_table(root/'explanation_analysis/consistency.csv'); fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    rr = [r for r in consistency if r['top_k'] == '5']
    axes[0].hist([float(r['spearman']) for r in rr], bins=15, color=COLORS[0]); axes[0].set(xlabel='Within-sequence IG/ISM rank correlation', ylabel='Parent sequences', title='Absolute importance ranking')
    for i, k in enumerate(['5', '10']):
        a = [float(r['top_k_jaccard']) for r in consistency if r['top_k'] == k]
        axes[1].scatter(np.full(len(a), i)+np.linspace(-.15,.15,len(a)), a, s=12, alpha=.5, color=COLORS[i])
        axes[1].plot([i-.2, i+.2], [np.mean(a)]*2, color='black', lw=2)
    axes[1].set_xticks([0,1], ['Top-5', 'Top-10']); axes[1].set(ylabel='IG/ISM Jaccard', title='Agreement varies by sequence', ylim=(-.03,1.05))
    save('04_attribution_consistency', fig, ['explanation_analysis/consistency.csv'], 'IG与ISM解释同一个CNN；一致性衡量模型解释方法的相似程度，不作为独立生物验证。')
    amplitude = read_table(root/'mutation/mutation_summary.csv'); fig, ax = plt.subplots(figsize=(7.5, 4.6))
    a = np.array([float(r['mean_absolute_delta_log10']) for r in amplitude]); low = a-[float(r['ci_low']) for r in amplitude]; high = np.array([float(r['ci_high']) for r in amplitude])-a
    ax.bar([r['group'] for r in amplitude], a, color=COLORS[:3], yerr=np.vstack([low,high]), capsize=5)
    ax.set(ylabel='Mean |mutant − wild-type| (log10)', title='Equal-count mutations: 5 positions, 10 repeats per parent')
    save('05_mutation_amplitude', fig, ['mutation/mutation_summary.csv'], 'High/Low/Random各640条变体，来自同一64条母序列。先按母序列汇总，区间按母序列重采样。')
    directions = read_table(root/'mutation/direction_summary.csv'); all_row = next(r for r in directions if r['expected_direction'] == 'all')
    fig, ax = plt.subplots(figsize=(8, 4.5)); keys = ['cnn_agreement','thermo_agreement','cross_model_agreement']; a = np.array([float(all_row[k]) for k in keys])
    low = a-[float(all_row[k+'_ci_low']) for k in keys]; high = np.array([float(all_row[k+'_ci_high']) for k in keys])-a
    ax.bar(['Gradient vs CNN','Gradient vs PC','CNN vs PC'], a, color=COLORS[:3], yerr=np.vstack([low,high]), capsize=5)
    ax.set(ylabel='Direction agreement fraction', ylim=(0,1.03), title='960 single substitutions; 64-parent clustered bootstrap')
    save('06_mutation_direction', fig, ['mutation/direction_summary.csv'], '预期方向来自野生型梯度一阶替换分数，随后与实际CNN变化及热力学变化比较；三者均为计算证据。')
    stages = read_table(root/'ablation/stage_summary.csv'); fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, key, title in zip(axes, ['r2_log10','rmse_log10'], ['Validation R² (log10)', 'Validation RMSE (log10)']):
        a = [float(r[key+'_mean']) for r in stages]; sd = [float(r[key+'_sd']) if r[key+'_sd'] else 0 for r in stages]
        ax.errorbar(range(4), a, yerr=sd, marker='o', capsize=5, color=COLORS[0]); ax.set_xticks(range(4), ['A0\nkmer','A1\n+CNN','A2\n+DNA','A3\n+guided regions']); ax.set_title(title)
    save('07_staged_ablation', fig, ['ablation/stage_summary.csv'], '各阶段统一使用训练集StandardScaler和Ridge头，alpha在验证集选择；A1至A3报告配对CNN种子均值及标准差。')
    transfer = read_table(root/'transfer/transfer_metrics.csv'); fig, ax = plt.subplots(figsize=(11, 5))
    labels = [r['condition'].replace('zero_shot_', 'zero\n').replace('few_shot_', '').replace('_groups','').replace('frozen_cnn','CNN').replace('_','\n') for r in transfer]
    a = np.array([float(r['spearman']) for r in transfer]); low = a-[float(r['spearman_ci_low']) for r in transfer]; high = np.array([float(r['spearman_ci_high']) for r in transfer])-a
    ax.bar(range(len(a)), a, color=[COLORS[0],COLORS[1]]*4, yerr=np.vstack([low,high]), capsize=4); ax.set_xticks(range(len(a)), labels, fontsize=9)
    ax.axhline(0, color='gray', lw=.8); ax.set(ylabel='Target test Spearman', title='B. subtilis 61 bp: test n=118; bootstrap by exact-sequence group')
    save('08_cross_species', fig, ['transfer/transfer_metrics.csv'], '目标域相同序列按组划分。few-shot只拟合目标train并在目标val选alpha；目标test已在旧实验中使用，本次为同基准扩展。')
    write_json(directory/'figure_manifest.json', dict(figures=records, n_figures=len(records)))
    finish(directory, cfg, dict(n_figures=len(records), visual_review='pending'), [root/s for s in ['evaluation/model_summary.csv','ablation/stage_summary.csv','mutation/direction_summary.csv','transfer/transfer_metrics.csv']])
    return directory
