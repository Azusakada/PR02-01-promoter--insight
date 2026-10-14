"""Generate a run-local, source-derived report without publishing final materials."""
from __future__ import annotations
from .data import read_json, read_table, write_json
from .m4_data_eval import run_root, seeds


def build(cfg, protocol):
    root = run_root(protocol); report = root/'reports'; report.mkdir(parents=True, exist_ok=True)
    def f(v): return f'{float(v):.4f}' if v not in ('', None) else '未定义'
    summary = read_table(root/'evaluation/model_summary.csv')
    stages = read_table(root/'ablation/stage_summary.csv')
    overlap = read_table(root/'explanation_analysis/overlap_summary.csv')
    agreement = read_table(root/'explanation_analysis/consistency_summary.csv')
    mutations = read_table(root/'mutation/mutation_summary.csv')
    directions = read_table(root/'mutation/direction_summary.csv')
    transfer = read_table(root/'transfer/transfer_metrics.csv')
    selected = read_json(root/'cnn/selected_model.json')
    sections = []
    def section(title, paragraphs, headers, rows, sources, image=None):
        sections.append(dict(title=title, paragraphs=paragraphs, table=dict(headers=headers, rows=rows),
                             sources=sources, image=(protocol['output']+'/figures/'+image+'.png') if image else None))
    section('数据与评价协议', [
        '主数据为TSS核验的87bp大肠杆菌天然表达记录，目标域为教师提供的61bp枯草杆菌数据。两域完整测量单位和条件仍不明确。',
        '拟合与归一化只使用train，val参与模型选择和描述性评价。继承随机划分可能包含邻近基因组窗口；历史test曾被使用，本轮源域test不评分。'],
        ['配置','本次实际值'], [['运行',protocol['output']], ['CNN种子',str(list(seeds(protocol)))],
        ['CNN候选',str(protocol['cnn_configurations'])], ['Ridge alpha',str(protocol['ridge_alpha'])],
        ['bootstrap',str(protocol['bootstrap_repeats'])+'次 / '+str(protocol['confidence'])+' / seed '+str(protocol['bootstrap_seed'])],
        ['归因train / val',str(protocol['attribution_train_n'])+' / '+str(protocol['attribution_val_n'])]], ['protocol/protocol.json'])
    section('必做1：性能比较', ['各run先独立计分，再汇总CNN种子均值和样本标准差。CNN选择配置为'+selected['condition']+'。验证集用于选参，不能把下表当作独立泛化成绩。'],
        ['方法','R²均值','Spearman均值','RMSE均值','种子SD(R²)'],
        [[r['method'],f(r['r2_log10_mean']),f(r['spearman_mean']),f(r['rmse_log10_mean']),f(r['r2_log10_sd'])] for r in summary],
        ['evaluation/model_summary.csv'], '01_model_comparison')
    section('必做2：位点与区域', ['IG和ISM解释选定的纯CNN。固定窗口与PC推定窗口分别统计；扣除长度/87作为均匀位置参照。区间按母序列重采样，未作多重比较显著性声明，也不等于实测结合注释。'],
        ['方法','窗口','区域','Top-k','富集均值','区间'],
        [[r['method'],r['annotation_status'],r['region'],r['top_k'],f(r['mean_excess_fraction']),f(r['excess_ci_low'])+' 至 '+f(r['excess_ci_high'])] for r in overlap],
        ['explanation_analysis/overlap_summary.csv'], '03_topk_overlap')
    section('必做3：逐步消融', ['A0为k-mer，A1添加CNN，A2添加冻结DNA表征，A3添加仅由train归因选区产生的特征。三个CNN种子配对，统一使用train拟合的StandardScaler和Ridge；alpha按本协议在val选择。',
        '各阶段相对前一阶段的变化直接列于表中。该结论仅针对本次特征、参数和划分，不能推出某类表征在其他任务无效。'],
        ['阶段','R²均值','Spearman均值','RMSE均值','相对前阶段ΔR²'],
        [[r['stage'],f(r['r2_log10_mean']),f(r['spearman_mean']),f(r['rmse_log10_mean']),f(r['delta_r2_from_previous'])] for r in stages],
        ['ablation/stage_summary.csv'], '07_staged_ablation')
    section('必做4：等数量计算突变', ['同一母序列在High/Low/Random组均替换5个位点，各重复10次。设计先保存再预测。母序列内先汇总，再按母序列bootstrap。'],
        ['组','母序列数','变体数','平均绝对变化','区间'],
        [[r['group'],r['n_parent'],r['n_mutants'],f(r['mean_absolute_delta_log10']),f(r['ci_low'])+' 至 '+f(r['ci_high'])] for r in mutations],
        ['mutation/mutation_summary.csv'], '05_mutation_amplitude')
    section('必做4：单点方向', ['预期方向来自野生型梯度的替换通道减原通道，不使用ISM实际变化回填预期。区间按母序列聚类，属于模型内部自洽性与跨模型比较，不能证明真实生物因果。'],
        ['预期类别','变体数','梯度/CNN','梯度/PC','CNN/PC'],
        [[r['expected_direction'],r['n_mutants'],f(r['cnn_agreement']),f(r['thermo_agreement']),f(r['cross_model_agreement'])] for r in directions],
        ['mutation/direction_summary.csv'], '06_mutation_direction')
    section('选做5：跨物种', ['相同目标序列按组划分，few-shot只拟合目标train，在目标val选alpha。目标test已在历史实验使用，属于原基准扩展；单位未知时主要解释Spearman。目标TSS和方向未知，不套用固定TSS热力学模型。'],
        ['条件','训练组数','Spearman','test组bootstrap区间'],
        [[r['condition'],r['n_train_groups'],f(r['spearman']),f(r['spearman_ci_low'])+' 至 '+f(r['spearman_ci_high'])] for r in transfer],
        ['transfer/transfer_metrics.csv'], '08_cross_species')
    section('选做6：归因一致性', ['同一CNN、同一母序列上的IG/ISM一致性与跨CNN种子稳定性是不同问题。下表只回答前者。'],
        ['Top-k','指标','均值','区间'],[[r['top_k'],r['metric'],f(r['mean']),f(r['ci_low'])+' 至 '+f(r['ci_high'])] for r in agreement],
        ['explanation_analysis/consistency_summary.csv'], '04_attribution_consistency')
    content = dict(title='启动子强度预测与位点解释',subtitle='PR02-01 本次运行报告（待定稿）',
        run_id=protocol['output'],date='2026年10月14日',members=['胡昊铭','李宇飞','李玘航','田惠今','陈思远'],
        sections=sections,references=[],final_materials_updated=False)
    write_json(report/'M5_REPORT_CONTENT.json',content)
    lines=['# '+content['title'],'',content['subtitle'],'','运行：`'+protocol['output']+'`','']
    for s in sections:
        lines+=['## '+s['title'],'']
        for paragraph in s['paragraphs']:lines+=[paragraph,'']
        t=s['table'];lines+=['| '+' | '.join(t['headers'])+' |','| '+' | '.join(['---']*len(t['headers']))+' |']
        lines+=['| '+' | '.join(map(str,r))+' |' for r in t['rows']];lines+=['']
        if s['image']:lines+=['![结果图](../figures/'+s['image'].split('/')[-1]+')','']
        lines+=['源表：'+ '、'.join('`'+protocol['output']+'/'+p+'`' for p in s['sources']),'']
    (report/'M5_FINAL_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return content
