"""Source-derived case explanations, distinct from experimental ground truth."""
from __future__ import annotations
import json
from .data import ROOT, coordinate, read_table
from .m4_data_eval import REGIONS, run_root


def build(cfg,protocol):
    root=run_root(protocol)
    attrs=read_table(root/'attribution/attributions.tsv');top={}
    for r in attrs:
        if int(r['rank_absolute'])<=5:top.setdefault((r['sample_id'],r['method']),[]).append((int(r['rank_absolute']),int(r['position_0based'])))
    lines=['# M4归因与突变案例分析','','以下均为固定选择规则下的计算案例，不代表真实结合或突变测量。',
           '','## IG与ISM的高、低一致性案例','','按Top-5 Jaccard最低及最高各3条选择，平分按sample_id；不根据生物解释是否顺眼筛选。',
           '','| sample_id | IG Top-5索引 | ISM Top-5索引 | 交集数 | Jaccard |','| --- | --- | --- | --- | --- |']
    cases=read_table(root/'explanation_analysis/consistency_cases.csv')
    for r in cases:
        a=[p for rank,p in sorted(top[r['sample_id'],'integrated_gradients'])]
        b=[p for rank,p in sorted(top[r['sample_id'],'ism_mean_absolute'])]
        lines.append(f"| {r['sample_id']} | {a} | {b} | {len(set(a)&set(b))} | {float(r['top_k_jaccard']):.4f} |")
    lines+=['','IG从均匀概率基线到序列输入积分，ISM直接计算离散替换后的有限变化。低重合表示这两种问题在该CNN上给出不同位置排序，不能仅凭低重合认定某方法错误。高重合也只表示同一模型内的解释一致。',
            '','全部索引均为0基。转换为TSS相对位置时，上游为索引−67，下游为索引−66，跳过0。固定与工具推定窗口的总体结论见overlap_summary.csv。',
            '','## CNN与热力学差异最大的单点变体','','从960条单点变体中，按|CNN delta−热力学 delta|降序取前10，平分按mutation_id。delta均为变体减野生型。',
            '','| 变体与母序列 | 位置/替换 | 固定窗口 | 预期梯度delta | CNN delta | 热力学delta |','| --- | --- | --- | --- | --- | --- |']
    mutation=read_table(root/'mutation/disagreement_cases.csv')
    for r in mutation:
        pos=json.loads(r['positions_0based'])[0];ref=json.loads(r['ref_bases'])[0];alt=json.loads(r['alt_bases'])[0]
        windows={**REGIONS,'discriminator':(61,67),'ITR':(67,87)}
        window=next((name for name,(start,end) in windows.items() if start<=pos<end),'other')
        lines.append(f"| {r['mutation_id']} / {r['sample_id']} | {pos} (TSS {coordinate(pos):+d}) {ref}>{alt} | {window} | {float(r['expected_delta_log10']):+.5f} | {float(r['delta_log10']):+.5f} | {float(r['thermo_delta_log10']):+.5f} |")
    lines+=['','这里的最强分歧不能解释为CNN或热力学预测的实测错误，因为没有相应变体的测量标签。CNN拟合天然表达相关性，PC描述固定TSS下的转录起始热力学，再通过训练集线性校准映射尺度。两者的函数形状及特征依赖不同，有限替换后可出现相反方向。',
            '','PC每个变体仍在固定TSS67处枚举30种状态，最优盒区状态允许变化；不是把野生型最优状态永久锁死。mutant_thermo_states.jsonl保存逐变体状态，可进一步区分碱基能量变化与最优状态切换。',
            '','可执行结论是优先把这些分歧案例纳入后续实测，而不是把任一模型输出当作真值。设计、替换预期、实际预测均由mutation_id连接，未按预测方向重选变体。',
            '','源表：'+protocol['output']+'/explanation_analysis/consistency_cases.csv；'+protocol['output']+'/mutation/disagreement_cases.csv。','']
    report=root/'reports';report.mkdir(parents=True,exist_ok=True)
    (report/'M4_CASE_ANALYSIS.md').write_text('\n'.join(lines),encoding='utf-8')
    notes=['# M4实验说明','','运行：`'+protocol['output']+'`。所有模块读取同一协议，不修改M3数据或已发布运行。',
        '','## 数据与评价','源域8267/1768/1774保持不变；源域test不评分。所有方法val成功覆盖1768条。每个run先计分，CNN汇总三种子均值和样本标准差。目标序列组冻结且不跨集合。',
        '','## 传统模型与DNA表征','Ridge与LinearSVR完成k=3/4/5网格。DNA固定官方117M权重，序列token平均排除padding与特殊token。编码器冻结，回归头和scaler只拟合train。性能比较以本次evaluation/model_summary.csv为准，完整候选及收敛记录均保留。',
        '','## CNN与融合','CNN候选和种子按本次协议执行，选定配置见cnn/selected_model.json。A0至A3统一Ridge头；种子逐一配对，归因选区只用协议指定的train样本。增益直接读取ablation/stage_summary.csv，不预写结论。',
        '','## 热力学与突变','复用经核验的TSS67及训练校准，突变重新枚举最优状态。1920幅度变体和960单点变体的设计先于实际变体预测保存。方向预期来自野生型梯度，而非将ISM真实变化回填为预期。统计按母序列聚类。',
        '','## 归因与选做','IG/ISM均解释纯CNN代表checkpoint，未将其解释套到DNA回归或融合模型。Top-k比较固定和推定窗口，扣除区域长度并保留随机对照。跨物种八条件及两方法一致性完成；区间较宽和多重比较未校正均在报告说明。',
        '','## 结论范围','本次性能、逐阶段变化和跨模型分歧均按源表解释。没有新增湿实验标签，无法证明生物因果。严格基因组分组验证和独立外部实验仍属于后续研究。',
        '','复现入口：`python -m pr02 m4 --stage verify`。完整重跑须复制协议并更换output，不覆盖本轮冻结目录。','']
    (report/'M4_EXPERIMENT_NOTES.md').write_text('\n'.join(notes),encoding='utf-8')
