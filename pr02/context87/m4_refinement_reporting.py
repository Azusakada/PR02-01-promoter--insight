"""Run-local review and figures; never rebuild or publish presentation materials."""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .data import ROOT, read_json, read_table, sha, write_json
from .m4_data_eval import completed, finish, run_root, seeds
from .m4_refinement import baseline


def build(cfg,protocol):
    root=run_root(protocol);old=baseline(protocol);directory=root/'figures';report=root/'reports'
    if completed(directory,cfg,protocol) and completed(report,cfg,protocol):return report
    directory.mkdir(parents=True,exist_ok=True);report.mkdir(parents=True,exist_ok=True)
    colors=['#386A91','#C17B36','#468773'];records=[]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    def save(name,fig,sources,caption):
        fig.tight_layout(pad=1.6)
        for suffix in ['png','svg']:fig.savefig(directory/(name+'.'+suffix),dpi=180)
        plt.close(fig)
        records.append(dict(figure_id=name,png=name+'.png',svg=name+'.svg',caption=caption,sha256=sha(directory/(name+'.png')),
            source_tables={str((root/p).relative_to(ROOT)):sha(root/p) for p in sources},visual_review='pending'))
    refined=read_table(root/'regularization/summary.csv');original=read_table(old/'ablation/stage_summary.csv')
    dna_old=next(r for r in read_table(old/'evaluation/model_summary.csv') if r['method']=='dnabert2_ridge')
    original_lookup={r['stage']:r for r in original};new_lookup={r['method']:r for r in refined}
    fig,ax=plt.subplots(figsize=(9,4.8));labels=['kmer_ridge','dnabert2_ridge','A1','A2','A3'];xx=np.arange(len(labels))
    previous=[float(dna_old['r2_log10_mean']) if m=='dnabert2_ridge' else float(original_lookup['A0' if m=='kmer_ridge' else m]['r2_log10_mean']) for m in labels]
    current=[float(new_lookup[m]['r2_log10_mean']) for m in labels]
    err=[float(new_lookup[m]['r2_log10_sd'] or 0) for m in labels]
    ax.bar(xx-.18,previous,.36,label='Original alpha <= 100',color='#ADBBC4');ax.bar(xx+.18,current,.36,yerr=err,capsize=4,label='Expanded alpha grid',color=colors[0])
    ax.set_xticks(xx,['k-mer','DNA+Ridge','A1 +CNN','A2 +DNA','A3 +guided']);ax.axhline(0,color='gray',lw=.8)
    ax.set(ylabel='Inherited validation R²',title='Controlled regularization comparison; same frozen representations');ax.legend()
    save('01_regularization_comparison',fig,['regularization/summary.csv','regularization/matched_comparison.csv'],'只扩大Ridge正则化范围；融合显示三个匹配种子的均值与样本标准差。验证集参与选参，数值不代表独立泛化。')
    searches=read_table(root/'regularization/search.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    for ax,method in zip(axes,['dnabert2_ridge','A3']):
        for i,seed in enumerate([''] if method=='dnabert2_ridge' else list(map(str,seeds(protocol)))):
            rr=[r for r in searches if r['method']==method and r['seed']==seed]
            ax.plot([float(r['alpha']) for r in rr],[float(r['val_mse_log10']) for r in rr],marker='o',color=colors[i],label='DNA head' if seed=='' else 'seed '+seed)
        ax.set(xscale='log',xlabel='Ridge alpha',ylabel='Validation MSE (log10)',title=method);ax.axvline(100,color='gray',ls='--',label='Old upper bound');ax.legend(fontsize=8)
    save('02_regularization_search',fig,['regularization/search.csv'],'完整保留预定七档alpha的误差曲线，不只展示获胜参数。')
    grouped=read_table(root/'grouped/summary.csv');audit=read_json(root/'grouped/split_audit.json')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7));labels=[r['method'].replace('dnabert2_ridge','DNA').replace('kmer_ridge','Ridge') for r in grouped]
    for ax,key in zip(axes,['r2_log10','spearman']):
        a=[float(r[key+'_mean']) if r[key+'_mean'] else np.nan for r in grouped];err=[float(r[key+'_sd'] or 0) for r in grouped]
        ax.bar(range(len(a)),a,color=colors[0],yerr=err,capsize=3);ax.set_xticks(range(len(a)),labels,rotation=40,ha='right');ax.axhline(0,color='gray',lw=.7)
        ax.set(ylabel=key,title='Spatial evaluation n='+str(audit['counts']['evaluation']))
    save('03_grouped_validation',fig,['grouped/summary.csv','grouped/split_audit.json'],'20kb区域分组，跨集合87bp窗口重叠为零；重新训练CNN、回归头、热力学校准和训练归因配方。超参数及早停只看dev。误差线为训练种子标准差。')
    fig,axes=plt.subplots(1,len(seeds(protocol)),figsize=(12,3.8),squeeze=False)
    for ax,seed in zip(axes[0],seeds(protocol)):
        history=read_table(root/f'grouped/cnn_s{seed}/training_history.csv')
        ax.plot([int(r['epoch']) for r in history],[float(r['train_mse']) for r in history],label='train')
        ax.plot([int(r['epoch']) for r in history],[float(r['val_mse']) for r in history],label='dev')
        ax.set(title='CNN seed '+str(seed),xlabel='Epoch',ylabel='MSE (training min-max scale)');ax.legend()
    save('04_grouped_learning_curves',fig,[f'grouped/cnn_s{s}/training_history.csv' for s in seeds(protocol)],'分组验证CNN的训练与开发集曲线；evaluation标签不参与早停。')
    region=read_table(root/'attribution_stability/region_by_seed.csv');fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for ax,method in zip(axes,['integrated_gradients','ism_mean_absolute']):
        for i,seed in enumerate(seeds(protocol)):
            rr=[r for r in region if r['method']==method and r['window_policy']=='fixed' and r['top_k']=='5' and r['seed']==str(seed)]
            ax.bar(np.arange(4)+(i-1)*.23,[float(r['mean_excess_fraction']) for r in rr],.23,label=str(seed),color=colors[i])
        ax.set_xticks(range(4),['UP','-35','spacer','-10']);ax.axhline(0,color='gray',lw=.7);ax.set(title='IG' if method=='integrated_gradients' else 'ISM',ylabel='Top-5 fraction minus region length/87');ax.legend(fontsize=8)
    save('05_attribution_seed_stability',fig,['attribution_stability/region_by_seed.csv'],'同一64条母序列、三个CNN种子的固定区域富集。跨种子稳定性与同模型IG/ISM一致性是不同指标；仍为描述性分析。')
    transfer=read_table(root/'transfer_stability/summary.csv');fig,ax=plt.subplots(figsize=(8.5,4.8))
    for i,kind in enumerate(['kmer','frozen_cnn']):
        rr=[next(r for r in transfer if r['condition']==f'few_shot_{kind}_{n}_groups') for n in protocol['few_shot_groups']]
        ax.errorbar(protocol['few_shot_groups'],[float(r['spearman_mean']) for r in rr],yerr=[float(r['spearman_sd']) for r in rr],marker='o',capsize=5,label=kind+' mean +/- draw SD',color=colors[i])
    for i,name in enumerate(['zero_shot_ridge','zero_shot_cnn']):
        r=next(r for r in transfer if r['condition']==name);ax.axhline(float(r['spearman_mean']),color=colors[i],ls='--',alpha=.7,label=name)
    ax.set(xlabel='Target training sequence groups',ylabel='Target test Spearman',title=f"{len(protocol['transfer_draw_seeds'])} predeclared nested training-group draws; same historical test");ax.legend(fontsize=8)
    save('06_transfer_training_draws',fig,['transfer_stability/summary.csv'],'误差线表示训练组抽样的标准差；每次抽样的目标test序列组bootstrap区间另存metrics.csv。目标test曾被使用，未称为新盲测。')
    write_json(directory/'figure_manifest.json',dict(figures=records,n_figures=len(records)))
    finish(directory,cfg,dict(n_figures=len(records),visual_review='pending'),[ROOT/p for row in records for p in row['source_tables']])
    def f(v):return f'{float(v):.4f}' if v not in ('',None) else '未定义'
    def table(headers,rows):return ['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows]+['']
    lines=['# PR02-01 项目完善核验','','本轮是实验修订与质量检查，项目最终内容待确认。PPT、Word及旧版正式报告保持原版本。',
        '',f"运行：`{protocol['output']}`。基线：`{protocol['baseline']}`。",'',
        '## 1. 原划分上的正则化对照','',
        '仅扩大回归头alpha范围，使用同样的特征、训练样本和三个CNN种子。所有候选都保留。此表的验证集参与选参，因此是受选择影响的描述性结果。', '']
    lines+=table(['模型','原R²','修订R²均值','种子SD','Spearman均值'],[[m,f(p),f(new_lookup[m]['r2_log10_mean']),f(new_lookup[m]['r2_log10_sd']),f(new_lookup[m]['spearman_mean'])] for m,p in zip(['kmer_ridge','dnabert2_ridge','A1','A2','A3'],previous)])
    lines+=['旧版负结果应限定为旧参数范围下的结果。扩展搜索后的变化不能直接归因于增加新生物信息，也不能代替独立数据评价。', '',
        '## 2. 基因组区域分组验证','',
        f"在原train+val中按{protocol['genomic_block_bp']}bp区域分组，train/dev/evaluation分别为{audit['counts']['train']}/{audit['counts']['dev']}/{audit['counts']['evaluation']}条。排除跨区域边界窗口{audit['exclusions']['excluded_block_boundary']}条，原test全部排除。双链窗口在任意两集合间的重叠均为0；相同序列及反向互补副本也不跨集合。",
        '', 'CNN采用预先固定的C0结构重新训练三个种子；归一化、scaler、热力学校准、归因选区均只拟合新train，回归头alpha和CNN早停只看dev。evaluation不用于这些选择。DNA编码器为固定的外部预训练权重，可复用未使用本项目标签的表征。','']
    lines+=table(['模型','R²均值','种子SD','Spearman均值','RMSE均值'],[[r['method'],f(r['r2_log10_mean']),f(r['r2_log10_sd']),f(r['spearman_mean']),f(r['rmse_log10_mean'])] for r in grouped])
    contrasts=read_table(root/'grouped/paired_contrasts.csv')
    lines+=table(['相对k-mer Ridge的对照','ΔR²','95%区域bootstrap区间'],[[r['contrast'],f(r['mean_delta_r2']),f(r['ci_low'])+' 至 '+f(r['ci_high'])] for r in contrasts])
    lines+=['误差线中的种子SD只反映训练随机性；区域bootstrap只反映固定模型在本次分组评价样本上的波动。原划分与分组划分样本组成不同，不能把分数差全部解释为数据泄漏的大小。这是单次内部空间分组对照，历史标签曾参与项目开发，仍不是独立外部盲测。','',
        '## 3. 位点解释的跨种子稳定性','',
        '原先的IG/ISM一致性仍保留；本轮另比较三个CNN种子。种子对在同一母序列内先汇总，再按64条母序列bootstrap，避免把种子对当作独立生物样本。','']
    agreement=read_table(root/'attribution_stability/cross_seed_summary.csv')
    lines+=table(['归因方法','Top-k','跨种子指标','均值','95%母序列区间'],[[r['method'],r['top_k'],r['metric'],f(r['mean']),f(r['ci_low'])+' 至 '+f(r['ci_high'])] for r in agreement])
    regions=read_table(root/'attribution_stability/region_across_seeds.csv')
    lines+=table(['方法','窗口政策','Top-k','−10富集均值','正向种子数/总数'],[[r['method'],r['window_policy'],r['top_k'],f(r['mean_excess_fraction']),r['positive_seed_count']+'/'+r['n_seeds']] for r in regions if r['region']=='-10'])
    lines+=['区域富集与具体位点稳定性分别解释。均匀位置参照没有消除所有TSS位置偏好；本轮没有进行多重检验显著性声明，固定或PC推定窗口也不是逐条实测结合注释。','',
        '## 4. 跨物种训练抽样稳定性','',
        f"目标域沿用原序列组划分。每次预定抽样内20/50/100组嵌套，共{len(protocol['transfer_draw_seeds'])}次，所有结果完整报告。目标训练组拟合，目标val选alpha；目标test不用于选抽样或条件。CNN源权重保持冻结，Ridge zero-shot采用本轮扩展alpha后的源域头。",'']
    lines+=table(['条件','抽样次数','Spearman均值','抽样SD','抽样最小/最大'],[[r['condition'],r['n_draws'],f(r['spearman_mean']),f(r['spearman_sd']),f(r['spearman_min'])+' / '+f(r['spearman_max'])] for r in transfer])
    lines+=['抽样SD与单次test序列组bootstrap区间分别保存。目标test曾在历史实验中使用；教师CSV缺少单位和测量条件，因此主要解释排序相关，跨物种绝对误差只作尺度诊断。','',
        '## 5. 工程修复与验收入口','',
        '配置中的种子、CNN候选、归因样本数和bootstrap参数实际生效。断点复用检查产物、上游文件、完整配置、协议及执行源码；配置或源码变化要求新run。旧源码按摘要归档，历史结果可核验，不能静默当作新代码运行。', '',
        '报告和图表写入各自run目录。正式PPT/Word没有随实验重新生成。图像检查必须显式登记，登记前不会通过完整验收。', '',
        '```bash','python -m pr02 m4-refine --protocol configs/m4_refinement_v2.json --stage verify','```','',
        '当前修订的数值与结论以本文件及其源表为准。旧版M5材料保持历史身份，待项目最终确认后统一更新。','',
        '## 图与源表索引','']
    for row in records:lines+=['### '+row['figure_id'],'',row['caption'],'',f"![{row['figure_id']}](../figures/{row['png']})",'', '源表：'+ '、'.join('`'+p+'`' for p in row['source_tables']),'']
    (report/'REFINEMENT_REVIEW.md').write_text('\n'.join(lines),encoding='utf-8')
    finish(report,cfg,dict(ppt_updated=False,word_updated=False,final_project_content_pending=True),
        [root/'regularization/summary.csv',root/'grouped/summary.csv',root/'attribution_stability/cross_seed_summary.csv',root/'transfer_stability/summary.csv'])
    return report
