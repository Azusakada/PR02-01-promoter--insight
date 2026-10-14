"""Build the course report from frozen experiment tables and prediction exports."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
BASE = 'runs/m4_complete_20261014_v1'
REV = 'runs/m4_refinement_20261014_v2'
SOURCES = {}
FIGURES = []
SEEDS = [20260928, 20260929, 20260930]
REGIONS = {'UP': (7, 31), '-35': (32, 38), 'spacer': (38, 55), '-10': (55, 61)}
COLORS = ['#386A91', '#C17B36', '#468773']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source(name):
    path = ROOT/name
    SOURCES[name] = digest(path)
    return path


def table_file(name):
    with source(name).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if name.endswith('.tsv') else ','))


def json_file(name):
    return json.loads(source(name).read_text())


def number(value, digits=4):
    if value in ('', None):
        return '未定义'
    result = f'{float(value):.{digits}f}'
    return result[1:] if result.startswith('-') and float(result) == 0 else result


def mean_sd(row, key):
    value = number(row[key+'_mean'])
    return value+'±'+number(row[key+'_sd']) if row[key+'_sd'] not in ('', None) else value


def build(output):
    SOURCES.clear(); FIGURES.clear()
    output.mkdir(parents=True, exist_ok=True); figures = output/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
    def save_figure(stem, fig, inputs):
        fig.tight_layout(pad=1.5)
        for ext in ('png', 'svg'): fig.savefig(figures/(stem+'.'+ext), dpi=200, bbox_inches='tight')
        plt.close(fig)
        for path in inputs: source(path)
        FIGURES.append(dict(path='figures/'+stem+'.png', inputs=inputs))
        return 'figures/'+stem+'.png'
    def copy_figure(stem, original, inputs):
        for ext in ('png', 'svg'):
            path = source(original+'.'+ext); shutil.copyfile(path, figures/(stem+'.'+ext))
        for path in inputs: source(path)
        FIGURES.append(dict(path='figures/'+stem+'.png', inputs=inputs, original=original+'.png'))
        return 'figures/'+stem+'.png'

    rows = table_file('data/03_Ecoli_strength87/samples.tsv')
    train = [r for r in rows if r['split'] == 'train']; val = [r for r in rows if r['split'] == 'val']
    ytrain = np.array([float(r['target_log10']) for r in train]); yval = np.array([float(r['target_log10']) for r in val])
    target = table_file('data/04_Bsub_strength61/samples.tsv')
    protocol = json_file('configs/m4_refinement_v2.json')
    json_file('project_config.json')
    json_file(BASE+'/cnn/selected_model.json')
    table_file(BASE+'/cnn/search.csv')
    original = {r['method']: r for r in table_file(BASE+'/evaluation/model_summary.csv')}
    refined = {r['method']: r for r in table_file(REV+'/regularization/summary.csv')}
    main = [dict(original['knn']), dict(refined['kmer_ridge']), dict(original['linear_svr']),
            dict(refined['dnabert2_ridge']), dict(original['thermo']), dict(original['cnn']), dict(original['train_mean'])]
    names = {'knn':'KNN', 'kmer_ridge':'k-mer Ridge', 'linear_svr':'LinearSVR',
             'dnabert2_ridge':'DNA + Ridge', 'thermo':'Promoter Calculator', 'cnn':'CNN', 'train_mean':'训练均值'}
    group = table_file(REV+'/grouped/summary.csv'); group_lookup = {r['method']:r for r in group}
    split = json_file(REV+'/grouped/split_audit.json')
    consistency = table_file(BASE+'/explanation_analysis/consistency_summary.csv')
    stability = table_file(REV+'/attribution_stability/cross_seed_summary.csv')
    region = table_file(REV+'/attribution_stability/region_across_seeds.csv')
    amplitude = table_file(BASE+'/mutation/mutation_summary.csv')
    paired = table_file(BASE+'/mutation/paired_amplitude.csv')
    direction = next(r for r in table_file(BASE+'/mutation/direction_summary.csv') if r['expected_direction']=='all')
    transfer = table_file(REV+'/transfer_stability/summary.csv')
    alpha_rows = table_file(REV+'/regularization/metrics.csv')
    assert {float(r['alpha']) for r in alpha_rows} == {10000.}
    assert [len(train),len(val),sum(r['split']=='test' for r in rows)] == [8267,1768,1774]
    assert split['counts'] == {'train':5985,'dev':2003,'evaluation':1998}
    assert not any(split['cross_subset_window_overlaps'].values())

    # Descriptive statistics use training labels only.
    feature_arrays = {'全序列 GC':np.array([sum(b in 'GC' for b in r['sequence'])/87 for r in train])}
    for label, name in [('−10 GC','-10'),('−35 GC','-35'),('UP GC','UP')]:
        a,b=REGIONS[name];feature_arrays[label]=np.array([sum(base in 'GC' for base in r['sequence'][a:b])/(b-a) for r in train])
    for label,name,consensus in [('−10 共识匹配','-10','TATAAT'),('−35 共识匹配','-35','TTGACA')]:
        a,b=REGIONS[name];feature_arrays[label]=np.array([sum(x==y for x,y in zip(r['sequence'][a:b],consensus))/6 for r in train])
    eda=[dict(feature=name,mean=float(values.mean()),spearman=float(spearmanr(values,ytrain).statistic)) for name,values in feature_arrays.items()]
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8))
    axes[0].hist(ytrain,bins=35,color=COLORS[0],edgecolor='white',linewidth=.3)
    axes[0].set(xlabel='log10 observed expression',ylabel='Training sequences',title='Training label distribution (n=8,267)')
    axes[1].bar(range(6),[r['spearman'] for r in eda],color=[COLORS[0]]*4+[COLORS[1]]*2)
    axes[1].set_xticks(range(6),['GC','-10 GC','-35 GC','UP GC','-10 match','-35 match'],rotation=35,ha='right')
    axes[1].axhline(0,color='gray',lw=.7);axes[1].set(ylabel='Spearman with log10 expression',title='Fixed-window sequence features')
    distribution=save_figure('01_training_distribution',fig,['data/03_Ecoli_strength87/samples.tsv'])
    fig,axes=plt.subplots(1,2,figsize=(11,4.2));short=['KNN','Ridge','LinearSVR','DNA+Ridge','PC','CNN','Mean']
    for ax,key,label in zip(axes,['r2_log10','spearman'],['R² on log10 expression','Spearman correlation']):
        values=[float(r[key+'_mean']) if r[key+'_mean'] not in ('',None) else np.nan for r in main]
        errors=[float(r[key+'_sd'] or 0) for r in main]
        ax.bar(range(7),values,color=COLORS[0],yerr=errors,capsize=3);ax.axhline(0,color='gray',lw=.7)
        ax.set_xticks(range(7),short,rotation=30,ha='right');ax.set(ylabel=label,title='Common validation n=1,768')
        if key=='spearman':ax.text(6,0.005,'undefined',ha='center',va='bottom',fontsize=8,rotation=90)
    performance=save_figure('02_model_comparison',fig,[BASE+'/evaluation/model_summary.csv',REV+'/regularization/summary.csv'])
    old_stages={r['stage']:r for r in table_file(BASE+'/ablation/stage_summary.csv')}
    search=table_file(REV+'/regularization/search.csv');fig,axes=plt.subplots(1,2,figsize=(11,3.8))
    labels=['kmer_ridge','dnabert2_ridge','A1','A2','A3'];x=np.arange(5)
    before=[float(original['dnabert2_ridge']['r2_log10_mean']) if key=='dnabert2_ridge' else float(old_stages['A0' if key=='kmer_ridge' else key]['r2_log10_mean']) for key in labels]
    after=[float(refined[key]['r2_log10_mean']) for key in labels]
    axes[0].bar(x-.18,before,.36,color='#ADBBC4',label='Original grid')
    axes[0].bar(x+.18,after,.36,color=COLORS[0],label='Expanded grid',yerr=[float(refined[key]['r2_log10_sd'] or 0) for key in labels],capsize=3)
    axes[0].set_xticks(x,['k-mer','DNA','A1','A2','A3']);axes[0].axhline(0,color='gray',lw=.7)
    axes[0].set(ylabel='Validation R²',title='Fixed-representation comparison');axes[0].legend(fontsize=8)
    for method,label,color in [('dnabert2_ridge','DNA head',COLORS[1]),('A3','A3 seed mean',COLORS[0])]:
        means=[];sds=[]
        for alpha in protocol['ridge_alpha']:
            values=[float(r['val_mse_log10']) for r in search if r['method']==method and float(r['alpha'])==alpha]
            means.append(np.mean(values));sds.append(np.std(values,ddof=1) if len(values)>1 else 0)
        axes[1].plot(protocol['ridge_alpha'],means,marker='o',ms=4,label=label,color=color)
        if method=='A3':axes[1].fill_between(protocol['ridge_alpha'],np.array(means)-sds,np.array(means)+sds,color=color,alpha=.12)
    axes[1].axvline(100,color='gray',ls='--',label='Old upper bound')
    axes[1].set(xscale='log',xlabel='Ridge alpha',ylabel='Validation MSE (log10)',title='Expanded regularization search');axes[1].legend(fontsize=8)
    ablation=save_figure('03_regularization_ablation',fig,[BASE+'/ablation/stage_summary.csv',REV+'/regularization/summary.csv',REV+'/regularization/search.csv'])
    grouped=copy_figure('05_genomic_validation',REV+'/figures/03_grouped_validation',[REV+'/grouped/summary.csv'])
    learning=copy_figure('06_learning_curves',REV+'/figures/04_grouped_learning_curves',[REV+f'/grouped/cnn_s{s}/training_history.csv' for s in SEEDS])
    regional=copy_figure('07_regional_attribution',REV+'/figures/05_attribution_seed_stability',[REV+'/attribution_stability/region_by_seed.csv'])
    matrices=[]
    for method in ['ig','ism']:
        for seed in SEEDS:
            file=source(REV+f'/attribution_stability/s{seed}/scores.npz')
            with np.load(file,allow_pickle=False) as z:values=np.abs(z[method])
            matrices.append((values/np.maximum(values.sum(1,keepdims=True),1e-12)).mean(0))
    fig,ax=plt.subplots(figsize=(10.5,3.7));im=ax.imshow(matrices,aspect='auto',cmap='YlGnBu',vmin=0)
    ax.set_yticks(range(6),['IG '+str(s) for s in SEEDS]+['ISM '+str(s) for s in SEEDS])
    ax.set_xticks([0,7,32,38,55,61,67,86],['-67','-60','-35','-29','-12','-6','+1','+20'])
    ax.set(xlabel='Position relative to TSS',title='Mean within-sequence normalized absolute importance')
    for pos in [31,38,55,61,67]:ax.axvline(pos-.5,color='white',lw=.6,alpha=.7)
    fig.colorbar(im,ax=ax,pad=.015,label='Mean importance fraction')
    position_map=save_figure('08_position_importance',fig,[REV+f'/attribution_stability/s{s}/scores.npz' for s in SEEDS])
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.7))
    a=np.array([float(r['mean_absolute_delta_log10']) for r in amplitude]);ci=np.array([[float(r['ci_low']),float(r['ci_high'])] for r in amplitude])
    axes[0].bar(['High','Low','Random'],a,yerr=np.vstack([a-ci[:,0],ci[:,1]-a]),capsize=4,color=COLORS)
    axes[0].set(ylabel='Mean |mutant - wild type| (log10)',title='5 substitutions; 10 replicates per parent')
    keys=['cnn_agreement','thermo_agreement','cross_model_agreement'];a=np.array([float(direction[k]) for k in keys])
    low=np.array([float(direction[k+'_ci_low']) for k in keys]);high=np.array([float(direction[k+'_ci_high']) for k in keys])
    axes[1].bar(['Gradient/CNN','Gradient/PC','CNN/PC'],a,yerr=np.vstack([a-low,high-a]),color=COLORS,capsize=4)
    axes[1].set(ylim=(0,1.05),ylabel='Direction agreement',title='960 single substitutions; 64 parents')
    mutation=save_figure('09_mutation_validation',fig,[BASE+'/mutation/mutation_summary.csv',BASE+'/mutation/direction_summary.csv'])
    fig,ax=plt.subplots(figsize=(10.5,3.2))
    for i,kind in enumerate(['kmer','frozen_cnn']):
        rr=[next(r for r in transfer if r['condition']==f'few_shot_{kind}_{n}_groups') for n in [20,50,100]]
        ax.errorbar([20,50,100],[float(r['spearman_mean']) for r in rr],yerr=[float(r['spearman_sd']) for r in rr],marker='o',capsize=4,color=COLORS[i],label=('k-mer' if i==0 else 'CNN')+' few-shot mean +/- SD')
    for i,key in enumerate(['zero_shot_ridge','zero_shot_cnn']):
        rr=next(r for r in transfer if r['condition']==key);ax.axhline(float(rr['spearman_mean']),color=COLORS[i],ls='--',alpha=.7,label=('Ridge' if i==0 else 'CNN')+' zero-shot')
    ax.set(xlabel='Target training sequence groups',ylabel='Target test Spearman',title='Five fixed training-group draws')
    ax.set_xticks([20,50,100]);ax.legend(fontsize=8,loc='lower right',ncol=2)
    cross_species=save_figure('10_cross_species',fig,[REV+'/transfer_stability/summary.csv'])

    # Group errors use the current heads and score each CNN seed separately.
    thresholds=np.quantile(ytrain,[1/3,2/3]);val_band=np.digitize(yval,thresholds)
    def predictions(name):
        records=table_file(name);lookup={r['sample_id']:r for r in records if r['subset']=='val'}
        assert set(lookup)=={r['sample_id'] for r in val}
        return np.array([float(lookup[r['sample_id']].get('predicted_log10',lookup[r['sample_id']].get('predicted_value_log10'))) for r in val])
    pred_ridge=predictions(REV+'/regularization/kmer_ridge/predictions.csv')
    pred_cnn=[predictions(BASE+f'/cnn/C0_s{s}/predictions_val.csv') for s in SEEDS]
    pred_pc=predictions('runs/ecoli87_thermo_20261013_v1/predictions_val.csv')
    errors=[]
    for i,name in enumerate(['低强度','中强度','高强度']):
        mask=val_band==i
        errors.append(dict(group=name,n=int(mask.sum()),cnn_mae=float(np.mean([np.abs(p[mask]-yval[mask]).mean() for p in pred_cnn])),
                           ridge_mae=float(np.abs(pred_ridge[mask]-yval[mask]).mean()),pc_mae=float(np.abs(pred_pc[mask]-yval[mask]).mean()),
                           mean_mae=float(np.abs(ytrain.mean()-yval[mask]).mean())))
    spacing=table_file('runs/ecoli87_m3_completion_20261013_v1/source_tables/spacing_correlations.csv')
    spacing_rho=next(r['spearman'] for r in spacing if r['subset']=='train')
    counts=table_file('runs/ecoli87_m3_completion_20261013_v1/source_tables/spacing_summary.csv')
    common=next(r for r in counts if r['subset']=='train' and r['spacer_length']=='17')
    sections=[];table_index=0;figure_index=0
    def section(title,paragraphs=(),new_page=True):
        s=dict(title=title,new_page=new_page,blocks=[dict(type='paragraph',text=p) for p in paragraphs]);sections.append(s);return s
    def paragraph(s,text):s['blocks'].append(dict(type='paragraph',text=text))
    def subheading(s,text):s['blocks'].append(dict(type='subheading',text=text))
    def add_table(s,title,headers,data,widths=None):
        nonlocal table_index
        table_index+=1;s['blocks'].append(dict(type='table',number=table_index,caption=title,headers=headers,rows=data,widths=widths));return table_index
    def figure(s,title,path,caption):
        nonlocal figure_index
        figure_index+=1;s['blocks'].append(dict(type='figure',number=figure_index,title=title,path=path,caption=caption));return figure_index

    s=section('摘要',[
        '本研究以11,809条TSS位置经核验的大肠杆菌87 bp序列为主数据，比较传统回归、卷积神经网络、冻结DNA预训练表征和Promoter Calculator对启动子相关表达的预测能力，并通过位点归因、计算突变和跨物种迁移分析模型的解释性。回归目标为原始表达关联值的log10变换。',
        '在共同的1,768条验证序列上，纯CNN的三种子平均R²为0.1160，Spearman为0.3077；扩大回归头正则化范围后，k-mer与CNN表征融合的R²为0.1174，继续加入DNA表征和归因区域特征后的R²分别为0.1100和0.1144。重新训练的基因组区域分组对照中，纯CNN和k-mer与CNN表征融合的R²分别为0.0993和0.1162。',
        'IG与ISM均显示−10区域的描述性富集，且该区域现象在三个CNN种子下保持一致；精确位点的跨种子稳定性较低。高归因位点的等数量替换产生更大的预测变化，梯度与CNN单点替换方向一致率为98.23%，反映模型内部的局部自洽性。枯草杆菌100组few-shot冻结CNN在五次训练抽样中的Spearman为0.2075±0.0162，未呈现相对zero-shot CNN的稳定收益。'],new_page=False)
    paragraph(s,'关键词  启动子强度  卷积神经网络  热力学模型  位点归因  跨物种迁移')
    subheading(s,'研究问题')
    paragraph(s,'RQ1比较数据驱动模型与热力学模型的预测表现；RQ2检验重要位点与−10、−35、spacer和UP区域的关系；RQ3检验高贡献位点替换对预测强度及方向的影响。')

    s=section('1 数据来源与研究设计',[
        '主数据取自Wang等发布的Deep_promoter仓库[1]。其天然序列预测任务以公开dRNA-seq转录组数据为基础，原始TSS图谱来自Thomason等[2]。本研究使用仓库中的50 bp序列及其表达标签，经历史参考基因组U00096.2双链精确定位并与原TSS表核对，恢复为覆盖TSS上游67 bp及下游20 bp的87 bp上下文。',
        '原始11,884条记录中，11,809条参考位置唯一，75条多位置命中记录单独排除。上下文恢复保留原标签和继承划分。天然表达关联值还受到细胞状态、转录后过程等因素影响，与孤立启动子的纯转录起始率属于不同观测量；保留文件尚不足以确定每条标签的完整单位及测量条件。',
        '枯草杆菌数据由课程教师提供，共591条61 bp记录、580种不同序列，用作跨物种目标域。重复序列按组划分。文件未附论文、强度单位、TSS和方向信息，迁移评价以Spearman排序相关为主。'])
    add_table(s,'数据集与划分',['数据','序列长度','训练 开发或验证 评价'],[
        ['大肠杆菌继承划分','87 bp','8,267 / 1,768 / 1,774'],
        ['大肠杆菌区域分组','87 bp','5,985 / 2,003 / 1,998'],
        ['枯草杆菌序列分组','61 bp','354 / 119 / 118']],widths=[1.8,1.1,3.8])
    paragraph(s,'源域继承划分中的test不参与本研究的模型计分；区域分组仅使用原train与val，并排除49条跨区域边界窗口。历史源域test与目标域test曾在项目早期分析中使用，故所有结果均按内部基准评价解释。')
    add_table(s,'固定分析窗口与TSS坐标',['区域','零基区间','相对TSS位置'],[
        ['UP','[7, 31)','−60至−37'],['−35盒','[32, 38)','−35至−30'],['spacer','[38, 55)','−29至−13'],
        ['−10盒','[55, 61)','−12至−7'],['discriminator','[61, 67)','−6至−1'],['初始转录区','[67, 87)','+1至+20']],widths=[1.8,2.0,2.9])
    paragraph(s,'序列索引67对应TSS，生物相对坐标跳过0。固定窗口用于统一比较；逐序列盒区另由Promoter Calculator推定。两类窗口均保留各自的假设与模型来源。')

    s=section('2 强度分布与区域特征',[
        '描述统计仅使用8,267条训练序列。原始表达值具有较长右尾，模型在log10尺度拟合和评价。CNN目标归一化的最小值与最大值由训练集确定；验证数据保留实际范围。',
        '−10窗口的平均GC比例较低，体现A/T富集；但单个盒区的共识匹配度与表达强度的单调关系较弱。区域GC、共识相似度和spacer长度分别描述不同性质，不能据某一指标的弱相关排除组合效应。'])
    add_table(s,'训练集序列特征与表达强度',['特征','平均特征值','Spearman'],[[r['feature'],number(r['mean']),number(r['spearman'])] for r in eda],widths=[2.8,1.9,2.0])
    figure(s,'训练集表达分布及区域特征',distribution,'左图为log10表达分布；右图为固定窗口特征与log10表达的Spearman相关。共识匹配采用TATAAT和TTGACA，统计单位为训练序列。')
    paragraph(s,f"热力学推定的spacer以17 bp最常见，训练集有{int(common['n']):,}条，占{100*int(common['n'])/len(train):.2f}%；推定长度与表达的Spearman为{number(spacing_rho)}。这些结果支持结合多个局部特征建模，未显示仅靠spacer长度即可有效排序。")

    s=section('3 预测模型与评价方法',[
        'KNN使用理化特征和距离加权回归；Ridge与LinearSVR使用按有效窗口数归一化的k-mer频率。StandardScaler仅拟合训练样本。回归头以验证集log10 MSE选择参数。Ridge正则化范围为1、10、100、1,000、10,000、100,000和1,000,000；扩大范围的对照保持输入表征和样本不变。',
        'CNN使用ACGT独热编码、两层一维卷积及自适应平均池化，再经全连接层输出。四组配置分别改变池化长度、dropout与weight decay，以三个训练种子的平均验证误差选择C0。代表模型固定采用种子20260928，用于位点解释和突变预测；性能统计保留全部三个种子。',
        'DNA表征采用冻结的DNABERT-2-117M[3]。完整编码87 bp序列，取最后一层有效序列token的均值，排除padding和特殊token，再拟合Ridge头。编码器不使用本项目标签更新参数。',
        'Promoter Calculator描述σ70启动子的转录起始热力学[4]。在已核验TSS处枚举spacer 15至20 bp及discriminator 6至10 bp的30种状态，选择最小自由能状态。原始Tx_rate经训练集拟合的log10线性关系映射到本数据的表达尺度。'])
    add_table(s,'模型设置',['模块','输入或结构','拟合与选择'],[
        ['KNN','理化特征 距离加权','邻居数21 101 501 1501'],
        ['Ridge与LinearSVR','k-mer频率 训练集标准化','Ridge扩展alpha；LinearSVR C为0.1 1 10'],
        ['CNN C0','通道32 64；卷积核7 5；池化长度5；隐藏层32','dropout 0.3；weight decay 0.001'],
        ['CNN训练','AdamW；学习率0.001；batch 128','最多100轮；patience 15；三个种子'],
        ['DNA回归','冻结768维表征与Ridge','训练集拟合；验证集选择alpha'],
        ['热力学','固定TSS与逐序列最优状态','训练集拟合尺度校准']],widths=[1.3,2.6,2.8])
    paragraph(s,'R²、MAE和RMSE均在log10表达尺度计算；Spearman评价排序。每个训练种子先独立计分，再报告均值及样本标准差，不将三份预测作为独立样本。置信区间按母序列、目标序列组或基因组区域重采样，统计单位随实验确定。')

    s=section('4 原划分上的预测性能',[
        '所有方法在同一1,768条验证序列上计分，覆盖率为100%。Ridge与DNA回归使用扩展正则化范围后的结果，KNN、LinearSVR、热力学及纯CNN保留相同训练设计。验证集承担模型选择，因此该组成绩反映内部基准上的描述性比较。'])
    add_table(s,'共同验证集预测性能',['模型','R²','Spearman','RMSE','MAE'],[[names[r['method']],mean_sd(r,'r2_log10'),mean_sd(r,'spearman'),mean_sd(r,'rmse_log10'),mean_sd(r,'mae_log10')] for r in main],widths=[1.65,1.25,1.25,1.25,1.3])
    paragraph(s,'CNN项显示三个种子的均值±样本标准差，其余为单次确定性拟合。训练均值预测的Spearman未定义。纯CNN的平均R²为0.1160，说明其具有一定预测能力，但表达方差的解释比例仍有限。')
    figure(s,'共同验证集模型比较',performance,'误差线仅表示CNN训练种子标准差；常数预测的排序相关留空。各方法采用相同的87 bp输入范围与log10评价尺度。')
    paragraph(s,'热力学模型的R²为0.0136，弱于本数据上的CNN与k-mer Ridge。其机制目标是转录起始，而标签来自天然表达观测；二者差异可能影响拟合，但本实验未分离测量噪声、上下文长度和模型容量各自造成的误差。')

    s=section('5 正则化与表征消融',[
        '表征按A0至A3逐步增加：A0为k-mer，A1加入CNN池化表征，A2加入冻结DNA表征，A3加入归因选择区域的k-mer、GC及适用盒区的共识匹配特征。全部阶段使用训练集StandardScaler与Ridge头，A1至A3按三个CNN种子配对。',
        '每个种子的归因区域由预定256条训练序列的IG按每碱基平均绝对贡献选择前两个区域；区域配方冻结后应用于验证序列。A3维度随所选区域是否含盒区共识特征略有不同。'])
    ablation_rows=[['A0 k-mer','256',number(refined['kmer_ridge']['r2_log10_mean']),number(refined['kmer_ridge']['spearman_mean']),number(refined['kmer_ridge']['rmse_log10_mean'])]]+[
        [label,dim,number(refined[key]['r2_log10_mean']),number(refined[key]['spearman_mean']),number(refined[key]['rmse_log10_mean'])]
        for key,label,dim in [('A1','A1 + CNN','576'),('A2','A2 + DNA','1,344'),('A3','A3 + 归因','1,475至1,476')]]
    add_table(s,'逐阶段表征消融',['阶段','特征维度','R²均值','Spearman均值','RMSE均值'],ablation_rows,widths=[1.4,1.25,1.35,1.35,1.35])
    figure(s,'正则化范围与表征消融',ablation,'左图比较原范围与扩展范围，深色误差线为三个种子的样本标准差；右图显示完整alpha搜索，A3阴影为种子标准差。所有输入表征、训练样本及CNN种子保持一致。')
    paragraph(s,'各源域回归头均在内部候选alpha=10,000处取得最低验证误差。A3的平均R²由−0.0273提高至0.1144，表明有限正则化范围明显影响了旧设置下的融合表现。修订后的A2仍低于A1，A3有所回升但未超过A1；该点估计未提供DNA或归因特征稳定增益的证据。')

    s=section('6 基因组区域分组验证',[
        '为检验相邻上下文跨集合的影响，将原train与val按20 kb基因组区域分组。跨区域边界的87 bp窗口被排除，相同序列及其反向互补副本绑定分组。新train、dev和evaluation分别为5,985、2,003和1,998条，任意两集合之间的双链基因组窗口重叠均为0。',
        '归一化、scaler、热力学校准、CNN和归因选区均按新train重新拟合。CNN固定C0结构，早停和回归头选择只使用dev；evaluation标签不参与这些选择。冻结的外部预训练DNA表征可复用。'])
    add_table(s,'区域分组评价性能',['模型','R²均值','种子SD','Spearman均值'],[[names.get(r['method'],r['method']),number(r['r2_log10_mean']),number(r['r2_log10_sd']),number(r['spearman_mean'])] for r in group],widths=[2.2,1.5,1.5,1.5])
    figure(s,'区域分组评价中的模型表现',grouped,'CNN和消融阶段分别报告三个种子的均值及标准差。evaluation与原验证集的组成不同，二者分数差不能直接解释为窗口重叠造成的偏差。')
    paragraph(s,'A1在区域分组对照中的平均R²为0.1162，k-mer Ridge为0.0868，纯CNN为0.0993。整体表现仍处于较低解释度范围。该对照是同一基因组上的一次内部空间划分，历史标签也曾用于项目开发；其结论范围限于此设计。')

    s=section('7 训练稳定性与分组误差',[
        '三个区域分组CNN均在较早阶段取得较低开发集误差，训练误差随后继续下降，开发集误差开始波动或回升。早停依据dev而非evaluation，延长训练不能据此视为可靠的性能改进。',
        '为比较不同强度范围的误差，按原训练集log10强度的三分位阈值划分验证样本。下表使用修订后的Ridge头、原纯CNN和热力学预测；CNN先按种子计算组内MAE，再取平均。'])
    add_table(s,'原验证集不同强度组的MAE',['强度组','样本数','CNN','Ridge','热力学','均值基线'],[[r['group'],r['n'],number(r['cnn_mae']),number(r['ridge_mae']),number(r['pc_mae']),number(r['mean_mae'])] for r in errors],widths=[1.1,.7,1.225,1.225,1.225,1.225])
    figure(s,'区域分组CNN学习曲线',learning,'纵轴为新train最小最大归一化尺度上的MSE。不同种子的最佳轮次分别依据开发集误差确定。')
    paragraph(s,'低强度和高强度组中，CNN的MAE低于Ridge及热力学。中强度组的热力学MAE为0.1723，低于CNN的0.2003，但训练均值基线更低，为0.1644；该区间的低误差可受预测向中部收缩影响。组内R²受较窄标签方差影响，故条件比较同时呈现MAE及常数参照。')

    s=section('8 重要位点与启动子区域',[
        '对固定64条验证母序列，以均匀ACGT概率0.25为基线计算64步梯形积分的Integrated Gradients[5]，并以每个位点三种替换造成的平均绝对log10变化定义ISM。按绝对重要性选择Top-5和Top-10，同分时按位置升序。',
        'Top-k区域命中比例减去区域长度/87，得到相对均匀随机位置的描述性富集。固定窗口和PC推定窗口分别统计。三个种子在每条母序列内先平均，再按64条母序列进行bootstrap。'])
    selected_rows=[r for r in region if r['window_policy']=='fixed' and r['top_k']=='5']
    add_table(s,'固定窗口Top-5区域富集',['方法','区域','平均富集','95%区间','正向种子数'],[
        ['IG' if r['method']=='integrated_gradients' else 'ISM',r['region'],number(r['mean_excess_fraction']),number(r['ci_low'])+' 至 '+number(r['ci_high']),r['positive_seed_count']+'/3'] for r in selected_rows],widths=[.75,1.0,1.3,2.6,1.1])
    figure(s,'三个CNN种子的区域归因',regional,'图中为固定窗口Top-5富集。两种方法在三个种子下均关注−10区域；−35、UP和spacer未显示同样一致的正向模式。')
    paragraph(s,'在IG和ISM、固定及PC推定窗口、Top-5及Top-10的所有组合中，−10区域的三个种子平均富集均为正。该现象与核心启动子区域的生物意义相容；均匀位置参照仍受TSS对齐和模型位置偏好的限制，区间按描述性结果报告。')

    s=section('9 归因方法与跨种子一致性',[
        '同一模型中IG与ISM的相关衡量归因定义之间的一致性；不同CNN种子之间的比较则衡量拟合随机性引起的位置变化。两者使用相同的87个位点和64条母序列，并分别计算排序Spearman及Top-k集合的Jaccard。'])
    within=lambda metric,k:next(r['mean'] for r in consistency if r['metric']==metric and r['top_k']==str(k))
    between=lambda method,metric,k:next(r['mean'] for r in stability if r['method']==method and r['metric']==metric and r['top_k']==str(k))
    add_table(s,'归因一致性与位置稳定性',['比较','位点排序相关','Top-5 Jaccard','Top-10 Jaccard'],[
        ['同一CNN的IG与ISM',number(within('spearman',5)),number(within('top_k_jaccard',5)),number(within('top_k_jaccard',10))],
        ['跨种子IG',number(between('integrated_gradients','spearman',5)),number(between('integrated_gradients','top_k_jaccard',5)),number(between('integrated_gradients','top_k_jaccard',10))],
        ['跨种子ISM',number(between('ism_mean_absolute','spearman',5)),number(between('ism_mean_absolute','top_k_jaccard',5)),number(between('ism_mean_absolute','top_k_jaccard',10))]],widths=[2.1,1.5,1.55,1.55])
    figure(s,'不同种子的重要性位置分布',position_map,'每条序列的绝对重要性先按全长总量归一化，再对64条母序列平均。颜色反映相对重要性比例，IG与ISM各自归一化，未直接比较两种方法的原始分数幅度。')
    paragraph(s,'跨种子Top-5 Jaccard仅约0.20至0.25，说明具体位点排序受模型训练随机性影响。−10区域的一致富集与精确碱基的不稳定可以同时存在，区域层面的证据强于单次Top-k清单。IG沿基线积分，ISM评价离散替换，方法差异也会改变局部排序。')
    completeness=table_file(REV+'/attribution_stability/ig_completeness.csv')
    paragraph(s,f"三个种子的IG最大绝对completeness误差为{max(abs(float(r['completeness_error_log10'])) for r in completeness):.6f} log10。该数值检验积分近似是否满足输出差分关系，用于归因计算的数值质量控制。")

    s=section('10 计算突变与方向检验',[
        '对每条母序列分别选择IG贡献最高5位、最低5位，以及从两组之外随机选择5位，构成High、Low和Random组。各组重复10次，每次均替换为其他碱基，共1,920条等数量变体。设计在预测前保存；统计先在母序列内汇总，再按母序列重采样。'])
    add_table(s,'等数量替换的预测幅度',['组','变体数','平均绝对变化','95%母序列区间'],[[r['group'],r['n_mutants'],number(r['mean_absolute_delta_log10']),number(r['ci_low'])+' 至 '+number(r['ci_high'])] for r in amplitude],widths=[1.2,1.1,1.7,2.7])
    paragraph(s,'High−Random的平均绝对变化差为0.0600 log10，95%区间为0.0367至0.0832；High−Low差为0.0795，区间为0.0563至0.1041。高归因位置对该CNN输出更敏感。High组的平均有符号变化接近0，随机改变重要位置并不预设增强或减弱。')
    paragraph(s,'方向检验对每条母序列的High 5位穷举其他3种碱基，共960条单点变体。预期方向来自野生型梯度的替换通道减原通道，随后与实际CNN变化及重新枚举状态后的热力学变化比较；近零阈值为10⁻⁴ log10。')
    add_table(s,'单点变化方向一致率',['比较','一致率','95%母序列聚类区间'],[[label,f"{100*float(direction[key]):.2f}%",f"{100*float(direction[key+'_ci_low']):.2f}% 至 {100*float(direction[key+'_ci_high']):.2f}%"] for key,label in [('cnn_agreement','梯度预期与CNN'),('thermo_agreement','梯度预期与热力学'),('cross_model_agreement','CNN与热力学')]],widths=[2.6,1.2,2.9])
    figure(s,'计算突变的幅度与方向',mutation,'左图为等数量替换的绝对效应；右图为单点方向一致率。两图区间均以64条母序列为重采样单位，960条替换未作为独立生物测量。')
    paragraph(s,'梯度与CNN的98.23%一致率提供局部线性近似与有限替换的自洽性证据；CNN与热力学的一致率为44.27%，显示两个预测函数存在明显分歧。变体缺少实测表达标签，方向一致性仍属于计算验证。')

    s=section('11 跨物种迁移与训练抽样',[
        '目标域保留354/119/118行的序列分组划分，相同61 bp序列不跨集合。zero-shot直接应用源域模型；few-shot固定源表征，在目标train拟合回归头，并在目标val选择alpha。k-mer采用窗口频率，CNN用相同源权重实例化61 bp自适应池化网络。',
        '采用五个固定训练组排列，每个排列内20、50和100组嵌套，完整保留30项few-shot拟合及2项zero-shot。跨抽样标准差衡量训练组选择的波动；每次抽样的目标test序列组bootstrap区间单独保存。'])
    def transfer_label(condition):
        if condition=='zero_shot_ridge':return 'Ridge zero-shot'
        if condition=='zero_shot_cnn':return 'CNN zero-shot'
        return ('CNN' if 'frozen_cnn' in condition else 'k-mer')+' few-shot'
    add_table(s,'目标test迁移结果',['条件','训练组数','抽样次数','Spearman','R²均值'],[[transfer_label(r['condition']),r['n_train_groups'],r['n_draws'],mean_sd(r,'spearman'),number(r['r2_log10_mean'])] for r in transfer],widths=[1.6,1.05,1.05,1.7,1.3])
    figure(s,'跨物种训练组抽样稳定性',cross_species,'误差线为五次训练组抽样的样本标准差。虚线为zero-shot结果。目标域test共118行，已在项目历史分析中使用。')
    paragraph(s,'100组few-shot冻结CNN的平均Spearman为0.2075，接近zero-shot CNN的0.2161；20组时波动较大。旧设置的一次抽样约0.263，而五次抽样同时采用了扩展alpha范围，故两组差异包含抽样与正则化变化。当前设计未显示稳定的迁移增益。目标单位未知，跨域zero-shot的绝对尺度误差仅作诊断。')

    s=section('12 讨论与结论')
    subheading(s,'RQ1 预测性能与模型适用范围')
    paragraph(s,'在天然表达关联标签、相同87 bp输入和共同验证样本下，CNN与k-mer回归的点估计优于校准热力学模型。区域分组评价中这种总体差异仍存在，但所有模型的解释度有限。k-mer与CNN表征融合的表现较好，继续增加DNA表征和归因区域特征未形成稳定的递增收益。')
    paragraph(s,'分强度观察时，CNN在低、高强度组的MAE较低；热力学在中强度组低于CNN，但仍高于常数基线。因此，中强度区间的结果尚不足以证明热力学机制解释更准确，模型比较需要同时考虑全体样本与条件基线。')
    paragraph(s,'正则化对照表明，高维表征的回归头设置会改变结论。扩大alpha后，融合R²由负值回到约0.11；这一变化应归于固定表征下的估计方式调整。对预训练模型的评价还受冻结编码、池化方式与天然表达任务的匹配程度影响，本实验仅覆盖一种实现。')
    subheading(s,'RQ2 核心区域与精确位点')
    paragraph(s,'−10在两种归因定义、两类窗口及三个CNN种子下均出现区域平均富集；−35、UP和spacer缺少同样一致的模式。位点层面的跨种子Jaccard较低，因此可支持模型学习了部分核心区域信号，但对某一个碱基的生物作用仍应保持不确定性。')
    subheading(s,'RQ3 计算敏感性与突变指导')
    paragraph(s,'高归因位置的等数量替换带来更大的CNN输出变化，一阶梯度较好地预测该模型的单点变化方向。跨模型方向分歧和缺少实测变体标签限制了因果解释。归因适合用于筛选候选突变，真实增强或减弱效应需要配对表达测量验证。')
    subheading(s,'研究限制')
    paragraph(s,'源域标签来源于天然转录组，目标域缺少完整测量元数据；不同实验标尺之间的可比性尚不明确。固定与工具推定盒区没有逐条结合位点实验注释，均匀随机位置也未消除所有位置背景差异。区域富集与多个模型对照的区间均按描述性结果解释，未作多重比较校正。')
    paragraph(s,'原验证集参与选参，历史源域与目标域test也曾被观察；空间分组是单次内部对照。训练种子标准差、母序列bootstrap和训练抽样标准差分别反映条件性波动，均未覆盖跨实验批次或外部新基因组的泛化不确定性。')
    subheading(s,'结论')
    paragraph(s,'本研究在统一数据口径下完成性能比较、区域归因、逐阶段消融与计算突变检验，并通过跨物种和多归因方法分析扩展解释。结果支持正则化与区域稳定性检查对启动子建模的重要性；预测精度、精确位点一致性和跨域收益仍有明显限制。')

    references=[
        ('Wang Y, Wang H, Wei L, Li S, Liu L, Wang X. Synthetic promoter design in Escherichia coli based on a deep generative network. Nucleic Acids Research. 2020;48(12):6403–6412.','https://doi.org/10.1093/nar/gkaa325'),
        ('Thomason MK, Bischler T, Eisenbart SK, et al. Global transcriptional start site mapping using differential RNA sequencing reveals novel antisense RNAs in Escherichia coli. Journal of Bacteriology. 2015;197(1):18–28.','https://doi.org/10.1128/JB.02096-14'),
        ('Zhou Z, Ji Y, Li W, Dutta P, Davuluri R, Liu H. DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome. ICLR 2024.','https://arxiv.org/abs/2306.15006'),
        ('LaFleur TL, Hossain A, Salis HM. Automated model-predictive design of synthetic promoters to control transcriptional profiles in bacteria. Nature Communications. 2022;13:5159.','https://doi.org/10.1038/s41467-022-32829-5'),
        ('Sundararajan M, Taly A, Yan Q. Axiomatic Attribution for Deep Networks. Proceedings of Machine Learning Research. 2017;70:3319–3328.','https://proceedings.mlr.press/v70/sundararajan17a.html'),
        ('HaochenW. Deep_promoter 源数据与代码仓库.','https://github.com/HaochenW/Deep_promoter'),
        ('PR02-01 项目代码与实验记录.','https://github.com/Azusakada/PR02-01-promoter--insight')]
    s=section('参考文献')
    for i,(text,url) in enumerate(references,1):s['blocks'].append(dict(type='reference',number=i,text=text,url=url))
    paragraph(s,'课程要求来源为project-pool(2).pdf中的PR02-01条目；枯草杆菌数据来源为教师提供的Bacillus-subtilis-promoter-strength.csv。')

    s=section('附录 复现说明与结果索引',[
        '基础依赖固定于requirements.txt。已交付DNA特征可直接用于数值核验；重新编码新序列时使用固定版本DNABERT-2权重。模型、预测、拟合ID、协议和源表以文件摘要关联。'])
    s['blocks'].append(dict(type='code',text='python -m pip install -r requirements.txt\npython -m pr02 m4-refine --stage verify\npython -m pr02 m4 --stage verify'))
    paragraph(s,'两个验证命令分别核验修订实验与原始归因、突变等记录。核验包括模型重载、训练集预处理来源、预测与指标复算、区域分组和文件摘要。')
    add_table(s,'实验与结果目录',['实验','目录'],[
        ['原划分性能与正则化','R1/regularization/'],['区域分组控制','R1/grouped/'],['三种子归因稳定性','R1/attribution_stability/'],
        ['跨物种五次训练抽样','R1/transfer_stability/'],['IG与ISM一致性','R0/explanation_analysis/'],['计算突变与方向','R0/mutation/'],
        ['逐条数据与坐标','data/03_Ecoli_strength87/']],widths=[2.3,4.4])
    paragraph(s,'R0 = '+BASE+'；R1 = '+REV+'。训练表征、模型配置和逐种子结果分别保留在相应目录中。')
    paragraph(s,'报告文字与表格由scripts/build_course_report.py从源表生成，Word由scripts/build_course_docx.py排版。report_manifest.json保存报告、图表和源文件摘要。')

    checks=dict(source_counts=dict(train=len(train),val=len(val),test=sum(r['split']=='test' for r in rows)),
        target_rows=len(target),target_unique_sequences=len({r['sequence'] for r in target}),
        grouped_counts=split['counts'],zero_cross_subset_overlaps=not any(split['cross_subset_window_overlaps'].values()),
        all_refined_heads_selected_alpha_10000=True,main_scores=[dict(method=r['method'],r2=float(r['r2_log10_mean'])) for r in main],
        group_mae=errors,eda=eda,source_test_scored=False)
    content=dict(title='启动子强度预测与关键位点归因',subtitle='PR02-01 课程项目研究报告',date='2026年10月14日',
        members=['胡昊铭','李宇飞','李玘航','田惠今','陈思远'],sections=sections,references=[dict(text=t,url=u) for t,u in references],
        source_hashes=SOURCES,figures=FIGURES,checks=checks)
    (output/'report_content.json').write_text(json.dumps(content,ensure_ascii=False,indent=2)+'\n')
    lines=['# '+content['title'],'',content['subtitle'],'',content['date'],'','项目组 '+'、'.join(content['members']),'']
    for s in sections:
        lines+=['## '+s['title'],'']
        for b in s['blocks']:
            if b['type']=='paragraph':lines+=[b['text'],'']
            elif b['type']=='subheading':lines+=['### '+b['text'],'']
            elif b['type']=='table':
                lines+=[f"表{b['number']} {b['caption']}",'','| '+' | '.join(b['headers'])+' |','| '+' | '.join(['---']*len(b['headers']))+' |']
                lines+=['| '+' | '.join(str(c).replace('|','/') for c in row)+' |' for row in b['rows']];lines+=['']
            elif b['type']=='figure':lines+=[f"![图{b['number']} {b['title']}]({b['path']})",'',f"图{b['number']} {b['title']}。{b['caption']}",'']
            elif b['type']=='reference':lines+=[f"[{b['number']}] {b['text']} [链接]({b['url']})",'']
            elif b['type']=='code':lines+=['```bash',b['text'],'```','']
    (output/'PR02-01_REPORT.md').write_text('\n'.join(lines))
    print(output/'PR02-01_REPORT.md')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'reports/final')
    build(parser.parse_args().output)
