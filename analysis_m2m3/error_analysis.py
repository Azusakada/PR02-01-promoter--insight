"""M3 validation-only diagnostics with full request coverage and common IDs."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from common import DATASET, ROOT, SAMPLES, SCHEMA, SEED, SPLITS, Run, load_data, require, sequence_features, single, write_json

PRED_FIELDS = {'dataset_id','sample_id','true_value','predicted_value','predicted_value_log10',
               'method_name','target_scale_model','model_checkpoint','predicted_tx_rate','thermo_mode',
               'calibration_id','run_id','split_id','subset','seed','prediction_status','error_reason',
               'predicted_value_normalized','label_transform_id','schema_version','data_version'}


def checked_predictions(path, data, subset='val', require_coverage=True):
    pred = pd.read_csv(path)
    require(PRED_FIELDS <= set(pred), f'{path}: missing {PRED_FIELDS-set(pred)}')
    require(pred.sample_id.notna().all() and pred.sample_id.is_unique, f'{path}: duplicate/missing IDs')
    require(single(pred, 'subset') == subset, f'{path}: expected only {subset}')
    for col in ['dataset_id','data_version','split_id','schema_version']:
        require(single(pred,col) == single(data,col), f'{path}: {col} mismatch')
    for col in ['method_name','run_id','target_scale_model']:
        single(pred,col)
    expected = set(data.loc[data['split'].eq(subset), 'sample_id'])
    require(set(pred.sample_id) <= expected, f'{path}: unknown or wrong-subset IDs')
    if require_coverage:
        require(set(pred.sample_id) == expected, f'{path}: missing requested IDs; supply raw request table including failures')
    require(pred.prediction_status.isin(['ok','failed']).all(), f'{path}: invalid statuses')
    failed = pred.prediction_status.eq('failed')
    require(pred.loc[failed,'error_reason'].notna().all(), f'{path}: failed row needs reason')
    values = ['predicted_value','predicted_value_log10','predicted_tx_rate','predicted_value_normalized']
    require(pred.loc[failed,values].isna().all().all(), f'{path}: failure contains prediction numbers')
    canonical = data.set_index('sample_id').loc[pred.sample_id]
    require(np.allclose(pred.true_value,canonical.strength,rtol=1e-8,atol=1e-8), f'{path}: true label mismatch')
    ok = ~failed
    scale = single(pred,'target_scale_model')
    require(scale in ['log10','log10_minmax','tool_raw'], f'{path}: unknown target scale')
    if scale == 'tool_raw':
        require(pred.loc[ok,'predicted_tx_rate'].gt(0).all() and np.isfinite(pred.loc[ok,'predicted_tx_rate']).all(), 'Invalid Tx_rate')
        has_cal = pred.loc[ok,'calibration_id'].notna()
        require(has_cal.all() or (~has_cal).all(), 'Mixed calibration states')
        if not has_cal.any():
            require(pred.loc[ok,['predicted_value','predicted_value_log10']].isna().all().all(), 'Uncalibrated Tx_rate cannot be strength prediction')
    else:
        require(pred.loc[ok,'model_checkpoint'].notna().all(), 'Missing checkpoint reference')
    return pred


def attach_calibration(raw, calibrated_path, calibration_path, train_path, data):
    cal = checked_predictions(calibrated_path,data,require_coverage=False)
    meta = json.loads(Path(calibration_path).read_text())
    train = checked_predictions(train_path,data,'train')
    require(meta.get('fit_subset') == 'train' and meta.get('kind') == 'log10_linear','Calibration is not train-only linear-log')
    for key in ['data_version','schema_version','split_id','run_id']:
        require(meta.get(key) == single(raw,key), f'Calibration {key} mismatch')
        require(single(train,key) == single(raw,key), f'Calibration training {key} mismatch')
    require(single(cal,'method_name') == single(raw,'method_name') and single(cal,'run_id') == single(raw,'run_id'), 'Raw/calibrated method or run mismatch')
    require(single(cal,'calibration_id') == meta.get('calibration_id'), 'Unknown calibration_id')
    expected_ok = set(raw.loc[raw.prediction_status.eq('ok'),'sample_id'])
    require(set(cal.sample_id) == expected_ok and cal.prediction_status.eq('ok').all(), 'Calibrated table does not cover exact successful raw requests')
    tr = train[train.prediction_status.eq('ok')]
    x = np.log10(tr.predicted_tx_rate.to_numpy())
    y = np.log10(tr.true_value.to_numpy())
    coef = np.linalg.lstsq(np.column_stack([x,np.ones(len(x))]),y,rcond=None)[0]
    require(meta.get('n_fit') == len(tr), 'Calibration n_fit mismatch')
    require(np.allclose(coef,[meta['coef_a'],meta['intercept_b']],rtol=1e-7,atol=1e-7), 'Saved calibration not reproduced from train-only data')
    rx = raw.set_index('sample_id').loc[cal.sample_id,'predicted_tx_rate'].to_numpy()
    require(np.allclose(rx,cal.predicted_tx_rate,rtol=1e-8), 'Calibrated Tx_rate mismatch')
    estimated = meta['coef_a']*np.log10(rx)+meta['intercept_b']
    require(np.allclose(estimated,cal.predicted_value_log10,rtol=1e-8,atol=1e-8), 'Calibration output mismatch')
    out = raw.copy().set_index('sample_id')
    for col in ['predicted_value','predicted_value_log10']:
        out.loc[cal.sample_id,col] = cal[col].to_numpy()
    out['calibration_id'] = meta['calibration_id']
    return out.reset_index(), {'calibration_id':meta['calibration_id'],'n_fit':len(tr), 'coef_a_recomputed':float(coef[0]),'intercept_b_recomputed':float(coef[1]),'status':'pass', 'failure_adapter':'success-only calibrated table joined onto full raw request table; failed request retained'}


def metric_value(name,y,p):
    if len(y)==0:
        return None,'undefined','empty_evaluation_set'
    if name == 'r2' and (len(y)<2 or np.ptp(y)==0):
        return None,'undefined','insufficient_or_constant_target'
    if name == 'spearman' and (len(y)<2 or np.ptp(y)==0 or np.ptp(p)==0):
        return None,'undefined','insufficient_or_constant_ranks'
    value = {'r2':r2_score,'mae':mean_absolute_error,'spearman':lambda a,b:spearmanr(a,b).statistic}[name](y,p)
    return float(value),'ok',''


def analyze(output, paths, calibrated=None, calibration=None, calibration_train=None):
    data = load_data()
    frames = [checked_predictions(p,data) for p in paths]
    calibration_qc = None
    inputs = [SAMPLES,SPLITS,*paths]
    if any(x is not None for x in [calibrated,calibration,calibration_train]):
        require(all(x is not None for x in [calibrated,calibration,calibration_train]), 'Provide calibrated, calibration, and calibration-train together')
        matches = [i for i,f in enumerate(frames) if single(f,'target_scale_model')=='tool_raw']
        require(len(matches)==1, 'Calibration option needs exactly one raw thermo method')
        i = matches[0]
        frames[i],calibration_qc = attach_calibration(frames[i],calibrated,calibration,calibration_train,data)
        inputs.extend([calibrated,calibration,calibration_train])
    methods = [single(f,'method_name') for f in frames]
    require(len(set(methods))==len(methods), 'Use one prediction run per method')
    thresholds = data.loc[data['split'].eq('train'),'target_log10'].quantile([1/3,2/3]).to_numpy()
    require(thresholds[0]<thresholds[1], 'Train tertile thresholds coincide')
    requested = set(data.loc[data['split'].eq('val'),'sample_id'])
    success = {m:set(f.loc[f.prediction_status.eq('ok'),'sample_id']) for m,f in zip(methods,frames)}
    common = set.intersection(*success.values())
    run = Run(output,'M3','pr02-error-analysis-m3',data,inputs,{'subset':'val','group_threshold_source':'train log10 tertiles','thresholds_log10':thresholds.tolist(),'case_seed':SEED,'case_rules':'top5 absolute errors per method, top5 pairwise log10 disagreement, 5 random common IDs','required_missing_methods':['kmer Ridge','CNN optional'],'calibration_handling':'verify imported train-only parameters; do not fit on val'})
    run.meta['evaluation_subsets'] = ['val']
    if calibration_qc:
        write_json(run.output/'calibration_qc.json',calibration_qc)
    metrics,used_ids,coverage,errors,groups = [],[],[],[],[]
    for method,frame in zip(methods,frames):
        ok = frame.prediction_status.eq('ok')
        numeric = frame.loc[ok,'predicted_value_log10'].notna().all() and frame.loc[ok,'predicted_value'].notna().all()
        if numeric and ok.any():
            require(np.isfinite(frame.loc[ok,'predicted_value_log10']).all() and frame.loc[ok,'predicted_value'].gt(0).all() and np.isfinite(frame.loc[ok,'predicted_value']).all(), f'{method}: invalid numeric predictions')
            require(np.allclose(10**frame.loc[ok,'predicted_value_log10'],frame.loc[ok,'predicted_value'],rtol=2e-8,atol=1e-8), f'{method}: raw/log10 prediction mismatch')
        elif single(frame,'target_scale_model') != 'tool_raw':
            require(not ok.any(),f'{method}: missing numerical prediction on successful request')
        elif not numeric:
            require(frame.loc[ok,['predicted_value','predicted_value_log10']].isna().all().all(), f'{method}: partially calibrated successful rows')
        coverage.append({'method_name':method,'run_id':single(frame,'run_id'),'subset':'val','n_requested':len(requested),'n_success':int(ok.sum()),'n_failed':int((~ok).sum()),'coverage':int(ok.sum())/len(requested),'n_common':len(common)})
        for set_name,ids in [('all_success',success[method]),('common',common)]:
            cid=f'{run.run_id}_val_{set_name}_{method}'
            subset = frame[frame.sample_id.isin(ids)].sort_values('sample_id')
            used_ids.extend({'comparison_set_id':cid,'method_name':method,'sample_id':sid,'subset':'val'} for sid in subset.sample_id)
            evaluation_scales = [('raw','true_value','predicted_value'),('log10','true_value','predicted_value_log10')] if numeric else []
            if single(frame,'target_scale_model')=='tool_raw':
                evaluation_scales.append(('tool_rank','true_value','predicted_tx_rate'))
            for scale,ycol,pcol in evaluation_scales:
                y=subset[ycol].to_numpy(dtype=float)
                if scale=='log10': y=np.log10(y)
                p=subset[pcol].to_numpy(dtype=float)
                for metric in (['spearman'] if scale=='tool_rank' else ['r2','spearman','mae']):
                    value,status,reason=metric_value(metric,y,p)
                    metrics.append({'dataset_id':DATASET,'run_id':single(frame,'run_id'),'method_name':method,'split_id':single(data,'split_id'),'evaluation_subset':f'val_{set_name}','metric_name':metric,'target_scale':scale,'value':value,'n_requested':len(requested),'n_success':int(ok.sum()),'n_used':len(y),'metric_status':status,'reason':reason,'evidence_level':'preliminary','comparison_set_id':cid,'schema_version':SCHEMA,'data_version':single(data,'data_version')})
        run.table(f'{method}_requested_predictions.csv',frame)
        if numeric:
            e=frame[ok].merge(data[['sample_id','sequence','annotation_status']],on='sample_id',validate='one_to_one')
            e['true_log10']=np.log10(e.true_value)
            e['residual_log10']=e.predicted_value_log10-e.true_log10
            e['abs_error_log10']=e.residual_log10.abs()
            e['strength_group']=np.select([e.true_log10<=thresholds[0],e.true_log10<=thresholds[1]],['low','mid'],default='high')
            e['in_common_set']=e.sample_id.isin(common)
            errors.append(e)
            for set_name,part in [('all_success',e),('common',e[e.in_common_set])]:
                for group in ['low','mid','high']:
                    g=part[part.strength_group.eq(group)]
                    groups.append({'method_name':method,'comparison_set':set_name,'strength_group':group,'n':len(g),'mae_log10':g.abs_error_log10.mean() if len(g) else None,'bias_log10':g.residual_log10.mean() if len(g) else None,'threshold_low':thresholds[0],'threshold_high':thresholds[1]})
    met=pd.DataFrame(metrics)
    metrics_path=run.table('metrics.csv',met)
    run.table('common_eval_ids.tsv',pd.DataFrame(used_ids,columns=['comparison_set_id','method_name','sample_id','subset']),'\t')
    coverage_path=run.table('coverage_summary.csv',pd.DataFrame(coverage))
    groups_df=pd.DataFrame(groups)
    groups_path=run.table('group_errors.csv',groups_df)
    error_df=pd.concat(errors,ignore_index=True) if errors else pd.DataFrame()
    errors_path=run.table('prediction_errors.csv',error_df)
    numeric_methods = sorted(error_df.method_name.unique()) if len(error_df) else []
    if numeric_methods:
        for fig_name,kind in [('m3_prediction_scatter','scatter'),('m3_residuals','residual')]:
            fig,axes=plt.subplots(1,len(numeric_methods),figsize=(6*len(numeric_methods),4.4),squeeze=False)
            for ax,method in zip(axes[0],numeric_methods):
                e=error_df[error_df.method_name.eq(method)&error_df.in_common_set]
                if kind=='scatter':
                    ax.scatter(e.true_log10,e.predicted_value_log10,s=10,alpha=.28,color='#386CB0',edgecolors='none')
                    if len(e):
                        lo=min(e.true_log10.min(),e.predicted_value_log10.min())-.1
                        hi=max(e.true_log10.max(),e.predicted_value_log10.max())+.1
                        ax.plot([lo,hi],[lo,hi],'--',color='#777777',lw=1)
                        ax.set(xlim=(lo,hi),ylim=(lo,hi))
                    ax.set(ylabel='Predicted log10 strength')
                else:
                    ax.scatter(e.true_log10,e.residual_log10,s=10,alpha=.28,color='#6C5B8E',edgecolors='none')
                    ax.axhline(0,color='#777777',ls='--',lw=1)
                    ax.set(ylabel='Residual: predicted - true (log10)')
                ax.set(xlabel='True log10 strength',title=f'{method}\nval common: n={len(e)}')
            run.figure(fig_name,fig,f'Validation {kind}',[errors_path],'val_common','log10',len(common),'Fixed val common IDs. Thermodynamic numerical output is train-only calibrated; no test predictions used.')
        fig,axes=plt.subplots(1,2,figsize=(12,4.2))
        x=np.arange(3);width=.75/len(numeric_methods)
        for i,method in enumerate(numeric_methods):
            g=groups_df[groups_df.method_name.eq(method)&groups_df.comparison_set.eq('common')].set_index('strength_group').reindex(['low','mid','high'])
            pos=x+(i-(len(numeric_methods)-1)/2)*width
            axes[0].bar(pos,g.mae_log10,width,label=method)
            axes[1].bar(pos,g.bias_log10,width,label=method)
        for ax in axes:
            ax.set_xticks(x,['low','mid','high'])
            ax.set_xlabel('Strength group (thresholds fixed from train)')
            ax.legend(frameon=False,fontsize=8)
        axes[0].set(ylabel='MAE (log10)',title='Validation errors by strength group')
        axes[1].set(ylabel='Mean residual (log10)',title='Under / overprediction')
        axes[1].axhline(0,color='gray',lw=.8)
        run.figure('m3_group_errors',fig,'Strength-group errors',[groups_path],'val_common','log10',len(common),'Strength boundaries come from train tertiles, not validation outcomes. Source table records group counts.')
    selected={}
    def select(ids,reason):
        for sid in ids: selected.setdefault(sid,set()).add(reason)
    for method in numeric_methods:
        e=error_df[error_df.method_name.eq(method)].sort_values(['abs_error_log10','sample_id'],ascending=[False,True])
        select(e.head(5).sample_id,f'largest_error:{method}')
    wide=error_df[error_df.in_common_set].pivot(index='sample_id',columns='method_name',values='predicted_value_log10') if len(error_df) else pd.DataFrame()
    differences=[]
    for left,right in combinations(numeric_methods,2):
        gap=(wide[left]-wide[right]).abs().sort_values(ascending=False)
        select(gap.head(5).index,f'model_disagreement:{left}:{right}')
        differences.extend({'sample_id':sid,'left_method':left,'right_method':right,'abs_disagreement_log10':v} for sid,v in gap.items())
    if common:
        select(np.random.default_rng(SEED).choice(sorted(common),size=min(5,len(common)),replace=False),'random_common')
    cases=error_df[error_df.sample_id.isin(selected)].copy() if len(error_df) else pd.DataFrame(columns=['sample_id'])
    cases['case_reason']=cases.sample_id.map(lambda x:';'.join(sorted(selected[x])))
    cases['data_check']='ID, split, label, 50bp sequence checked against immutable main table'
    cases['interpretation_status']='biological_mechanism_unconfirmed'
    cases['extra_observation']='Observed prediction discrepancy; no experimental mechanism established'
    run.table('case_review.csv',cases)
    diff_path=run.table('model_disagreements.csv',pd.DataFrame(differences))
    if len(numeric_methods)==2:
        fig,axes=plt.subplots(1,2,figsize=(11,4.2))
        a,b=numeric_methods
        axes[0].scatter(wide[a],wide[b],s=10,alpha=.25,color='#45A37A',edgecolors='none')
        axes[0].set(xlabel=a+' (log10)',ylabel=b+' (log10)',title=f'Model agreement | common n={len(common)}')
        vals=pd.DataFrame(differences).abs_disagreement_log10
        axes[1].hist(vals,bins=35,color='#E18B35',edgecolor='white')
        axes[1].set(xlabel='Absolute prediction difference (log10)',ylabel='Sample count',title='Disagreement distribution')
        run.figure('m3_model_disagreement',fig,'Model disagreement',[errors_path,diff_path],'val_common','log10',len(common),'Disagreement is a diagnostic, not evidence that either model is biologically correct.')
    numerical=met[(met.target_scale=='log10')&(met.evaluation_subset=='val_common')]
    rows=[]
    for method in numeric_methods:
        m=numerical[numerical.method_name.eq(method)].set_index('metric_name').value
        rows.append(f'| {method} | {len(common)} | {m.get("r2",float("nan")):.6f} | {m.get("spearman",float("nan")):.6f} | {m.get("mae",float("nan")):.6f} |')
    report='''# M3 验证集误差分析

本轮仅读取已有模型的 val 逐样本预测，未训练模型、未使用 test 选参。热力学结果来自固定 teammate commit；原始 Tx_rate 保留独立列，数值比较使用经过 train-only 参数复算验证的校准结果。

## 同一批有效样本上的结果

| 方法 | n_common | log10 R² | Spearman | log10 MAE |
|---|---:|---:|---:|---:|
'''+ '\n'.join(rows)+f'''

## 覆盖和解释

请求 val 样本数 {len(requested)}；共同成功样本数 {len(common)}。各方法的请求、成功、失败数见 source_tables/coverage_summary.csv；精确评价 ID 见 common_eval_ids.tsv。指标均由逐样本预测重新计算，不使用队友汇总表中的全数据分母。

低/中/高边界来自 train log10 strength 的 1/3 和 2/3 分位数：{thresholds[0]:.6f}、{thresholds[1]:.6f}。每组 n、MAE 和偏差见 group_errors.csv。正偏差表示高估，负偏差表示低估；请结合散点与残差图解释均值收缩，不能把预测误差直接归因于某个生物学元件。

case_review.csv 包含每方法最大误差前5、方法分歧前5以及固定种子的5个随机共同样本；可能重叠，保留所有选择理由。所有案例先核对样本、划分、标签、序列，具体机制保持 unconfirmed。难例未删除。

## 当前完成边界

KNN 与热力学的验证集图表、分组误差、覆盖和案例可交接；正式 k-mer Ridge 预测仍未提交，CNN 结果为可选。最低中期所要求的 Ridge+热力学比较仍待 Ridge 交接，不能把本轮标成全组 M3 已完成。

热力学校准表只保存成功记录，本模块用原始完整请求表承载失败记录后再按 ID 合并校准值，未修改队友文件。KNN val 预测来自 train-only 模型，但仓库只保存了 train+val 的 test 模型 checkpoint；不要用该 checkpoint 重新生成 val 预测。当前热力学上下文来自基因组恢复，测量上下文一致性、工具扫描位置和 assay 方向仍需生物学核实。两种模型使用的信息范围不同，应在汇报中说明。
'''
    (run.output/'m3_error_analysis.md').write_text(report,encoding='utf-8')
    run.finish()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,nargs='+',required=True)
    ap.add_argument('--calibrated',type=Path)
    ap.add_argument('--calibration',type=Path)
    ap.add_argument('--calibration-train',type=Path)
    a=ap.parse_args()
    analyze(a.output,a.predictions,a.calibrated,a.calibration,a.calibration_train)


if __name__=='__main__':
    main()
