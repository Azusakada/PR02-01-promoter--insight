"""Reload refined predictors and independently check fit cohorts and score tables."""
from __future__ import annotations
import json
import joblib
import numpy as np
from scipy.stats import spearmanr

from .data import ROOT, TARGET, load, read_json, read_table, require, sha, write_json
from .experiment import cnn_predict, metrics
from .features import encode, kmers, physchem
from .m4_attribution import guided_features, log_model
from .m4_cnn import embedding, load_model
from .m4_data_eval import REGIONS, completed, read_bundle, run_root, seeds, verify_sealed_release
from .m4_grouped_validation import genomic_interval, overlaps, partition
from .m4_refinement import aggregate, baseline
from .research import ranks


def check_aggregate(file, records, key='method'):
    clean=[]
    for record in records:
        row=dict(record);row['n']=int(row['n'])
        for metric in ['r2_log10','spearman','mae_log10','rmse_log10']:
            row[metric]=float(row[metric]) if row[metric] not in ('',None) else None
        clean.append(row)
    actual={r[key]:r for r in read_table(file)};expected=aggregate(clean,key)
    require(set(actual)=={r[key] for r in expected},'Summary methods differ from raw scores')
    for row in expected:
        for name,value in row.items():
            if name==key:continue
            if value is None:require(actual[row[key]][name]=='','Undefined summary value replaced')
            else:require(np.isclose(float(actual[row[key]][name]),value,atol=1e-12),'Summary aggregation mismatch: '+row[key]+'/'+name)


def compare_export(folder, rows, y, prediction, assignment):
    rr=read_table(folder/'predictions.csv'); expected=[(r,a,p,s) for r,a,p,s in zip(rows,y,prediction,assignment) if s in ('train','val','dev','evaluation')]
    require(len(rr)==len(expected) and len({r['sample_id'] for r in rr})==len(rr),'Prediction ID coverage mismatch')
    for actual,(row,target,p,subset) in zip(rr,expected):
        require(actual['sample_id']==row['sample_id'] and actual['subset']==subset,'Prediction cohort mismatch')
        require(np.isclose(float(actual['observed_log10']),target,atol=1e-12),'Prediction label mismatch')
        require(np.isclose(float(actual['predicted_log10']),p,atol=2e-6,rtol=1e-6),'Reloaded prediction mismatch')


def selected_alpha(folder,head,protocol):
    search=read_table(folder/'search.csv')
    require({float(r['alpha']) for r in search}==set(protocol['ridge_alpha']) and len(search)==len(protocol['ridge_alpha']),'Incomplete regularization grid')
    require(float(head[-1].alpha)==float(min(search,key=lambda r:float(r['val_mse_log10']))['alpha']),'Selected head differs from minimum development loss')


def check_score(row,y,p):
    score=metrics(y,p)
    for key,value in score.items():
        if value is None:require(row[key] in (None,''),'Undefined metric replaced with a number')
        else:require(np.isclose(float(row[key]),value,atol=1e-10),'Score mismatch: '+key)


def verify(cfg,protocol,visual=True):
    import torch
    torch.set_num_threads(cfg['num_threads'])
    root=run_root(protocol);old=baseline(protocol);checks=[]
    verify_sealed_release(old);checks.append('original_sealed_outputs_reports_and_ppt_preserved')
    for file in root.rglob('run_manifest.json'):completed(file.parent,cfg,protocol,archived_code=True)
    checks.append('artifact_upstream_config_protocol_and_executed_source_checksums')
    require(read_json(root/'protocol/protocol.json')==protocol,'Protocol identity changed')
    all_rows,_=load(cfg);ids=[r['sample_id'] for r in all_rows];y=np.array([float(r['target_log10']) for r in all_rows]);assignment=np.array([r['split'] for r in all_rows])
    train=assignment=='train';val=assignment=='val';xk=read_bundle(old/'ml/kmer',ids);xd=read_bundle(old/'dna_features',ids)
    representations={'kmer_ridge':xk,'dnabert2_ridge':xd}
    for seed in seeds(protocol):
        xc=read_bundle(old/f'cnn/features_s{seed}',ids);xg=read_bundle(old/f'guided_s{seed}',ids)
        for stage,parts in [('A1',[xk,xc]),('A2',[xk,xc,xd]),('A3',[xk,xc,xd,xg])]:representations[f'{stage}_s{seed}']=np.column_stack(parts)
    refined=read_table(root/'regularization/metrics.csv')
    require(len(refined)==2+3*len(seeds(protocol)),'Missing refined heads')
    for name,x in representations.items():
        folder=root/'regularization'/name;head=joblib.load(folder/'model.joblib');selected_alpha(folder,head,protocol)
        require(np.allclose(head[0].mean_,x[train].mean(0,dtype=np.float64),atol=2e-5),'Refined scaler includes validation')
        require({r['sample_id'] for r in read_table(folder/'fit_ids.tsv')}==set(np.array(ids)[train]),'Refined fit ID mismatch')
        p=head.predict(x);compare_export(folder,all_rows,y,p,assignment)
        row=read_json(folder/'result.json');check_score(row,y[val],p[val])
        recorded=next(r for r in refined if r['method']==row['method'] and r['seed']==('' if row['seed'] is None else str(row['seed'])))
        check_score(recorded,y[val],p[val])
    check_aggregate(root/'regularization/summary.csv',refined)
    checks.append('all_11_refined_heads_reloaded_scalers_and_validation_scores_checked')
    directory=root/'grouped';split=read_table(directory/'split.tsv');expected=partition(all_rows,protocol)
    require(split==[{k:str(v) for k,v in r.items()} for r in expected],'Genomic partition differs from frozen rule')
    assigned={r['sample_id']:r['split'] for r in split};rows=[r for r in all_rows if assigned[r['sample_id']] in ('train','dev','evaluation')]
    sets=np.array([assigned[r['sample_id']] for r in rows]);train=sets=='train';dev=sets=='dev';evaluation=sets=='evaluation'
    require(all(r['split']!='test' for r in rows),'Historical source test entered grouped study')
    intervals={s:[genomic_interval(r) for r,a in zip(rows,sets) if a==s] for s in ('train','dev','evaluation')}
    require(all(overlaps(intervals[a],intervals[b])==0 for a,b in [('train','dev'),('train','evaluation'),('dev','evaluation')]),'Grouped windows overlap')
    y=np.array([float(r['target_log10']) for r in rows]);seq=[r['sequence'] for r in rows];ids=[r['sample_id'] for r in rows]
    xk=kmers(seq,protocol['group_kmer_k']).astype(np.float32);xd=read_bundle(old/'dna_features',ids)
    representations={'kmer_ridge':xk,'dnabert2_ridge':xd,'knn':physchem(seq),'linear_svr':xk}
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f:states={r['sample_id']:float(r['Tx_rate']) for r in map(json.loads,f)}
    representations['thermo']=np.array([[np.log10(states[r['sample_id']])] for r in rows])
    group_records=read_table(directory/'metrics.csv')
    require(len(group_records)==6+4*len(seeds(protocol)),'Missing grouped model scores')
    check_score(next(r for r in group_records if r['method']=='train_mean'),y[evaluation],np.full(evaluation.sum(),y[train].mean()))
    for seed in seeds(protocol):
        folder=directory/f'cnn_s{seed}';model,t=load_model(folder)
        require(t['min_log10']==float(y[train].min()) and t['max_log10']==float(y[train].max()),'Grouped target transform leakage')
        fit={r['sample_id'] for r in read_table(folder/'fit_ids.tsv')};selection={r['sample_id'] for r in read_table(folder/'selection_ids.tsv')}
        require(fit==set(np.array(ids)[train]) and selection==set(np.array(ids)[dev]),'Grouped CNN fit/selection leakage')
        p=t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model,encode(seq,87))
        compare_export(folder,rows,y,p,sets);check_score(read_json(folder/'result.json'),y[evaluation],p[evaluation])
        check_score(next(r for r in group_records if r['method']=='cnn' and r['seed']==str(seed)),y[evaluation],p[evaluation])
        xc=embedding(model,encode(seq,87));recipe=directory/f'guided_s{seed}';fitted={r['sample_id'] for r in read_table(recipe/'fit_ids.tsv')}
        require(len(fitted)==protocol['attribution_train_n'] and fitted<=fit,'Grouped attribution selection leakage')
        with np.load(recipe/'training_ig.npz',allow_pickle=False) as z:
            require(set(z['sample_ids'])==fitted,'Grouped IG sample IDs mismatch')
            order=sorted(REGIONS,key=lambda n:(-float(np.abs(z['scores'][:,slice(*REGIONS[n])]).mean()),list(REGIONS).index(n)))
        require(read_json(recipe/'feature_recipe.json')['selected_regions']==order[:2],'Grouped recipe selection mismatch')
        xg,_=guided_features(seq,order[:2])
        for stage,parts in [('A1',[xk,xc]),('A2',[xk,xc,xd]),('A3',[xk,xc,xd,xg])]:representations[f'{stage}_s{seed}']=np.column_stack(parts)
    for name,x in representations.items():
        folder=directory/name;head=joblib.load(folder/'model.joblib')
        if name!='thermo':require(np.allclose(head[0].mean_,x[train].mean(0,dtype=np.float64),atol=2e-5),'Grouped scaler leakage')
        else:
            from sklearn.linear_model import LinearRegression
            recalibrated=LinearRegression().fit(x[train],y[train]);require(np.allclose(head.coef_,recalibrated.coef_) and np.allclose(head.intercept_,recalibrated.intercept_),'Thermodynamic calibration leakage')
        require({r['sample_id'] for r in read_table(folder/'fit_ids.tsv')}==set(np.array(ids)[train]),'Grouped head fit leakage')
        require({r['sample_id'] for r in read_table(folder/'selection_ids.tsv')}==set(np.array(ids)[dev]),'Grouped head selection leakage')
        if name not in ('knn','linear_svr','thermo'):selected_alpha(folder,head,protocol)
        p=head.predict(x);compare_export(folder,rows,y,p,sets);check_score(read_json(folder/'result.json'),y[evaluation],p[evaluation])
        method=name.split('_s')[0] if name.startswith(('A1_','A2_','A3_')) else name
        seed=name.rsplit('_s',1)[1] if name.startswith(('A1_','A2_','A3_')) else ''
        check_score(next(r for r in group_records if r['method']==method and r['seed']==seed),y[evaluation],p[evaluation])
    check_aggregate(directory/'summary.csv',group_records)
    checks.append('zero_genomic_overlap_and_all_grouped_models_refit_reloaded_with_training_only_preprocessing')
    # Independent recomputation of regional and across-seed agreement summaries.
    directory=root/'attribution_stability';ids=[r['sample_id'] for r in read_table(old/'protocol/attribution_val_ids.tsv')]
    master={r['sample_id']:r for r in all_rows};x=encode([master[sid]['sequence'] for sid in ids],87);selected=read_json(old/'cnn/selected_model.json')
    matrices={}
    for seed in seeds(protocol):
        with np.load(directory/f's{seed}/scores.npz',allow_pickle=False) as z:
            require(z['sample_ids'].tolist()==ids and z['ig'].shape==(len(ids),87) and np.isfinite(z['ism']).all(),'Attribution matrix identity mismatch')
            model=log_model(ROOT/selected['runs'][str(seed)]);difference=cnn_predict(model,x)-cnn_predict(model,np.full_like(x,.25))
            require(np.allclose(z['ig'].sum(1)-difference,z['errors'],atol=2e-6),'IG completeness mismatch')
            require(float(np.abs(z['errors']).max())<.02,'IG completeness error exceeds predeclared tolerance')
            matrices[seed,'integrated_gradients']=np.abs(z['ig']);matrices[seed,'ism_mean_absolute']=np.abs(z['ism'])
    seed_pairs=read_table(directory/'cross_seed_pairs.csv')
    require(len(seed_pairs)==len(ids)*2*len(protocol['top_k'])*(len(seeds(protocol))*(len(seeds(protocol))-1)//2),'Missing cross-seed comparisons')
    for row in seed_pairs:
        i=ids.index(row['sample_id']);a=matrices[int(row['seed_a']),row['method']][i];b=matrices[int(row['seed_b']),row['method']][i];k=int(row['top_k'])
        one=set(ranks(a)[:k]);two=set(ranks(b)[:k])
        require(np.isclose(float(row['spearman']),spearmanr(a,b).statistic),'Cross-seed ranking correlation mismatch')
        require(np.isclose(float(row['top_k_jaccard']),len(one&two)/len(one|two)),'Cross-seed Jaccard mismatch')
    for row in read_table(directory/'cross_seed_summary.csv'):
        rr=[r for r in seed_pairs if r['method']==row['method'] and r['top_k']==row['top_k']]
        require(np.isclose(float(row['mean']),np.mean([float(r[row['metric']]) for r in rr])),'Cross-seed summary mismatch')
    for row in read_table(directory/'region_across_seeds.csv'):
        rr=[r for r in read_table(directory/'topk_overlap.csv') if all(r[k]==row[k] for k in ['method','window_policy','region','top_k'])]
        require(np.isclose(np.mean([float(r['excess_fraction']) for r in rr]),float(row['mean_excess_fraction'])),'Region aggregate mismatch')
    checks.append('three_seed_IG_completeness_and_regional_aggregates_checked')
    directory=root/'transfer_stability';target=read_table(TARGET/'samples.tsv');split=read_table(directory/'target_split.tsv')
    require(split==read_table(old/'protocol/target_split.tsv'),'Target split changed')
    assignment=np.array([r['split'] for r in split]);groups=np.array([r['sequence_group'] for r in target]);yt=np.array([float(r['target_log10']) for r in target]);test=assignment=='test'
    k=read_json(old/'ml/ridge/selected_parameters.json')['k'];xk=kmers([r['sequence'] for r in target],k).astype(np.float32)
    model,t=load_model(ROOT/selected['representative_run'],61);encoded=encode([r['sequence'] for r in target],61);xc=embedding(model,encoded)
    exported=read_table(directory/'predictions.csv');lookup={r['sample_id']:i for i,r in enumerate(target)}
    result=read_table(directory/'metrics.csv');require(len(result)==2+2*len(protocol['transfer_draw_seeds'])*len(protocol['few_shot_groups']),'Missing target draws')
    for row in result:
        name=row['condition'];draw=row['draw_seed']
        if name=='zero_shot_ridge':p=joblib.load(root/'regularization/kmer_ridge/model.joblib').predict(xk)
        elif name=='zero_shot_cnn':p=t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model,encoded)
        else:
            folder=directory/f'{name}_s{draw}';head=joblib.load(folder/'model.joblib');selected_alpha(folder,head,protocol)
            fitted={r['sample_id'] for r in read_table(folder/'fit_ids.tsv')};fit=np.array([r['sample_id'] in fitted for r in target])
            n=int(row['n_train_groups']);order=read_table(directory/f'draw_s{draw}_group_order.tsv')
            require(set(groups[fit])=={r['sequence_group'] for r in order[:n]} and np.all(assignment[fit]=='train'),'Few-shot group selection leakage')
            x=xc if 'frozen_cnn' in name else xk
            require(np.allclose(head[0].mean_,x[fit].mean(0,dtype=np.float64),atol=2e-5),'Target scaler leakage');p=head.predict(x)
        rr=[r for r in exported if r['condition']==name and r['draw_seed']==draw]
        require(len(rr)==int((assignment!='train').sum()) and len({r['sample_id'] for r in rr})==len(rr),'Target coverage mismatch')
        require(np.allclose([float(r['predicted_log10']) for r in rr],[p[lookup[r['sample_id']]] for r in rr],atol=2e-6),'Target reload mismatch')
        check_score(row,yt[test],p[test])
    check_aggregate(directory/'summary.csv',result,'condition')
    checks.append('all_32_target_conditions_reloaded_and_nested_training_group_draws_verified')
    figures=read_json(root/'figures/figure_manifest.json')
    for row in figures['figures']:
        require(sha(root/'figures'/row['png'])==row['sha256'],'Refinement figure changed')
        for name,value in row['source_tables'].items():require(sha(ROOT/name)==value,'Figure source changed')
        if visual:require(row['visual_review']=='reviewed','Inspect and acknowledge every refinement figure')
    checks.append('source_linked_figures_and_visual_review' if visual else 'source_linked_figures_visual_review_pending')
    result=dict(status='passed' if visual else 'numerical_checks_passed_visual_pending',run_id=protocol['output'],checks=checks,
        source_test_evaluated=False,ppt_updated=False,n_refined_heads=len(refined),n_target_conditions=len(result),
        n_grouped_evaluation=int(evaluation.sum()),n_attribution_parents=len(ids),visual_review_required=visual)
    write_json(root/'verification.json',result)
    return result
