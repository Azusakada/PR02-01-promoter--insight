"""Across-checkpoint explanation stability and repeated target training-group draws."""
from __future__ import annotations

import itertools
import json
import joblib
import numpy as np
from scipy.stats import spearmanr

from .data import ROOT, TARGET, load, read_json, read_table, require, write_json, write_table
from .experiment import cnn_predict, metrics
from .features import encode, kmers
from .m4_attribution import log_model
from .m4_cnn import embedding, load_model
from .m4_data_eval import REGIONS, bootstrap, completed, finish, run_root, seeds
from .m4_ml_dnalm import fit_head
from .m4_refinement import aggregate, baseline
from .research import integrated_gradients, ranks


def attribution(cfg, protocol):
    import torch
    torch.set_num_threads(cfg['num_threads'])
    root=run_root(protocol);old=baseline(protocol);directory=root/'attribution_stability'
    if completed(directory,cfg,protocol):return directory
    directory.mkdir(parents=True,exist_ok=True)
    rows,_=load(cfg);master={r['sample_id']:r for r in rows}
    ids=[r['sample_id'] for r in read_table(old/'protocol/attribution_val_ids.tsv')]
    require(all(master[sid]['split']=='val' for sid in ids),'Stability cohort is not validation')
    selected=read_json(old/'cnn/selected_model.json');sequences=[master[sid]['sequence'] for sid in ids];x=encode(sequences,87)
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f:states={r['sample_id']:r for r in map(json.loads,f)}
    values={};output=[];errors_out=[];overlap=[];pairwise=[];region_summary=[]
    methods=['integrated_gradients','ism_mean_absolute']
    for seed in seeds(protocol):
        folder=directory/f's{seed}'
        if completed(folder,cfg,protocol):
            with np.load(folder/'scores.npz',allow_pickle=False) as z:
                require(z['sample_ids'].tolist()==ids,'Stability ID order changed')
                ig=z['ig'];ism=z['ism'];errors=z['errors']
        else:
            folder.mkdir(parents=True,exist_ok=True);model=log_model(ROOT/selected['runs'][str(seed)])
            ig,errors=integrated_gradients(model,x,protocol['ig_steps']);ism=[]
            wild=cnn_predict(model,x)
            for i,sequence in enumerate(sequences):
                mutants=[sequence[:pos]+alt+sequence[pos+1:] for pos,ref in enumerate(sequence) for alt in 'ACGT' if alt!=ref]
                delta=cnn_predict(model,encode(mutants,87))-wild[i];ism.append(np.abs(delta.reshape(87,3)).mean(1))
            ism=np.asarray(ism)
            np.savez_compressed(folder/'scores.npz',ig=ig,ism=ism,errors=errors,sample_ids=np.array(ids))
            finish(folder,cfg,dict(seed=seed,n_parent=len(ids)),[ROOT/selected['runs'][str(seed)]/'model.pt',old/'protocol/attribution_val_ids.tsv'])
        print('Attribution stability seed',seed,'IG max error',float(np.abs(errors).max()),flush=True)
        for i,sid in enumerate(ids):
            errors_out.append(dict(sample_id=sid,seed=seed,completeness_error_log10=float(errors[i])))
            for method,scores in [('integrated_gradients',ig),('ism_mean_absolute',ism)]:
                score=scores[i];values[seed,sid,method]=np.abs(score)
                for pos,value in enumerate(score):output.append(dict(sample_id=sid,seed=seed,method=method,position_0based=pos,score_log10=float(value)))
                inferred={name:tuple(states[sid][key]) for name,key in [('UP','UP_position'),('-35','hex35_position'),('spacer','spacer_position'),('-10','hex10_position')]}
                for status,windows in [('fixed',REGIONS),('PC_inferred',inferred)]:
                    for region,(start,end) in windows.items():
                        for k in protocol['top_k']:
                            top=ranks(score)[:k];fraction=float(((top>=start)&(top<end)).mean())
                            overlap.append(dict(sample_id=sid,seed=seed,method=method,window_policy=status,region=region,top_k=k,
                                overlap_fraction=fraction,random_fraction=(end-start)/87,excess_fraction=fraction-(end-start)/87))
    for a,b in itertools.combinations(seeds(protocol),2):
        for sid in ids:
            for method in methods:
                one=values[a,sid,method];two=values[b,sid,method];rho=float(spearmanr(one,two).statistic)
                for k in protocol['top_k']:
                    u=set(ranks(one)[:k]);v=set(ranks(two)[:k])
                    pairwise.append(dict(sample_id=sid,seed_a=a,seed_b=b,method=method,top_k=k,spearman=rho,top_k_jaccard=len(u&v)/len(u|v)))
    for seed,method,status,region,k in itertools.product(seeds(protocol),methods,['fixed','PC_inferred'],REGIONS,protocol['top_k']):
        rr=[r for r in overlap if (r['seed'],r['method'],r['window_policy'],r['region'],r['top_k'])==(seed,method,status,region,k)]
        a=np.array([r['excess_fraction'] for r in rr]);ci=bootstrap(a,protocol=protocol)
        region_summary.append(dict(seed=seed,method=method,window_policy=status,region=region,top_k=k,n_parent=len(ids),
            mean_excess_fraction=float(a.mean()),ci_low=ci[0],ci_high=ci[1]))
    across=[]
    for method,status,region,k in itertools.product(methods,['fixed','PC_inferred'],REGIONS,protocol['top_k']):
        rr=[r for r in overlap if (r['method'],r['window_policy'],r['region'],r['top_k'])==(method,status,region,k)]
        parent=np.array([np.mean([r['excess_fraction'] for r in rr if r['sample_id']==sid]) for sid in ids])
        seed_means=[np.mean([r['excess_fraction'] for r in rr if r['seed']==seed]) for seed in seeds(protocol)]
        ci=bootstrap(parent,protocol=protocol)
        across.append(dict(method=method,window_policy=status,region=region,top_k=k,n_parent=len(ids),n_seeds=len(seeds(protocol)),
            mean_excess_fraction=float(parent.mean()),min_seed_mean=float(min(seed_means)),max_seed_mean=float(max(seed_means)),
            positive_seed_count=sum(v>0 for v in seed_means),ci_low=ci[0],ci_high=ci[1],bootstrap_unit='parent; average seeds within parent first'))
    agreement=[]
    for method,k in itertools.product(methods,protocol['top_k']):
        rr=[r for r in pairwise if r['method']==method and r['top_k']==k]
        for metric in ['spearman','top_k_jaccard']:
            # Three seed pairs from one parent do not become three independent parents.
            a=np.array([np.mean([r[metric] for r in rr if r['sample_id']==sid]) for sid in ids]);ci=bootstrap(a,protocol=protocol)
            agreement.append(dict(method=method,top_k=k,metric=metric,n_parent=len(ids),mean=float(a.mean()),ci_low=ci[0],ci_high=ci[1]))
    write_table(directory/'scores.csv',output);write_table(directory/'ig_completeness.csv',errors_out)
    write_table(directory/'topk_overlap.csv',overlap);write_table(directory/'region_by_seed.csv',region_summary)
    write_table(directory/'region_across_seeds.csv',across);write_table(directory/'cross_seed_pairs.csv',pairwise);write_table(directory/'cross_seed_summary.csv',agreement)
    finish(directory,cfg,dict(n_parent=len(ids),n_seeds=len(seeds(protocol)),
        interpretation='Across fitted CNNs, distinct from IG-vs-ISM agreement within one CNN. Descriptive intervals, no multiple-testing claim; uniform position reference is not experimental binding annotation.'),
        [old/'release_manifest.json'])
    return directory


def transfer(cfg,protocol):
    root=run_root(protocol);old=baseline(protocol);directory=root/'transfer_stability'
    if completed(directory,cfg,protocol):return directory
    directory.mkdir(parents=True,exist_ok=True)
    rows=read_table(TARGET/'samples.tsv');split=read_table(old/'protocol/target_split.tsv')
    require([r['sample_id'] for r in rows]==[r['sample_id'] for r in split],'Target split order mismatch')
    sets=np.array([r['split'] for r in split]);groups=np.array([r['sequence_group'] for r in rows]);y=np.array([float(r['target_log10']) for r in rows])
    group_sets={s:set(groups[sets==s]) for s in ['train','val','test']}
    require(all(not group_sets[a]&group_sets[b] for a,b in [('train','val'),('train','test'),('val','test')]),'Target sequence leakage')
    source=joblib.load(root/'regularization/kmer_ridge/model.joblib')
    k=read_json(old/'ml/ridge/selected_parameters.json')['k'];xk=kmers([r['sequence'] for r in rows],k).astype(np.float32)
    selected=read_json(old/'cnn/selected_model.json');model,t=load_model(ROOT/selected['representative_run'],61)
    encoded=encode([r['sequence'] for r in rows],61);xc=embedding(model,encoded)
    zero=[('zero_shot_ridge',source.predict(xk)),('zero_shot_cnn',t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model,encoded))]
    predictions=[];records=[];all_search=[]
    def score(condition,draw_seed,n,p):
        test=sets=='test';z=np.column_stack([y[test],p[test]])
        ci=bootstrap(z,statistic=lambda a:float(spearmanr(a[:,0],a[:,1]).statistic),groups=groups[test],protocol=protocol)
        records.append(dict(condition=condition,draw_seed=draw_seed,n_train_groups=n,**metrics(y[test],p[test]),spearman_ci_low=ci[0],spearman_ci_high=ci[1]))
        for r,subset,actual,value in zip(rows,sets,y,p):
            if subset not in ('val','test'):continue
            predictions.append(dict(sample_id=r['sample_id'],sequence_group=r['sequence_group'],subset=subset,condition=condition,draw_seed=draw_seed,
                observed_log10=float(actual),predicted_log10=float(value)))
    for name,p in zero:score(name,None,0,p)
    for draw in protocol['transfer_draw_seeds']:
        order=sorted(group_sets['train']);np.random.default_rng(draw).shuffle(order)
        write_table(directory/f'draw_s{draw}_group_order.tsv',[dict(rank=i+1,sequence_group=g) for i,g in enumerate(order)])
        for n in protocol['few_shot_groups']:
            fit=np.isin(groups,order[:n]);require(np.all(sets[fit]=='train'),'Target heldout used in fitting')
            for kind,x in [('kmer',xk),('frozen_cnn',xc)]:
                name=f'few_shot_{kind}_{n}_groups';folder=directory/f'{name}_s{draw}';folder.mkdir(parents=True,exist_ok=True)
                head,search=fit_head(x,y,fit,sets=='val',protocol['ridge_alpha'])
                joblib.dump(head,folder/'model.joblib');write_table(folder/'fit_ids.tsv',[dict(sample_id=r['sample_id'],sequence_group=r['sequence_group']) for r,use in zip(rows,fit) if use])
                write_table(folder/'search.csv',search);all_search.extend(dict(condition=name,draw_seed=draw,**r) for r in search)
                p=head.predict(x);score(name,draw,n,p)
                finish(folder,cfg,dict(draw_seed=draw,n_train_groups=n,selected_alpha=float(head[-1].alpha),fit='target_train',selection='target_val'),[old/'protocol/target_split.tsv'])
        print('Target training-group draw',draw,'completed',flush=True)
    write_table(directory/'target_split.tsv',split);write_table(directory/'metrics.csv',records)
    write_table(directory/'predictions.csv',predictions);write_table(directory/'search.csv',all_search)
    summary=aggregate(records,'condition')
    for r in summary:
        rr=[x for x in records if x['condition']==r['condition']]
        r.update(n_train_groups=rr[0]['n_train_groups'],n_draws=len(rr),
                 spearman_min=min(x['spearman'] for x in rr),spearman_max=max(x['spearman'] for x in rr),
                 uncertainty='SD across training draws; per-draw test-group bootstrap CIs are in metrics.csv')
    write_table(directory/'summary.csv',summary)
    finish(directory,cfg,dict(n_draws=len(protocol['transfer_draw_seeds']),nested_groups_within_draw=True,
        target_test_history='Previously observed target test reused; all predeclared draws retained, none selected by test score',
        primary_metric='Spearman; source/target units unknown; absolute scale errors diagnostic only',
        uncertainty='Training-draw SD and conditional test-group bootstrap are reported separately; neither is external assay uncertainty'),
        [old/'protocol/target_split.tsv',root/'regularization/kmer_ridge/model.joblib',ROOT/selected['representative_run']/'model.pt'])
    return directory
