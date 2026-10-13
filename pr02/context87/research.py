from __future__ import annotations

import copy
import hashlib
import itertools
import json

import joblib
import numpy as np
from scipy.stats import spearmanr, wilcoxon
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .analysis import REGIONS
from .data import TARGET, coordinate, digest_ids, load, read_json, read_table, require, sha, write_json, write_table
from .experiment import (cnn_predict, fresh, load_cnn, manifest, masks, metrics, path,
                         seed_torch, train_network)
from .features import encode, kmers, thermal_states


def ranks(scores):
    return np.lexsort((np.arange(len(scores)),-np.abs(scores)))


def integrated_gradients(model,x,steps):
    import torch
    require(type(steps) is int and steps>=2,'IG requires at least two integration steps')
    inputs=torch.from_numpy(x); baseline=torch.full_like(inputs,.25); total=torch.zeros_like(inputs)
    for i in range(steps+1):
        point=(baseline+(inputs-baseline)*(i/steps)).detach().requires_grad_(True)
        gradient=torch.autograd.grad(model(point).sum(),point)[0]
        total+=gradient*(.5 if i in (0,steps) else 1)
    attr=(inputs-baseline)*total/steps
    with torch.no_grad(): difference=model(inputs)-model(baseline)
    return attr.sum(dim=1).numpy(),(attr.sum(dim=(1,2))-difference).numpy()


def explain(cfg):
    rows,transform=load(cfg); val=[r for r in rows if r['split']=='val']; directory=fresh(path(cfg,'explain'))
    rng=np.random.default_rng(cfg['seed']); count=min(cfg['research']['explain_samples'],len(val))
    selected=[val[i] for i in sorted(rng.choice(len(val),count,replace=False))]
    model,t=load_cnn(cfg); torch=seed_torch(cfg); x=encode([r['sequence'] for r in selected],87)
    span=t['max_log10']-t['min_log10']; model.eval()
    ig,errors=integrated_gradients(model,x,cfg['research']['integrated_gradients_steps']); ig*=span; errors*=span
    inputs=torch.from_numpy(x).requires_grad_(True)
    gradient=torch.autograd.grad(model(inputs).sum(),inputs)[0].numpy()
    saliency=np.abs(gradient*(x-.25)).sum(axis=1)*span
    wild=cnn_predict(model,x); ism=[]
    for row,baseline in zip(selected,wild):
        alternatives=[row['sequence'][:p]+b+row['sequence'][p+1:] for p in range(87) for b in 'ACGT' if b!=row['sequence'][p]]
        deltas=(cnn_predict(model,encode(alternatives,87))-baseline)*span
        ism.append(np.abs(deltas.reshape(87,3)).mean(axis=1))
    methods={'integrated_gradients':ig,'saliency':saliency,'ism_mean_absolute':np.array(ism)}
    attribution_rows=[]; consistency=[]; region_rows=[]
    for method,values in methods.items():
        for r,score in zip(selected,values):
            ranking=ranks(score); rank_at=np.empty(87,int); rank_at[ranking]=np.arange(1,88)
            for p in range(87): attribution_rows.append(dict(sample_id=r['sample_id'],method=method,position_0based=p,
                   position_relative_tss=coordinate(p),nucleotide=r['sequence'][p],score_log10=float(score[p]),
                   importance_absolute=float(abs(score[p])),rank_absolute=int(rank_at[p]),sequence_length=87))
        for name,(start,end) in REGIONS.items(): region_rows.append(dict(method=method,region=name,n_samples=count,
                  annotation_status='canonical_window_assumption',mean_absolute_per_base=float(np.abs(values[:,start:end]).mean())))
    for i,r in enumerate(selected):
        for a,b in itertools.combinations(methods,2):
            one=np.abs(methods[a][i]); two=np.abs(methods[b][i]); k=cfg['research']['mutations_per_group']
            topa=set(ranks(one)[:k]); topb=set(ranks(two)[:k])
            consistency.append(dict(sample_id=r['sample_id'],method_a=a,method_b=b,spearman=float(spearmanr(one,two).statistic),
                                    top_k=k,top_k_jaccard=len(topa&topb)/len(topa|topb)))
    write_table(directory/'attributions.tsv',attribution_rows); write_table(directory/'consistency.csv',consistency)
    write_table(directory/'region_attribution.csv',region_rows)
    write_table(directory/'ig_completeness.csv',[dict(sample_id=r['sample_id'],error_log10=float(error)) for r,error in zip(selected,errors)])
    write_table(directory/'sample_ids.tsv',[dict(sample_id=r['sample_id']) for r in selected])
    write_json(directory/'interpretation.json',dict(n_samples=count,subset='val',selection='seeded uniform random, independent of labels and prediction errors',
          baseline='uniform A/C/G/T probability 0.25 per position',attribution_scale='log10 strength',
          ig_integration_steps=cfg['research']['integrated_gradients_steps'],max_ig_completeness_error_log10=float(np.abs(errors).max()),
          box_coordinates='canonical assumptions or model inferred; TSS alone experimentally verified',
          limitations='Attribution reflects this fitted model. Correlated sequence context and validation reuse limit biological conclusions.'))
    manifest(cfg,'explain',directory,dict(n_samples=count,methods=list(methods),test_evaluated=False))
    return directory


def mutate(cfg):
    rows,_=load(cfg); master={r['sample_id']:r for r in rows}; directory=fresh(path(cfg,'mutate'))
    attrs=read_table(path(cfg,'explain')/'attributions.tsv'); scores={}
    for r in attrs:
        if r['method']=='integrated_gradients': scores.setdefault(r['sample_id'],np.zeros(87))[int(r['position_0based'])]=float(r['score_log10'])
    rng=np.random.default_rng(cfg['seed']); k=cfg['research']['mutations_per_group']; require(3*k<=87,'High/Low/Random groups must fit sequence')
    model,t=load_cnn(cfg); span=t['max_log10']-t['min_log10']; design=[]
    for sid,score in scores.items():
        sequence=master[sid]['sequence']; order=ranks(score); high=list(map(int,order[:k])); low=list(map(int,order[-k:]))
        pool=[p for p in range(87) if p not in high+low]
        for repeat in range(cfg['research']['mutation_random_repeats']):
            groups={'high':high,'low':low,'random':list(map(int,rng.choice(pool,k,replace=False)))}
            for group,positions in groups.items():
                mutant=list(sequence)
                for p in positions: mutant[p]=rng.choice([b for b in 'ACGT' if b!=sequence[p]])
                mutant=''.join(mutant); require(sum(a!=b for a,b in zip(sequence,mutant))==k,'Incorrect Hamming distance')
                design.append(dict(sample_id=sid,group=group,repeat=repeat,k=k,positions_0based=json.dumps(positions),
                                   positions_relative_tss=json.dumps([coordinate(p) for p in positions]),
                                   wild_type_sequence=sequence,mutant_sequence=mutant))
    wild=cnn_predict(model,encode([r['wild_type_sequence'] for r in design],87)); mutant=cnn_predict(model,encode([r['mutant_sequence'] for r in design],87))
    raw=thermal_states([r['mutant_sequence'] for r in design]); wildstates=thermal_states([master[sid]['sequence'] for sid in scores]);
    wildtx={sid:s['Tx_rate'] for sid,s in zip(scores,wildstates)}
    calibration=read_json(path(cfg,'thermo')/'calibration.json')
    for r,w,p,s in zip(design,wild,mutant,raw):
        r.update(predicted_wild_log10=float(t['min_log10']+span*w),predicted_mutant_log10=float(t['min_log10']+span*p),
                 predicted_delta_log10=float(span*(p-w)),
                 thermo_calibrated_delta_log10=float(calibration['coef_a']*np.log10(s['Tx_rate']/wildtx[r['sample_id']])),
                 evidence='computational_prediction')
    write_table(directory/'mutation_predictions.csv',design)
    summary=[dict(group=group,n=len([r for r in design if r['group']==group]),
                  mean_absolute_delta_log10=float(np.mean([abs(r['predicted_delta_log10']) for r in design if r['group']==group])),
                  mean_delta_log10=float(np.mean([r['predicted_delta_log10'] for r in design if r['group']==group]))) for group in ('high','low','random')]
    write_table(directory/'mutation_summary.csv',summary)
    write_json(directory/'interpretation.json',dict(k=k,repeats=cfg['research']['mutation_random_repeats'],
               random_positions='exclude High and Low sets; same mutation count and alternative-base sampling',
               attribution='absolute integrated gradients',evidence='model predictions only; no measured mutant strength or causal validation'))
    manifest(cfg,'mutate',directory,dict(n_mutants=len(design),evidence='computational_prediction'))
    return directory


def ablate(cfg):
    rows,t=load(cfg); m=masks(rows); directory=fresh(path(cfg,'ablate')); seed_torch(cfg)
    full=encode([r['sequence'] for r in rows],87); low=t['min_log10']; span=t['max_log10']-low
    y=np.array([float(r['target_log10']) for r in rows]); results=[]
    controls={'remove_recovered_context':[(0,17),(67,87)],'mask_canonical_minus10':[REGIONS['-10']],
              'mask_canonical_minus35':[REGIONS['-35']],'mask_canonical_UP':[REGIONS['UP']]}
    baseline=next(r for r in read_table(path(cfg,'comparison')/'metrics.csv') if r['method']=='cnn')
    results.append(dict(condition='full_87bp',**{k:float(baseline[k]) for k in ['n','r2_log10','mae_log10','rmse_log10','spearman']}))
    import torch
    for name,windows in controls.items():
        x=full.copy()
        for start,end in windows: x[:,:,start:end]=.25
        print('ablation '+name,flush=True)
        model,history=train_network(cfg,x,(y-low)/span,m['train'],m['val'])
        torch.save(dict(state_dict=model.state_dict(),model_config=cfg['cnn']['model'],windows=windows,transform=t),directory/f'{name}.pt')
        write_table(directory/f'{name}_history.csv',history)
        p=low+span*cnn_predict(model,x[m['val']]); values=metrics(y[m['val']],p)
        results.append(dict(condition=name,**values))
        write_table(directory/f'{name}_predictions_val.csv',[dict(sample_id=r['sample_id'],predicted_log10=float(value)) for r,value in zip([r for r in rows if r['split']=='val'],p)])
    write_table(directory/'ablation_metrics.csv',results)
    write_json(directory/'interpretation.json',dict(mask='uniform base probability 0.25',retrained=True,seed=cfg['seed'],
               selection='early stopping on validation MSE; inherited same train/val assignment',
               limitations='Single-seed exploratory controls; canonical boxes are assumed windows. Test remains unused.'))
    manifest(cfg,'ablate',directory,dict(controls=controls,retrained=True))
    return directory


def target_splits(rows,cfg):
    groups=np.array([r['sequence_group'] for r in rows]); indices=np.arange(len(rows)); settings=cfg['transfer']
    trainval,test=next(GroupShuffleSplit(n_splits=1,test_size=settings['test_fraction'],random_state=cfg['seed']).split(indices,groups=groups))
    train,val=next(GroupShuffleSplit(n_splits=1,test_size=settings['val_fraction']/(1-settings['test_fraction']),random_state=cfg['seed']+1).split(trainval,groups=groups[trainval]))
    assignment=np.full(len(rows),'test',dtype='<U5'); assignment[trainval[train]]='train'; assignment[trainval[val]]='val'
    sets={s:set(groups[assignment==s]) for s in ('train','val','test')}
    require(all(not sets[a]&sets[b] for a,b in itertools.combinations(sets,2)),'Target duplicate group leakage')
    return assignment


def transfer(cfg):
    directory=fresh(path(cfg,'transfer')); rows=read_table(TARGET/'samples.tsv'); split=target_splits(rows,cfg)
    write_table(directory/'target_split.tsv',[dict(sample_id=r['sample_id'],sequence_group=r['sequence_group'],split=s) for r,s in zip(rows,split)])
    y=np.array([float(r['target_log10']) for r in rows]); sequences=[r['sequence'] for r in rows]
    train=split=='train'; val=split=='val'; test=split=='test'; results=[]; exports=[]
    frozen=joblib.load(path(cfg,'ridge')/'model.joblib'); k=frozen['parameters']['k']; x=kmers(sequences,k)
    zero_ridge=frozen['model'].predict(x)
    cnn,t=load_cnn(cfg,length=61); cnn_features=encode(sequences,61)
    zero_cnn=t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(cnn,cnn_features)
    # Freeze feature extractor. Learned target regression head uses training groups only.
    torch=seed_torch(cfg)
    with torch.no_grad(): embeddings=cnn.pool(cnn.features(torch.from_numpy(cnn_features))).flatten(start_dim=1).numpy()
    for name,p in [('zero_shot_ridge',zero_ridge),('zero_shot_cnn',zero_cnn)]:
        results.append(dict(condition=name,n_train_groups=0,**metrics(y[test],p[test]),
                            scale_interpretation='source label scale; MAE/R2 diagnostic only, primary comparison is Spearman'))
        exports.extend(dict(sample_id=r['sample_id'],condition=name,observed_target_log10=float(v),predicted_log10=float(q))
                       for r,v,q in zip(np.array(rows,dtype=object)[test],y[test],p[test]))
    group_ids=sorted({r['sequence_group'] for r,s in zip(rows,split) if s=='train'}); rng=np.random.default_rng(cfg['seed']); rng.shuffle(group_ids)
    for n_groups in cfg['transfer']['few_shot_groups']:
        chosen=set(group_ids[:n_groups]); few=np.array([r['sequence_group'] in chosen for r in rows])
        for feature_name,features in [('kmer',x),('frozen_cnn',embeddings)]:
            best=None; best_loss=np.inf; search=[]
            for alpha in [1.,10.,100.]:
                model=make_pipeline(StandardScaler(),Ridge(alpha=alpha)); model.fit(features[few],y[few]); p=model.predict(features[val])
                loss=np.mean((y[val]-p)**2); search.append(dict(alpha=alpha,val_mse=float(loss)))
                if loss<best_loss: best=model; best_loss=loss
            name=f'few_shot_{feature_name}_{n_groups}_groups'; p=best.predict(features[test]); joblib.dump(best,directory/f'{name}.joblib')
            write_table(directory/f'{name}_selection.csv',search)
            write_table(directory/f'{name}_fit_ids.tsv',[dict(sample_id=r['sample_id']) for r,use in zip(rows,few) if use])
            results.append(dict(condition=name,n_train_groups=n_groups,**metrics(y[test],p),scale_interpretation='teacher target scale; training target labels fit the regression head'))
            exports.extend(dict(sample_id=r['sample_id'],condition=name,observed_target_log10=float(v),predicted_log10=float(q)) for r,v,q in zip(np.array(rows,dtype=object)[test],y[test],p))
    write_table(directory/'transfer_metrics.csv',results); write_table(directory/'target_test_predictions.csv',exports)
    write_json(directory/'transfer_manifest.json',dict(source_species='E. coli',source_length=87,target_species='B. subtilis',target_length=61,
               target_source_sha256=sha(TARGET/'raw/promoter_strength.csv'),target_samples_sha256=sha(TARGET/'samples.tsv'),
               target_counts={s:int((split==s).sum()) for s in ['train','val','test']},target_units='unknown',target_tss='unknown',
               split_policy='exact sequence groups disjoint across train/val/test; repeated labels retained',
               primary_zero_shot_metric='Spearman; raw regression scale is not comparable across unknown assays',
               length_policy='kmer frequencies normalized by window count; CNN adaptive pooling with unchanged weights',
               few_shot_policy='nested seeded target training groups; alpha selected on target validation; frozen source CNN feature extractor',
               target_test_protocol='predeclared conditions; target test used only for final comparison, never fitting or selecting',
               limitations='Unknown target assay, units and orientation; direct CNN transfer is exploratory, no target thermodynamic claims'))
    manifest(cfg,'transfer',directory,dict(target_test_evaluated=True,source_test_evaluated=False,target_label_usage='few-shot train and val only for fitting/selection'))
    return directory
