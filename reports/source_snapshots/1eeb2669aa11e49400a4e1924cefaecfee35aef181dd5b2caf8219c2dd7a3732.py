"""Spatially separated train/dev/evaluation; all supervised components are refit."""
from __future__ import annotations

import bisect
import copy
import warnings
import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR

from .data import ROOT, load, rc, read_json, read_table, require, write_json, write_table
from .experiment import cnn_predict, metrics, train_network
from .features import encode, kmers, physchem
from .m4_attribution import guided_features
from .m4_cnn import embedding, load_model
from .m4_data_eval import REGIONS, bootstrap, completed, finish, read_bundle, run_root, seeds
from .m4_ml_dnalm import fit_head
from .m4_refinement import aggregate, baseline, save_predictions
from .research import integrated_gradients


def genomic_interval(row):
    """Zero-based genomic interval; reverse strand places 19 bases below the TSS."""
    start = int(row['tss_1based'])-1-(67 if row['strand'] == '+' else 19)
    return start, start+87


def overlaps(reference, queries):
    if not reference: return 0
    reference = sorted(reference); starts = [a for a, b in reference]
    ends = np.maximum.accumulate([b for a, b in reference])
    count = 0
    for start, end in queries:
        index = bisect.bisect_left(starts, end)-1
        count += int(index >= 0 and ends[index] > start)
    return count


def partition(rows, protocol):
    block_bp = protocol['genomic_block_bp']; length = protocol['genome_length']
    eligible = []; result = []
    for row in rows:
        start, end = genomic_interval(row)
        status = 'eligible'; block = start//block_bp
        if row['split'] == 'test': status = 'excluded_historical_test'
        elif start < 0 or end > length or start//block_bp != (end-1)//block_bp: status = 'excluded_block_boundary'
        record = dict(sample_id=row['sample_id'], inherited_split=row['split'], strand=row['strand'],
                      genomic_start_0based=start, genomic_end_exclusive=end, block_id=block, split=status)
        result.append(record)
        if status == 'eligible': eligible.append(record)
    # Exact reverse-complement duplicates are tied even when their coordinates differ.
    parent = {r['block_id']: r['block_id'] for r in eligible}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    seq_groups = {}; master = {r['sample_id']: r for r in rows}
    for record in eligible:
        seq = master[record['sample_id']]['sequence']; key = min(seq, rc(seq)); block = record['block_id']
        if key in seq_groups: parent[find(block)] = find(seq_groups[key])
        else: seq_groups[key] = block
    blocks = sorted({find(r['block_id']) for r in eligible})
    np.random.default_rng(protocol['group_split_seed']).shuffle(blocks)
    n_eval = int(np.ceil(len(blocks)*protocol['group_evaluation_fraction']))
    n_dev = int(np.ceil(len(blocks)*protocol['group_dev_fraction']))
    assignment = {b: ('evaluation' if i < n_eval else 'dev' if i < n_eval+n_dev else 'train') for i, b in enumerate(blocks)}
    for r in eligible:
        r['split'] = assignment[find(r['block_id'])]
    return result


def run(cfg, protocol):
    import torch
    root = run_root(protocol); directory = root/'grouped'
    if completed(directory, cfg, protocol): return directory
    directory.mkdir(parents=True, exist_ok=True)
    all_rows, _ = load(cfg); split = partition(all_rows, protocol)
    if (directory/'split.tsv').exists():
        require(read_table(directory/'split.tsv') == [{k: str(v) for k, v in r.items()} for r in split], 'Grouped split changed')
    else: write_table(directory/'split.tsv', split)
    assigned = {r['sample_id']: r['split'] for r in split}
    rows = [r for r in all_rows if assigned[r['sample_id']] in ('train', 'dev', 'evaluation')]
    sets = np.array([assigned[r['sample_id']] for r in rows]); train = sets == 'train'; dev = sets == 'dev'; evaluation = sets == 'evaluation'
    intervals = {s: [genomic_interval(r) for r, subset in zip(rows, sets) if subset == s] for s in ('train', 'dev', 'evaluation')}
    overlap_audit = {a+'_vs_'+b: overlaps(intervals[a], intervals[b]) for a, b in [('train','dev'), ('train','evaluation'), ('dev','evaluation')]}
    require(not any(overlap_audit.values()), 'Genomic windows overlap across grouped subsets')
    seq_sets = {s: {min(r['sequence'], rc(r['sequence'])) for r, subset in zip(rows, sets) if subset == s} for s in intervals}
    require(all(not seq_sets[a] & seq_sets[b] for a,b in [('train','dev'), ('train','evaluation'), ('dev','evaluation')]), 'Identical sequence leakage')
    y = np.array([float(r['target_log10']) for r in rows]); low = float(y[train].min()); high = float(y[train].max())
    transform = dict(min_log10=low, max_log10=high, fit_subset='grouped_train', formula='(log10_strength-min)/(max-min)')
    write_json(directory/'split_audit.json', dict(counts={s:int((sets == s).sum()) for s in intervals},
        exclusions={s:sum(r['split'] == s for r in split) for s in ['excluded_historical_test','excluded_block_boundary']},
        block_bp=protocol['genomic_block_bp'], split_seed=protocol['group_split_seed'], cross_subset_window_overlaps=overlap_audit,
        reverse_complement_duplicates_cross_subsets=False, label_transform=transform,
        selection='fit on train, select alpha and CNN stopping epoch on dev; evaluation labels excluded from these decisions',
        scope='One spatially separated partition of original train+val. Historical source test excluded. Internal robustness control, not a new external blind dataset.'))
    old = baseline(protocol); ids = [r['sample_id'] for r in rows]; seqs = [r['sequence'] for r in rows]
    k = protocol['group_kmer_k']; xk = kmers(seqs, k).astype(np.float32)
    xd = read_bundle(old/'dna_features', ids); xencoded = encode(seqs,87)
    records = []; searches = []
    def persist(name, method, seed, x, head, search, extra=None):
        output = directory/name; output.mkdir(parents=True, exist_ok=True)
        joblib.dump(head, output/'model.joblib'); write_table(output/'fit_ids.tsv', [dict(sample_id=r['sample_id']) for r, use in zip(rows,train) if use])
        write_table(output/'selection_ids.tsv', [dict(sample_id=r['sample_id']) for r, use in zip(rows,dev) if use])
        p = head.predict(x); save_predictions(output, rows, y, p, sets, method, seed)
        row = dict(method=method, seed=seed, **metrics(y[evaluation],p[evaluation]))
        write_json(output/'result.json', row)
        if search: write_table(output/'search.csv', search)
        searches.extend(dict(method=method, seed=seed, **r) for r in search)
        finish(output, cfg, dict(fit='grouped_train', selection='grouped_dev', evaluation='grouped_evaluation', **(extra or {})), [directory/'split.tsv'])
        records.append(row)
        print('Grouped',name,'R2',row['r2_log10'],'rho',row['spearman'],flush=True)
        return p
    for method, x in [('kmer_ridge', xk), ('dnabert2_ridge', xd)]:
        head, search = fit_head(x,y,train,dev,protocol['ridge_alpha'])
        persist(method,method,None,x,head,search,dict(selected_alpha=float(head[-1].alpha)))
    # The same baseline families are independently fit; no original source calibrator is reused.
    xp = physchem(seqs); candidates = []
    for item in cfg['search']['knn']:
        head = make_pipeline(StandardScaler(),KNeighborsRegressor(n_neighbors=item['k'],weights='distance',n_jobs=cfg['num_threads']))
        head.fit(xp[train],y[train]); loss = float(np.mean((head.predict(xp[dev])-y[dev])**2))
        candidates.append((loss,head,dict(neighbors=item['k'],val_mse_log10=loss)))
    persist('knn','knn',None,xp,min(candidates,key=lambda t:t[0])[1],[c[2] for c in candidates])
    candidates=[]
    for c in protocol['svr_C']:
        head=make_pipeline(StandardScaler(),LinearSVR(C=c,epsilon=0.,loss='squared_epsilon_insensitive',dual=False,max_iter=50000,tol=1e-5,random_state=cfg['seed']))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always',ConvergenceWarning); head.fit(xk[train],y[train])
        require(not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Grouped SVR convergence failure')
        loss=float(np.mean((head.predict(xk[dev])-y[dev])**2)); candidates.append((loss,head,dict(C=c,val_mse_log10=loss)))
    persist('linear_svr','linear_svr',None,xk,min(candidates,key=lambda t:t[0])[1],[c[2] for c in candidates])
    import json
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f: states={r['sample_id']:float(r['Tx_rate']) for r in map(json.loads,f)}
    xt=np.array([[np.log10(states[r['sample_id']])] for r in rows]); head=LinearRegression().fit(xt[train],y[train])
    persist('thermo','thermo',None,xt,head,[])
    baseline_prediction=np.full(len(rows),y[train].mean())
    records.append(dict(method='train_mean',seed=None,**metrics(y[evaluation],baseline_prediction[evaluation])))
    for seed in seeds(protocol):
        output=directory/f'cnn_s{seed}'
        if completed(output,cfg,protocol):
            model,t=load_model(output); require(t==transform,'Grouped CNN transform changed')
        else:
            output.mkdir(parents=True,exist_ok=True); current=copy.deepcopy(cfg); current['seed']=seed
            model,history=train_network(current,xencoded,(y-low)/(high-low),train,dev)
            torch.save(dict(state_dict=model.state_dict(),model_config=current['cnn']['model'],transform=transform),output/'model.pt')
            write_json(output/'config.json',current); write_table(output/'training_history.csv',history)
            write_table(output/'fit_ids.tsv',[dict(sample_id=r['sample_id']) for r,use in zip(rows,train) if use])
            write_table(output/'selection_ids.tsv',[dict(sample_id=r['sample_id']) for r,use in zip(rows,dev) if use])
            p=low+(high-low)*cnn_predict(model,xencoded); save_predictions(output,rows,y,p,sets,'cnn',seed)
            write_json(output/'result.json',dict(method='cnn',seed=seed,**metrics(y[evaluation],p[evaluation])))
            finish(output,cfg,dict(fit='grouped_train',early_stopping='grouped_dev',architecture='fixed original C0; no evaluation architecture search'),[directory/'split.tsv'])
        records.append(read_json(output/'result.json'))
        xc=embedding(model,xencoded)
        # Derive the attribution recipe only from this partition's training samples.
        rng=np.random.default_rng(seed); chosen=np.sort(rng.choice(np.flatnonzero(train),protocol['attribution_train_n'],replace=False))
        from pr02_cnn.model import Log10Predictor
        scores,errors=integrated_gradients(Log10Predictor(model,transform).eval(),xencoded[chosen],protocol['ig_steps'])
        ranking=sorted(REGIONS,key=lambda name:(-float(np.abs(scores[:,slice(*REGIONS[name])]).mean()),list(REGIONS).index(name)))
        xg,names=guided_features(seqs,ranking[:2])
        recipe=directory/f'guided_s{seed}';recipe.mkdir(parents=True,exist_ok=True)
        write_json(recipe/'feature_recipe.json',dict(selected_regions=ranking[:2],fit_subset='grouped_train',feature_names=names,
             mean_absolute_completeness_error=float(np.abs(errors).mean())))
        write_table(recipe/'fit_ids.tsv',[dict(sample_id=rows[i]['sample_id']) for i in chosen])
        np.savez_compressed(recipe/'training_ig.npz',scores=scores,sample_ids=np.array([rows[i]['sample_id'] for i in chosen]))
        finish(recipe,cfg,dict(seed=seed),[output/'model.pt',directory/'split.tsv'])
        for stage,parts in [('A1',[xk,xc]),('A2',[xk,xc,xd]),('A3',[xk,xc,xd,xg])]:
            x=np.column_stack(parts);head,search=fit_head(x,y,train,dev,protocol['ridge_alpha'])
            persist(f'{stage}_s{seed}',stage,seed,x,head,search,dict(selected_alpha=float(head[-1].alpha)))
    write_table(directory/'metrics.csv',records);write_table(directory/'summary.csv',aggregate(records));write_table(directory/'search.csv',searches)
    # Paired block bootstrap quantifies evaluation-sample uncertainty separately from seed SD.
    model_predictions={}
    for p in directory.glob('*/predictions.csv'):
        rr=[r for r in read_table(p) if r['subset']=='evaluation']
        if rr:model_predictions[p.parent.name]=np.array([float(r['predicted_log10']) for r in rr])
    groups=np.array([str(genomic_interval(r)[0]//protocol['genomic_block_bp']) for r,use in zip(rows,evaluation) if use])
    contrasts=[];ye=y[evaluation]
    for method in ['cnn','A1','A2','A3','dnabert2_ridge']:
        keys=[f'{method}_s{s}' for s in seeds(protocol)] if method in ['cnn','A1','A2','A3'] else [method]
        predictions=np.column_stack([model_predictions[key] for key in keys]);reference=model_predictions['kmer_ridge']
        z=np.column_stack([ye,reference,predictions])
        def difference(a):
            denominator=np.sum((a[:,0]-a[:,0].mean())**2)
            return float((np.sum((a[:,0]-a[:,1])**2)-np.mean(np.sum((a[:,0,None]-a[:,2:])**2,axis=0)))/denominator)
        ci=bootstrap(z,statistic=difference,groups=groups,protocol=protocol)
        contrasts.append(dict(contrast=method+'_minus_kmer_ridge',mean_delta_r2=difference(z),ci_low=ci[0],ci_high=ci[1],bootstrap_unit='20kb genomic block',seed_aggregation='mean of per-seed scores, not ensemble prediction'))
    write_table(directory/'paired_contrasts.csv',contrasts)
    finish(directory,cfg,dict(source_test_evaluated=False,n_evaluation=int(evaluation.sum()),n_seeds=len(seeds(protocol)),
        frozen_configuration_before_evaluation=True,limitations='One spatial partition; historical labels informed project development. Not external prospective validation.'),
        [directory/'split.tsv',old/'dna_features/feature_manifest.json'])
    return directory
