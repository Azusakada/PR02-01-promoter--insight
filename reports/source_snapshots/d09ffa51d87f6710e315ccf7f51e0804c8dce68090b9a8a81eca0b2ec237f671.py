"""Independent reload, provenance, coordinate and statistical checks for M4."""
from __future__ import annotations
import json
import joblib
import numpy as np
from .data import ROOT, TARGET, load, read_json, read_table, require, sha, write_json
from .experiment import cnn_predict, masks, metrics
from .features import encode, kmers, thermal_states
from .m4_data_eval import SEEDS, read_bundle, run_root, validate_prediction_rows
from .m4_cnn import load_model, embedding


def verify(cfg, protocol, visual=True):
    root = run_root(protocol); rows, _ = load(cfg); m = masks(rows); ids = [r['sample_id'] for r in rows]
    y = np.array([float(r['target_log10']) for r in rows]); seq = [r['sequence'] for r in rows]; checks = []
    for file in sorted(root.rglob('run_manifest.json')):
        meta = read_json(file)
        require(meta['source_test_evaluated'] is False, 'Source test scoring enabled')
        for name, expected in meta['artifacts'].items(): require(sha(file.parent/name) == expected, 'Artifact mismatch: '+str(file.parent/name))
        for name, expected in meta['upstream'].items(): require(sha(ROOT/name) == expected, 'Upstream mismatch: '+name)
        for name, expected in meta['code_hashes'].items():
            if sha(ROOT/name) != expected:
                snapshot = ROOT/'reports/source_snapshots'/(expected+'.py')
                require(snapshot.exists() and sha(snapshot) == expected, 'Missing executed source snapshot: '+name)
    checks.append('all_manifest_artifact_upstream_and_executed_code_hashes')
    def prediction_match(directory, p):
        for subset in ('train', 'val'):
            truth = [r for r in rows if r['split'] == subset]
            out = validate_prediction_rows(read_table(directory/f'predictions_{subset}.csv'), truth, cfg, subset)
            require(np.allclose([float(r['predicted_value_log10']) for r in out], p[m[subset]], atol=2e-6, rtol=1e-6), 'Model reload mismatch: '+str(directory))
    for kind in ('ridge', 'svr', 'dnabert2'):
        directory = root/'ml'/kind; saved = joblib.load(directory/'model.joblib')
        x = kmers(seq, saved['parameters']['k']).astype(np.float32) if kind != 'dnabert2' else read_bundle(root/'dna_features', ids)
        require(np.allclose(saved['model'][0].mean_, x[m['train']].mean(0, dtype=np.float64), atol=2e-5), 'ML scaler fit identity mismatch')
        prediction_match(directory, saved['model'].predict(x))
        require({r['sample_id'] for r in read_table(directory/'fit_ids.tsv')} == set(np.array(ids)[m['train']]), 'ML fit includes heldout data')
    checks.append('ridge_svr_dna_head_reload_and_training_only_scalers')
    search = read_table(root/'cnn/search.csv'); selected = read_json(root/'cnn/selected_model.json')
    require(len(search) == 12 and {(r['condition'],int(r['seed'])) for r in search} == {(c,s) for c in ['C0','C1','C2','C3'] for s in SEEDS}, 'Incomplete CNN grid')
    winner = min(['C0','C1','C2','C3'], key=lambda c: np.mean([float(r['val_mse_log10']) for r in search if r['condition'] == c]))
    require(selected['condition'] == winner and selected['representative_seed'] == SEEDS[0], 'CNN selection mismatch')
    for record in search:
        directory = root/f"cnn/{record['condition']}_s{record['seed']}"; model, t = load_model(directory)
        p = t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model, encode(seq,87))
        prediction_match(directory,p)
        require(np.isclose(np.mean((p[m['val']]-y[m['val']])**2),float(record['val_mse_log10']),atol=1e-8), 'CNN search score mismatch')
    checks.append('all_12_cnn_runs_reloaded_and_mean_validation_selection_verified')
    kmer = read_bundle(root/'ml/kmer',ids); dna = read_bundle(root/'dna_features',ids)
    for seed in SEEDS:
        guided = root/f'guided_s{seed}'; recipe = read_json(guided/'feature_recipe.json')
        expected_train = {r['sample_id'] for r in read_table(root/'protocol/attribution_train_ids.tsv')}
        require(expected_train <= set(np.array(ids)[m['train']]) and len(expected_train) == 256, 'Attribution feature selection leakage')
        with np.load(guided/'training_ig.npz',allow_pickle=False) as z:
            require(set(z['sample_ids']) == expected_train,'Training IG identity mismatch')
            from .m4_data_eval import REGIONS
            ranked = sorted(REGIONS,key=lambda name:(-float(np.abs(z['scores'][:,slice(*REGIONS[name])]).mean()),list(REGIONS).index(name)))
            require(recipe['selected_regions'] == ranked[:2], 'Guided region selection mismatch')
        from .m4_attribution import guided_features
        expected,_=guided_features(seq,recipe['selected_regions'])
        xguided=read_bundle(guided,ids);require(np.allclose(expected,xguided),'Guided features mismatch')
        cnn=read_bundle(root/f'cnn/features_s{seed}',ids)
        model,_=load_model(ROOT/selected['runs'][str(seed)])
        require(np.allclose(cnn[:8],embedding(model,encode(seq[:8],87)),atol=1e-6),'CNN feature extraction mismatch')
        for stage,parts in [('A0',[kmer]),('A1',[kmer,cnn]),('A2',[kmer,cnn,dna]),('A3',[kmer,cnn,dna,xguided])]:
            if stage=='A0' and seed!=SEEDS[0]:continue
            directory=root/'ablation'/(stage if stage=='A0' else f'{stage}_s{seed}')
            x=np.column_stack(parts);head=joblib.load(directory/'model.joblib')
            require(np.allclose(head[0].mean_,x[m['train']].mean(0,dtype=np.float64),atol=2e-5),'Fusion scaler leakage')
            prediction_match(directory,head.predict(x))
            search_head=read_table(directory/'search.csv')
            require(float(head[-1].alpha)==float(min(search_head,key=lambda r:float(r['val_mse_log10']))['alpha']),'Fusion head selection mismatch')
    checks.append('10_fusion_runs_reloaded_and_training_attribution_recipes_verified')
    attrs=read_table(root/'attribution/attributions.tsv'); master={r['sample_id']:r for r in rows}; attribute_groups={}
    for r in attrs:
        pos=int(r['position_0based']);require(r['nucleotide']==master[r['sample_id']]['sequence'][pos],'Attribution base mismatch')
        attribute_groups.setdefault((r['sample_id'],r['method']),[]).append(pos)
    require(len(attribute_groups)==128 and all(sorted(p)==list(range(87)) for p in attribute_groups.values()),'Incomplete attribution positions')
    design=read_table(root/'mutation/mutation_design.csv'); pred=read_table(root/'mutation/mutation_predictions.csv')
    require(len(design)==2880 and len({r['mutation_id'] for r in design})==2880,'Mutation design count mismatch')
    require([r['mutation_id'] for r in pred]==[r['mutation_id'] for r in design],'Mutation ID order mismatch')
    require(sha(root/'mutation/mutation_design.csv')==read_json(root/'mutation/design_frozen.json')['design_sha256'],'Frozen design changed')
    for r in design:
        require(r['wild_type_sequence']==master[r['sample_id']]['sequence'],'Mutation parent mismatch')
        changed=[i for i,(a,b) in enumerate(zip(r['wild_type_sequence'],r['mutant_sequence'])) if a!=b]
        require(changed==sorted(json.loads(r['positions_0based'])) and len(changed)==int(r['k']) and len(r['mutant_sequence'])==87,'Mutation coordinate mismatch')
    from .m4_attribution import log_model
    model=log_model(ROOT/selected['representative_run'])
    wild=cnn_predict(model,encode([r['wild_type_sequence'] for r in design],87)); mutant=cnn_predict(model,encode([r['mutant_sequence'] for r in design],87))
    require(np.allclose(mutant-wild,[float(r['delta_log10']) for r in pred],atol=2e-6),'Mutation CNN delta mismatch')
    require(sum(r['experiment_type']=='amplitude' for r in design)==1920 and sum(r['experiment_type']=='single_direction' for r in design)==960,'Mutation condition counts mismatch')
    # Independently recompute PC states for evenly spaced variants rather than trust exported deltas.
    chosen=np.linspace(0,len(design)-1,12,dtype=int);states=thermal_states([design[i]['mutant_sequence'] for i in chosen])
    calibration=read_json(ROOT/'runs'/cfg['runs']['thermo']/'calibration.json')
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f: original={r['sample_id']:float(r['Tx_rate']) for r in map(json.loads,f)}
    for index,state in zip(chosen,states):
        expected=calibration['coef_a']*np.log10(state['Tx_rate']/original[design[index]['sample_id']])
        require(np.isclose(expected,float(pred[index]['thermo_delta_log10']),atol=1e-10),'Mutant thermodynamic delta mismatch')
    checks.append('128_attribution_vectors_2880_mutations_and_sampled_PC_states_verified')
    # Recompute the actual summary estimators, including all direction denominators.
    joined=[dict(**r,**{k:v for k,v in p.items() if k not in r}) for r,p in zip(design,pred)]
    for summary in read_table(root/'mutation/direction_summary.csv'):
        rr=[r for r in joined if r['experiment_type']=='single_direction' and (summary['expected_direction']=='all' or r['expected_direction']==summary['expected_direction'])]
        require(len(rr)==int(summary['n_mutants']),'Direction denominator mismatch')
        require(np.isclose(np.mean([r['actual_direction']==r['expected_direction'] for r in rr]),float(summary['cnn_agreement'])),'Direction agreement mismatch')
    checks.append('mutation_direction_summaries_independently_recomputed')
    target=read_table(TARGET/'samples.tsv'); split=read_table(root/'transfer/target_split.tsv')
    require(split==read_table(ROOT/'runs'/cfg['runs']['transfer']/'target_split.tsv'),'Target split was changed')
    train_ids={r['sample_id'] for r in split if r['split']=='train'}
    for file in (root/'transfer').glob('*_fit_ids.tsv'):require({r['sample_id'] for r in read_table(file)}<=train_ids,'Target fit leakage')
    tp=read_table(root/'transfer/predictions.csv');require(len(read_table(root/'transfer/transfer_metrics.csv'))==8,'Missing transfer conditions')
    for summary in read_table(root/'transfer/transfer_metrics.csv'):
        rr=[r for r in tp if r['condition']==summary['condition'] and r['subset']=='test']
        actual=metrics([float(r['observed_target_log10']) for r in rr],[float(r['predicted_log10']) for r in rr])
        for key,value in actual.items():require(np.isclose(value,float(summary[key]),atol=1e-10),'Transfer score mismatch')
    checks.append('8_transfer_conditions_frozen_group_split_and_metrics_verified')
    for summary in read_table(root/'evaluation/metrics.csv'):
        if summary['method']=='train_mean':continue
        out=read_table(ROOT/summary['run_id']/f"predictions_{summary['subset']}.csv")
        actual=metrics([float(r['true_value_log10']) for r in out],[float(r['predicted_value_log10']) for r in out])
        for key,value in actual.items():require(np.isclose(value,float(summary[key]),atol=1e-10),'Unified metric mismatch')
    checks.append('unified_metrics_independently_recomputed')
    figures=read_json(root/'figures/figure_manifest.json')
    for r in figures['figures']:
        require(sha(root/'figures'/r['png'])==r['sha256'],'Figure checksum mismatch')
        for name,value in r['source_tables'].items():require(sha(ROOT/name)==value,'Figure source mismatch')
        if visual:require(r['visual_review']=='reviewed','View and acknowledge every M4 figure before release')
    checks.append('figure_hashes_source_tables_and_visual_review' if visual else 'figure_hashes_source_tables')
    frozen=ROOT/'reports/M4_FROZEN_INPUTS.json'
    if frozen.exists():
        for name,value in read_json(frozen).items():require(sha(ROOT/name)==value,'Historical artifact changed: '+name)
        checks.append('all_frozen_historical_artifacts_unchanged')
    from .audit import verify as verify_m3
    old=verify_m3(cfg)
    require(old['status']=='passed','Legacy M3 verification failed');checks.append('original_M3_verification_passed')
    result=dict(status='passed',run_id=protocol['output'],checks=checks,n_source_val=1768,n_attribution_parents=64,
        n_mutants=2880,n_transfer_conditions=8,source_test_evaluated=False,visual_review_required=visual)
    write_json(root/'verification.json',result)
    return result
