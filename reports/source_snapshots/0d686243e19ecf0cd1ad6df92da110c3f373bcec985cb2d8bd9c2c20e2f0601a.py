from __future__ import annotations

import json

import joblib
import numpy as np

from .data import ROOT, build, digest_ids, load, read_json, read_table, require, sha, validate, write_json
from .experiment import METHODS, cnn_predict, input_hashes, load_cnn, masks, metrics, path
from .features import encode, kmers, physchem


def reproduce(cfg,output):
    output=ROOT/output if not output.is_absolute() else output
    require(output.resolve().is_relative_to(ROOT),'Rebuild destination must remain inside repository')
    build(output)
    from .data import BASE
    same={name:(output/name).read_bytes()==(BASE/name).read_bytes() for name in ['samples.tsv','split_manifest.tsv','label_transform.json','data_manifest.json']}
    require(all(same.values()),'Canonical data reproduction failed')
    write_json(output/'reproduction_check.json',dict(status='passed',byte_identical=same))
    return same


def verify(cfg,visual=False):
    proof=validate(cfg); rows,transform=load(cfg); m=masks(rows); sequences=[r['sequence'] for r in rows]
    train_hash=digest_ids(r['sample_id'] for r in rows if r['split']=='train'); val_ids=[r['sample_id'] for r in rows if r['split']=='val']
    y=np.array([float(r['target_log10']) for r in rows]); metric_rows=read_table(path(cfg,'comparison')/'metrics.csv')
    results=[]
    for kind in (*METHODS,'comparison','analysis'):
        directory=path(cfg,kind); meta=read_json(directory/'run_manifest.json')
        require(meta['config']==cfg and meta['input_hashes']==input_hashes(cfg),'Run is bound to a different configuration/dataset: '+kind)
        require(meta['train_ids_sha256']==train_hash and meta['test_metrics_enabled'] is False,'Invalid fit identity/test policy')
        for name,expected in meta['code_hashes'].items():
            current=ROOT/name
            if sha(current)!=expected:
                snapshot=ROOT/'reports/source_snapshots'/(expected+current.suffix)
                require(snapshot.exists() and sha(snapshot)==expected,'Missing exact executed source snapshot: '+name)
        for name,expected in meta['artifacts'].items(): require(sha(directory/name)==expected,'Artifact hash mismatch: '+kind+'/'+name)
        results.append(kind+'_artifacts')
    for kind in METHODS:
        directory=path(cfg,kind); export=read_table(directory/'predictions_val.csv')
        require([r['sample_id'] for r in export]==val_ids,'Wrong validation IDs')
        saved=np.array([float(r['predicted_value_log10']) for r in export])
        require(all(r['method_name']==kind and r['run_id']==cfg['runs'][kind] and r['data_version']==cfg['data_version']
                    and r['split_id']==cfg['split_id'] and r['subset']=='val' for r in export),'Wrong prediction provenance')
        require(np.allclose([float(r['true_value_log10']) for r in export],y[m['val']]),'Wrong validation labels')
        if kind in ('knn','ridge','svr'):
            checkpoint=joblib.load(directory/'model.joblib'); require(checkpoint['input_hashes']==input_hashes(cfg),'Wrong checkpoint data')
            model=checkpoint['model']; x=physchem(sequences) if kind=='knn' else kmers(sequences,checkpoint['parameters']['k'])
            require(np.allclose(model[0].mean_,x[m['train']].mean(axis=0),rtol=1e-12,atol=1e-12),'Scaler was not fitted on training rows')
            reconstructed=model.predict(x[m['val']])
            selection=read_table(directory/'hyperparameter_search.csv'); selected=read_json(directory/'selected_parameters.json')
            winner=min(selection,key=lambda r:float(r['val_mse_log10']))
            require(all(float(winner[k])==value for k,value in selected.items()),'Selection is not validation-only minimum')
            require(digest_ids(r['sample_id'] for r in read_table(directory/'fit_ids.tsv'))==train_hash,'Fit rows differ from train')
        elif kind=='cnn':
            model,t=load_cnn(cfg); require(t==transform,'CNN target transform mismatch')
            reconstructed=t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model,encode([r['sequence'] for r in rows if r['split']=='val'],87))
        else:
            with (directory/'states.jsonl').open() as f: states=[json.loads(line) for line in f]
            require([r['sample_id'] for r in states]==[r['sample_id'] for r in rows],'Thermodynamic state ID order mismatch')
            require(all(r['TSS']==67 for r in states),'Thermodynamic inference did not use fixed TSS')
            x=np.log10([r['Tx_rate'] for r in states]); a,b=np.linalg.lstsq(np.column_stack([x[m['train']],np.ones(m['train'].sum())]),y[m['train']],rcond=None)[0]
            calibration=read_json(directory/'calibration.json')
            require(calibration['fit_subset']=='train' and calibration['train_ids_sha256']==train_hash
                    and np.allclose([a,b],[calibration['coef_a'],calibration['intercept_b']]),'Calibration leakage or mismatch')
            reconstructed=a*x[m['val']]+b
        require(np.allclose(saved,reconstructed,rtol=1e-6,atol=1e-6),'Reloaded model predictions differ: '+kind)
        observed=metrics(y[m['val']],saved); stored=next(r for r in metric_rows if r['method']==kind)
        require(all(np.isclose(float(stored[k]),value,rtol=1e-10,atol=1e-10) for k,value in observed.items()),'Comparison metrics differ: '+kind)
        results.append(kind+'_independent_prediction_and_metrics')
    figure_meta=read_json(path(cfg,'analysis')/'figure_manifest.json')
    for figure in figure_meta['figures']:
        require(sha(path(cfg,'analysis')/figure['figure'])==figure['sha256'],'Figure hash mismatch')
        if visual: require(figure['visual_review']=='reviewed','View and acknowledge figures before publication')
    for kind in ['explain','mutate','ablate','transfer']:
        directory=path(cfg,kind)
        if not (directory/'run_manifest.json').exists(): continue
        meta=read_json(directory/'run_manifest.json')
        require(meta['config']==cfg and meta['input_hashes']==input_hashes(cfg),'Research run data identity mismatch')
        for name,expected in meta['artifacts'].items(): require(sha(directory/name)==expected,'Research artifact hash mismatch: '+kind+'/'+name)
        if kind=='explain':
            attributes=read_table(directory/'attributions.tsv'); groups={}; master={r['sample_id']:r for r in rows}
            for r in attributes:
                position=int(r['position_0based']); sid=r['sample_id']
                require(sid in val_ids and 0<=position<87 and r['nucleotide']==master[sid]['sequence'][position], 'Attribution coordinate mismatch')
                groups.setdefault((sid,r['method']),[]).append(position)
            require(all(sorted(indices)==list(range(87)) for indices in groups.values()),'Incomplete attribution positions')
        elif kind=='mutate':
            master={r['sample_id']:r for r in rows}; mutants=read_table(directory/'mutation_predictions.csv')
            for r in mutants:
                wild=r['wild_type_sequence']; mutant=r['mutant_sequence']; positions=json.loads(r['positions_0based'])
                require(wild==master[r['sample_id']]['sequence'] and len(mutant)==87 and set(mutant)<=set('ACGT'),'Mutation source mismatch')
                changed=[i for i,(a,b) in enumerate(zip(wild,mutant)) if a!=b]
                require(sorted(positions)==changed and len(changed)==int(r['k']),'Mutation count/position mismatch')
            model,t=load_cnn(cfg); span=t['max_log10']-t['min_log10']
            delta=span*(cnn_predict(model,encode([r['mutant_sequence'] for r in mutants],87))-cnn_predict(model,encode([r['wild_type_sequence'] for r in mutants],87)))
            require(np.allclose(delta,[float(r['predicted_delta_log10']) for r in mutants],atol=1e-6),'Mutation prediction mismatch')
        elif kind=='transfer':
            from .data import TARGET
            target=read_table(TARGET/'samples.tsv'); assignment=read_table(directory/'target_split.tsv')
            require([r['sample_id'] for r in assignment]==[r['sample_id'] for r in target],'Target split coverage mismatch')
            sets={s:{r['sequence_group'] for r in assignment if r['split']==s} for s in ['train','val','test']}
            require(not sets['train']&sets['val'] and not sets['train']&sets['test'] and not sets['val']&sets['test'],'Target group leakage')
            train_ids={r['sample_id'] for r in assignment if r['split']=='train'}
            for file in directory.glob('*_fit_ids.tsv'):
                require({r['sample_id'] for r in read_table(file)}<=train_ids,'Few-shot fitting includes heldout targets')
            proof_target=read_json(directory/'transfer_manifest.json')
            require(sha(TARGET/'samples.tsv')==proof_target['target_samples_sha256'],'Transfer target identity mismatch')
        results.append(kind+'_research_integrity')
    return dict(status='passed',dataset=proof,checks=results,common_validation_count=len(val_ids),test_evaluated=False,visual_review_required=visual)


def publish(cfg):
    proof=verify(cfg,visual=True)
    directory=ROOT/'results/context87'; directory.mkdir(parents=True,exist_ok=True)
    import shutil
    for name in ['metrics.csv','predictions_val.csv','comparison_manifest.json']:
        shutil.copy2(path(cfg,'comparison')/name,directory/name)
    write_json(directory/'verification.json',proof)
    write_json(directory/'current.json',dict(schema_version='3.0.0',dataset_id=cfg['dataset_id'],data_version=cfg['data_version'],
               split_id=cfg['split_id'],runs=cfg['runs'],metrics='results/context87/metrics.csv',test_evaluated=False))
    current=ROOT/'results/current.json'; historical=ROOT/'results/legacy50_current.json'
    if current.exists() and not historical.exists(): shutil.copy2(current,historical)
    write_json(current,read_json(directory/'current.json'))
    return directory
