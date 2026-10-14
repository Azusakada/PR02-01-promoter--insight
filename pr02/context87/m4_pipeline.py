"""Reproducible M4 entry point and model adapters, separate from frozen M3 runs."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from .data import ROOT, read_json, require, sha, write_json
from .features import encode, kmers, thermal_states
from .m4_data_eval import initialize, run_root


def write_model_refs(cfg, protocol):
    """Publish explicit adapters without modifying completed model directories."""
    from .experiment import input_hashes
    root=run_root(protocol); selected=read_json(root/'cnn/selected_model.json'); folder=root/'model_refs'
    for kind in ('ridge','svr','dnabert2'):
        source=root/'ml'/kind; ref=read_json(source/'model_ref.json')
        ref.update(input_hashes=input_hashes(cfg), transform_path=cfg['label_transform'],
                   model_config_path=(source/'selected_parameters.json').relative_to(ROOT).as_posix() if kind!='dnabert2'
                     else (source/'hyperparameter_search.csv').relative_to(ROOT).as_posix())
        write_json(folder/(kind+'.json'),ref)
    for seed,run in selected['runs'].items():
        source=ROOT/run
        write_json(folder/f'cnn_s{seed}.json',dict(model_kind='cnn',method_name='cnn',run_id=run,
            checkpoint_path=(source/'model.pt').relative_to(ROOT).as_posix(),checkpoint_sha256=sha(source/'model.pt'),
            model_config_path=(source/'config.json').relative_to(ROOT).as_posix(),transform_path=cfg['label_transform'],
            input_hashes=input_hashes(cfg),path_base='repository',seed=int(seed),input_lengths=[87,61],
            predictor_entry='pr02.context87.m4_pipeline:predict_log10'))
    source=ROOT/'runs'/cfg['runs']['thermo']
    write_json(folder/'thermo.json',dict(model_kind='thermo',method_name='thermo',run_id=cfg['runs']['thermo'],
        checkpoint_path=(source/'calibration.json').relative_to(ROOT).as_posix(),checkpoint_sha256=sha(source/'calibration.json'),
        path_base='repository',model_config_path='project_config.json',transform_path=(source/'calibration.json').relative_to(ROOT).as_posix(),
        input_hashes=input_hashes(cfg),input_length=87,tss_index_0based=67,seed=None,predictor_entry='pr02.context87.m4_pipeline:predict_log10'))


def predict_log10(sequences, model_ref):
    """Predict from an explicit repository-relative model reference, never implicit latest."""
    if not isinstance(model_ref, dict): model_ref = read_json(model_ref)
    checkpoint = ROOT/model_ref['checkpoint_path']
    require(sha(checkpoint) == model_ref['checkpoint_sha256'], 'Model checksum mismatch')
    kind = model_ref['model_kind']
    if kind in ('ridge', 'svr'):
        saved = joblib.load(checkpoint)
        return saved['model'].predict(kmers(sequences, saved['parameters']['k']).astype(np.float32))
    if kind == 'cnn':
        from .m4_cnn import load_model
        from .experiment import cnn_predict
        require(len({len(s) for s in sequences}) == 1, 'CNN batch requires one explicit length')
        length = len(sequences[0]); require(length in (87, 61), 'Unsupported CNN input length')
        model, t = load_model(checkpoint.parent, length)
        return t['min_log10']+(t['max_log10']-t['min_log10'])*cnn_predict(model, encode(sequences, length))
    if kind == 'thermo':
        calibration = read_json(checkpoint)
        return np.array([calibration['coef_a']*np.log10(r['Tx_rate'])+calibration['intercept_b'] for r in thermal_states(sequences)])
    if kind == 'dnabert2_ridge':
        from .m4_dnalm_features import encode_sequences
        from huggingface_hub import snapshot_download
        details = model_ref['details']; model_dir = snapshot_download(details['encoder'], revision=details['revision'])
        x, _, _ = encode_sequences(sequences, model_dir)
        return joblib.load(checkpoint)['model'].predict(x)
    raise ValueError('Unknown model reference: '+kind)


def run(cfg, protocol, stage='all', model_dir=None, device='cpu', reviewed=None):
    if stage == 'verify':
        from .m4_verify import verify
        return verify(cfg, protocol)
    initialize(cfg, protocol)
    if stage in ('all', 'ml'):
        from .m4_ml_dnalm import run_traditional
        run_traditional(cfg, protocol)
    if stage in ('all', 'dna'):
        from .m4_dnalm_features import run as dna
        from .m4_ml_dnalm import run_dna_head
        dna(cfg, protocol, model_dir=model_dir, device=device); run_dna_head(cfg, protocol)
    if stage in ('all', 'cnn'):
        from .m4_cnn import run as cnn
        cnn(cfg, protocol)
    if stage in ('all', 'attribution'):
        from .m4_attribution import run as attribution
        from .m4_explanation_analysis import run as explanation
        attribution(cfg, protocol); explanation(cfg, protocol)
    if stage in ('all', 'mutation'):
        from .m4_thermo_mutation import run as mutation
        mutation(cfg, protocol)
    if stage in ('all', 'ablation'):
        from .m4_ablation import run as ablation
        ablation(cfg, protocol)
    if stage in ('all', 'transfer'):
        from .m4_transfer import run as transfer
        transfer(cfg, protocol)
    if stage in ('all', 'evaluate'):
        from .m4_data_eval import evaluate
        evaluate(cfg, protocol)
    if stage in ('all', 'figures'):
        from .m4_reporting import figures
        figures(cfg, protocol)
    if stage in ('all', 'report'):
        from .m4_final_report import build
        from .m4_case_notes import build as build_cases
        write_model_refs(cfg, protocol)
        build_cases(cfg, protocol)
        build(cfg, protocol)
    if stage == 'acknowledge':
        from .m4_reporting import acknowledge
        return acknowledge(cfg, protocol, reviewed or [])
    return run_root(protocol)
