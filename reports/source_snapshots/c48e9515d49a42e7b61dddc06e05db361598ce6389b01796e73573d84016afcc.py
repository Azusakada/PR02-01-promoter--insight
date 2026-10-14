"""Reproducible M4 entry point and model adapters, separate from frozen M3 runs."""
from __future__ import annotations
from pathlib import Path
import joblib
import numpy as np
from .data import ROOT, read_json, require, sha
from .features import encode, kmers, thermal_states
from .m4_data_eval import initialize, run_root


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


def run(cfg, protocol, stage='all', model_dir=None, device='cpu'):
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
    if stage == 'verify':
        from .m4_verify import verify
        return verify(cfg, protocol)
    return run_root(protocol)
