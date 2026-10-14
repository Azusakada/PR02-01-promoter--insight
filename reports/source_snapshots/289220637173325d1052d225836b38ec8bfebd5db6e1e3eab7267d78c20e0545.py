"""Pinned official DNABERT-2, frozen float32 inference, padding-aware mean pooling."""
from __future__ import annotations
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np

from .data import ROOT, load, require, sha, write_json, write_table
from .m4_data_eval import bundle, completed, finish, run_root


def encode_sequences(sequences, model_dir, device='cpu', batch_size=32):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from transformers.models.bert.configuration_bert import BertConfig
    torch.set_num_threads(2)
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True, local_files_only=True)
    config = BertConfig.from_pretrained(str(model_dir), local_files_only=True)
    model, info = AutoModel.from_pretrained(str(model_dir), config=config, trust_remote_code=True,
                   local_files_only=True, output_loading_info=True)
    require(not info['missing_keys'] and not info['mismatched_keys'], 'Encoder has missing or mismatched pretrained weights')
    model.eval().to(device)
    for parameter in model.parameters(): parameter.requires_grad_(False)
    outputs = []; token_records = []; start = time.monotonic()
    with torch.inference_mode():
        for offset in range(0, len(sequences), batch_size):
            batch = sequences[offset:offset+batch_size]
            encoded = tokenizer(batch, padding=True, truncation=False, return_tensors='pt', return_special_tokens_mask=True)
            ids = encoded['input_ids']; special = encoded.pop('special_tokens_mask')
            valid = encoded['attention_mask'].bool() & ~special.bool()
            require(bool(valid.sum(1).min() > 0) and ids.shape[1] <= config.max_position_embeddings, 'Invalid/truncated DNA tokens')
            for i, sequence in enumerate(batch):
                tokens = tokenizer.convert_ids_to_tokens(ids[i][valid[i]].tolist())
                require(''.join(tokens) == sequence, 'Tokenization does not preserve full DNA sequence')
                token_records.append(dict(sequence_length=len(sequence), n_sequence_tokens=int(valid[i].sum()), n_total_tokens=int(encoded['attention_mask'][i].sum()), truncated=False))
            encoded['token_type_ids'] = torch.zeros_like(ids)
            hidden = model(**{k: v.to(device) for k, v in encoded.items()})[0]
            weights = valid.to(device=device, dtype=hidden.dtype).unsqueeze(-1)
            pooled = (hidden*weights).sum(1)/weights.sum(1)
            outputs.append(pooled.cpu().numpy().astype(np.float32))
            if offset == 0 or offset//batch_size % 25 == 0:
                print(f'DNABERT-2 {min(offset+batch_size,len(sequences))}/{len(sequences)} elapsed={time.monotonic()-start:.1f}s', flush=True)
    return np.concatenate(outputs), token_records, info


def run(cfg, protocol, model_dir=None, device='cpu'):
    directory = run_root(protocol)/'dna_features'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    if model_dir is None:
        from huggingface_hub import snapshot_download
        model_dir = snapshot_download(protocol['dna_model'], revision=protocol['dna_revision'])
    model_dir = Path(model_dir)
    rows, _ = load(cfg)
    x, tokens, info = encode_sequences([r['sequence'] for r in rows], model_dir, device=device)
    for r, token in zip(rows, tokens): token['sample_id'] = r['sample_id']
    hashes = {p.name: sha(p) for p in model_dir.iterdir() if p.is_file() and p.suffix in ('.bin', '.json', '.py')}
    details = dict(model=protocol['dna_model'], revision=protocol['dna_revision'], frozen=True,
        course_label_usage='none', pooling='mean sequence tokens, excluding padding and CLS/SEP',
        truncation=False, dtype='float32', device=device, local_files=hashes,
        transformers_version=importlib.metadata.version('transformers'), loading_info=info,
        provenance='https://huggingface.co/'+protocol['dna_model']+'/tree/'+protocol['dna_revision'])
    bundle(directory, x, rows, 'frozen_pretrained_dnabert2', cfg, details)
    write_table(directory/'tokenization_audit.csv', tokens)
    finish(directory, cfg, details)
    return directory
