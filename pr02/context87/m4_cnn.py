"""Predeclared four configurations x three seeds, with fixed representative seed."""
from __future__ import annotations
import copy
import json
import shutil

import numpy as np

from .data import ROOT, load, read_json, read_table, write_json, write_table
from .experiment import cnn_class, cnn_predict, input_hashes, masks, train_network
from .features import encode
from .m4_data_eval import SEEDS, bundle, completed, export_predictions, finish, run_root


def load_model(directory, length=87):
    import torch
    checkpoint = torch.load(directory/'model.pt', map_location='cpu', weights_only=True)
    config = dict(checkpoint['model_config']); config['sequence_length'] = length
    model = cnn_class()(config); model.load_state_dict(checkpoint['state_dict']); model.eval()
    return model, checkpoint['transform']


def embedding(model, x):
    import torch
    model.eval()
    with torch.no_grad():
        return np.concatenate([model.pool(model.features(torch.from_numpy(x[i:i+128]))).flatten(1).numpy() for i in range(0, len(x), 128)])


def run(cfg, protocol):
    import torch
    root = run_root(protocol)/'cnn'
    if completed(root): return root
    root.mkdir(parents=True, exist_ok=True)
    rows, transform = load(cfg); m = masks(rows); x = encode([r['sequence'] for r in rows], 87)
    y = np.array([float(r['target_log10']) for r in rows]); low = transform['min_log10']; span = transform['max_log10']-low
    all_search = []
    for condition, pooling, dropout, decay in [('C0', 5, .3, .001), ('C1', 20, .3, .001), ('C2', 20, .1, .001), ('C3', 20, .1, .0001)]:
        for seed in SEEDS:
            directory = root/f'{condition}_s{seed}'
            if completed(directory):
                all_search.append(read_json(directory/'selection_record.json')); continue
            directory.mkdir(parents=True, exist_ok=True)
            current = copy.deepcopy(cfg); current['seed'] = seed
            current['cnn']['model'].update(pooled_length=pooling, dropout=dropout)
            current['cnn']['training']['weight_decay'] = decay
            print(f'M4 CNN {condition} seed={seed}', flush=True)
            if condition == 'C0' and seed == SEEDS[0]:
                old = ROOT/'runs'/cfg['runs']['cnn']
                checkpoint = torch.load(old/'model.pt', map_location='cpu', weights_only=True)
                assert checkpoint['model_config'] == current['cnn']['model']
                assert read_json(old/'run_manifest.json')['config'] == current
                shutil.copyfile(old/'model.pt', directory/'model.pt')
                history = read_table(old/'training_history.csv')
                model, _ = load_model(directory)
                reuse = old.relative_to(ROOT).as_posix()
            else:
                model, history = train_network(current, x, (y-low)/span, m['train'], m['val'])
                torch.save(dict(state_dict=model.state_dict(), model_config=current['cnn']['model'], transform=transform,
                                input_hashes=input_hashes(cfg)), directory/'model.pt')
                reuse = None
            write_json(directory/'config.json', current)
            write_table(directory/'training_history.csv', history)
            p = low+span*cnn_predict(model, x)
            export_predictions(cfg, 'cnn', directory, rows, p)
            record = dict(condition=condition, seed=seed, pooling=pooling, dropout=dropout, weight_decay=decay,
                          val_mse_log10=float(np.mean((p[m['val']]-y[m['val']])**2)),
                          best_epoch=int(min(history, key=lambda r: float(r['val_mse']))['epoch']), epochs=len(history), reused_run=reuse)
            write_json(directory/'selection_record.json', record)
            finish(directory, cfg, record)
            all_search.append(record)
    winner = min(('C0', 'C1', 'C2', 'C3'), key=lambda c: np.mean([r['val_mse_log10'] for r in all_search if r['condition'] == c]))
    chosen = {str(seed): (root/f'{winner}_s{seed}').relative_to(ROOT).as_posix() for seed in SEEDS}
    for seed, source in chosen.items():
        directory = ROOT/source; model, _ = load_model(directory)
        bundle(root/f'features_s{seed}', embedding(model, x), rows, 'supervised_cnn_pooled', cfg,
               dict(source_run=source, checkpoint_sha256=__import__('hashlib').sha256((directory/'model.pt').read_bytes()).hexdigest(),
                    fit_subset='train', seed=int(seed), pooling=model.pool.output_size))
    write_table(root/'search.csv', all_search)
    write_json(root/'selected_model.json', dict(condition=winner, runs=chosen, representative_seed=SEEDS[0],
                representative_run=chosen[str(SEEDS[0])], selection='minimum mean validation log10 MSE over three seeds',
                length_policy='87/61 bp explicit adaptive-average-pooling model instances, identical trained weights'))
    finish(root, cfg, dict(selected_condition=winner, n_configurations=4, n_seeds=3))
    return root
