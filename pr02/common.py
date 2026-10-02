from __future__ import annotations

import csv
import importlib.metadata
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/01_Ecoli_strength/data_v1.tsv'
SPLITS = ROOT / 'data/01_Ecoli_strength/split_manifest.tsv'
TRANSFORM = ROOT / 'data/01_Ecoli_strength/label_transform.json'
CONTRACTS = ROOT / 'contracts/scripts'
sys.path.insert(0, str(CONTRACTS))
import validate as v
import label_transform as label
from api_contract import ModelArtifact, PredictionBundle


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def resolve(path):
    path = Path(path)
    path = (ROOT / path if not path.is_absolute() else path).resolve()
    v.require(path.is_relative_to(ROOT), 'Path must remain inside repository')
    return path


def fresh_dir(path):
    path = resolve(path)
    v.require(not path.exists() or not any(path.iterdir()), f'Refusing to overwrite run: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(run_id):
    v.require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}', run_id)), 'Invalid run_id')
    return fresh_dir(ROOT / 'runs' / run_id)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_table(path, rows, columns=None, delimiter=','):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None:
        v.require(bool(rows), 'Empty table requires explicit columns')
        columns = list(rows[0])
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter, lineterminator='\n')
        writer.writeheader()
        for row in rows:
            writer.writerow({key: '' if value is None else value for key, value in row.items()})


def read_table(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def load_bundle():
    v.bundle_check(DATA, SPLITS, transform=TRANSFORM)
    samples = v.load_table('dataset', DATA)
    splits = v.load_table('splits', SPLITS)
    transform = read_json(TRANSFORM)
    split_id = transform['split_id']
    split_map = {r['sample_id']:r['split'] for r in splits if r['split_id'] == split_id}
    groups = {sub:sorted([r for r in samples if split_map[r['sample_id']] == sub], key=lambda r:r['sample_id'])
              for sub in ['train','val','test']}
    return samples, groups, transform


def prediction_base(sample, method, run_id, split_id, subset, scale, *, checkpoint=None, seed=None, calibration_id=None):
    return dict(dataset_id=sample['dataset_id'], sample_id=sample['sample_id'], true_value=sample['strength'],
                predicted_value=None, predicted_value_log10=None, method_name=method, target_scale_model=scale,
                model_checkpoint=checkpoint, predicted_tx_rate=None, thermo_mode=None, calibration_id=calibration_id,
                run_id=run_id, split_id=split_id, subset=subset, seed=seed, prediction_status='failed',
                error_reason='not_processed', predicted_value_normalized=None, label_transform_id=None,
                schema_version='2.0.0', data_version=sample['data_version'])


def artifact(kind, path):
    return {'kind':kind, 'path':relative(path), 'sha256':v.sha256(Path(path))}


def git(*args, binary=False):
    executable = os.environ.get('PR02_GIT') or shutil.which('git')
    v.require(bool(executable), 'Set PR02_GIT or add Git to PATH')
    raw = subprocess.check_output([executable,'-C',str(ROOT),*args])
    return raw if binary else raw.decode('utf-8').strip()


def write_manifest(output, run_id, method, artifacts, *, parameters, fit=(), selection=(), evaluation=('val',),
                   status='success', seed=None, evidence='preliminary'):
    _, _, transform = load_bundle()
    dirty = bool(git('status','--porcelain'))
    if dirty:
        # All executable source is committed before formal runs. Capture subsequent tracked edits.
        patch = Path(output)/'source_patch.diff'
        patch.write_bytes(git('diff','HEAD',binary=True))
        artifacts = [*artifacts, artifact('source_patch', patch)]
    packages = {}
    for name in ['numpy','scipy','scikit-learn','pandas','biopython','torch','PyYAML','matplotlib']:
        try: packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    manifest = dict(dataset_id=transform['dataset_id'], schema_version='2.0.0', run_id=run_id,
                    stage='M3', task_id='pr02_integrated_m2_m3', method_name=method,
                    data_version=transform['data_version'], split_id=transform['split_id'],
                    evidence_level=evidence, execution_status=status, seed=seed,
                    created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),
                    command=[sys.executable,*sys.argv], code_commit=git('rev-parse','HEAD'), dirty_worktree=dirty,
                    environment=dict(python=platform.python_version(), platform=platform.platform(),packages=packages),
                    parameters=parameters, fit_subsets=list(fit), selection_subsets=list(selection),
                    evaluation_subsets=list(evaluation), artifacts=list(artifacts))
    path = Path(output)/'run_manifest.json'
    write_json(path, manifest)
    v.run_check(path, ROOT)
    return path
