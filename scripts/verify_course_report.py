"""Verify report tables, prediction metrics, document content and release hashes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/final'
R0 = ROOT / 'runs/m4_complete_20261014_v1'
R1 = ROOT / 'runs/m4_refinement_20261014_v2'
SEEDS = [20260928, 20260929, 20260930]
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
METRICS = ['r2_log10', 'spearman', 'rmse_log10', 'mae_log10']
INPUTS = {}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = digest(path)
    return json.loads(path.read_text())


def rows(path):
    INPUTS[path.relative_to(ROOT).as_posix()] = digest(path)
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter='\t' if path.suffix == '.tsv' else ','))


def close(actual, expected):
    if expected is None:
        assert actual in (None, ''), (actual, expected)
    else:
        assert np.isclose(float(actual), expected, rtol=1e-9, atol=1e-10), (actual, expected)


def score(y, p):
    return dict(r2_log10=1 - np.square(y-p).sum()/np.square(y-y.mean()).sum(),
                spearman=float(spearmanr(y, p).statistic) if np.ptp(p) else None,
                rmse_log10=float(np.sqrt(np.square(y-p).mean())),
                mae_log10=float(np.abs(y-p).mean()))


def aggregate(summary, records, key='method'):
    assert {r[key] for r in summary} == {r[key] for r in records}
    for row in summary:
        selected = [r for r in records if r[key] == row[key]]
        assert len(selected) == int(row['n_runs'])
        for metric in METRICS:
            values = [float(r[metric]) for r in selected if r[metric] not in ('', None)]
            close(row[metric+'_mean'], float(np.mean(values)) if values else None)
            close(row[metric+'_sd'], float(np.std(values, ddof=1)) if len(values) > 1 else None)


def verify(record=False):
    content = read(OUT/'report_content.json')
    for path, expected in content['source_hashes'].items():
        assert digest(ROOT/path) == expected, path
    INPUTS.update(content['source_hashes'])
    source = rows(ROOT/'data/03_Ecoli_strength87/samples.tsv')
    lookup = {r['sample_id']: r for r in source}
    assert len(lookup) == 11809
    assert Counter(r['split'] for r in source) == {'train': 8267, 'val': 1768, 'test': 1774}
    val_ids = [r['sample_id'] for r in source if r['split'] == 'val']
    ytrain = np.array([float(r['target_log10']) for r in source if r['split'] == 'train'])
    yval = np.array([float(lookup[i]['target_log10']) for i in val_ids])
    assert all(len(r['sequence']) == 87 and r['tss_index_0based'] == '67' for r in source)

    checked_predictions = 0
    def predictions(path, ids, subset, method=None):
        nonlocal checked_predictions
        data = [r for r in rows(path) if r['subset'] == subset and
                (method is None or r['method_name'] == method)]
        by_id = {r['sample_id']: r for r in data}
        assert len(data) == len(by_id) == len(ids) and set(by_id) == set(ids), path
        p = []
        for i in ids:
            r = by_id[i]
            label = r.get('observed_log10', r.get('true_value_log10'))
            close(label, float(lookup[i]['target_log10']))
            p.append(float(r.get('predicted_log10', r.get('predicted_value_log10'))))
        checked_predictions += 1
        return np.array(p)

    refined_records = rows(R1/'regularization/metrics.csv')
    refined = rows(R1/'regularization/summary.csv')
    for row in refined_records:
        name = row['method'] + ('_s'+row['seed'] if row['seed'] else '')
        values = score(yval, predictions(R1/'regularization'/name/'predictions.csv', val_ids, 'val'))
        for key, value in values.items(): close(row[key], value)
        assert float(row['alpha']) == 10000 and row['alpha_at_grid_boundary'] == 'False'
    aggregate(refined, refined_records)

    original = {r['method']: r for r in rows(R0/'evaluation/model_summary.csv')}
    original_records = []
    for method in ['knn', 'linear_svr', 'thermo', 'train_mean']:
        p = np.full(len(yval), ytrain.mean()) if method == 'train_mean' else predictions(
            R0/'evaluation/predictions.csv', val_ids, 'val', 'svr' if method == 'linear_svr' else method)
        original_records.append(dict(method=method, **score(yval, p)))
    cnn = []
    for seed in SEEDS:
        p = predictions(R0/f'cnn/C0_s{seed}/predictions_val.csv', val_ids, 'val')
        cnn.append(p);original_records.append(dict(method='cnn', **score(yval, p)))
    aggregate([original[k] for k in ['knn', 'linear_svr', 'thermo', 'cnn', 'train_mean']], original_records)
    for item in content['checks']['main_scores']:
        row = next((r for r in refined if r['method'] == item['method']), original.get(item['method']))
        close(item['r2'], float(row['r2_log10_mean']))

    group_split = rows(R1/'grouped/split.tsv')
    group_ids = [r['sample_id'] for r in group_split if r['split'] == 'evaluation']
    group_train = [r['sample_id'] for r in group_split if r['split'] == 'train']
    assert len(group_ids) == 1998 and len(group_train) == 5985
    yg = np.array([float(lookup[i]['target_log10']) for i in group_ids])
    group_mean = np.mean([float(lookup[i]['target_log10']) for i in group_train])
    grouped_records = rows(R1/'grouped/metrics.csv')
    for row in grouped_records:
        name = row['method'] + ('_s'+row['seed'] if row['seed'] else '')
        p = np.full(len(yg), group_mean) if row['method'] == 'train_mean' else predictions(
            R1/'grouped'/name/'predictions.csv', group_ids, 'evaluation')
        for key, value in score(yg, p).items(): close(row[key], value)
    aggregate(rows(R1/'grouped/summary.csv'), grouped_records)

    ridge = predictions(R1/'regularization/kmer_ridge/predictions.csv', val_ids, 'val')
    pc = predictions(ROOT/'runs/ecoli87_thermo_20261013_v1/predictions_val.csv', val_ids, 'val')
    bands = np.digitize(yval, np.quantile(ytrain, [1/3, 2/3]))
    for band, item in enumerate(content['checks']['group_mae']):
        mask = bands == band
        assert item['n'] == int(mask.sum())
        close(item['cnn_mae'], np.mean([np.abs(p[mask]-yval[mask]).mean() for p in cnn]))
        for key, p in [('ridge', ridge), ('pc', pc), ('mean', np.full(len(yval), ytrain.mean()))]:
            close(item[key+'_mae'], np.abs(p[mask]-yval[mask]).mean())
    assert content['checks']['group_mae'][1]['mean_mae'] < content['checks']['group_mae'][1]['pc_mae']

    transfer = rows(R1/'transfer_stability/summary.csv')
    transfer_records = rows(R1/'transfer_stability/metrics.csv')
    assert len(transfer_records) == 32
    aggregate(transfer, transfer_records, key='condition')
    target = rows(ROOT/'data/04_Bsub_strength61/samples.tsv')
    assert len(target) == 591 and len({r['sequence'] for r in target}) == 580
    target_split = rows(R1/'transfer_stability/target_split.tsv')
    assert Counter(r['split'] for r in target_split) == {'train': 354, 'val': 119, 'test': 118}
    target_assignment = {r['sample_id']: r['split'] for r in target_split}
    assert set(target_assignment) == {r['sample_id'] for r in target}
    for sequence in {r['sequence'] for r in target}:
        assert len({target_assignment[r['sample_id']] for r in target if r['sequence'] == sequence}) == 1

    parents = rows(R0/'mutation/mutation_parent_summary.csv')
    for row in rows(R0/'mutation/mutation_summary.csv'):
        selected = [r for r in parents if r['group'] == row['group']]
        assert len(selected) == 64 and sum(int(r['n_mutants']) for r in selected) == 640
        for metric in ['mean_absolute_delta_log10', 'mean_delta_log10']:
            close(row[metric], np.mean([float(r[metric]) for r in selected]))
    direction = next(r for r in rows(R0/'mutation/direction_summary.csv') if r['expected_direction'] == 'all')
    dparents = [r for r in rows(R0/'mutation/direction_parent_summary.csv') if r['expected_direction'] == 'all']
    assert len(dparents) == 64 and sum(int(r['n_mutants']) for r in dparents) == 960
    for key in ['cnn_agreement', 'thermo_agreement', 'cross_model_agreement']:
        close(direction[key], np.average([float(r[key]) for r in dparents], weights=[int(r['n_mutants']) for r in dparents]))
    regional = rows(R1/'attribution_stability/region_across_seeds.csv')
    assert len([r for r in regional if r['region'] == '-10']) == 8
    assert all(r['positive_seed_count'] == '3' for r in regional if r['region'] == '-10')

    blocks = [b for s in content['sections'] for b in s['blocks']]
    tables = [b for b in blocks if b['type'] == 'table']
    figures = [b for b in blocks if b['type'] == 'figure']
    assert len(content['sections']) == 15 and len(tables) == 14 and len(figures) == 9
    assert [b['number'] for b in tables] == list(range(1, 15))
    assert [b['number'] for b in figures] == list(range(1, 10))
    assert {b['path'] for b in figures} == {f['path'] for f in content['figures']}
    for b in figures:
        assert (OUT/b['path']).is_file() and (OUT/b['path']).with_suffix('.svg').is_file()
    md = (OUT/'PR02-01_REPORT.md').read_text()
    for text in [b['text'] for b in blocks if b['type'] in ('paragraph', 'subheading', 'code')]:
        assert text in md
    with zipfile.ZipFile(OUT/'PR02-01_REPORT.docx') as z:
        document = ET.fromstring(z.read('word/document.xml'))
        paragraphs = [''.join(e.itertext()) for e in document.iter(W+'t')]
        flat_text = ''.join(paragraphs)
        for b in blocks:
            if b['type'] in ('paragraph', 'subheading'): assert b['text'] in flat_text
        word_tables = document.findall('.//'+W+'tbl')
        assert len(word_tables) == len(tables)
        for actual, expected in zip(word_tables, tables):
            cells = [[''.join(t.text or '' for t in c.iter(W+'t')) for c in r.findall(W+'tc')] for r in actual.findall(W+'tr')]
            assert cells == [[str(c) for c in row] for row in [expected['headers']]+expected['rows']], expected['number']
        assert len(document.findall('.//'+W+'drawing')) == 9
        assert not any(document.findall('.//'+W+tag) for tag in ['ins', 'del', 'commentRangeStart', 'commentReference'])
        assert not any('comments' in name.lower() for name in z.namelist())
        core = ET.fromstring(z.read('docProps/core.xml'))
        assert core.find('{http://purl.org/dc/elements/1.1/}creator').text == 'PR02-01 项目组'
        assert not core.find('{http://schemas.openxmlformats.org/package/2006/metadata/core-properties}lastModifiedBy').text
        for node in ET.fromstring(z.read('docProps/app.xml')):
            if node.tag.rsplit('}', 1)[-1] in ['Application', 'AppVersion', 'Company', 'Manager', 'Template']:
                assert not node.text
    unwanted = re.compile(r'\b(?:agent|codex|chatgpt|openai)\b|本轮|我会|我已|已帮|按你的|用户要求|待确认|下一步我|作为AI|TODO|TBD|/Users/|turn\d+(?:search|view)|filecite', re.I)
    current_docs = [ROOT/'README.md', ROOT/'reports/PROJECT_STATUS.md', ROOT/'reports/PROJECT_REFINEMENT_20261014.md']
    for text in [md, flat_text]+[p.read_text() for p in current_docs]:
        assert not unwanted.search(text), unwanted.search(text)
    assert len(content['references']) == 7
    assert {int(n) for n in re.findall(r'\[(\d+)\]', md)} == set(range(1, 8))

    # Layout review is tied to the exact delivered Word file, not a previous draft.
    layout = read(OUT/'layout_review.json')
    assert layout['status'] == 'passed' and layout['document_sha256'] == digest(OUT/'PR02-01_REPORT.docx')
    assert layout['page_count'] == 15 and layout['reviewed_pages'] == list(range(1, 16))
    release = read(R0/'release_manifest.json')
    for name, expected in release['artifacts'].items():
        path = ROOT/name
        if path.is_file() and digest(path) == expected: continue
        assert name == 'README.md' or name.startswith('pr02/'), name
        assert digest(ROOT/'reports/release_snapshots'/(expected+path.suffix)) == expected, name

    checks = dict(status='passed', sections=15, tables=14, figures=9, reviewed_pages=15,
                  source_records=11809, target_records=591, prediction_exports_recomputed=checked_predictions,
                  refined_heads=11, transfer_fits=32, numerical_aggregates='passed',
                  conditional_baseline='passed', document_content_and_tables='passed',
                  current_document_prose='passed', metadata_and_revisions='passed',
                  original_release_and_ppt='preserved', source_test_scored=False)
    if record:
        (OUT/'report_checks.json').write_text(json.dumps(checks, indent=2)+'\n')
        artifacts = [p for p in OUT.rglob('*') if p.is_file() and p.name != 'report_manifest.json']
        artifacts += current_docs + [ROOT/'requirements_report.txt']
        artifacts += [ROOT/'scripts'/n for n in ['build_course_report.py', 'build_course_docx.py', 'verify_course_report.py']]
        manifest = dict(schema_version='course_report_v1', report_date='2026-10-14',
                        experimental_runs=[R0.relative_to(ROOT).as_posix(), R1.relative_to(ROOT).as_posix()],
                        sources=INPUTS, artifacts={p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(artifacts))})
        (OUT/'report_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
        index = dict(report='reports/final/PR02-01_REPORT.md', word='reports/final/PR02-01_REPORT.docx',
                     manifest='reports/final/report_manifest.json', manifest_sha256=digest(OUT/'report_manifest.json'),
                     checks='reports/final/report_checks.json')
        (ROOT/'results/context87/report_current.json').write_text(json.dumps(index, indent=2)+'\n')
    else:
        manifest = json.loads((OUT/'report_manifest.json').read_text())
        for path, expected in {**manifest['sources'], **manifest['artifacts']}.items():
            assert digest(ROOT/path) == expected, path
        assert json.loads((OUT/'report_checks.json').read_text()) == checks
        index = json.loads((ROOT/'results/context87/report_current.json').read_text())
        assert index['manifest_sha256'] == digest(OUT/'report_manifest.json')
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--record', action='store_true', help='Record verified hashes after final content and layout review.')
    verify(parser.parse_args().record)
