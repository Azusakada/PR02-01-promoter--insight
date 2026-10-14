"""Pre-saved mutation designs, local direction tests and fixed-TSS PC comparison."""
from __future__ import annotations
import json
import numpy as np
from .data import ROOT, load, read_json, read_table, require, sha, write_json, write_table
from .experiment import cnn_predict
from .features import encode, thermal_states
from .m4_attribution import direction, log_model
from .m4_data_eval import SEEDS, bootstrap, completed, finish, run_root
from .research import ranks


def design_mutations(master, scores, expectations, protocol, expectation_sha):
    rng = np.random.default_rng(SEEDS[0]); output = []
    def add(sid, positions, alternatives, experiment, group, repeat):
        sequence = master[sid]['sequence']; mutant = list(sequence)
        for pos, alt in zip(positions, alternatives): mutant[pos] = alt
        mutant = ''.join(mutant)
        require(sum(a != b for a, b in zip(sequence, mutant)) == len(positions), 'Incorrect mutation count')
        record = dict(mutation_id=f'm4mut_{len(output)+1:05d}', sample_id=sid, experiment_type=experiment, group=group,
            repeat=repeat, k=len(positions), positions_0based=json.dumps(list(map(int, positions))),
            ref_bases=json.dumps([sequence[pos] for pos in positions]), alt_bases=json.dumps(list(alternatives)),
            wild_type_sequence=sequence, mutant_sequence=mutant, design_seed=SEEDS[0],
            expected_delta_log10='', expected_direction='', expectation_source_sha256=expectation_sha)
        if experiment == 'single_direction':
            value = expectations[sid, int(positions[0]), alternatives[0]]
            record.update(expected_delta_log10=value, expected_direction=direction(value, protocol))
        output.append(record)
    for sid in scores:
        order = ranks(scores[sid]); high = order[:5]; low = order[-5:]
        pool = [p for p in range(87) if p not in set(high) | set(low)]
        for repeat in range(10):
            for group, positions in [('high', high), ('low', low), ('random', rng.choice(pool, 5, replace=False))]:
                alts = [str(rng.choice([b for b in 'ACGT' if b != master[sid]['sequence'][pos]])) for pos in positions]
                add(sid, positions, alts, 'amplitude', group, repeat)
        for pos in high:
            for alt in 'ACGT':
                if alt != master[sid]['sequence'][pos]: add(sid, [pos], [alt], 'single_direction', 'high', 0)
    return output


def run(cfg, protocol):
    root = run_root(protocol); directory = root/'mutation'
    if completed(directory): return directory
    directory.mkdir(parents=True, exist_ok=True)
    rows, _ = load(cfg); master = {r['sample_id']: r for r in rows}
    attrs = read_table(root/'attribution/attributions.tsv'); scores = {}
    for r in attrs:
        if r['method'] == 'integrated_gradients': scores.setdefault(r['sample_id'], np.zeros(87))[int(r['position_0based'])] = float(r['score_log10'])
    file = root/'attribution/substitution_expectation.csv'
    expectations = {(r['sample_id'], int(r['position_0based']), r['alt_base']): float(r['expected_delta_log10']) for r in read_table(file)}
    design = design_mutations(master, scores, expectations, protocol, sha(file))
    # Persist design and its identity before any mutant prediction.
    if (directory/'mutation_design.csv').exists():
        require(read_table(directory/'mutation_design.csv') == [{k: str(v) for k, v in r.items()} for r in design], 'Existing mutation design differs')
    else: write_table(directory/'mutation_design.csv', design)
    write_json(directory/'design_frozen.json', dict(design_sha256=sha(directory/'mutation_design.csv'), n_mutants=len(design),
               amplitude_count=1920, single_direction_count=960, frozen_before_predictions=True))
    selected = read_json(root/'cnn/selected_model.json'); model = log_model(ROOT/selected['representative_run'])
    unique = sorted({r['mutant_sequence'] for r in design})
    cnn_values = cnn_predict(model, encode(unique, 87)); cnn_map = dict(zip(unique, map(float, cnn_values)))
    print('M4 fixed-TSS thermodynamics for', len(unique), 'unique mutants', flush=True)
    states_file = directory/'mutant_thermo_states.jsonl'
    if not states_file.exists():
        states = thermal_states(unique)
        with states_file.open('w') as f:
            for sequence, state in zip(unique, states): f.write(json.dumps(dict(mutant_sequence=sequence, **state))+'\n')
    else:
        with states_file.open() as f: states = [json.loads(line) for line in f]
        require([r['mutant_sequence'] for r in states] == unique, 'Cached mutant PC identities differ')
    thermo = {seq: float(state['Tx_rate']) for seq, state in zip(unique, states)}
    wild = dict(zip(scores, map(float, cnn_predict(model, encode([master[sid]['sequence'] for sid in scores], 87)))))
    with (ROOT/'runs'/cfg['runs']['thermo']/'states.jsonl').open() as f: original = {r['sample_id']: float(r['Tx_rate']) for r in map(json.loads, f)}
    calibration = read_json(ROOT/'runs'/cfg['runs']['thermo']/'calibration.json')
    predictions = []
    for r in design:
        sid = r['sample_id']; p = cnn_map[r['mutant_sequence']]; w = wild[sid]
        td = float(calibration['coef_a']*np.log10(thermo[r['mutant_sequence']]/original[sid])); delta = p-w
        predictions.append(dict(mutation_id=r['mutation_id'], source_cnn_run_id=selected['representative_run'],
            source_thermo_run_id=cfg['runs']['thermo'], predicted_wild_log10=w, predicted_mutant_log10=p,
            delta_log10=delta, thermo_delta_log10=td, prediction_status='ok',
            actual_direction=direction(delta, protocol), thermo_direction=direction(td, protocol), evidence='computational_prediction'))
    write_table(directory/'mutation_predictions.csv', predictions)
    joined = [dict(**r, **{k: v for k, v in p.items() if k not in r}) for r, p in zip(design, predictions)]
    parent = []; summaries = []
    for group in ['high', 'low', 'random']:
        a = []
        for sid in scores:
            rr = [r for r in joined if r['experiment_type'] == 'amplitude' and r['group'] == group and r['sample_id'] == sid]
            record = dict(sample_id=sid, group=group, n_mutants=len(rr),
                mean_absolute_delta_log10=float(np.mean([abs(r['delta_log10']) for r in rr])),
                mean_delta_log10=float(np.mean([r['delta_log10'] for r in rr])),
                positive_fraction=float(np.mean([r['actual_direction'] == 'positive' for r in rr])),
                negative_fraction=float(np.mean([r['actual_direction'] == 'negative' for r in rr])))
            parent.append(record); a.append(record['mean_absolute_delta_log10'])
        low, high = bootstrap(a)
        summaries.append(dict(group=group, n_parent=64, n_mutants=640, mean_absolute_delta_log10=float(np.mean(a)),
            ci_low=low, ci_high=high, mean_delta_log10=float(np.mean([r['mean_delta_log10'] for r in parent if r['group'] == group]))))
    paired = []
    for control in ['low', 'random']:
        a = np.array([next(r['mean_absolute_delta_log10'] for r in parent if r['sample_id'] == sid and r['group'] == 'high')-
                      next(r['mean_absolute_delta_log10'] for r in parent if r['sample_id'] == sid and r['group'] == control) for sid in scores])
        low, high = bootstrap(a)
        paired.append(dict(contrast='high_minus_'+control, n_parent=64, mean_absolute_delta_difference=float(a.mean()), ci_low=low, ci_high=high))
    direction_parent = []; direction_summary = []
    for expected in ['all', 'positive', 'negative', 'near_zero']:
        rr = [r for r in joined if r['experiment_type'] == 'single_direction' and (expected == 'all' or r['expected_direction'] == expected)]
        for sid in scores:
            ss = [r for r in rr if r['sample_id'] == sid]
            if not ss: continue
            direction_parent.append(dict(sample_id=sid, expected_direction=expected, n_mutants=len(ss),
                cnn_agreement=float(np.mean([r['actual_direction'] == r['expected_direction'] for r in ss])),
                thermo_agreement=float(np.mean([r['thermo_direction'] == r['expected_direction'] for r in ss])),
                cross_model_agreement=float(np.mean([r['thermo_direction'] == r['actual_direction'] for r in ss]))))
        pp = [r for r in direction_parent if r['expected_direction'] == expected]
        if not pp: continue
        # Mutation-weighted estimator, with parent-cluster resampling.
        a = np.array([[float(r['actual_direction'] == r['expected_direction']), float(r['thermo_direction'] == r['expected_direction']),
                       float(r['thermo_direction'] == r['actual_direction'])] for r in rr])
        row = dict(expected_direction=expected, n_parent=len(pp), n_mutants=len(rr), near_zero_count=sum(r['actual_direction'] == 'near_zero' for r in rr),
                   bootstrap_unit='parent_sequence', estimator='mutation-weighted agreement; clustered parent bootstrap')
        for i, name in enumerate(['cnn_agreement', 'thermo_agreement', 'cross_model_agreement']):
            low, high = bootstrap(a[:, i], groups=[r['sample_id'] for r in rr])
            row[name] = float(a[:, i].mean()); row[name+'_ci_low'] = low; row[name+'_ci_high'] = high
        direction_summary.append(row)
    write_table(directory/'mutation_parent_summary.csv', parent); write_table(directory/'mutation_summary.csv', summaries)
    write_table(directory/'paired_amplitude.csv', paired); write_table(directory/'direction_parent_summary.csv', direction_parent)
    write_table(directory/'direction_summary.csv', direction_summary)
    cases = sorted([r for r in joined if r['experiment_type'] == 'single_direction'], key=lambda r: (-abs(r['delta_log10']-r['thermo_delta_log10']), r['mutation_id']))[:10]
    write_table(directory/'disagreement_cases.csv', cases)
    finish(directory, cfg, dict(n_parent=64, n_mutants=2880, fixed_tss=67, direction_threshold_log10=protocol['mutation_direction_threshold_log10'],
        interpretation='local linear explanation consistency and cross-model comparison; no experimentally measured mutant outcome'),
        [file, root/'attribution/attributions.tsv', ROOT/selected['representative_run']/'model.pt'])
    return directory
