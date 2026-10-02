"""Both-strand thermodynamic inference, train-only calibration and full coverage."""
from __future__ import annotations
import json
import math
import sys
import time

from .common import (ROOT, DATA, SPLITS, artifact, load_bundle, prediction_base, read_table,
                     run_dir, relative, v, write_json, write_manifest, write_table)
from .metrics import metric_rows

INPUTS = ROOT/'thermo/input/calculator_inputs.tsv'
METHOD = 'thermo_regseq2'


def select_state(output, length):
    candidates = [(strand, int(mapped), state)
                  for strand, key in [('forward','Forward_Predictions_per_TSS'),('reverse','Reverse_Predictions_per_TSS')]
                  for mapped, state in output[key].items()]
    if not candidates: raise ValueError('no_valid_promoter_state')
    strand, mapped, state = min(candidates,key=lambda x:x[2]['dG_total'])
    expected = int(state['TSS']) if strand=='forward' else length-int(state['TSS'])
    if mapped!=expected: raise ValueError('Unexpected tool coordinate mapping')
    boxes = {}
    for field in ['UP_position','hex35_position','spacer_position','hex10_position','disc_position']:
        start,end = state[field]
        boxes[field] = [int(start),int(end)] if strand=='forward' else [length-int(end),length-int(start)]
    return strand,mapped,state,boxes,len(candidates)


def fit_calibration(predictions, train_ids, run_id, split_id, data_version):
    train = [r for r in predictions if r['sample_id'] in train_ids and r['prediction_status']=='ok']
    if len(train)<2: raise ValueError('Insufficient successful train rows for calibration')
    xs = [math.log10(float(r['predicted_tx_rate'])) for r in train]
    ys = [math.log10(float(r['true_value'])) for r in train]
    mx,my = sum(xs)/len(xs),sum(ys)/len(ys)
    denominator = sum((x-mx)**2 for x in xs)
    if denominator==0: raise ValueError('Constant training Tx_rate')
    a = sum((x-mx)*(y-my) for x,y in zip(xs,ys))/denominator
    b = my-a*mx
    import hashlib
    ids = sorted(r['sample_id'] for r in train)
    return dict(calibration_id=run_id+'_train_log_linear',kind='log10_linear',fit_subset='train',
                n_fit=len(train),train_ids_sha256=hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest(),
                coef_a=a,intercept_b=b,split_id=split_id,run_id=run_id,schema_version='2.0.0',data_version=data_version)


def apply_calibration(predictions, calibration):
    rows = []
    for raw in predictions:
        row = dict(raw,calibration_id=calibration['calibration_id'])
        if row['prediction_status']=='ok':
            try:
                z = calibration['coef_a']*math.log10(float(row['predicted_tx_rate']))+calibration['intercept_b']
                strength = 10**z
                if not math.isfinite(strength) or strength<=0: raise ValueError('nonfinite_or_underflow')
                row.update(predicted_value=strength,predicted_value_log10=z)
            except (ValueError,OverflowError) as exc:
                row.update(prediction_status='failed',error_reason='calibration: '+str(exc),
                           predicted_value=None,predicted_value_log10=None,predicted_tx_rate=None)
        rows.append(row)
    return rows


def run(run_id='thermo_regseq2_integrated_20261002_v2'):
    samples,groups,transform = load_bundle()
    v.bundle_check(DATA,SPLITS,calculator_inputs=INPUTS)
    inputs = v.load_table('calculator_inputs', INPUTS)
    master = {r['sample_id']:r for r in samples}
    v.require(len(inputs)==len(master) and {r['sample_id'] for r in inputs}==set(master), 'Incomplete thermo requests')
    subset_map = {r['sample_id']:sub for sub,rs in groups.items() for r in rs}
    context = {r['sample_id']:r for r in read_table(ROOT/'thermo/input/context_map.tsv',delimiter='\t')}
    output = run_dir(run_id)
    sys.path.insert(0,str(ROOT/'thermo/tools/regseq2'))
    from promoter_calculator import Promoter_Calculator
    calculator = Promoter_Calculator()
    raw_rows, failures = [], []
    start = time.perf_counter()
    raw_path = output/'thermo_raw_outputs.jsonl'
    with raw_path.open('w',encoding='utf-8') as log:
        for i,source in enumerate(inputs,1):
            sid = source['sample_id']
            row = prediction_base(master[sid],METHOD,run_id,transform['split_id'],subset_map[sid],'tool_raw')
            row['thermo_mode'] = source['tss_mode']
            diagnostic = dict(sample_id=sid,input_origin=source['input_origin'],context_source_ref=source['context_source_ref'],
                              context_match_status=context[sid]['match_status'])
            try:
                if source['input_status']!='ready': raise ValueError(source['error_reason'] or 'input_not_ready')
                sequence = source['sequence']
                calculator.run(sequence,[0,len(sequence)])
                strand,tss,best,boxes,count = select_state(calculator.output(),len(sequence))
                tx = float(best['Tx_rate'])
                if not math.isfinite(tx) or tx<=0: raise ValueError('invalid_Tx_rate')
                row.update(prediction_status='ok',error_reason=None,predicted_tx_rate=tx)
                offset = source['source_sequence_start_0index']
                diagnostic.update(status='ok',selected_strand=strand,best_TSS_0index=tss,
                                  tss_coordinate_kind='0_based_boundary_on_provided_input',
                                  tool_internal_TSS=int(best['TSS']),TSS_inside_original50=offset<=tss<offset+50,
                                  candidate_count=count,input_length=len(sequence),Tx_rate=tx,dG_total=float(best['dG_total']),
                                  boxes_input_0index_halfopen=boxes,
                                  boxes_original50_0index_halfopen={k:[a-offset,b-offset] if offset<=a<b<=offset+50 else None for k,(a,b) in boxes.items()},
                                  selected_state={k:float(best[k]) for k in ['dG_10','dG_35','dG_spacer','dG_UP','dG_disc','dG_ITR']},
                                  hex35=best['hex35'],hex10=best['hex10'],spacer_len=len(best['spacer']))
            except Exception as exc:
                row['error_reason'] = type(exc).__name__+': '+str(exc)
                diagnostic.update(status='failed',error_reason=row['error_reason'])
                failures.append(dict(sample_id=sid,subset=row['subset'],error_reason=row['error_reason']))
            raw_rows.append(row)
            log.write(json.dumps(diagnostic,ensure_ascii=False,allow_nan=False)+'\n')
            if i%1000==0:
                log.flush()
                print(f'thermo {i}/{len(inputs)} failed={len(failures)} elapsed={time.perf_counter()-start:.1f}s',flush=True)
    calibration = fit_calibration(raw_rows,{r['sample_id'] for r in groups['train']},run_id,transform['split_id'],transform['data_version'])
    calibrated = apply_calibration(raw_rows,calibration)
    write_json(output/'thermo_calibration.json',calibration)
    write_table(output/'train_ids.tsv',[{'sample_id':r['sample_id']} for r in groups['train']],delimiter='\t')
    write_table(output/'calculator_failures.csv',failures,columns=['sample_id','subset','error_reason'])
    artifacts = [artifact('raw_outputs',raw_path),artifact('calibration',output/'thermo_calibration.json'),
                 artifact('fit_ids',output/'train_ids.tsv'),artifact('failures',output/'calculator_failures.csv'),
                 artifact('calculator_inputs',INPUTS),artifact('dataset',DATA),artifact('splits',SPLITS)]
    metrics,coverage = [],[]
    for sub in ['train','val','test']:
        raw = [r for r in raw_rows if r['subset']==sub]
        cal = [r for r in calibrated if r['subset']==sub]
        for name,rows in [('raw',raw),('calibrated',cal)]:
            path = output/f'predictions/thermo_{sub}_{name}.csv'
            write_table(path,rows)
            v.bundle_check(DATA,SPLITS,predictions=path)
            artifacts.append(artifact('predictions',path))
        success = sum(r['prediction_status']=='ok' for r in cal)
        coverage.append(dict(method_name=METHOD,subset=sub,n_requested=len(cal),n_success=success,n_failed=len(cal)-success,coverage=success/len(cal)))
        if sub!='test': metrics.extend(metric_rows(cal,sub,comparison_set_id=run_id+'_'+sub+'_available'))
    write_table(output/'metrics.csv',metrics)
    write_table(output/'coverage_summary.csv',coverage)
    artifacts.extend([artifact('metrics',output/'metrics.csv'),artifact('coverage',output/'coverage_summary.csv')])
    parameters = dict(input_mode='recovered_context',both_strands=True,selection='minimum dG_total among all (strand,TSS) states',
                      context_match_counts={kind:sum(r['match_status']==kind for r in context.values()) for kind in ['found','ambiguous','not_found']},
                      tool_coefficient_sha256=v.sha256(ROOT/'thermo/tools/regseq2/free_energy_coeffs.npy'),
                      tool_intercept_sha256=v.sha256(ROOT/'thermo/tools/regseq2/model_intercept.npy'),
                      test_metrics_enabled=False,elapsed_seconds=time.perf_counter()-start)
    write_json(output/'config.json',parameters)
    artifacts.append(artifact('config',output/'config.json'))
    write_manifest(output,run_id,METHOD,artifacts,parameters=parameters,fit=['train'],selection=[],
                   evaluation=['train','val','test'],status='partial' if any(r['prediction_status']=='failed' for r in calibrated) else 'success')
    print(f'Thermo completed: {output}',flush=True)
    return output
