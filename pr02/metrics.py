from __future__ import annotations
import math
import numpy as np
from scipy.stats import spearmanr


def metric_value(name, y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    if not len(y): return None, 'undefined', 'empty_evaluation_set'
    if not (np.isfinite(y).all() and np.isfinite(p).all()):
        raise ValueError('Non-finite metric inputs')
    if name == 'mae': return float(np.abs(y-p).mean()), 'ok', ''
    if len(y)<2 or np.ptp(y)==0: return None, 'undefined', 'insufficient_or_constant_target'
    if name == 'r2': return float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)), 'ok', ''
    if name == 'spearman':
        if np.ptp(p)==0: return None, 'undefined', 'constant_prediction'
        return float(spearmanr(y,p).statistic), 'ok', ''
    raise ValueError(name)


def metric_rows(predictions, subset, *, comparison_set_id, common_ids=None):
    requested = list(predictions)
    ok = [r for r in requested if r['prediction_status']=='ok']
    used = [r for r in ok if common_ids is None or r['sample_id'] in common_ids]
    first = requested[0]
    rows = []
    scales = ['raw','log10'] if first['target_scale_model']!='tool_raw' or first['calibration_id'] else ['tool_rank']
    if first['target_scale_model']=='tool_raw' and first['calibration_id']: scales.append('tool_rank')
    for scale in scales:
        y = [math.log10(float(r['true_value'])) if scale=='log10' else float(r['true_value']) for r in used]
        p = [float(r[{'raw':'predicted_value','log10':'predicted_value_log10','tool_rank':'predicted_tx_rate'}[scale]]) for r in used]
        for metric in ['spearman'] if scale=='tool_rank' else ['r2','mae','spearman']:
            value, status, reason = metric_value(metric,y,p)
            rows.append(dict(dataset_id=first['dataset_id'],run_id=first['run_id'],method_name=first['method_name'],
                             split_id=first['split_id'],evaluation_subset=subset,metric_name=metric,target_scale=scale,
                             value=value,n_requested=len(requested),n_success=len(ok),n_used=len(used),
                             metric_status=status,reason=reason,evidence_level='preliminary',
                             comparison_set_id=comparison_set_id,schema_version='2.0.0',data_version=first['data_version']))
    return rows
