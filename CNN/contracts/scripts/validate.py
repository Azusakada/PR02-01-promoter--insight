#!/usr/bin/env python3
"""Read-only PR02-01 table and cross-file checks (Python >= 3.10, stdlib).

This validates formatting and explicit invariants, not scientific correctness,
model training, numeric tensor contents, or access-control compliance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unittest
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

BASE = Path(__file__).resolve().parents[1]
CONTRACTS = json.loads((BASE / 'assets/tables.json').read_text(encoding='utf-8'))

class ContractError(ValueError):
    """A user-actionable data contract violation."""

def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)

def close(a: float, b: float, label: str, *, rel: float = 1e-6, abs_tol: float = 1e-8) -> None:
    require(math.isclose(a, b, rel_tol=rel, abs_tol=abs_tol), f'{label}: {a!r} != {b!r}')

def read_json(path: Path) -> dict:
    def invalid(value: str):
        raise ContractError(f'{path}: JSON不接受 {value}')
    def pairs(items):
        d = {}
        for k,v in items:
            require(k not in d, f'{path}: JSON重复键 {k}')
            d[k]=v
        return d
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'),
                     parse_constant=invalid, object_pairs_hook=pairs)
    require(isinstance(obj, dict), f'{path}: JSON顶层必须是对象')
    return obj

def parse_cell(raw: str | None, spec: dict, loc: str) -> Any:
    require(raw is not None, f'{loc}: 缺单元格')
    if raw == '':
        require(spec['nullable'], f'{loc}: 不可空')
        return None
    typ = spec['type']
    if typ == 'str':
        require(bool(raw.strip()), f'{loc}: 不接受纯空白文本')
        require(raw == raw.strip(), f'{loc}: 不允许首尾空白，请显式清洗并记录')
        value: Any = raw
    elif typ == 'int':
        require(bool(re.fullmatch(r'-?(0|[1-9][0-9]*)', raw)), f'{loc}: 需要整数，收到{raw!r}')
        value = int(raw)
    elif typ == 'float':
        try:
            value = float(raw)
        except ValueError as e:
            raise ContractError(f'{loc}: 不是浮点数 {raw!r}') from e
        require(math.isfinite(value), f'{loc}: 不接受NaN/Infinity')
    elif typ == 'bool':
        require(raw in ('true', 'false'), f'{loc}: 布尔必须为小写true/false')
        value = raw == 'true'
    else:
        raise ContractError(f'{loc}: 未知契约类型{typ}')
    if 'enum' in spec:
        require(value in spec['enum'], f'{loc}: 值{value!r}不在{spec["enum"]}')
    if 'const' in spec:
        require(value == spec['const'], f'{loc}: 应为{spec["const"]!r}')
    if 'pattern' in spec:
        require(bool(re.fullmatch(spec['pattern'], value)), f'{loc}: 不符合{spec["pattern"]}')
    if 'min' in spec:
        require(value >= spec['min'], f'{loc}: 小于{spec["min"]}')
    if 'max' in spec:
        require(value <= spec['max'], f'{loc}: 大于{spec["max"]}')
    if 'exclusive_min' in spec:
        require(value > spec['exclusive_min'], f'{loc}: 必须大于{spec["exclusive_min"]}')
    return value

def one_value(rows: list[dict], columns: list[str], label: str) -> None:
    for c in columns:
        require(len({r[c] for r in rows}) == 1, f'{label}: {c}在同一文件中不一致')

def load_table(kind: str, path: Path) -> list[dict]:
    require(kind in CONTRACTS['tables'], f'未知表类型: {kind}')
    spec = CONTRACTS['tables'][kind]
    columns = spec['columns']
    rows: list[dict] = []
    seen: set[tuple] = set()
    with Path(path).open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.DictReader(fh, delimiter=spec['delimiter'], strict=True)
        headers = reader.fieldnames
        require(bool(headers), f'{path}: 没有表头')
        require(len(headers) == len(set(headers)), f'{path}: 列名重复')
        missing = set(columns) - set(headers)
        require(not missing, f'{path}: 缺列 {sorted(missing)}；请确认CSV/TSV分隔符')
        extras = set(headers) - set(columns)
        bad = [c for c in extras if not c.startswith(spec['allow_extra_prefix'])]
        require(not bad, f'{path}: 未声明扩展列{bad}，新增字段应使用extra_或升级契约')
        for raw in reader:
            require(None not in raw, f'{path}:{reader.line_num}: 单元格多于列数')
            loc = f'{path.name}:{reader.line_num}'
            row = {c: parse_cell(raw.get(c), s, f'{loc}/{c}') for c, s in columns.items()}
            for c in extras:
                require(raw.get(c) is not None, f'{loc}/{c}: 缺单元格')
                row[c] = raw[c]
            key = tuple(row[c] for c in spec['primary_key'])
            require(key not in seen, f'{loc}: 主键重复 {key}')
            seen.add(key)
            rows.append(row)
    require(bool(rows), f'{path}: 表只有表头，没有数据')
    check_semantics(kind, rows)
    return rows


def check_direction(r: dict) -> None:
    loc=r['sample_id']
    if not r['has_direction_expectation']:
        require(r['expected_direction']=='none' and r['direction_basis'] is None,
                f'{loc}: 无方向依据时expected_direction=none，basis为空')
        require(r['replacement_rule']=='random_other', f'{loc}: 定向替换不能没有方向依据')
    else:
        require(r['annotation_status'] in ['reliable_external','tool_inferred'],
                f'{loc}: 定向替换缺可追溯区域依据，当前missing不能做方向先验')
        require(r['expected_direction'] in ['up','down'] and bool(r['direction_basis']), f'{loc}: 缺方向先验记录')
        require(r['replacement_rule'] in ['toward_consensus','away_from_consensus'], f'{loc}: 定向与默认幅度条件须分开')
    if 'direction_match' in r:
        if not r['has_direction_expectation']:
            require(r['direction_match'] is None and r['direction_epsilon_log10'] is None, f'{loc}: 无方向先验不能填写方向成绩')
        else:
            e=r['direction_epsilon_log10'];require(e is not None and r['direction_match'] is not None,f'{loc}: 方向结果缺预设epsilon/命中状态')
            match=r['delta_log10']>e if r['expected_direction']=='up' else r['delta_log10'] < -e
            require(r['direction_match']==match,f'{loc}: 方向命中与delta/epsilon不符')

def check_semantics(kind: str, rows: list[dict]) -> None:
    one_value(rows,['data_version','dataset_id'],kind)
    if kind=='dataset':
        require(len({r['source_row'] for r in rows})==len(rows),'dataset: 原始source_row重复')
        require(len({r['sequence'] for r in rows})==len(rows),'dataset: 重复序列，需回查处理而非静默保留')
        for r in rows:close(r['target_log10'],math.log10(r['strength']),f'{r["sample_id"]}: target_log10错误')
    elif kind=='splits':
        groups=defaultdict(list)
        for r in rows:groups[r['split_id']].append(r)
        for key,g in groups.items():
            one_value(g,['seed','split_strategy'],key)
            require({r['split'] for r in g}=={'train','val','test'},f'{key}: 三个集合均需非空')
            if g[0]['split_strategy']=='sequence_group':
                members=defaultdict(set)
                for r in g:
                    require(r['group_id'] is not None,f'{key}: 分组策略缺group_id')
                    members[r['group_id']].add(r['split'])
                require(all(len(v)==1 for v in members.values()),f'{key}: 序列分组跨集合')
    elif kind=='targets':
        one_value(rows,['split_id','transform_id'],'targets')
    elif kind=='feature_index':
        one_value(rows,['feature_version'],'feature_index')
        require(sorted(r['row_index'] for r in rows)==list(range(len(rows))),'feature_index: 行号不完整或重复')
    elif kind=='annotations':
        one_value(rows,['annotation_version'],'annotations')
        for r in rows:
            a,b=r['start_0index'],r['end_0index_exclusive'];status=r['annotation_status']
            if status in ['missing','not_applicable']:
                require(a is None and b is None and r['score'] is None,f'{r["sample_id"]}: 未知/不适用区间和分数必须空')
                if status=='missing':require(r['strand']=='unknown',f'{r["sample_id"]}: missing不能断言方向')
                else:require(bool(r['evidence_ref']),f'{r["sample_id"]}: not_applicable需范围依据')
            else:
                require(a is not None and b is not None and 0<=a<b<=50,f'{r["sample_id"]}: 注释区间非法')
                require(bool(r['evidence_ref']) and r['strand']!='unknown',f'{r["sample_id"]}: 注释缺来源或方向')
                if r['region_type']=='TSS':require(b-a==1,'TSS须单个位点')
    elif kind=='calculator_inputs':
        one_value(rows,['input_version'],'calculator_inputs')
        for r in rows:
            loc=r['sample_id'];n=r['sequence_length']
            require(len(r['sequence'])==n,f'{loc}: 工具输入长度与声明不符')
            require(r['source_sequence_start_0index']+50<=n,f'{loc}: 原50bp无法落在输入内')
            if r['input_origin']=='original50':
                require(n==50 and r['source_sequence_start_0index']==0 and r['context_source_ref'] is None,f'{loc}: original50不能拼接上下文')
            else:require(bool(r['context_source_ref']),f'{loc}: 补取上下文缺来源')
            if r['input_status']=='blocked':require(bool(r['error_reason']),f'{loc}: blocked缺原因')
            else:
                require(r['error_reason'] is None,f'{loc}: ready不应有错误')
                if r['tss_mode']=='fixed_tss':
                    p=r['tss_position_1index'];require(p is not None and 1<=p<=n and bool(r['tss_evidence_ref']),f'{loc}: fixed_tss缺合法位置/独立依据')
            if r['tss_mode']=='scan':require(r['tss_position_1index'] is None and r['tss_evidence_ref'] is None,f'{loc}: scan不应写默认固定TSS')
    elif kind=='predictions':
        one_value(rows,['run_id','method_name','split_id','subset','target_scale_model','thermo_mode','calibration_id','label_transform_id','seed','model_checkpoint'],'predictions')
        for r in rows:
            loc=r['sample_id'];nums=['predicted_value','predicted_value_log10','predicted_value_normalized','predicted_tx_rate']
            if r['subset']=='mutation':require(r['true_value'] is None,f'{loc}: 未测突变不能复制母本真值')
            if r['prediction_status']=='failed':
                require(bool(r['error_reason']) and all(r[c] is None for c in nums),f'{loc}: failed须空预测+原因');continue
            require(r['error_reason'] is None,f'{loc}: ok不能带错误')
            if r['target_scale_model']=='tool_raw':
                require(r['predicted_tx_rate'] is not None and r['thermo_mode'] is not None,f'{loc}: tool_raw缺输出/模式')
                if r['calibration_id'] is None:
                    require(all(r[c] is None for c in ['predicted_value','predicted_value_log10','predicted_value_normalized','label_transform_id']),f'{loc}: 未校准工具不能填写strength预测')
                else:require(r['predicted_value'] is not None and r['predicted_value_log10'] is not None,f'{loc}: 已校准缺strength尺度结果')
            else:
                require(r['predicted_tx_rate'] is None and r['thermo_mode'] is None and r['calibration_id'] is None,f'{loc}: 学习器不应混入工具输出')
                require(r['model_checkpoint'] is not None and r['predicted_value'] is not None and r['predicted_value_log10'] is not None,f'{loc}: 缺模型/原尺度/log10预测')
                if r['target_scale_model']=='log10_minmax':
                    require(r['predicted_value_normalized'] is not None and r['label_transform_id'] is not None,f'{loc}: minmax输出缺normalized或transform')
                else:require(r['predicted_value_normalized'] is None and r['label_transform_id'] is None,f'{loc}: 直接log10模型不能冒用归一化值')
            if r['predicted_value'] is not None:
                require(r['predicted_value_log10'] is not None,f'{loc}: 缺log10值')
                close(math.log10(r['predicted_value']),r['predicted_value_log10'],f'{loc}: raw/log10不一致')
            if r['predicted_value_normalized'] is not None:require(r['label_transform_id'] is not None,f'{loc}: normalized缺transform')
    elif kind=='attributions':
        one_value(rows,['run_id','split_id','model_checkpoint','coordinate_version','baseline_type','n_steps','label_transform_id','explanation_scale'],'attributions')
        groups=defaultdict(list)
        for r in rows:
            groups[r['sample_id']].append(r)
            require(r['position_1index']==r['position_0index']+1,'归因0/1编号不一致')
            close(r['contribution_abs'],abs(r['contribution_signed']),'归因绝对值不一致')
        for sid,g in groups.items():
            require(len(g)==50 and {r['position_0index'] for r in g}==set(range(50)),f'{sid}: 每样本必须完整50位')
            one_value(g,['predicted_value_log10','annotation_status'],sid)
            order=sorted(g,key=lambda r:(-r['contribution_abs'],r['position_0index']))
            require(all(r['rank_abs']==i for i,r in enumerate(order,1)),f'{sid}: 排名错误')
    elif kind=='attribution_qc':
        one_value(rows,['run_id','n_steps','baseline_type','explanation_scale'],'attribution_qc')
        for r in rows:
            err=abs(r['contribution_sum']-(r['predicted_value_log10']-r['baseline_value_log10']))
            close(err,r['completeness_error'],'完备性误差不符')
            if r['qc_status']=='unassessed':require(r['tolerance_abs'] is None,'未评估不能填已冻结阈值')
            else:
                require(r['tolerance_abs'] is not None,'QC缺阈值')
                require((r['qc_status']=='pass')==(err<=r['tolerance_abs']),'QC状态与阈值/误差不符')
    elif kind in ['mutation_design','mutations']:
        one_value(rows,['design_id','split_id','coordinate_version','k'],'mutation')
        if kind=='mutations':one_value(rows,['run_id','model_checkpoint','label_transform_id'],'mutations')
        for r in rows:
            require(r['original_base']!=r['mutant_base'],'突变没有改变碱基')
            require(r['sequence'][r['position_1index']-1]==r['mutant_base'],'突变位置与mutant_base不一致')
            check_direction(r)
            if kind=='mutations':
                close(r['delta_log10'],r['predicted_value_log10_mutant']-r['predicted_value_log10_original'],'delta错误')
                close(r['abs_delta_log10'],abs(r['delta_log10']),'abs_delta错误')
        if kind=='mutation_design':
            groups=defaultdict(lambda:defaultdict(set));counts=Counter()
            for r in rows:
                if r['replacement_rule']!='random_other':continue
                key=(r['design_id'],r['parent_id']);p=r['position_1index'];g=groups[key][r['group']]
                require(p not in g,'默认单替换同组重复位置');g.add(p);counts[key]+=1
            for key,g in groups.items():
                k=rows[0]['k'];require(set(g)=={'High','Low','Random'},f'{key}: 默认幅度设计缺组')
                require(all(len(v)==k for v in g.values()),f'{key}: 每组数量不等于k')
                require(len(set.union(*g.values()))==3*k,f'{key}: 三组位置交叠')
    elif kind=='metrics':
        for r in rows:
            require(r['n_used']<=r['n_success']<=r['n_requested'],'指标覆盖数量矛盾')
            if r['target_scale']=='tool_rank':require(r['metric_name']=='spearman','未校准工具仅比较排序')
            if r['metric_status']=='undefined':require(r['value'] is None and bool(r['reason']),'undefined需空值+原因')
            else:
                require(r['value'] is not None and r['reason'] is None and r['n_used']>0,'有效指标缺数值/样本或存在矛盾原因')
                if r['metric_name'] in ['r2','spearman']:require(r['n_used']>=2,'样本不足')
                if r['metric_name']=='r2':require(r['value']<=1+1e-10,'R²不能>1，负值允许')
                if r['metric_name']=='spearman':require(-1<=r['value']<=1,'Spearman越界')
                if r['metric_name']=='mae':require(r['value']>=0,'MAE为负')

def read_ids(path: Path) -> set[str]:
    with Path(path).open(encoding='utf-8-sig',newline='') as fh:
        rr=csv.DictReader(fh,delimiter='\t');require(rr.fieldnames==['sample_id'],'请求ID表须单列sample_id')
        ids=[]
        for r in rr:
            require(None not in r and r['sample_id'] and r['sample_id'].strip()==r['sample_id'],'ID无效')
            ids.append(r['sample_id'])
    require(ids and len(ids)==len(set(ids)),'ID为空/重复');return set(ids)

def scale_check(meta: dict, samples: list[dict], splits: list[dict], samples_path: Path) -> dict:
    req={'schema_version','dataset_id','data_version','split_id','transform_id','kind','fit_subset','train_ids','train_ids_sha256','samples_sha256','min_log10','max_log10','clip'}
    require(req<=set(meta),f'变换缺字段{sorted(req-set(meta))}')
    require(meta['schema_version']=='2.0.0' and meta['kind']=='log10_minmax' and meta['fit_subset']=='train' and meta['clip'] is False,'变换类型/fit集合/裁剪设置错误')
    require(meta['dataset_id']==samples[0]['dataset_id'] and meta['data_version']==samples[0]['data_version'],'变换数据身份不符')
    require(isinstance(meta['transform_id'],str) and meta['transform_id'],'变换ID缺失')
    expected={r['sample_id'] for r in splits if r['split_id']==meta['split_id'] and r['split']=='train'}
    require(expected and isinstance(meta['train_ids'],list) and all(isinstance(i,str) for i in meta['train_ids']),'缺训练ID')
    require(len(meta['train_ids'])==len(expected) and set(meta['train_ids'])==expected,'minmax拟合ID不是当前train完整集合')
    h=hashlib.sha256(''.join(i+'\n' for i in sorted(expected)).encode()).hexdigest()
    require(h==meta['train_ids_sha256'] and sha256(samples_path)==meta['samples_sha256'],'标签变换哈希不符')
    vals=[r['target_log10'] for r in samples if r['sample_id'] in expected]
    a,b=meta['min_log10'],meta['max_log10']
    require(type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) and a<b,'minmax参数必须有限且a<b')
    close(a,min(vals),'minmax下界不是train最小值');close(b,max(vals),'minmax上界不是train最大值')
    return meta

def mutation_config_check(d: dict) -> dict:
    require(d.get('schema_version')=='2.0.0' and d.get('sequence_length')==50,'突变配置不是本版50bp')
    k=d.get('k');require(type(k) is int and 1<=k and 3*k<=50,'互斥三组必须3*k<=50，不能继续k20/29')
    require(d.get('groups')==['High','Low','Random'] and d.get('random_group_excludes_high_low') is True,'三组互斥规则不符')
    require(d.get('one_base_per_variant') is True and d.get('analysis_scale')=='log10','单点/尺度不符')
    if d.get('direction_enabled'):
        require(d.get('direction_evidence_ref') and d.get('direction_epsilon_log10') is not None,'定向实验尚缺证据或阈值')
    return {'k':k,'minimum_distinct_positions':3*k,'sequence_length':50}

def bundle_check(samples_path: Path, splits_path: Path, *, feature_index: Path|None=None,
                 calculator_inputs: Path|None=None,predictions: Path|None=None,
                 attributions: Path|None=None,attribution_qc: Path|None=None,
                 mutations: Path|None=None,mutation_design: Path|None=None,
                 requested_ids: Path|None=None,annotations: Path|None=None,
                 transform: Path|None=None,targets: Path|None=None) -> dict:
    samples=load_table('dataset',samples_path);splits=load_table('splits',splits_path)
    master={r['sample_id']:r for r in samples};dv=samples[0]['data_version'];ds=samples[0]['dataset_id']
    def version(rows,kind):
        require(all(r['data_version']==dv and r['dataset_id']==ds for r in rows),f'{kind}: 数据身份不一致')
    version(splits,'splits');maps=defaultdict(dict)
    for r in splits:
        require(r['sample_id'] in master,'split含未知ID');maps[r['split_id']][r['sample_id']]=r
    for key,m in maps.items():require(set(m)==set(master),f'{key}: split不能完整覆盖主表')
    out={'samples':len(samples),'split_ids':list(maps),'checked':['dataset','splits'],'limitations':'未验证原始数据、NPY/NPZ、训练过程、工具适用性或生物学依据。'}
    requested=read_ids(requested_ids) if requested_ids else None
    scale=scale_check(read_json(transform),samples,splits,samples_path) if transform else None
    if scale:out['checked'].append('label_transform')
    if targets:
        require(scale is not None,'targets需提供--transform');rs=load_table('targets',targets);version(rs,'targets')
        require({r['sample_id'] for r in rs}==set(master),'targets需完整覆盖主表')
        for r in rs:
            require(r['split_id']==scale['split_id'] and r['transform_id']==scale['transform_id'],'目标变换ID不符')
            z=master[r['sample_id']]['target_log10'];close(r['target_log10'],z,'target_log10错位')
            close(r['target_normalized'],(z-scale['min_log10'])/(scale['max_log10']-scale['min_log10']),'target_normalized错误/被裁剪')
        out['checked'].append('targets')
    if feature_index:
        rs=load_table('feature_index',feature_index);version(rs,'feature_index')
        require({r['sample_id'] for r in rs}==set(master),'特征ID覆盖不符');out['checked'].append('feature_index')
    if annotations:
        rs=load_table('annotations',annotations);version(rs,'annotations')
        require(all(r['sample_id'] in master for r in rs),'注释含未知ID');out['checked'].append('annotations')
    if calculator_inputs:
        rs=load_table('calculator_inputs',calculator_inputs);version(rs,'calculator_inputs')
        for r in rs:
            sid=r['sample_id'];require(sid in master,'工具输入未知ID');i=r['source_sequence_start_0index']
            require(r['sequence'][i:i+50]==master[sid]['sequence'],'工具输入未按声明保留原50bp；方向变换应由adapter另记录')
        out['checked'].append('calculator_inputs')
    designs=load_table('mutation_design',mutation_design) if mutation_design else []
    results=load_table('mutations',mutations) if mutations else []
    def check_mutation_parents(rs):
        for r in rs:
            sid=r['sample_id'];pid=r['parent_id'];require(pid in master and sid not in master,'母本未知或突变ID撞主表')
            require(r['split_id'] in maps,'突变split未知')
            seq=master[pid]['sequence'];pos=r['position_1index']-1
            require(seq[pos]==r['original_base'],'original_base与母本不符')
            require(seq[:pos]+r['mutant_base']+seq[pos+1:]==r['sequence'],'突变并非声明位置的单点替换')
    for kind,rs in [('mutation_design',designs),('mutations',results)]:
        if rs:version(rs,kind);check_mutation_parents(rs);out['checked'].append(kind)
    design_map={r['sample_id']:r for r in designs}
    if designs and results:
        for r in results:
            require(r['sample_id'] in design_map,'突变结果无冻结设计')
            require(all(r[k]==v for k,v in design_map[r['sample_id']].items()),'突变结果与冻结设计不符')
    ps=load_table('predictions',predictions) if predictions else []
    pmap={r['sample_id']:r for r in ps}
    if ps:
        version(ps,'predictions');first=ps[0];sp=first['split_id'];sub=first['subset'];require(sp in maps,'预测split未知')
        if sub=='mutation':
            require(designs,'突变预测覆盖检查需--mutation-design（包含失败请求）');expected=set(design_map)
        else:expected={sid for sid,r in maps[sp].items() if r['split']==sub}
        if requested is not None:require(requested<=expected,'请求ID不属于声明集合');expected=requested
        require(set(pmap)==expected,'预测未完整覆盖请求，失败也需保留行')
        for r in ps:
            if sub!='mutation':
                require(r['true_value'] is not None,'真实样本缺true_value');close(r['true_value'],master[r['sample_id']]['strength'],'true_value与新主表不符')
            if r['prediction_status']=='ok' and r['predicted_value_normalized'] is not None:
                require(scale is not None,'归一化预测跨文件核验需--transform')
                require(r['split_id']==scale['split_id'] and r['label_transform_id']==scale['transform_id'],'预测变换ID不符')
                z=scale['min_log10']+(scale['max_log10']-scale['min_log10'])*r['predicted_value_normalized']
                close(r['predicted_value_log10'],z,'预测反变换错误')
        out['checked'].append('predictions')
    elif requested is not None:require(requested<=set(master),'未知请求ID')
    if attributions:
        rs=load_table('attributions',attributions);version(rs,'attributions');groups=defaultdict(list)
        for r in rs:
            sid=r['sample_id'];require(sid in master and r['split_id'] in maps,'归因ID/split未知')
            require(r['nucleotide']==master[sid]['sequence'][r['position_0index']],'归因碱基错位')
            if r['label_transform_id'] is not None:
                require(scale is not None and r['label_transform_id']==scale['transform_id'] and r['split_id']==scale['split_id'],'归因变换ID未核验')
            groups[sid].append(r)
            if sid in pmap:
                p=pmap[sid];require(p['prediction_status']=='ok' and p['model_checkpoint']==r['model_checkpoint'] and p['label_transform_id']==r['label_transform_id'],'预测归因模型/变换不一致')
                close(p['predicted_value_log10'],r['predicted_value_log10'],'归因输出错位')
        if requested is not None:require(set(groups)==requested,'归因覆盖不符')
        require(attribution_qc is not None,'完整归因bundle需QC')
        qc=load_table('attribution_qc',attribution_qc);version(qc,'attribution_qc')
        require({r['sample_id'] for r in qc}==set(groups),'QC覆盖不符')
        for r in qc:
            g=groups[r['sample_id']];f=g[0]
            require(all(r[k]==f[k] for k in ['run_id','baseline_type','n_steps','explanation_scale']),'QC设置不符')
            close(r['contribution_sum'],math.fsum(x['contribution_signed'] for x in g),'QC贡献和错误')
            close(r['predicted_value_log10'],f['predicted_value_log10'],'QC输出错误')
        for r in designs:
            pid=r['parent_id'];require(pid in groups,'突变母本缺归因')
            if r['replacement_rule']=='random_other':
                rank=next(x['rank_abs'] for x in groups[pid] if x['position_1index']==r['position_1index'])
                if r['group']=='High':require(rank<=r['k'],'High不符合top-k')
                if r['group']=='Low':require(rank>50-r['k'],'Low不符合bottom-k')
        for r in results:
            f=groups[r['parent_id']][0];require(r['model_checkpoint']==f['model_checkpoint'] and r['label_transform_id']==f['label_transform_id'],'突变与归因模型/变换不一致')
            close(r['predicted_value_log10_original'],f['predicted_value_log10'],'突变母本预测不符')
        out['checked']+=['attributions','attribution_qc']
    elif attribution_qc:raise ContractError('QC必须与attributions同时核验')
    return out

def safe_file(root: Path, raw: str) -> Path:
    require(isinstance(raw,str) and bool(raw),'artifact path必须非空字符串')
    require('\\' not in raw and not PurePosixPath(raw).is_absolute(),'artifact path使用仓库相对POSIX路径')
    p=(root/raw).resolve();r=root.resolve()
    require(p.is_relative_to(r),f'artifact逃出root: {raw}')
    require(p.is_file(),f'缺少artifact文件: {raw}')
    return p

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda:fh.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def run_check(path: Path, root: Path) -> dict:
    d=read_json(path)
    required={'dataset_id','schema_version','run_id','stage','task_id','method_name','data_version','split_id','evidence_level',
              'execution_status','seed','created_at','command','code_commit','dirty_worktree','environment','parameters',
              'fit_subsets','selection_subsets','evaluation_subsets','artifacts'}
    require(not(required-set(d)),f'run_manifest缺字段{sorted(required-set(d))}')
    require(d['schema_version']=='2.0.0','run schema版本不支持')
    for k in ['dataset_id','run_id','task_id','method_name','data_version','split_id']:
        require(isinstance(d[k],str) and bool(d[k].strip()),f'run {k}不能为空')
    require(d['stage'] in ['M2','M3','M4','M5'],'run stage无效')
    require(d['evidence_level'] in ['synthetic','smoke','preliminary','final'],'run evidence_level无效')
    require(d['execution_status'] in ['success','partial','failed'],'run execution_status无效')
    require(d['seed'] is None or (type(d['seed']) is int and d['seed']>=0),'run seed必须非负整数/null')
    require(type(d['dirty_worktree']) is bool,'run dirty_worktree必须bool')
    require(isinstance(d['command'],list) and bool(d['command']) and all(isinstance(v,str) and v for v in d['command']),'run command应为非空字符串数组')
    require(isinstance(d['environment'],dict) and {'python','platform','packages'}<=set(d['environment']),'run environment缺python/platform/packages')
    require(isinstance(d['environment']['packages'],dict),'run environment.packages必须对象')
    require(isinstance(d['parameters'],dict),'run parameters必须对象')
    try:
        dt=datetime.fromisoformat(d['created_at'].replace('Z','+00:00'))
    except (ValueError,AttributeError) as e:raise ContractError('run created_at不是ISO时间') from e
    require(dt.tzinfo is not None,'run时间必须有时区')
    for k,allowed in [('fit_subsets',{'train'}),('selection_subsets',{'val'}),('evaluation_subsets',{'train','val','test','mutation'})]:
        require(isinstance(d[k],list) and all(isinstance(x,str) for x in d[k]) and set(d[k])<=allowed,f'run {k}非法或包含test训练/选参')
    if d['evidence_level'] in ['preliminary','final']:
        require(isinstance(d['code_commit'],str) and bool(re.fullmatch('[a-f0-9]{40,64}',d['code_commit'])),'真实实验需实际code_commit')
    require(isinstance(d['artifacts'],list),'run artifacts需数组')
    if d['execution_status']=='success': require(bool(d['artifacts']),'success run不能没有产物')
    if d['evidence_level']=='final' and d['method_name']!='thermo_regseq2':
        require(any(isinstance(a,dict) and a.get('kind')=='model_manifest' for a in d['artifacts']),
                'final learned-model run必须声明model_manifest，不能省略smoke检查')
    if d['evidence_level'] in ['preliminary','final'] and d['dirty_worktree']:
        require(any(isinstance(a,dict) and a.get('kind')=='source_patch' for a in d['artifacts']),
                '未提交代码的真实实验须声明source_patch哈希以便复现')
    seen=set()
    for a in d['artifacts']:
        require(isinstance(a,dict) and {'kind','path','sha256'}<=set(a),'artifact缺kind/path/sha256')
        require(isinstance(a['kind'],str) and bool(a['kind']),'artifact kind须字符串')
        require(isinstance(a['sha256'],str) and bool(re.fullmatch('[a-f0-9]{64}',a['sha256'])),'SHA256须64位小写hex')
        p=safe_file(root,a['path']);require(a['path'] not in seen,'artifact重复路径');seen.add(a['path'])
        require(sha256(p)==a['sha256'],f'artifact hash不符: {a["path"]}')
        if a['kind'] in CONTRACTS['tables']:
            rows=load_table(a['kind'],p)
            for row in rows:
                require(row['dataset_id']==d['dataset_id'] and row['data_version']==d['data_version'],'artifact与run数据版本不一致')
                if 'run_id' in row:require(row['run_id']==d['run_id'],'artifact与run_id不一致')
                if 'method_name' in row:require(row['method_name']==d['method_name'],'artifact方法与run不一致')
                if 'split_id' in row:require(row['split_id']==d['split_id'],'artifact划分与run不一致')
                if 'evidence_level' in row:require(row['evidence_level']==d['evidence_level'],'artifact证据等级与run不一致')
                if d['execution_status']=='success' and row.get('prediction_status')=='failed':
                    raise ContractError('success run包含failed预测，需标partial或failed')
        if a['kind']=='model_manifest':
            meta=read_json(p)
            if d['evidence_level']=='final':require(meta.get('smoke_test') is False,'formal run不能使用smoke/未知状态模型')
    return {'run_id':d['run_id'],'artifacts_checked':len(seen),'evidence_level':d['evidence_level'],
            'limitations':'仅检查列、关键元数据和已声明文件哈希；不证明训练过程和科学结论。'}



def main(argv: list[str]|None=None) -> int:
    p=argparse.ArgumentParser(description=__doc__);sp=p.add_subparsers(dest='command',required=True)
    t=sp.add_parser('table');t.add_argument('kind',choices=list(CONTRACTS['tables']));t.add_argument('path',type=Path)
    b=sp.add_parser('bundle');b.add_argument('--samples',type=Path,required=True);b.add_argument('--splits',type=Path,required=True)
    opts=['feature-index','calculator-inputs','predictions','attributions','attribution-qc','mutations','mutation-design','requested-ids','annotations','transform','targets']
    for key in opts:b.add_argument('--'+key,type=Path)
    r=sp.add_parser('run');r.add_argument('path',type=Path);r.add_argument('--root',type=Path,required=True)
    m=sp.add_parser('mutation-config');m.add_argument('path',type=Path)
    sp.add_parser('self-test')
    args=p.parse_args(argv)
    try:
        if args.command=='self-test':
            suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent),pattern='test_contracts.py')
            result=unittest.TextTestRunner(verbosity=2).run(suite);return 0 if result.wasSuccessful() else 1
        if args.command=='table':result={'table':args.kind,'rows':len(load_table(args.kind,args.path))}
        elif args.command=='run':result=run_check(args.path,args.root)
        elif args.command=='mutation-config':result=mutation_config_check(read_json(args.path))
        else:result=bundle_check(args.samples,args.splits,**{k.replace('-','_'):getattr(args,k.replace('-','_')) for k in opts})
        print(json.dumps({'status':'passed','checks':result},ensure_ascii=False,indent=2));return 0
    except (ContractError,OSError,csv.Error,json.JSONDecodeError,UnicodeError,KeyError,TypeError,OverflowError) as e:
        print(json.dumps({'status':'failed','error':str(e)},ensure_ascii=False,indent=2),file=sys.stderr);return 2

if __name__=='__main__':raise SystemExit(main())
