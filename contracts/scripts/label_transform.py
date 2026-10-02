#!/usr/bin/env python3
"""Train-only log10/min-max helper. Never overwrites existing outputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import validate as v


def compute_label_transform(samples_path: Path, splits_path: Path, split_id: str) -> dict:
    v.bundle_check(samples_path, splits_path)
    rows=v.load_table('dataset',samples_path); splits=v.load_table('splits',splits_path)
    ids=sorted(r['sample_id'] for r in splits if r['split_id']==split_id and r['split']=='train')
    v.require(ids, '没有当前split的训练样本')
    wanted=set(ids); zs=[r['target_log10'] for r in rows if r['sample_id'] in wanted]
    a,b=min(zs),max(zs);v.require(a<b,'训练集log10标签恒定，不能拟合min-max')
    sig={'samples':v.sha256(samples_path),'split_id':split_id,'train_ids':ids,'min_log10':a,'max_log10':b}
    tid='log10mm_'+hashlib.sha256(json.dumps(sig,sort_keys=True).encode()).hexdigest()[:16]
    result={'schema_version':'2.0.0','dataset_id':rows[0]['dataset_id'],'data_version':rows[0]['data_version'],
      'split_id':split_id,'transform_id':tid,'kind':'log10_minmax','fit_subset':'train','train_ids':ids,
      'train_ids_sha256':hashlib.sha256(''.join(x+'\n' for x in ids).encode()).hexdigest(),
      'samples_sha256':v.sha256(samples_path),'min_log10':a,'max_log10':b,'clip':False}
    v.scale_check(result,rows,splits,samples_path)
    return result


def fit_label_transform(samples_path: Path, splits_path: Path, *, split_id: str, output_path: Path) -> Path:
    """Persist a verified train-only transform; fail rather than overwrite."""
    meta=compute_label_transform(samples_path,splits_path,split_id)
    output_path=Path(output_path)
    output_path.parent.mkdir(parents=True,exist_ok=True)
    with output_path.open('x',encoding='utf-8') as f:
        json.dump(meta,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')
    return output_path


def normalise_strength(strength: float, meta: dict) -> float:
    v.require(math.isfinite(strength) and strength>0,'strength必须有限且>0')
    a,b=meta['min_log10'],meta['max_log10'];v.require(math.isfinite(a) and math.isfinite(b) and a<b,'min-max参数无效')
    return (math.log10(strength)-a)/(b-a)


def restore_prediction(value: float, meta: dict) -> tuple[float,float]:
    v.require(math.isfinite(value),'预测非有限')
    a,b=meta['min_log10'],meta['max_log10'];v.require(math.isfinite(a) and math.isfinite(b) and a<b,'min-max参数无效')
    z=a+(b-a)*value
    try:raw=10.0**z
    except OverflowError as exc:raise v.ContractError('反变换溢出；记录预测失败，不裁剪掩盖') from exc
    v.require(math.isfinite(raw) and raw>0,'反变换非有限或下溢')
    return z,raw


def main() -> int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--samples',type=Path,required=True);p.add_argument('--splits',type=Path,required=True)
    p.add_argument('--split-id',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        fit_label_transform(a.samples,a.splits,split_id=a.split_id,output_path=a.output)
        print(a.output);return 0
    except (v.ContractError,OSError,ValueError) as e:
        p.exit(2,f'{e}\n')

if __name__=='__main__':raise SystemExit(main())
