"""Evaluation identity includes methods and prediction versions as well as IDs."""
import hashlib
import json
from pathlib import Path


def comparison_id(frames,paths,ids,kind='common',subset='val'):
    first=frames[0][0]
    members=sorted([dict(method_name=rows[0]['method_name'],run_id=rows[0]['run_id'],
                         predictions_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                    for rows,path in zip(frames,paths)],key=lambda member:member['method_name'])
    payload=dict(dataset_id=first['dataset_id'],data_version=first['data_version'],split_id=first['split_id'],
                 subset=subset,methods=members,sample_ids=sorted(ids))
    digest=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    return subset+'_'+kind+'_'+digest
