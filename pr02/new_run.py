from __future__ import annotations
import re
import yaml
from .common import ROOT, read_json, v, write_json


def create(tag, base):
    v.require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,64}',tag)), 'Invalid tag')
    plan = read_json(base)
    cnn = yaml.safe_load((ROOT/plan['cnn_config']).read_text(encoding='utf-8'))
    plan_path = ROOT/'configs'/f'integration_{tag}.json'
    cnn_path = ROOT/'CNN/configs'/f'cnn_{tag}.yaml'
    v.require(not plan_path.exists() and not cnn_path.exists(),'Config already exists; use a new tag')
    plan['runs'] = dict(knn='knn_'+tag,thermo='thermo_'+tag,comparison='comparison_'+tag)
    if 'ridge' in read_json(base)['runs']:
        plan['runs'].update(ridge='ridge_'+tag,svr='svr_'+tag)
    cnn['run_id'] = 'cnn_'+tag
    cnn['output_dir'] = 'CNN/runs/'+cnn['run_id']
    plan['cnn_config'] = cnn_path.relative_to(ROOT).as_posix()
    plan['validation_predictions'] = [f"runs/{plan['runs']['knn']}/predictions_knn_val.csv",
                                     f"runs/{plan['runs']['thermo']}/predictions/thermo_val_calibrated.csv",
                                     f"{cnn['output_dir']}/validation/predictions_cnn.csv"]
    if 'ridge' in plan['runs']:
        plan['validation_predictions'].extend([f"runs/{plan['runs']['ridge']}/predictions_ridge_val.csv",
                                               f"runs/{plan['runs']['svr']}/predictions_svr_val.csv"])
    cnn_path.write_text(yaml.safe_dump(cnn,allow_unicode=True,sort_keys=False),encoding='utf-8',newline='\n')
    write_json(plan_path,plan)
    print(f'Created {plan_path}; run: python -m pr02 --config {plan_path.relative_to(ROOT).as_posix()} run-all')
    return plan_path
