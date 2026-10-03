"""Generate diagnostics for exactly the prediction files in a project plan."""
import subprocess
import sys
from .common import ROOT, resolve, v


def run(cfg):
    v.require('analysis_runs' in cfg and 'errors' in cfg['analysis_runs'],'No error-analysis run configured')
    output=resolve(cfg['analysis_runs']['errors'])
    subprocess.run([sys.executable,str(ROOT/'analysis_m2m3/error_analysis.py'),'--output',str(output),
                    '--predictions',*[str(resolve(p)) for p in cfg['validation_predictions']]],cwd=ROOT,check=True)
    return output
