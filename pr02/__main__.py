from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from .common import ROOT, DATA, SPLITS, TRANSFORM, load_bundle, read_json, resolve, v


def main(argv=None):
    parser = argparse.ArgumentParser(description='PR02-01 shared experiment management')
    parser.add_argument('--config',type=Path,default=ROOT/'project_config.json')
    sub = parser.add_subparsers(dest='command',required=True)
    for name in ['run-knn','run-thermo','evaluate']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--run-id')
    sub.add_parser('validate-data')
    sub.add_parser('run-all')
    sub.add_parser('verify')
    args = parser.parse_args(argv)
    cfg = read_json(resolve(args.config))
    try:
        load_bundle()
        if args.command=='validate-data':
            print(v.bundle_check(DATA,SPLITS,transform=TRANSFORM))
        elif args.command=='run-knn':
            from .knn import run
            run(args.run_id or cfg['runs']['knn'])
        elif args.command=='run-thermo':
            from .thermo import run
            run(args.run_id or cfg['runs']['thermo'])
        elif args.command=='evaluate':
            from .evaluate import run
            run([resolve(p) for p in cfg['validation_predictions']],args.run_id or cfg['runs']['comparison'])
        elif args.command=='verify':
            from .verify import verify
            verify(cfg)
        else:
            from .knn import run as knn
            from .thermo import run as thermo
            from .evaluate import run as evaluate
            # Preflight all destinations before any work. No partly overwritten rerun.
            import yaml
            cnn = yaml.safe_load(resolve(cfg['cnn_config']).read_text(encoding='utf-8'))
            destinations = [ROOT/'runs'/r for r in cfg['runs'].values()]+[resolve(cnn['output_dir'])]
            for path in destinations:
                v.require(not path.exists() or not any(path.iterdir()),f'Use new run IDs; existing output: {path}')
            knn(cfg['runs']['knn'])
            thermo(cfg['runs']['thermo'])
            subprocess.run([sys.executable,str(ROOT/'CNN/run_cnn.py'),'train','--config',str(resolve(cfg['cnn_config']))],cwd=ROOT,check=True)
            evaluate([resolve(p) for p in cfg['validation_predictions']],cfg['runs']['comparison'])
            from .verify import verify
            verify(cfg)
    except (v.ContractError,ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f'PR02 failed: {exc}\n')


if __name__=='__main__': main()
