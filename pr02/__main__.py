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
    sub.add_parser('run-ridge-svr')
    sub.add_parser('analyze')
    sub.add_parser('validate-data')
    sub.add_parser('run-all')
    sub.add_parser('verify')
    sub.add_parser('publish')
    cmd = sub.add_parser('new-run')
    cmd.add_argument('--tag',required=True)
    args = parser.parse_args(argv)
    cfg = read_json(resolve(args.config))
    try:
        v.require(cfg.get('schema_version')=='2.0.0','Unsupported project schema')
        _,_,identity = load_bundle()
        for name,path in [('samples',DATA),('splits',SPLITS),('label_transform',TRANSFORM)]:
            v.require(resolve(cfg[name])==path,f'Unsupported shared {name}; do not silently change frozen data')
        for name in ['dataset_id','data_version','split_id']:
            v.require(cfg[name]==identity[name],f'Project {name} mismatch')
        v.require(cfg.get('test_metrics_enabled') is False,'This M3 entry point does not enable test model selection/evaluation')
        if args.command=='new-run':
            from .new_run import create
            create(args.tag,resolve(args.config))
        elif args.command=='validate-data':
            print(v.bundle_check(DATA,SPLITS,transform=TRANSFORM))
        elif args.command=='run-knn':
            from .knn import run
            run(args.run_id or cfg['runs']['knn'])
        elif args.command=='run-thermo':
            from .thermo import run
            run(args.run_id or cfg['runs']['thermo'])
        elif args.command=='run-ridge-svr':
            from .kmer import run
            run(cfg['runs']['ridge'],cfg['runs']['svr'])
        elif args.command=='analyze':
            from .analysis import run
            run(cfg)
        elif args.command=='evaluate':
            from .evaluate import run
            run([resolve(p) for p in cfg['validation_predictions']],args.run_id or cfg['runs']['comparison'])
        elif args.command=='publish':
            from .publish import publish
            publish(cfg)
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
            if 'analysis_runs' in cfg: destinations.append(resolve(cfg['analysis_runs']['errors']))
            for path in destinations:
                v.require(not path.exists() or not any(path.iterdir()),f'Use new run IDs; existing output: {path}')
            knn(cfg['runs']['knn'])
            thermo(cfg['runs']['thermo'])
            if 'ridge' in cfg['runs']:
                from .kmer import run as kmer
                kmer(cfg['runs']['ridge'],cfg['runs']['svr'])
            subprocess.run([sys.executable,str(ROOT/'CNN/run_cnn.py'),'train','--config',str(resolve(cfg['cnn_config']))],cwd=ROOT,check=True)
            evaluate([resolve(p) for p in cfg['validation_predictions']],cfg['runs']['comparison'])
            if 'analysis_runs' in cfg:
                from .analysis import run as analyze
                output=analyze(cfg)
                print(f'Computations complete. Review figures in {output}/figures, acknowledge them with analysis_m2m3/verify_run.py, then run pr02 publish. Current published results have not changed.')
                return
            from .publish import publish
            publish(cfg)
    except (v.ContractError,ValueError,OSError,RuntimeError,subprocess.CalledProcessError) as exc:
        parser.exit(2,f'PR02 failed: {exc}\n')


if __name__=='__main__': main()
