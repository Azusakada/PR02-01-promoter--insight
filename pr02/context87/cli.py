from __future__ import annotations

import argparse
import copy
import json
import re
from pathlib import Path

from .data import BASE, ROOT, load, read_json, require, validate, write_json
from .experiment import METHODS, evaluate, fresh, path, run_cnn, run_ml, run_thermo


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR02-01 verified 87 bp context workflow (schema 3)')
    parser.add_argument('--config',type=Path,default=ROOT/'project_config.json')
    sub=parser.add_subparsers(dest='command',required=True)
    for command in ['validate-data','run-knn','run-ridge-svr','run-cnn','run-thermo','evaluate','analyze','explain','mutate','ablate','transfer','verify','publish','acknowledge-figures']:
        sub.add_parser(command)
    cmd=sub.add_parser('run-all'); cmd.add_argument('--research',action='store_true',help='Also run attribution, mutations, retrained ablations and cross-species transfer')
    cmd=sub.add_parser('rebuild-data'); cmd.add_argument('--output',type=Path,required=True)
    cmd=sub.add_parser('new-run'); cmd.add_argument('--tag',required=True)
    for command in ['analyze-m2m3', 'verify-m2m3', 'acknowledge-m2m3', 'publish-m2m3']:
        cmd=sub.add_parser(command)
        cmd.add_argument('--output',type=Path,default=Path('runs/ecoli87_m2m3_20261013_v1'))
        if command=='acknowledge-m2m3':
            cmd.add_argument('--reviewed',nargs='+',required=True,help='All ten figure IDs actually inspected by the operator')
    args=parser.parse_args(argv); cfg=read_json(args.config)
    try:
        load(cfg)
        if args.command=='new-run':
            require(bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,40}',args.tag)),'Invalid experiment tag')
            new=copy.deepcopy(cfg); new['runs']={k:f'ecoli87_{k}_{args.tag}' for k in cfg['runs']}
            dest=ROOT/'configs'/f'project_ecoli87_{args.tag}.json'
            require(not dest.exists(),'Configuration already exists'); write_json(dest,new); print(dest)
        elif args.command=='validate-data': print(json.dumps(validate(cfg),ensure_ascii=False,indent=2))
        elif args.command=='rebuild-data':
            from .audit import reproduce
            print(reproduce(cfg,args.output))
        elif args.command=='run-knn': run_ml(cfg,'knn')
        elif args.command=='run-ridge-svr':
            run_ml(cfg,'ridge'); run_ml(cfg,'svr')
        elif args.command=='run-cnn': run_cnn(cfg)
        elif args.command=='run-thermo': run_thermo(cfg)
        elif args.command=='evaluate': evaluate(cfg)
        elif args.command=='analyze':
            from .analysis import analyze
            analyze(cfg)
        elif args.command in ('analyze-m2m3','verify-m2m3','acknowledge-m2m3','publish-m2m3'):
            from . import m2m3
            if args.command=='analyze-m2m3': print(m2m3.analyze(cfg,args.output))
            elif args.command=='verify-m2m3': print(json.dumps(m2m3.verify(cfg,args.output),ensure_ascii=False,indent=2))
            elif args.command=='acknowledge-m2m3': m2m3.acknowledge(cfg,args.output,args.reviewed)
            else: print(m2m3.publish(cfg,args.output))
        elif args.command=='acknowledge-figures':
            from .analysis import acknowledge
            acknowledge(cfg)
        elif args.command=='verify':
            from .audit import verify
            result=verify(cfg); write_json(ROOT/'results/context87/verification.json',result); print(json.dumps(result,indent=2))
        elif args.command=='publish':
            from .audit import publish
            print(publish(cfg))
        elif args.command in ('explain','mutate','ablate','transfer'):
            from . import research
            print(getattr(research,args.command)(cfg))
        else:
            stages=[*METHODS,'comparison','analysis']+(['explain','mutate','ablate','transfer'] if args.research else [])
            for kind in stages: require(not path(cfg,kind).exists() or not any(path(cfg,kind).iterdir()),'Existing run; use new-run: '+kind)
            validate(cfg)
            for kind in ('knn','ridge','svr'): run_ml(cfg,kind)
            run_cnn(cfg); run_thermo(cfg); evaluate(cfg)
            from .analysis import analyze
            analyze(cfg)
            if args.research:
                from .research import explain,mutate,ablate,transfer
                explain(cfg); mutate(cfg); ablate(cfg); transfer(cfg)
            from .audit import verify
            proof=verify(cfg); write_json(ROOT/'results/context87/verification.json',proof)
            print('Completed. View runs/'+cfg['runs']['analysis']+'/figures, then acknowledge-figures and publish.')
    except (ValueError,OSError,RuntimeError) as exc:
        parser.exit(2,f'PR02 failed: {exc}\n')


if __name__=='__main__': main()
