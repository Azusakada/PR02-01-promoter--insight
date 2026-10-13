"""Choose a versioned workflow from configuration, never by sequence truncation."""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[1]/'project_config.json')
    args,_=parser.parse_known_args(argv)
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('schema_version')=='3.0.0':
        from .context87.cli import main as execute
    else:
        from .legacy_cli import main as execute
    return execute(argv)


if __name__=='__main__': main()
