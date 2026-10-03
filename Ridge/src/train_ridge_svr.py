"""Train published Ridge/SVR selections through the shared immutable-run CLI."""
from pathlib import Path
import argparse
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pr02.__main__ import main

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'project_config.json')
    args=parser.parse_args()
    main(['--config',str(args.config),'run-ridge-svr'])
