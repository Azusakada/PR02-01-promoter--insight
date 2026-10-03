"""Compatibility entry point for the shared frozen project inputs."""
from pathlib import Path
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from pr02.kmer import check_inputs

if __name__=='__main__':
    print(json.dumps(check_inputs(),ensure_ascii=False,indent=2))
