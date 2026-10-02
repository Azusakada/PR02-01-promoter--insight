"""Compatibility entry point; shared configuration and immutable run outputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pr02.__main__ import main

if __name__ == "__main__":
    main(['evaluate',*sys.argv[1:]])
