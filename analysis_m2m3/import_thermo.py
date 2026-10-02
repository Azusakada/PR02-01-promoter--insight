"""Import immutable teammate artifacts without merging teammate code."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common import ROOT, relative, require, sha256, write_json

FILES = [
    "thermo/input/context_map.tsv",
    "thermo/results/predictions/thermo_train.csv",
    "thermo/results/predictions/thermo_val.csv",
    "thermo/results/predictions/thermo_val_calibrated.csv",
    "thermo/results/thermo_calibration.json",
    "thermo/thermo_applicability.md",
    "thermo/handoff.md",
    "thermo/recover_context.py",
    "thermo/run_thermo.py",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True, help="Prefer a full immutable commit SHA")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", f"{args.ref}^{{commit}}"], cwd=ROOT, text=True).strip()
    payloads = {p: subprocess.check_output(["git", "show", f"{commit}:{p}"], cwd=ROOT) for p in FILES}
    require(not args.output.exists() or not any(args.output.iterdir()), "Import output already nonempty")
    relative(args.output)
    args.output.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for original, payload in payloads.items():
        path = args.output / Path(original).name
        path.write_bytes(payload)
        artifacts.append({"source_path": original, "path": relative(path), "sha256": sha256(path)})
    write_json(args.output / "import_manifest.json", {
        "source_repository": "https://github.com/Azusakada/PR02-01-promoter--insight",
        "source_commit": commit, "evidence_level": "imported_teammate_preliminary",
        "scope": "Teammate data snapshot; no model training or calculator execution in this module",
        "artifacts": artifacts,
    })
    print(relative(args.output))


if __name__ == "__main__":
    main()
