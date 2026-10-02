"""Strict ID/version checks and evidence metadata for analysis-only runs."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "analysis_m2m3"
SAMPLES = ROOT / "data/01_Ecoli_strength/data_v1.tsv"
SPLITS = ROOT / "data/01_Ecoli_strength/split_manifest.tsv"
SCHEMA = "2.0.0"
DATASET = "course_ecoli50_strength"
SEED = 20260928
COLORS = {"train": "#386CB0", "val": "#E18B35", "test": "#45A37A"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def single(frame, column):
    require(column in frame, f"Missing {column}")
    values = frame[column].dropna().unique()
    require(len(values) == 1 and frame[column].notna().all(), f"Mixed/missing {column}")
    return str(values[0])


def load_data(samples=SAMPLES, splits=SPLITS):
    data = pd.read_csv(samples, sep="\t")
    split = pd.read_csv(splits, sep="\t")
    required = {"dataset_id", "sample_id", "source_row", "sequence", "sequence_length", "strength", "target_log10", "annotation_status", "schema_version", "data_version"}
    require(required <= set(data), f"Missing dataset fields: {required - set(data)}")
    require(data.sample_id.notna().all() and data.sample_id.is_unique, "Invalid/duplicate sample_id")
    require(data.source_row.notna().all() and data.source_row.is_unique and (data.source_row >= 1).all(), "Invalid source_row")
    require(data.sequence.str.fullmatch("[ACGT]{50}").fillna(False).all(), "Invalid 50 bp sequence")
    require(data.sequence_length.eq(50).all(), "Invalid sequence_length")
    require(np.isfinite(data.strength).all() and data.strength.gt(0).all(), "Invalid strength")
    require(np.allclose(np.log10(data.strength), data.target_log10, rtol=0, atol=1e-10), "target_log10 mismatch")
    require(data.annotation_status.isin(["missing", "partial", "reliable_external", "tool_inferred"]).all(), "Unknown annotation status")
    for column in ("dataset_id", "data_version", "schema_version"):
        require(single(data, column) == single(split, column), f"Dataset/split {column} mismatch")
    require(single(data, "dataset_id") == DATASET and single(data, "schema_version") == SCHEMA, "Unsupported dataset/schema")
    require(split.sample_id.notna().all() and split.sample_id.is_unique, "Duplicate/missing split ID")
    require(set(data.sample_id) == set(split.sample_id), "Split does not cover exact dataset IDs")
    require(set(split['split']) == {"train", "val", "test"}, "Invalid split subsets")
    single(split, "split_id")
    return data.merge(split[["sample_id", "split", "split_id"]], on="sample_id", validate="one_to_one")


def sequence_features(data):
    counts = {base: data.sequence.str.count(base).to_numpy(dtype=float) for base in "ACGT"}
    out = data[["sample_id", "split", "strength", "target_log10"]].copy()
    out["gc_content"] = (counts['G'] + counts['C']) / 50
    out["at_content"] = 1 - out.gc_content
    for base in "ACGT":
        out[f"{base.lower()}_fraction"] = counts[base] / 50
    p = np.column_stack([counts[base] / 50 for base in "ACGT"])
    out["entropy"] = -(p * np.log2(np.where(p > 0, p, 1))).sum(axis=1)
    return out


class Run:
    def __init__(self, output, stage, task_id, data, inputs, parameters):
        self.output = Path(output).resolve()
        relative(self.output)
        require(not self.output.exists() or not any(self.output.iterdir()), f"Output already nonempty: {self.output}; choose a new run")
        self.output.mkdir(parents=True, exist_ok=True)
        (self.output / "source_tables").mkdir()
        (self.output / "figures").mkdir()
        self.run_id = self.output.name
        self.data = data
        self.figures = []
        self.meta = {
            "stage": stage, "task_id": task_id, "run_id": self.run_id,
            "method_name": "descriptive_analysis", "dataset_id": DATASET,
            "data_version": single(data, "data_version"), "schema_version": SCHEMA,
            "split_id": single(data, "split_id"), "evidence_level": "preliminary",
            "execution_status": "success", "seed": SEED,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "command": [sys.executable, *sys.argv],
            "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "dirty_worktree": bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()),
            "environment": {"python": platform.python_version(), "platform": platform.platform(),
                            "packages": {x: importlib.metadata.version(x) for x in ["numpy", "pandas", "scipy", "matplotlib", "scikit-learn"]}},
            "parameters": parameters, "fit_subsets": [], "selection_subsets": [], "evaluation_subsets": [],
            "inputs": [{"path": relative(p), "sha256": sha256(p)} for p in inputs],
        }
        patch = subprocess.check_output(["git", "diff", "HEAD", "--", "analysis_m2m3"], cwd=ROOT)
        if patch:
            (self.output / "source_patch.diff").write_bytes(patch)
            self.meta['source_patch'] = relative(self.output / "source_patch.diff")
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold", "figure.dpi": 120})

    def table(self, name, frame, sep=","):
        path = self.output / "source_tables" / name
        frame.to_csv(path, index=False, sep=sep, na_rep="", lineterminator="\n")
        return relative(path)

    def figure(self, name, figure, title, source_tables, subset, scale, n, caption):
        figure.tight_layout()
        png = self.output / "figures" / f"{name}.png"
        svg = self.output / "figures" / f"{name}.svg"
        figure.savefig(png, dpi=180, bbox_inches="tight")
        figure.savefig(svg, bbox_inches="tight")
        plt.close(figure)
        self.figures.append({"figure_id": name, "title": title, "run_id": self.run_id,
                             "data_version": self.meta['data_version'], "split_id": self.meta['split_id'],
                             "subset": subset, "target_scale": scale, "n_samples": int(n),
                             "annotation_level": "missing", "reliable_annotation_coverage": 0,
                             "parameters": self.meta['parameters'],
                             "source_tables": [{"path": p, "sha256": sha256(ROOT / p)} for p in source_tables],
                             "images": [{"path": relative(p), "sha256": sha256(p)} for p in [png, svg]],
                             "caption": caption, "visual_check_status": "unassessed"})

    def finish(self):
        write_json(self.output / "figure_manifest.json", self.figures)
        artifacts = [p for p in sorted(self.output.rglob('*')) if p.is_file() and p.name != "run_manifest.json"]
        self.meta['artifacts'] = [{"kind": "source_patch" if p.name == "source_patch.diff" else p.suffix.lstrip('.'), "path": relative(p), "sha256": sha256(p)} for p in artifacts]
        write_json(self.output / "run_manifest.json", self.meta)
        print(relative(self.output))
