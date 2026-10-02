from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
import platform
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
CONTRACT_DIR = ROOT / "contracts" / "scripts"
# Keep the supplied validator byte-identical; import it by its file path.
spec = importlib.util.spec_from_file_location("pr02_contract_validator", CONTRACT_DIR / "validate.py")
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)
sys.modules.setdefault("validate", v)
spec = importlib.util.spec_from_file_location("pr02_contract_transform", CONTRACT_DIR / "label_transform.py")
label = importlib.util.module_from_spec(spec)
spec.loader.exec_module(label)
spec = importlib.util.spec_from_file_location("pr02_api_contract", CONTRACT_DIR / "api_contract.py")
api = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = api
spec.loader.exec_module(api)
ModelArtifact = api.ModelArtifact
PredictionBundle = api.PredictionBundle


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_table(path: Path, rows: list[dict], columns=None, delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns or list(rows[0]), delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: "" if val is None else str(val).lower() if isinstance(val, bool) else val for k, val in row.items()})


def ids_hash(ids):
    return hashlib.sha256("".join(x + "\n" for x in sorted(ids)).encode("utf-8")).hexdigest()


def relative(path: Path) -> str:
    return Path(path).resolve().relative_to(ROOT).as_posix()


def resolve(path) -> Path:
    p = Path(path)
    p = p if p.is_absolute() else ROOT / p
    p = p.resolve()
    v.require(p.is_relative_to(ROOT), f"路径必须位于仓库内: {p}")
    return p


def fresh_dir(path: Path):
    path = resolve(path)
    v.require(not path.exists() or not any(path.iterdir()), f"输出目录非空，拒绝覆盖: {path}")
    path.mkdir(parents=True, exist_ok=True)


def config(path: Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as f:
        cfg = yaml.safe_load(f)
    v.require(cfg["schema_version"] == "2.0.0", "schema_version 必须是 2.0.0")
    v.require(cfg["model"]["sequence_length"] == 50 and cfg["model"]["channels"] == list("ACGT"), "模型输入必须为50bp和ACGT通道")
    v.require(cfg["evidence_level"] in {"smoke", "preliminary"}, "本模块只交付smoke或preliminary")
    t = cfg["training"]
    for key in ["batch_size", "max_epochs", "patience", "num_threads"]:
        v.require(type(t[key]) is int and t[key] > 0, f"training.{key}必须是正整数")
    v.require(type(t["seed"]) is int and t["seed"] >= 0, "seed必须非负整数")
    v.require(t["num_workers"] >= 0 and t["learning_rate"] > 0 and t["weight_decay"] >= 0, "训练配置无效")
    return cfg


def seed_everything(cfg):
    t = cfg["training"]
    seed = t["seed"]
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(t["num_threads"])
    torch.use_deterministic_algorithms(t["deterministic"])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = t["deterministic"]


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    v.require(name in {"cpu", "cuda"}, "device必须为cpu/cuda/auto")
    v.require(name != "cuda" or torch.cuda.is_available(), "当前PyTorch环境没有可用CUDA")
    return torch.device(name)


def git_info():
    executable = os.environ.get("PR02_GIT") or shutil.which("git")
    v.require(executable, "请将Git加入PATH或设置PR02_GIT")
    def git(*args):
        return subprocess.check_output([executable, "-C", str(ROOT), *args], text=True, encoding="utf-8").strip()
    return {"code_commit": git("rev-parse", "HEAD"), "dirty_worktree": bool(git("status", "--porcelain")), "branch": git("branch", "--show-current")}


def source_snapshot(output: Path):
    """Capture tracked changes and complete new source files without git add/commit."""
    executable = os.environ.get("PR02_GIT") or shutil.which("git")
    patch = subprocess.check_output([executable, "-C", str(ROOT), "diff", "HEAD", "--", "CNN", "README.md", ".gitignore"])
    # `git diff HEAD` does not include untracked source; add its actual content.
    import difflib
    untracked = subprocess.check_output([executable, "-C", str(ROOT), "ls-files", "--others", "--exclude-standard", "--", "CNN", ".gitignore"], text=True, encoding="utf-8").splitlines()
    for name in untracked:
        path = resolve(name)
        if path.suffix in {".py", ".md", ".yaml", ".json", ".txt"} or path.name == ".gitignore":
            lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
            patch += (f"diff --git a/{name} b/{name}\nnew file mode 100644\n" +
                      "".join(difflib.unified_diff([], lines, fromfile="/dev/null", tofile="b/"+name))).encode("utf-8")
    patch_path = output / "source_patch.diff"
    patch_path.write_bytes(patch)
    files = {}
    archive = output / "source_snapshot.zip"
    import zipfile
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted((ROOT / "CNN").rglob("*")):
            if not path.is_file() or any(part in {"runs", "__pycache__", ".venv"} for part in path.relative_to(ROOT / "CNN").parts):
                continue
            rel = relative(path)
            files[rel] = v.sha256(path)
            z.write(path, rel)
    write_json(output / "source_files.json", files)
    return [artifact("source_patch", patch_path), artifact("source_snapshot", archive), artifact("source_files", output / "source_files.json")]


def artifact(kind, path):
    return {"kind": kind, "path": relative(path), "sha256": v.sha256(path)}


def run_manifest(cfg, output, data_version, artifacts, *, fit, selection, evaluation, status="success"):
    git = git_info()
    if git["dirty_worktree"]:
        artifacts.extend(source_snapshot(output))
    manifest = {
        "dataset_id": "course_ecoli50_strength", "schema_version": "2.0.0",
        "stage": "M3", "task_id": cfg["task_id"], "run_id": cfg["run_id"],
        "method_name": "cnn_1d", "data_version": data_version, "split_id": cfg["data"]["split_id"],
        "evidence_level": cfg["evidence_level"], "execution_status": status,
        "seed": cfg["training"]["seed"], "created_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "command": [sys.executable, *sys.argv], "code_commit": git["code_commit"],
        "dirty_worktree": git["dirty_worktree"], "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {"torch": str(torch.__version__), "numpy": np.__version__, "PyYAML": yaml.__version__},
            "device": cfg["training"]["device"], "git_branch": git["branch"]},
        "parameters": cfg, "fit_subsets": fit, "selection_subsets": selection,
        "evaluation_subsets": evaluation, "artifacts": artifacts,
    }
    write_json(output / "run_manifest.json", manifest)
    v.run_check(output / "run_manifest.json", ROOT)
    return output / "run_manifest.json"
