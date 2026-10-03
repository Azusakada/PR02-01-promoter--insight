# -*- coding: utf-8 -*-
"""
从课程原始 E_coli.txt 重建与李宇飞 data_v1 对齐的主表，
并用已公开的划分算法恢复固定 split。

划分已用官方 label_transform.train_ids（8318）和截断的 split_manifest
前 8729 行交叉验证：sklearn train_test_split(train_size=8318, random_state=20260928)
再对剩余 50/50 划分 val/test，完全吻合。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
LOCAL = PROJECT / "_local_course"
SNAPSHOT = ROOT / "data_snapshot"
LABEL_TRANSFORM_CANDIDATES = [
    SNAPSHOT / "label_transform.json",
    Path(
        r"C:\Users\li\.cursor\projects\c-Users-li-Desktop-promotorC\agent-tools\40edaf7d-5627-4671-b8da-9ebf8ee8ba60.txt"
    ),
]

DATASET_ID = "course_ecoli50_strength"
DATA_VERSION = "ecoli50_strength_v1"
SPLIT_ID = "ecoli50_random_20260928_v1"
SEED = 20260928
N = 11884
OFFICIAL_SAMPLES_SHA256 = "0599c5b29198c78c481d035bf32596a53411310f401463a62e160bfdbe93ca7e"
OFFICIAL_SPLITS_SHA256 = "5d162f64e68a305749033e5f9e7f4d5bdfeb6b28af757883446f75682f9fd9b0"
TRANSFORM_ID = "log10mm_59711a1c2217859e"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_data_v1() -> pd.DataFrame:
    raw = pd.read_csv(LOCAL / "E_coli.txt", sep="\t")
    raw.columns = [c.strip() for c in raw.columns]
    if list(raw.columns)[:2] != ["strength", "promoter"]:
        raise ValueError(f"unexpected E_coli.txt columns: {list(raw.columns)}")
    if len(raw) != N:
        raise ValueError(f"expected {N} rows, got {len(raw)}")
    seq = raw["promoter"].astype(str).str.strip().str.upper()
    strength = raw["strength"].astype(float)
    if not seq.str.fullmatch(r"[ACGT]{50}").all():
        raise ValueError("sequences are not all 50 bp ACGT")
    if (strength <= 0).any():
        raise ValueError("non-positive strength")
    if seq.duplicated().any():
        raise ValueError("duplicate sequences")
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "sample_id": [f"ecoli50_r{i:06d}" for i in range(1, N + 1)],
            "source_row": np.arange(1, N + 1, dtype=int),
            "sequence": seq.to_numpy(),
            "sequence_length": 50,
            "strength": strength.to_numpy(),
            "target_log10": np.log10(strength.to_numpy()),
            "annotation_status": "missing",
            "schema_version": "2.0.0",
            "data_version": DATA_VERSION,
        }
    )


def try_load_official_data_v1() -> pd.DataFrame | None:
    candidates = [
        SNAPSHOT / "official_data_v1.tsv",
        SNAPSHOT / "data_v1.tsv",
    ]
    for path in candidates:
        if path.exists() and sha256_file(path) == OFFICIAL_SAMPLES_SHA256:
            return pd.read_csv(path, sep="\t")
    urls = [
        "https://raw.githubusercontent.com/Azusakada/PR02-01-promoter--insight/main/data/01_Ecoli_strength/data_v1.tsv",
        "https://cdn.jsdelivr.net/gh/Azusakada/PR02-01-promoter--insight@main/data/01_Ecoli_strength/data_v1.tsv",
    ]
    for url in urls:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "promotorC-ridge"})
            with urllib.request.urlopen(req, timeout=60) as response:
                blob = response.read()
            tmp = SNAPSHOT / "official_data_v1.tsv"
            SNAPSHOT.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(blob)
            if sha256_file(tmp) == OFFICIAL_SAMPLES_SHA256:
                return pd.read_csv(tmp, sep="\t")
        except Exception:
            continue
    return None


def build_split(sample_ids: list[str], official_train: set[str]) -> pd.DataFrame:
    idx = np.arange(len(sample_ids))
    train_idx, rest_idx = train_test_split(
        idx, train_size=8318, random_state=SEED, shuffle=True
    )
    val_idx, test_idx = train_test_split(
        rest_idx, test_size=0.5, random_state=SEED, shuffle=True
    )
    split = np.empty(len(sample_ids), dtype=object)
    split[train_idx] = "train"
    split[val_idx] = "val"
    split[test_idx] = "test"
    recovered_train = {sample_ids[i] for i in train_idx}
    if recovered_train != official_train:
        missing = sorted(official_train - recovered_train)[:5]
        extra = sorted(recovered_train - official_train)[:5]
        raise RuntimeError(f"split train set mismatch missing={missing} extra={extra}")
    counts = pd.Series(split).value_counts().to_dict()
    if counts != {"train": 8318, "val": 1783, "test": 1783}:
        raise RuntimeError(f"unexpected split sizes {counts}")
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "sample_id": sample_ids,
            "split": split,
            "group_id": "",
            "split_strategy": "random",
            "seed": SEED,
            "schema_version": "2.0.0",
            "data_version": DATA_VERSION,
        }
    )


def load_label_transform() -> tuple[dict, Path]:
    for path in LABEL_TRANSFORM_CANDIDATES:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("transform_id") == TRANSFORM_ID:
                return payload, path
    urls = [
        "https://raw.githubusercontent.com/Azusakada/PR02-01-promoter--insight/main/data/01_Ecoli_strength/label_transform.json",
    ]
    for url in urls:
        try:
            import urllib.request

            req = urllib.request.Request(url, headers={"User-Agent": "promotorC-ridge"})
            with urllib.request.urlopen(req, timeout=60) as response:
                blob = response.read()
            dest = SNAPSHOT / "label_transform.json"
            SNAPSHOT.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(blob)
            payload = json.loads(dest.read_text(encoding="utf-8"))
            if payload.get("transform_id") == TRANSFORM_ID:
                return payload, dest
        except Exception:
            continue
    raise FileNotFoundError("label_transform.json not found locally or from official repo")


def main() -> None:
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    official, lt_src = load_label_transform()
    if official["transform_id"] != TRANSFORM_ID:
        raise RuntimeError("unexpected transform_id")
    official_table = try_load_official_data_v1()
    source = "official_repo_data_v1"
    if official_table is None:
        data = build_data_v1()
        source = "reconstructed_from_E_coli_txt"
    else:
        data = official_table
    required = [
        "dataset_id",
        "sample_id",
        "source_row",
        "sequence",
        "sequence_length",
        "strength",
        "target_log10",
        "annotation_status",
        "schema_version",
        "data_version",
    ]
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise RuntimeError(f"data_v1 missing columns: {missing}")
    data = data[required].copy()
    if len(data) != N:
        raise RuntimeError(f"expected {N} rows, got {len(data)}")
    splits = build_split(data["sample_id"].tolist(), set(official["train_ids"]))
    data_path = SNAPSHOT / "data_v1.tsv"
    split_path = SNAPSHOT / "split_manifest.tsv"
    official_bytes = SNAPSHOT / "official_data_v1.tsv"
    if official_bytes.exists() and sha256_file(official_bytes) == OFFICIAL_SAMPLES_SHA256:
        shutil.copyfile(official_bytes, data_path)
        data = pd.read_csv(data_path, sep="\t")[required]
    elif data_path.exists() and sha256_file(data_path) == OFFICIAL_SAMPLES_SHA256:
        data = pd.read_csv(data_path, sep="\t")[required]
    else:
        data.to_csv(data_path, sep="\t", index=False, lineterminator="\n")
    splits.to_csv(split_path, sep="\t", index=False, lineterminator="\n")
    dest_lt = SNAPSHOT / "label_transform.json"
    if lt_src.resolve() != dest_lt.resolve():
        shutil.copyfile(lt_src, dest_lt)
    phys_src = LOCAL / "promoter_physicochemical.csv"
    phys = pd.read_csv(phys_src)
    expected_cols = [
        "gc_content",
        "at_content",
        "gc_skew",
        "at_skew",
        "melting_temp",
        "bendability",
        "stacking_energy",
        "entropy",
    ]
    if list(phys.columns) != expected_cols or len(phys) != N:
        raise ValueError("local physicochemical table does not align with E_coli.txt")
    phys.insert(0, "sample_id", data["sample_id"].to_numpy())
    phys_path = SNAPSHOT / "promoter_physicochemical.tsv"
    phys.to_csv(phys_path, sep="\t", index=False, lineterminator="\n")
    note = {
        "data_v1_sha256": sha256_file(data_path),
        "official_data_v1_sha256": OFFICIAL_SAMPLES_SHA256,
        "data_v1_hash_match": sha256_file(data_path) == OFFICIAL_SAMPLES_SHA256,
        "split_sha256": sha256_file(split_path),
        "official_split_sha256": OFFICIAL_SPLITS_SHA256,
        "split_hash_match": sha256_file(split_path) == OFFICIAL_SPLITS_SHA256,
        "split_membership_verified_against_official_train_ids": True,
        "data_source": source,
        "note": (
            "优先使用李宇飞仓库中的冻结 data_v1；若下载失败则按清洗日志从课程 E_coli.txt 重建官方列。"
            "sample_id/sequence/strength/target_log10 与划分成员已核对。"
            "理化特征来自课程 promoter_physicochemical.csv，按源表行序接入，不是湿实验新测。"
        ),
        "transform_id": TRANSFORM_ID,
        "split_id": SPLIT_ID,
        "n": N,
        "split_sizes": {"train": 8318, "val": 1783, "test": 1783},
    }
    (SNAPSHOT / "input_audit.json").write_text(
        json.dumps(note, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(note, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
