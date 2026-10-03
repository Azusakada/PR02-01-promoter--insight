# -*- coding: utf-8 -*-
"""
固定词表的 k-mer 计数特征。词表按 ACGT 字典序枚举全部 4^k 个 k-mer，
不依赖训练集出现与否，保证行序=sample_id 源序、列序可复现。
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data_snapshot"
FEAT_DIR = ROOT / "features"
BASES = ("A", "C", "G", "T")


def vocabulary(k: int) -> list[str]:
    return ["".join(p) for p in itertools.product(BASES, repeat=k)]


def kmer_counts(seq: str, k: int, vocab_index: dict[str, int]) -> np.ndarray:
    x = np.zeros(len(vocab_index), dtype=np.float32)
    for i in range(len(seq) - k + 1):
        mer = seq[i : i + k]
        j = vocab_index.get(mer)
        if j is not None:
            x[j] += 1.0
    return x


def build_kmer_matrix(sequences: list[str], k: int) -> tuple[np.ndarray, list[str]]:
    vocab = vocabulary(k)
    index = {mer: i for i, mer in enumerate(vocab)}
    x = np.stack([kmer_counts(seq, k, index) for seq in sequences], axis=0)
    return x, vocab


def main() -> None:
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(SNAPSHOT / "data_v1.tsv", sep="\t")
    ids = data["sample_id"].astype(str).to_numpy()
    seqs = data["sequence"].astype(str).tolist()
    np.save(FEAT_DIR / "sample_ids.npy", ids)
    (FEAT_DIR / "sample_ids.txt").write_text("\n".join(ids.tolist()) + "\n", encoding="utf-8")
    feature_index = pd.DataFrame(
        {
            "dataset_id": "course_ecoli50_strength",
            "row_index": np.arange(len(ids), dtype=int),
            "sample_id": ids,
            "feature_version": "kmer_count_acgt_lex_v1",
            "schema_version": "2.0.0",
            "data_version": "ecoli50_strength_v1",
        }
    )
    feature_index.to_csv(FEAT_DIR / "feature_index.tsv", sep="\t", index=False, lineterminator="\n")

    manifest = {
        "data_version": "ecoli50_strength_v1",
        "n_samples": int(len(ids)),
        "row_order": "data_v1 source order, sample_id ecoli50_r000001..r011884",
        "feature_version": "kmer_count_acgt_lex_v1",
        "feature_index": "features/feature_index.tsv",
        "features": [],
    }
    for k in (3, 4, 5):
        x, vocab = build_kmer_matrix(seqs, k)
        path = FEAT_DIR / f"kmer{k}.npz"
        np.savez_compressed(
            path,
            X=x,
            sample_ids=ids,
            vocabulary=np.array(vocab),
            k=np.array([k]),
        )
        (FEAT_DIR / f"kmer{k}_vocabulary.txt").write_text("\n".join(vocab) + "\n", encoding="utf-8")
        manifest["features"].append(
            {
                "name": f"kmer{k}",
                "kind": "kmer_count",
                "k": k,
                "n_features": int(x.shape[1]),
                "windows_per_seq": 50 - k + 1,
                "path": f"features/kmer{k}.npz",
                "vocabulary_path": f"features/kmer{k}_vocabulary.txt",
                "note": "全部 4^k 个 k-mer，ACGT 字典序；等长序列下计数与频率只差常数",
            }
        )

    phys = pd.read_csv(SNAPSHOT / "promoter_physicochemical.tsv", sep="\t")
    if not phys["sample_id"].astype(str).equals(pd.Series(ids)):
        raise RuntimeError("physchem sample_id order does not match data_v1")
    phys_path = FEAT_DIR / "physchem8.tsv"
    phys.to_csv(phys_path, sep="\t", index=False, lineterminator="\n")
    manifest["features"].append(
        {
            "name": "physchem8",
            "kind": "derived_physicochemical",
            "n_features": 8,
            "columns": [c for c in phys.columns if c != "sample_id"],
            "path": "features/physchem8.tsv",
            "experimental_measurement": False,
            "note": "课程已有 8 类派生理化特征，按源表行序接入，不是湿实验新测。",
        }
    )
    (FEAT_DIR / "feature_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"n": len(ids), "features": [f["name"] for f in manifest["features"]]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
