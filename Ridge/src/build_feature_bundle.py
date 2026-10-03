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
    import argparse
    import sys
    sys.path.insert(0,str(ROOT.parent))
    from pr02.kmer import build_features, validate_features
    parser=argparse.ArgumentParser(description='Verify frozen features, or rebuild into a fresh directory')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    result=build_features(args.output) if args.output else validate_features()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
