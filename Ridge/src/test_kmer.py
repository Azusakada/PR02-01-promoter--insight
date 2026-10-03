# -*- coding: utf-8 -*-
"""k-mer 词表、计数和行序的单元测试。不依赖正式训练结果。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_feature_bundle import build_kmer_matrix, kmer_counts, vocabulary  # noqa: E402


def test_vocabulary_is_full_acgt_lexicographic() -> None:
    vocab3 = vocabulary(3)
    assert len(vocab3) == 64
    assert vocab3[0] == "AAA"
    assert vocab3[-1] == "TTT"
    assert vocab3 == sorted(vocab3)
    assert len(set(vocab3)) == 64
    vocab5 = vocabulary(5)
    assert len(vocab5) == 1024
    assert vocab5[0] == "AAAAA"
    assert vocab5[-1] == "TTTTT"


def test_known_sequence_counts() -> None:
    seq = "AAACGT"
    vocab = vocabulary(3)
    index = {mer: i for i, mer in enumerate(vocab)}
    counts = kmer_counts(seq, 3, index)
    assert counts.sum() == 4
    assert counts[index["AAA"]] == 1
    assert counts[index["AAC"]] == 1
    assert counts[index["ACG"]] == 1
    assert counts[index["CGT"]] == 1
    assert counts[index["TTT"]] == 0


def test_equal_length_window_sum() -> None:
    seqs = ["A" * 50, "ACGT" * 12 + "AC", "T" * 50]
    for k in (3, 4, 5):
        x, vocab = build_kmer_matrix(seqs, k)
        assert x.shape == (3, 4**k)
        assert np.allclose(x.sum(axis=1), 50 - k + 1)
        assert vocab[0] == "A" * k


def test_feature_bundle_row_order_if_present() -> None:
    snap = ROOT / "data_snapshot" / "data_v1.tsv"
    ids_txt = ROOT / "features" / "sample_ids.txt"
    kmer3 = ROOT / "features" / "kmer3.npz"
    if not (snap.exists() and ids_txt.exists() and kmer3.exists()):
        return
    data = pd.read_csv(snap, sep="\t")
    ids = ids_txt.read_text(encoding="utf-8").splitlines()
    blob = np.load(kmer3, allow_pickle=True)
    assert ids == data["sample_id"].astype(str).tolist()
    assert list(blob["sample_ids"].astype(str)) == ids
    assert blob["X"].shape == (11884, 64)
    assert np.allclose(blob["X"].sum(axis=1), 48)


if __name__ == "__main__":
    test_vocabulary_is_full_acgt_lexicographic()
    test_known_sequence_counts()
    test_equal_length_window_sum()
    test_feature_bundle_row_order_if_present()
    print("kmer unit tests passed")
