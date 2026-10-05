# -*- coding: utf-8 -*-
"""
build_thermo_inputs.py  —  PR02-01 热力学基线 / 工具输入适配

把统一主表 data_v1.tsv 与上下文映射 context_map.tsv 转成 Promoter Calculator /
regseq2 的工具输入表（契约表 calculator_inputs，schema 2.0.0）。

关键设计
--------
- 50 bp 本身不足以让 regseq2 运行（有效 TSS 需 >=58 bp 上游 + 20 bp 下游）。
  因此对能在参考基因组唯一定位的样本，补取两侧各 50 bp 上下文，构造 150 bp 输入
  （input_origin=recovered_context）；定位失败的样本保持原始 50 bp 并标 blocked。
- 不推断 TSS/-10/-35；tss_mode=scan（由工具自行扫描）。
- 保留 sample_id 与运行模式，逐样本记录来源坐标。

输出
----
thermo/input/calculator_inputs_both_strands_v2.tsv
thermo/input/thermo_input_summary_v2.json

用法
----
python thermo/build_thermo_inputs.py
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DATA_V1 = os.path.join(REPO, "data", "01_Ecoli_strength", "data_v1.tsv")
CONTEXT_MAP = os.path.join(HERE, "input", "context_map.tsv")
OUT_TSV = os.path.join(HERE, "input", "calculator_inputs_both_strands_v2.tsv")
OUT_JSON = os.path.join(HERE, "input", "thermo_input_summary_v2.json")

DATASET_ID = "course_ecoli50_strength"
SCHEMA_VERSION = "2.0.0"
INPUT_VERSION = "thermo_input_both_strands_v2"
CONTEXT_FLANK = 50
CACHE_DIR = os.path.join(HERE, ".cache")

_COMP = str.maketrans("ACGT", "TGCA")


def revcomp(s: str) -> str:
    return s.translate(_COMP)[::-1]


def read_tsv(path):
    raw = open(path, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header = lines[0].split("\t")
    return header, [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def load_genome():
    import gzip
    gz = os.path.join(CACHE_DIR, "ecoli_mg1655.fna.gz")
    with gzip.open(gz, "rt") as f:
        parts, cur = [], []
        for line in f:
            if line.startswith(">"):
                if cur:
                    parts.append("".join(cur)); cur = []
            else:
                cur.append(line.strip())
        if cur:
            parts.append("".join(cur))
    return "".join(parts).upper()


def circular_slice(genome, start, end):
    """半开区间 [start, end)，支持环形基因组。"""
    n = len(genome)
    if start >= 0 and end <= n:
        return genome[start:end]
    return "".join(genome[i % n] for i in range(start, end))


def main():
    if os.path.exists(OUT_TSV) or os.path.exists(OUT_JSON):
        raise ValueError("Output already exists; preserve frozen inputs and choose new output paths")
    hdr1, data = read_tsv(DATA_V1)
    _, cmap = read_tsv(CONTEXT_MAP)
    cmap_by_id = {r["sample_id"]: r for r in cmap}
    data_version = data[0]["data_version"]
    genome = load_genome()
    L = int(data[0]["sequence_length"])

    rows = []
    n_ready = n_blocked = 0
    for r in data:
        sid = r["sample_id"]
        s = r["sequence"].upper()
        c = cmap_by_id.get(sid)
        if c is None or c["match_status"] == "not_found":
            rows.append({
                "dataset_id": DATASET_ID, "sample_id": sid, "sequence": s,
                "sequence_length": L, "input_origin": "original50",
                "source_sequence_start_0index": 0, "context_source_ref": "",
                "tss_mode": "scan", "tss_position_1index": "",
                "tss_evidence_ref": "", "orientation_policy": "both_strands",
                "feasibility_ref": "thermo/thermo_applicability.md",
                "input_status": "blocked",
                "error_reason": "no_exact_match_in_reference_genome_50bp_too_short_for_scan",
                "input_version": INPUT_VERSION, "schema_version": SCHEMA_VERSION,
                "data_version": data_version,
            })
            n_blocked += 1
            continue

        pos = int(c["genome_start_0index"])
        strand = c["strand"]
        lo = pos - CONTEXT_FLANK
        hi = pos + L + CONTEXT_FLANK
        window = circular_slice(genome, lo, hi)          # 基因组坐标 + strand 方向
        if strand == "-":
            window = revcomp(window)
        # window 中第 CONTEXT_FLANK 位起即为原始 50 bp（正负链都应成立）
        assert window[CONTEXT_FLANK:CONTEXT_FLANK + L] == s, sid
        ref = "%s:%s:%d-%d" % (c["genome_accession"], strand, lo, hi)
        rows.append({
            "dataset_id": DATASET_ID, "sample_id": sid, "sequence": window,
            "sequence_length": len(window), "input_origin": "recovered_context",
            "source_sequence_start_0index": CONTEXT_FLANK,
            "context_source_ref": ref, "tss_mode": "scan",
            "tss_position_1index": "", "tss_evidence_ref": "",
            "orientation_policy": "both_strands",
            "feasibility_ref": "thermo/thermo_applicability.md",
            "input_status": "ready", "error_reason": "",
            "input_version": INPUT_VERSION, "schema_version": SCHEMA_VERSION,
            "data_version": data_version,
        })
        n_ready += 1

    cols = ["dataset_id", "sample_id", "sequence", "sequence_length", "input_origin",
            "source_sequence_start_0index", "context_source_ref", "tss_mode",
            "tss_position_1index", "tss_evidence_ref", "orientation_policy",
            "feasibility_ref", "input_status", "error_reason", "input_version",
            "schema_version", "data_version"]
    with open(OUT_TSV, "w", encoding="utf-8", newline="") as f:
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")

    summary = {
        "input_version": INPUT_VERSION, "schema_version": SCHEMA_VERSION,
        "data_version": data_version, "n_total": len(rows),
        "n_ready": n_ready, "n_blocked": n_blocked,
        "input_length_ready": CONTEXT_FLANK * 2 + L,
        "context_flank": CONTEXT_FLANK,
        "tool": "regseq2 Promoter_Calculator (MG1655 sigma70)",
        "output": os.path.relpath(OUT_TSV, REPO).replace("\\", "/"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("[out] %s" % OUT_TSV)
    print("[stat] ready=%d blocked=%d total=%d" % (n_ready, n_blocked, len(rows)))


if __name__ == "__main__":
    main()
