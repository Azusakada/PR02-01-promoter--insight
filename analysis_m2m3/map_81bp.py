"""Transfer 81 bp promoter layout onto uniquely matched 50 bp sequences.

Positive 81 bp sequences use the dataset window [-60, +20], with TSS at
1-based position 61. Direction follows that layout: -35 is upstream of -10.
Unmatched, ambiguous, and negative-only sequences stay missing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

from common import ROOT, SAMPLES, SPLITS, load_data, relative, require, single, write_json
from region_annotation import COLUMNS, REGION_TYPES, _spearman, summarize_regions, write_annotations

ECOLI81 = ROOT / "data/02_reg_and_gen_six_species/Escherichia coli/Dataset.csv"
ANNOTATION_VERSION = "region_map_81bp_20261004_v1"
ANNOTATION_ID = "ecoli81_layout_v1"
EVIDENCE_REF = "data/02_reg_and_gen_six_species/Escherichia coli/Dataset.csv#positive_layout_tss_1based_61"
# Canonical offsets on the 81 bp promoter, 0-based, end exclusive.
# TSS is 1-based position 61. -35/-10 are the hexamers -35..-30 and -12..-7.
LAYOUT = {
    "UP_element": (0, 21),
    "minus35": (25, 31),
    "spacer": (31, 48),
    "minus10": (48, 54),
    "TSS": (60, 61),
}
CONSENSUS = {"minus35": "TTGACA", "minus10": "TATAAT"}


def reverse_complement(sequence):
    return sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def index_windows(frame):
    """Map each 50 bp window to the loci where it occurs.

    A locus is the start of that window on the original 81 bp sequence.
    Reverse means the stored 50 bp is the reverse complement of that window.
    """
    index = defaultdict(set)
    for record in frame.itertuples(index=False):
        sequence = str(record.seq).upper()
        length = len(sequence)
        for start in range(0, length - 49):
            index[sequence[start:start + 50]].add((start, "forward", length))
        reversed_sequence = reverse_complement(sequence)
        for start in range(0, length - 49):
            original = length - 50 - start
            index[reversed_sequence[start:start + 50]].add((original, "reverse", length))
    return index


def classify_matches(samples, positive, negative):
    """Return one audit row per sample. Only a single positive locus is usable."""
    rows = []
    usable = {}
    for record in samples.sort_values("source_row").itertuples(index=False):
        sequence = record.sequence.upper()
        positive_loci = positive.get(sequence, set())
        negative_loci = negative.get(sequence, set())
        positive_keys = {(item[0], item[1]) for item in positive_loci}
        negative_keys = {(item[0], item[1]) for item in negative_loci}
        offset = strand = parent_length = None
        if positive_keys and negative_keys:
            status = "both"
        elif len(positive_keys) == 1:
            status = "unique_positive"
            offset, strand = next(iter(positive_keys))
            parent_length = min(item[2] for item in positive_loci if (item[0], item[1]) == (offset, strand))
            usable[record.sample_id] = (offset, strand, parent_length)
        elif len(positive_keys) > 1:
            status = "ambiguous_positive"
        elif negative_keys:
            status = "negative_only"
        else:
            status = "no_hit"
        rows.append({
            "sample_id": record.sample_id,
            "match_status": status,
            "offset_0index": "" if offset is None else int(offset),
            "strand": "" if strand is None else strand,
            "parent_length": "" if parent_length is None else int(parent_length),
            "n_positive_loci": len(positive_keys),
            "n_negative_loci": len(negative_keys),
        })
    return pd.DataFrame(rows), usable


def map_interval(region_start, region_end, offset, strand, parent_length, window=50):
    """Return local bounds when the whole region lies inside the stored 50 bp."""
    if region_end > parent_length:
        return None
    if strand == "forward":
        start = region_start - offset
        end = region_end - offset
    elif strand == "reverse":
        start = offset + window - region_end
        end = offset + window - region_start
    else:
        raise ValueError(f"Unknown strand {strand}")
    if 0 <= start < end <= window:
        return int(start), int(end)
    return None


def build_mapped_annotations(samples, usable):
    dataset_id = single(samples, "dataset_id")
    data_version = single(samples, "data_version")
    schema_version = single(samples, "schema_version")
    rows = []
    for record in samples.sort_values("source_row").itertuples(index=False):
        locus = usable.get(record.sample_id)
        for region in REGION_TYPES:
            row = {
                "dataset_id": dataset_id,
                "sample_id": record.sample_id,
                "region_type": region,
                "annotation_id": "missing" if locus is None else ANNOTATION_ID,
                "start_0index": pd.NA,
                "end_0index_exclusive": pd.NA,
                "strand": "unknown",
                "annotation_status": "missing",
                "evidence_ref": EVIDENCE_REF,
                "score": pd.NA,
                "annotation_version": ANNOTATION_VERSION,
                "schema_version": schema_version,
                "data_version": data_version,
            }
            if locus is not None:
                offset, strand, parent_length = locus
                bounds = map_interval(*LAYOUT[region], offset, strand, parent_length)
                row["strand"] = strand
                if bounds is None:
                    row["annotation_status"] = "not_applicable"
                else:
                    row["annotation_status"] = "reliable_external"
                    row["start_0index"] = bounds[0]
                    row["end_0index_exclusive"] = bounds[1]
            rows.append(row)
    frame = pd.DataFrame(rows, columns=COLUMNS)
    require(len(frame) == len(samples) * len(REGION_TYPES), "Annotation row count mismatch")
    return frame


def _promoter_chunk(sequence, start, end, strand):
    chunk = sequence[int(start):int(end)]
    return reverse_complement(chunk) if strand == "reverse" else chunk


def motif_relationships(samples, annotations):
    """Identity to the canonical hexamer versus log10 strength, matched subset only."""
    sample_cols = samples[["sample_id", "sequence", "target_log10"]].drop_duplicates("sample_id")
    results = []
    for region, consensus in CONSENSUS.items():
        part = annotations[
            annotations.region_type.eq(region) & annotations.annotation_status.eq("reliable_external")
        ]
        merged = part.merge(sample_cols, on="sample_id", how="left", validate="one_to_one")
        identities = []
        for record in merged.itertuples(index=False):
            chunk = _promoter_chunk(record.sequence, record.start_0index, record.end_0index_exclusive, record.strand)
            require(len(chunk) == len(consensus), f"{region} length is not the canonical hexamer")
            identities.append(sum(base == expected for base, expected in zip(chunk, consensus)) / len(consensus))
        coefficient = _spearman(identities, merged.target_log10.to_numpy()) if len(merged) else None
        row = {
            "region_type": region,
            "consensus": consensus,
            "n": int(len(merged)),
            "relationship_status": "computed" if coefficient is not None else "not_computed",
        }
        if coefficient is not None:
            row["spearman_consensus_identity_vs_log10"] = coefficient
        results.append(row)
    return results


def write_relationship_report(path, audit, summary, motifs):
    counts = audit.match_status.value_counts().to_dict()
    lines = [
        "# 81 bp 启动子布局转移到 50 bp 强度序列",
        "",
        "日期：2026-10-04。注释版本 `region_map_81bp_20261004_v1`。",
        "",
        "81 bp 正样本使用数据集窗口 [-60, +20]，TSS 在 1-based 第 61 位。",
        "−35 取 −35 到 −30，−10 取 −12 到 −7。方向按这个布局确定：−35 在 −10 上游。",
        "只有和 label=1 的 81 bp 序列精确、唯一重合的 50 bp 才转移坐标。",
        "重合位置不唯一、只落在负样本、或正负样本都命中的序列保持 missing，不猜测坐标。",
        "区域必须完整落在当前 50 bp 内；探出窗口的区域记 not_applicable，坐标留空。",
        "这是 81 bp 数据集构建坐标的转移，不是每条 50 bp 单独重测的实验盒区。",
        "",
        f"主表分母 {len(audit)}。",
        f"唯一正样本匹配 {counts.get('unique_positive', 0)}。",
        f"正样本多重位置 {counts.get('ambiguous_positive', 0)}。",
        f"同时命中正负样本 {counts.get('both', 0)}。",
        f"只命中负样本 {counts.get('negative_only', 0)}。",
        f"没有命中 {counts.get('no_hit', 0)}。",
        "",
        "覆盖率分母仍是全部主表样本。Spearman 只在 reliable_external 且坐标完整的子集上计算。",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['region_type']}`：可靠坐标 {row['n_reliable_external_with_coordinates']} / "
            f"{row['n_samples_denominator']}，覆盖率 {row['coverage']:.6f}，状态 `{row['relationship_status']}`。"
        )
        if "spearman_gc_vs_log10" in row:
            lines.append(
                f"  区间 GC 与 log10(strength) 的 Spearman 为 {row['spearman_gc_vs_log10']:.6f}"
                f"（n={row['relationship_n']}）。"
            )
    lines.append("")
    for row in motifs:
        if "spearman_consensus_identity_vs_log10" in row:
            lines.append(
                f"- `{row['region_type']}` 与共识 `{row['consensus']}` 的一致比例和 log10(strength) 的 Spearman 为 "
                f"{row['spearman_consensus_identity_vs_log10']:.6f}（n={row['n']}）。"
            )
        else:
            lines.append(f"- `{row['region_type']}` 共识 `{row['consensus']}` 未计算 Spearman（n={row['n']}）。")
    lines.extend([
        "",
        "n 小于 30 的 Spearman 只保留数值，不解释为稳定关系。UP 元件几乎都落在 50 bp 窗口外面。",
        "−10、−35 与强度的相关都很弱，不能据此声称盒区序列决定了 strength。",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def export(output, samples, ecoli_path=ECOLI81):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), f"Output already nonempty: {output}")
    ecoli = pd.read_csv(ecoli_path)
    require({"seq_id", "seq", "label"} <= set(ecoli.columns), "81 bp table is missing required columns")
    positive = index_windows(ecoli[ecoli.label.eq(1)])
    negative = index_windows(ecoli[ecoli.label.eq(0)])
    audit, usable = classify_matches(samples, positive, negative)
    annotations = build_mapped_annotations(samples, usable)
    summary = summarize_regions(samples, annotations)
    motifs = motif_relationships(samples, annotations)
    output.mkdir(parents=True, exist_ok=True)
    write_annotations(output / "annotations.tsv", annotations)
    audit.to_csv(output / "match_audit.tsv", sep="\t", index=False, lineterminator="\n")
    matched = samples[samples.sample_id.isin(usable)].copy()
    split_counts = matched["split"].value_counts().to_dict() if "split" in matched else {}
    write_json(output / "region_coverage.json", {
        "annotation_version": ANNOTATION_VERSION,
        "evidence_ref": EVIDENCE_REF,
        "layout_81bp_0based_end_exclusive": {key: list(value) for key, value in LAYOUT.items()},
        "n_samples": int(samples.sample_id.nunique()),
        "match_status_counts": audit.match_status.value_counts().to_dict(),
        "unique_positive_by_split": {str(key): int(value) for key, value in split_counts.items()},
        "regions": summary,
        "motif_relationships": motifs,
        "limitation": "Coordinates are transferred from the 81 bp layout only for a unique positive exact match.",
    })
    write_relationship_report(output / "region_relationship.md", audit, summary, motifs)
    print(relative(output))
    return audit, annotations, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--ecoli81", type=Path, default=ECOLI81)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, load_data(args.samples, args.splits), args.ecoli81)
