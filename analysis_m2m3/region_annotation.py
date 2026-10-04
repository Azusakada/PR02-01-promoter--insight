"""Export missing region annotations and withhold strength correlations.

Coordinates stay empty unless a row is reliable_external with both bounds.
Tool-inferred boxes are counted and are not used as experimental evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import ROOT, SAMPLES, SPLITS, load_data, relative, require, single, write_json

CONTRACT = json.loads((ROOT / "contracts/assets/tables.json").read_text(encoding="utf-8"))
ANNOTATION_SPEC = CONTRACT["tables"]["annotations"]
COLUMNS = list(ANNOTATION_SPEC["columns"])
REGION_TYPES = tuple(ANNOTATION_SPEC["columns"]["region_type"]["enum"])
ANNOTATION_VERSION = "region_gap_20261003_v1"
EVIDENCE_REF = "analysis_m2m3/reports/annotation_source_review.md"
MIN_RELATIONSHIP_N = 3


def build_annotations(data):
    """One missing row per sample and region. Unknown bounds stay empty."""
    require(data.sample_id.is_unique, "Duplicate sample_id")
    dataset_id = single(data, "dataset_id")
    data_version = single(data, "data_version")
    schema_version = single(data, "schema_version")
    ordered = data.sort_values("source_row")
    n_regions = len(REGION_TYPES)
    frame = pd.DataFrame({
        "dataset_id": dataset_id,
        "sample_id": np.repeat(ordered.sample_id.to_numpy(), n_regions),
        "region_type": np.tile(np.array(REGION_TYPES, dtype=object), len(ordered)),
        "annotation_id": "missing",
        "start_0index": pd.NA,
        "end_0index_exclusive": pd.NA,
        "strand": "unknown",
        "annotation_status": "missing",
        "evidence_ref": EVIDENCE_REF,
        "score": pd.NA,
        "annotation_version": ANNOTATION_VERSION,
        "schema_version": schema_version,
        "data_version": data_version,
    })
    require(list(frame.columns) == COLUMNS, "Annotation columns drifted from the contract")
    require(len(frame) == len(ordered) * n_regions, "Annotation row count mismatch")
    return frame


def _region_gc(sequence, start, end):
    start, end = int(start), int(end)
    require(0 <= start < end <= len(sequence), "Region bounds fall outside the stored sequence")
    chunk = sequence[start:end]
    return (chunk.count("G") + chunk.count("C")) / len(chunk)


def _spearman(x, y):
    rx = pd.Series(x, dtype=float).rank(method="average").to_numpy()
    ry = pd.Series(y, dtype=float).rank(method="average").to_numpy()
    if len(rx) < MIN_RELATIONSHIP_N or np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize_regions(samples, annotations):
    """Coverage for every region. Correlation only on reliable external coordinates."""
    require(set(REGION_TYPES) <= set(annotations.region_type), "Annotation table is missing a region type")
    n_samples = int(samples.sample_id.nunique())
    require(n_samples > 0, "No samples")
    sample_cols = samples[["sample_id", "sequence", "target_log10"]].drop_duplicates("sample_id")
    rows = []
    for region in REGION_TYPES:
        part = annotations[annotations.region_type.eq(region)]
        require(part.sample_id.is_unique, f"Duplicate {region} annotation")
        reliable = part[part.annotation_status.eq("reliable_external")]
        for record in reliable.itertuples(index=False):
            require(pd.notna(record.start_0index) and pd.notna(record.end_0index_exclusive),
                    f"{record.sample_id}: reliable_external requires both coordinates")
            require(record.strand in {"forward", "reverse"} and bool(record.evidence_ref),
                    f"{record.sample_id}: reliable_external requires strand and evidence_ref")
        usable = reliable.merge(sample_cols, on="sample_id", how="left", validate="one_to_one")
        require(usable.sequence.notna().all(), "Reliable annotation sample_id is absent from the dataset")
        row = {
            "region_type": region,
            "n_rows": int(len(part)),
            "n_samples_denominator": n_samples,
            "n_missing": int(part.annotation_status.eq("missing").sum()),
            "n_tool_inferred_not_used": int(part.annotation_status.eq("tool_inferred").sum()),
            "n_reliable_external_with_coordinates": int(len(usable)),
            "coverage": float(len(usable) / n_samples),
        }
        coefficient = None
        if len(usable):
            gc = [_region_gc(seq, start, end) for seq, start, end in zip(
                usable.sequence, usable.start_0index, usable.end_0index_exclusive)]
            coefficient = _spearman(gc, usable.target_log10.to_numpy())
        if coefficient is None:
            row["relationship_status"] = (
                "not_computed_no_reliable_coordinates" if len(usable) == 0 else "not_computed_insufficient_variation"
            )
        else:
            row["relationship_status"] = "computed"
            row["spearman_gc_vs_log10"] = coefficient
            row["relationship_feature"] = "gc_content_of_annotated_interval"
            row["relationship_n"] = int(len(usable))
        rows.append(row)
    return rows


def _cell(value):
    if value is None or value is pd.NA or (isinstance(value, float) and np.isnan(value)):
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def write_annotations(path, annotations):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for record in annotations.itertuples(index=False):
            writer.writerow({column: _cell(getattr(record, column)) for column in COLUMNS})


def write_report(path, summary):
    lines = [
        "# 区域注释与盒区–强度关系缺口",
        "",
        "日期：2026-10-03。注释版本 `region_gap_20261003_v1`。",
        f"证据指向 `{EVIDENCE_REF}`。作者 PromoA 数组和主表都没有逐条实验 TSS、方向或盒区坐标。",
        "",
        "本目录的 `annotations.tsv` 按公共 annotations 契约，为每条样本的五种区域各写一行。",
        "状态是 `missing`，链方向是 `unknown`，起点、终点和分数留空。",
        "这些空坐标不是固定窗口，工具扫描结果也没有被写成 `reliable_external`。",
        "",
        "覆盖率的分母是当前主表样本数。分子只计 `reliable_external` 且坐标完整的行。",
        "`tool_inferred` 单独计数，不进入盒区与强度的计算。",
        "",
    ]
    for row in summary:
        lines.append(
            f"- `{row['region_type']}`：行数 {row['n_rows']}，分母 {row['n_samples_denominator']}，"
            f"缺失 {row['n_missing']}，未采用的工具推断 {row['n_tool_inferred_not_used']}，"
            f"可靠且坐标完整 {row['n_reliable_external_with_coordinates']}，覆盖率 {row['coverage']:.6f}。"
            f"关系状态 `{row['relationship_status']}`。"
        )
        if "spearman_gc_vs_log10" in row:
            lines.append(
                f"  该区域在 {row['relationship_n']} 条可靠坐标上计算了区间 GC 与 log10(strength) 的 Spearman："
                f"{row['spearman_gc_vs_log10']:.6f}。"
            )
    if all("spearman_gc_vs_log10" not in row for row in summary):
        lines.extend([
            "",
            "五种区域的可靠坐标子集都为空或不足以比较，因此本次没有输出 Spearman 或 Pearson。",
            "恢复可追溯实验坐标后，重新运行本脚本才会在对应子集上计算区间 GC 与 log10(strength) 的 Spearman，并同时保留分母和覆盖率。",
        ])
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def export(output, data):
    output = Path(output)
    require(not output.exists() or not any(output.iterdir()), f"Output already nonempty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    annotations = build_annotations(data)
    summary = summarize_regions(data, annotations)
    write_annotations(output / "annotations.tsv", annotations)
    write_json(output / "region_coverage.json", {
        "annotation_version": ANNOTATION_VERSION,
        "evidence_ref": EVIDENCE_REF,
        "n_samples": int(data.sample_id.nunique()),
        "n_annotation_rows": int(len(annotations)),
        "regions": summary,
        "limitation": "No Spearman is stored unless relationship_status is computed.",
    })
    write_report(output / "region_relationship_gap.md", summary)
    print(relative(output))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, load_data(args.samples, args.splits))
