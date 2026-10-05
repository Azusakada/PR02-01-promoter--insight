"""Transfer a canonical layout onto position-consistent positive matches.

The layout convention uses the 81 bp window [-60, +20], with reference TSS
at 1-based position 61. These are inferred windows, not measured box boundaries.
Unmatched, ambiguous, and negative-only sequences stay missing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

import json
import shutil

from common import ROOT, SAMPLES, SPLITS, Run, load_data, relative, require, single, write_json
from region_annotation import COLUMNS, REGION_TYPES, _spearman, summarize_regions, write_annotations

ECOLI81 = ROOT / "data/02_reg_and_gen_six_species/Escherichia coli/Dataset.csv"
ANNOTATION_VERSION = "region_map_81bp_20261005_v2"
ANNOTATION_ID = "ecoli81_canonical_layout_v2"
EVIDENCE_REF = "data/02_reg_and_gen_six_species/Escherichia coli/Dataset.csv"
UPSTREAM_COMMIT = "3c779b9d07f369432604551b2b020287b224232d"
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
    require(frame.seq_id.notna().all() and frame.seq_id.is_unique, "Invalid/duplicate parent seq_id")
    for record in frame.itertuples(index=False):
        sequence = str(record.seq).upper()
        parent_id = str(record.seq_id)
        length = len(sequence)
        for start in range(0, length - 49):
            index[sequence[start:start + 50]].add((start, "forward", length, parent_id))
        reversed_sequence = reverse_complement(sequence)
        for start in range(0, length - 49):
            original = length - 50 - start
            index[reversed_sequence[start:start + 50]].add((original, "reverse", length, parent_id))
    return index


def classify_matches(samples, positive, negative):
    """Retain every parent identity; accept only identical positive layout mappings."""
    rows = []
    usable = {}
    for record in samples.sort_values("source_row").itertuples(index=False):
        sequence = record.sequence.upper()
        positive_loci = positive.get(sequence, set())
        negative_loci = negative.get(sequence, set())
        positive_keys = {(item[0], item[1], item[2]) for item in positive_loci}
        negative_keys = {(item[0], item[1], item[2]) for item in negative_loci}
        positive_parents = sorted({item[3] for item in positive_loci})
        negative_parents = sorted({item[3] for item in negative_loci})
        offset = strand = parent_length = None
        if positive_keys and negative_keys:
            status = "both"
        elif len(positive_keys) == 1:
            status = "position_consistent_positive"
            offset, strand, parent_length = next(iter(positive_keys))
            usable[record.sample_id] = (offset, strand, parent_length, positive_parents)
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
            "n_positive_loci": len(positive_loci),
            "n_negative_loci": len(negative_loci),
            "n_positive_positions": len(positive_keys),
            "n_negative_positions": len(negative_keys),
            "n_positive_parents": len(positive_parents),
            "n_negative_parents": len(negative_parents),
            "positive_parent_ids": json.dumps(positive_parents, separators=(",", ":")),
            "negative_parent_ids": json.dumps(negative_parents, separators=(",", ":")),
            "positive_parent_loci": json.dumps(sorted(positive_loci), separators=(",", ":")),
            "negative_parent_loci": json.dumps(sorted(negative_loci), separators=(",", ":")),
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
                offset, strand, parent_length, parent_ids = locus
                row["evidence_ref"] = EVIDENCE_REF + "#seq_id=" + ",".join(parent_ids) + ";layout=" + ANNOTATION_ID
                bounds = map_interval(*LAYOUT[region], offset, strand, parent_length)
                row["strand"] = strand
                if bounds is None:
                    row["annotation_status"] = "not_applicable"
                else:
                    row["annotation_status"] = "tool_inferred"
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
            annotations.region_type.eq(region) & annotations.annotation_status.isin(["reliable_external", "tool_inferred"])
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


def export(output, samples, ecoli_path=ECOLI81, samples_path=SAMPLES, splits_path=SPLITS):
    from region_presentation import presentation

    ecoli = pd.read_csv(ecoli_path)
    require({"seq_id", "seq", "label"} <= set(ecoli.columns), "81 bp table is missing required columns")
    require(ecoli.seq_id.notna().all() and ecoli.seq_id.is_unique, "Invalid/duplicate parent seq_id")
    require(ecoli.label.isin([0, 1]).all(), "Invalid parent label")
    positive_frame = ecoli[ecoli.label.eq(1)]
    require(positive_frame.seq.str.len().eq(81).all(), "Canonical layout requires 81 bp positive parents")
    positive = index_windows(positive_frame)
    negative = index_windows(ecoli[ecoli.label.eq(0)])
    audit, usable = classify_matches(samples, positive, negative)
    annotations = build_mapped_annotations(samples, usable)
    summary = summarize_regions(samples, annotations, include_inferred=True)
    motifs = motif_relationships(samples, annotations)
    parameters = {
        "upstream_commit": UPSTREAM_COMMIT,
        "annotation_version": ANNOTATION_VERSION,
        "coordinate_basis": "canonical dataset layout assumption",
        "parent_matching_policy": "all positive parent matches must have identical offset, strand, and length; negative overlap excluded",
        "reference_tss_1based": 61,
        "layout_81bp_0based_end_exclusive": {key: list(value) for key, value in LAYOUT.items()},
        "annotation_status": "tool_inferred",
        "feature_source": "computed_from_sequence",
        "inference": "descriptive relationships, no model fitting or selection",
    }
    run = Run(output, "M2", "huhaoming_region_analysis", samples,
              [samples_path, splits_path, ecoli_path], parameters)
    snapshot = run.output / "source_code"
    snapshot.mkdir()
    source_files = [Path(__file__), ROOT / "analysis_m2m3/region_annotation.py",
                    ROOT / "analysis_m2m3/region_presentation.py", ROOT / "analysis_m2m3/common.py"]
    for source in source_files:
        shutil.copyfile(source, snapshot / source.name)
    run.meta["upstream_commit"] = UPSTREAM_COMMIT
    run.meta["source_snapshot"] = [relative(snapshot / source.name) for source in source_files]
    run.meta["annotation_version"] = ANNOTATION_VERSION
    write_annotations(run.output / "annotations.tsv", annotations)
    audit.to_csv(run.output / "match_audit.tsv", sep="\t", index=False, lineterminator="\n")
    matched = samples[samples.sample_id.isin(usable)].copy()
    split_counts = matched["split"].value_counts().to_dict() if "split" in matched else {}
    eligible = audit[audit.match_status.eq("position_consistent_positive")]
    write_json(run.output / "region_coverage.json", {
        "annotation_version": ANNOTATION_VERSION,
        "evidence_ref": EVIDENCE_REF,
        "layout_81bp_0based_end_exclusive": {key: list(value) for key, value in LAYOUT.items()},
        "n_samples": int(samples.sample_id.nunique()),
        "match_status_counts": audit.match_status.value_counts().to_dict(),
        "n_position_consistent_positive": int(len(eligible)),
        "n_unique_positive_parent": int(eligible.n_positive_parents.eq(1).sum()),
        "n_multiple_positive_parents_same_position": int(eligible.n_positive_parents.gt(1).sum()),
        "position_consistent_positive_by_split": {str(key): int(value) for key, value in split_counts.items()},
        "regions": summary,
        "motif_relationships": motifs,
        "coordinate_basis": "canonical dataset layout assumption; tool_inferred",
    })
    presentation(run, samples, annotations, audit, summary, motifs)
    run.finish()
    return audit, annotations, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, default=SAMPLES)
    parser.add_argument("--splits", type=Path, default=SPLITS)
    parser.add_argument("--ecoli81", type=Path, default=ECOLI81)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export(args.output, load_data(args.samples, args.splits), args.ecoli81, args.samples, args.splits)
