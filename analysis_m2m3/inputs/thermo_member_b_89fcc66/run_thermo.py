# -*- coding: utf-8 -*-
"""
run_thermo.py  —  PR02-01 热力学基线 / 批量运行 regseq2

读取契约表 calculator_inputs.tsv，对每个 ready 输入调用开源 regseq2
Promoter_Calculator（宿主 MG1655，sigma70），取 dG_total 最小的启动子状态作为该样本
预测，输出原始 Tx_rate、覆盖情况、失败清单。

契约约束（重要）
----------------
团队接口契约要求 `predictions` 表**每个文件的 subset 必须一致**，且
`run_id/method_name/split_id/subset/target_scale_model/thermo_mode/calibration_id/
label_transform_id/seed/model_checkpoint` 在文件内恒定。因此逐样本预测按
train/val/test **分文件**输出；另外给一份合并文件供人工查看（不是契约表）。

输出
----
thermo/results/predictions/thermo_{train,val,test}.csv   契约 predictions（未校准）
thermo/results/calculator_predictions_all.csv            合并件（非契约表，便于查看）
thermo/results/calculator_failures.csv                   失败/阻塞清单
thermo/results/calculator_raw_outputs/thermo_raw_outputs.jsonl  原始结果（dG 分解）
thermo/results/thermo_run_manifest.json                  运行元数据

用法
----
python thermo/run_thermo.py                 # 全量运行（约 16 分钟）
python thermo/run_thermo.py --limit 200     # 调试
python thermo/run_thermo.py --rebuild-from-raw  # 用已有 jsonl 重建预测表（不重跑工具）
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
from regseq2.promoter_calculator import Promoter_Calculator  # noqa: E402

INPUTS = os.path.join(HERE, "input", "calculator_inputs.tsv")
DATA_V1 = os.path.join(REPO, "data", "01_Ecoli_strength", "data_v1.tsv")
SPLITS = os.path.join(REPO, "data", "01_Ecoli_strength", "split_manifest.tsv")
RESULTS = os.path.join(HERE, "results")
PRED_DIR = os.path.join(RESULTS, "predictions")
RAW_DIR = os.path.join(RESULTS, "calculator_raw_outputs")
RAW_JSONL = os.path.join(RAW_DIR, "thermo_raw_outputs.jsonl")

RUN_ID = "thermo_regseq2_ecoli50_v1"
METHOD_NAME = "thermo_regseq2"
SCHEMA_VERSION = "2.0.0"
DATASET_ID = "course_ecoli50_strength"
SUBSETS = ["train", "val", "test"]

PRED_COLS = ["dataset_id", "sample_id", "true_value", "predicted_value",
             "predicted_value_log10", "method_name", "target_scale_model",
             "model_checkpoint", "predicted_tx_rate", "thermo_mode",
             "calibration_id", "run_id", "split_id", "subset", "seed",
             "prediction_status", "error_reason", "predicted_value_normalized",
             "label_transform_id", "schema_version", "data_version"]


def read_tsv(path):
    raw = open(path, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip() != ""]
    header = lines[0].split("\t")
    return [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]


def write_predictions(rows, subset, path):
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(PRED_COLS)
        for r in rows:
            w.writerow([r.get(c, "") for c in PRED_COLS])


def build_row(sid, split, true_val, status, error_reason, tx, split_id, data_version,
              thermo_mode):
    return {
        "dataset_id": DATASET_ID, "sample_id": sid, "true_value": true_val,
        "predicted_value": "", "predicted_value_log10": "", "method_name": METHOD_NAME,
        "target_scale_model": "tool_raw", "model_checkpoint": "",
        "predicted_tx_rate": ("%.10g" % tx) if (status == "ok" and tx is not None) else "",
        "thermo_mode": thermo_mode, "calibration_id": "", "run_id": RUN_ID,
        "split_id": split_id, "subset": split, "seed": "",
        "prediction_status": status, "error_reason": error_reason,
        "predicted_value_normalized": "", "label_transform_id": "",
        "schema_version": SCHEMA_VERSION, "data_version": data_version,
    }


def run_full(args):
    os.makedirs(PRED_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    inputs = read_tsv(INPUTS)
    if args.limit:
        inputs = inputs[: args.limit]
    data = {r["sample_id"]: r for r in read_tsv(DATA_V1)}
    split_rows = read_tsv(SPLITS)
    split_by_id = {r["sample_id"]: r["split"] for r in split_rows}
    split_id = split_rows[0]["split_id"]
    data_version = inputs[0]["data_version"]

    calc = Promoter_Calculator()
    calc_meta = {"organism": calc.organism, "K": calc.K, "beta": calc.BETA,
                 "sigmaLevels": calc.sigmaLevels}

    t0 = time.time()
    n_ok = n_fail = 0
    by_subset = {s: [] for s in SUBSETS}
    failures = []
    with open(RAW_JSONL, "w", encoding="utf-8") as fr:
        for i, row in enumerate(inputs, 1):
            sid = row["sample_id"]
            split = split_by_id.get(sid, "")
            true_val = data.get(sid, {}).get("strength", "")
            if row["input_status"] != "ready":
                n_fail += 1
                failures.append([DATASET_ID, sid, RUN_ID, row["tss_mode"],
                                 "failed", row["error_reason"]])
                by_subset.setdefault(split, []).append(
                    build_row(sid, split, true_val, "failed", row["error_reason"],
                              None, split_id, data_version, row["tss_mode"]))
                fr.write(json.dumps({"sample_id": sid, "status": "failed",
                                     "error_reason": row["error_reason"]},
                                    ensure_ascii=False) + "\n")
                continue
            seq = row["sequence"]
            try:
                calc.run(seq, [0, len(seq)])
                out = calc.output()
                states = {}
                states.update(out["Forward_Predictions_per_TSS"])
                states.update(out["Reverse_Predictions_per_TSS"])
                if not states:
                    raise RuntimeError("no_valid_promoter_state")
                best = min(states.values(), key=lambda x: x["dG_total"])
                n_ok += 1
                by_subset.setdefault(split, []).append(
                    build_row(sid, split, true_val, "ok", None, best["Tx_rate"],
                              split_id, data_version, row["tss_mode"]))
                fr.write(json.dumps({
                    "sample_id": sid, "status": "ok", "input_origin": row["input_origin"],
                    "input_length": int(row["sequence_length"]),
                    "context_source_ref": row["context_source_ref"],
                    "best_TSS_0index": best["TSS"],
                    "TSS_inside_original50": bool(
                        int(row["source_sequence_start_0index"]) <= best["TSS"]
                        < int(row["source_sequence_start_0index"]) + 50),
                    "Tx_rate": best["Tx_rate"], "dG_total": best["dG_total"],
                    "dG_10": best["dG_10"], "dG_35": best["dG_35"],
                    "dG_spacer": best["dG_spacer"], "dG_UP": best["dG_UP"],
                    "dG_disc": best["dG_disc"], "dG_ITR": best["dG_ITR"],
                    "hex35": best["hex35"], "hex10": best["hex10"],
                    "spacer_len": len(best["spacer"]),
                }, ensure_ascii=False) + "\n")
            except Exception as e:  # noqa: BLE001
                n_fail += 1
                reason = "%s: %s" % (type(e).__name__, e)
                failures.append([DATASET_ID, sid, RUN_ID, row["tss_mode"], "failed", reason])
                by_subset.setdefault(split, []).append(
                    build_row(sid, split, true_val, "failed", reason, None,
                              split_id, data_version, row["tss_mode"]))
                fr.write(json.dumps({"sample_id": sid, "status": "failed",
                                     "error_reason": reason}, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                print("[%d/%d] ok=%d fail=%d elapsed=%.1fs"
                      % (i, len(inputs), n_ok, n_fail, time.time() - t0), flush=True)

    write_outputs(by_subset, failures, split_id, data_version)
    manifest = {
        "run_id": RUN_ID, "method_name": METHOD_NAME, "dataset_id": DATASET_ID,
        "data_version": data_version, "split_id": split_id,
        "schema_version": SCHEMA_VERSION, "evidence_level": "preliminary",
        "execution_status": "completed",
        "n_requested": len(inputs), "n_success": n_ok, "n_failed": n_fail,
        "tool": "regseq2 Promoter_Calculator (open-source Salis thermodynamic model)",
        "tool_version": "bnjenner/PromoterCalc_Comparison ext/regseq2",
        "tool_meta": calc_meta, "command": "python thermo/run_thermo.py",
        "environment": {"python": sys.version.split()[0], "platform": platform.platform()},
        "created_at_epoch": int(time.time()),
        "elapsed_sec": round(time.time() - t0, 1),
        "prediction_scale": "tool_raw Tx_rate (not calibrated to strength)",
    }
    with open(os.path.join(RESULTS, "thermo_run_manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print("[done] ok=%d fail=%d total=%d elapsed=%.1fs"
          % (n_ok, n_fail, len(inputs), time.time() - t0))


def write_outputs(by_subset, failures, split_id, data_version):
    os.makedirs(PRED_DIR, exist_ok=True)
    for sub in SUBSETS:
        write_predictions(by_subset.get(sub, []), sub,
                          os.path.join(PRED_DIR, "thermo_%s.csv" % sub))
    # 合并件（非契约表，仅供人工查看）
    write_predictions([r for s in SUBSETS for r in by_subset.get(s, [])], "",
                      os.path.join(RESULTS, "calculator_predictions_all.csv"))
    with open(os.path.join(RESULTS, "calculator_failures.csv"), "w",
              encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "sample_id", "run_id", "thermo_mode",
                    "prediction_status", "error_reason"])
        w.writerows(failures)


def rebuild_from_raw():
    """用已有 jsonl 重建预测表，避免重跑 16 分钟的工具。"""
    data = {r["sample_id"]: r for r in read_tsv(DATA_V1)}
    split_rows = read_tsv(SPLITS)
    split_by_id = {r["sample_id"]: r["split"] for r in split_rows}
    split_id = split_rows[0]["split_id"]
    inputs = {r["sample_id"]: r for r in read_tsv(INPUTS)}
    data_version = next(iter(inputs.values()))["data_version"]
    by_subset = {s: [] for s in SUBSETS}
    failures = []
    n_ok = n_fail = 0
    for line in open(RAW_JSONL, encoding="utf-8"):
        rec = json.loads(line)
        sid = rec["sample_id"]
        split = split_by_id.get(sid, "")
        true_val = data.get(sid, {}).get("strength", "")
        mode = inputs.get(sid, {}).get("tss_mode", "scan")
        if rec["status"] == "ok":
            n_ok += 1
            by_subset.setdefault(split, []).append(
                build_row(sid, split, true_val, "ok", None, rec["Tx_rate"],
                          split_id, data_version, mode))
        else:
            n_fail += 1
            failures.append([DATASET_ID, sid, RUN_ID, mode, "failed", rec["error_reason"]])
            by_subset.setdefault(split, []).append(
                build_row(sid, split, true_val, "failed", rec["error_reason"], None,
                          split_id, data_version, mode))
    write_outputs(by_subset, failures, split_id, data_version)
    print("[rebuild] ok=%d fail=%d" % (n_ok, n_fail))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rebuild-from-raw", action="store_true")
    args = ap.parse_args()
    if args.rebuild_from_raw:
        rebuild_from_raw()
    else:
        run_full(args)


if __name__ == "__main__":
    main()
