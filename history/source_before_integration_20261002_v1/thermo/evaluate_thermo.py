# -*- coding: utf-8 -*-
"""
evaluate_thermo.py  —  PR02-01 热力学基线 / 评测与标尺处理

regseq2 输出的是热力学转录速率 Tx_rate，与主表 strength 不是同一标尺，因此：

- 未经校准：只做排序比较（Spearman），target_scale=tool_rank；
- 若要数值比较：只在 **train** 上拟合 log10(strength) ~ a*log10(Tx_rate)+b，
  再用于 val/test；原始与校准结果分开保存（契约 8.3 / interface-semantics §5）。

契约注意：predictions 表要求同一文件内 subset 恒定，故校准后预测同样按
train/val/test 分文件输出，且 target_scale_model 保持 tool_raw、填 calibration_id。

输出
----
thermo/results/thermo_metrics.csv              统一指标表（契约 metrics）
thermo/results/thermo_coverage.csv             覆盖率/失败统计
thermo/results/thermo_calibration.json         校准参数（train-only）
thermo/results/predictions/thermo_{sub}_calibrated.csv  契约 predictions（已校准）

用法
----
python thermo/evaluate_thermo.py
"""
from __future__ import annotations

import csv
import json
import math
import os

from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RESULTS = os.path.join(HERE, "results")
PRED_DIR = os.path.join(RESULTS, "predictions")
DATA_V1 = os.path.join(REPO, "data", "01_Ecoli_strength", "data_v1.tsv")
SPLITS = os.path.join(REPO, "data", "01_Ecoli_strength", "split_manifest.tsv")

RUN_ID = "thermo_regseq2_ecoli50_v1"
METHOD = "thermo_regseq2"
SCHEMA = "2.0.0"
DATASET = "course_ecoli50_strength"
SUBSETS = ["train", "val", "test"]


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_tsv(path):
    raw = open(path, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else raw.decode("utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    h = lines[0].split("\t")
    return [dict(zip(h, ln.split("\t"))) for ln in lines[1:]]


def r2(y, p):
    n = len(y)
    if n < 2:
        return None
    my = sum(y) / n
    ss_res = sum((a - b) ** 2 for a, b in zip(y, p))
    ss_tot = sum((a - my) ** 2 for a in y)
    return None if ss_tot == 0 else 1 - ss_res / ss_tot


def spearman(y, p):
    if len(y) < 3 or len(set(y)) < 2 or len(set(p)) < 2:
        return None
    return float(stats.spearmanr(y, p).statistic)


def main():
    preds = read_csv(os.path.join(RESULTS, "calculator_predictions_all.csv"))
    data = {r["sample_id"]: r for r in read_tsv(DATA_V1)}
    split_by_id = {r["sample_id"]: r["split"] for r in read_tsv(SPLITS)}
    split_id = read_tsv(SPLITS)[0]["split_id"]
    data_version = preds[0]["data_version"]

    ok = []
    for r in preds:
        sid = r["sample_id"]
        if r["prediction_status"] != "ok" or r["predicted_tx_rate"] == "":
            continue
        s = float(data[sid]["strength"])
        ok.append({"sample_id": sid, "tx": float(r["predicted_tx_rate"]),
                   "strength": s, "log10_strength": math.log10(s) if s > 0 else None,
                   "split": split_by_id.get(sid, "")})

    # ---- train-only 校准：log10(strength) = a*log10(Tx) + b ----
    tr = [x for x in ok if x["split"] == "train" and x["tx"] > 0]
    xs = [math.log10(x["tx"]) for x in tr]
    ys = [x["log10_strength"] for x in tr]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    sxy = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    a = sxy / sxx
    b = my - a * mx
    calib = {"calibration_id": "thermo_regseq2_train_lin_log_v1",
             "kind": "log10_linear", "fit_subset": "train", "n_fit": len(tr),
             "coef_a": a, "intercept_b": b, "split_id": split_id, "run_id": RUN_ID,
             "schema_version": SCHEMA, "data_version": data_version}
    with open(os.path.join(RESULTS, "thermo_calibration.json"), "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)

    for x in ok:
        x["pred_log10"] = a * math.log10(x["tx"]) + b if x["tx"] > 0 else None

    # ---- 指标 ----
    metric_rows = []

    def emit(subset_name, ids, metric_name, target_scale, value, status, reason):
        metric_rows.append([DATASET, RUN_ID, METHOD, split_id, subset_name,
                            metric_name, target_scale,
                            "" if value is None else "%.6f" % value,
                            len(preds), len(ok), len(ids), status, reason,
                            "preliminary", subset_name + "_thermo_v1", SCHEMA,
                            data_version])

    for sub in SUBSETS + ["all"]:
        ids = ok if sub == "all" else [x for x in ok if x["split"] == sub]
        rho = spearman([x["strength"] for x in ids], [x["tx"] for x in ids])
        emit(sub + "_tool_rank", ids, "spearman", "tool_rank", rho,
             "ok" if rho is not None else "undefined",
             "" if rho is not None else "insufficient_or_constant")
        ids_c = [x for x in ids if x["pred_log10"] is not None]
        yl = [x["log10_strength"] for x in ids_c]
        pl = [x["pred_log10"] for x in ids_c]
        r2v = r2(yl, pl)
        emit(sub + "_calibrated", ids_c, "r2", "log10", r2v,
             "ok" if r2v is not None else "undefined",
             "" if r2v is not None else "constant_target")
        mae = (sum(abs(u - v) for u, v in zip(yl, pl)) / len(yl)) if yl else None
        emit(sub + "_calibrated", ids_c, "mae", "log10", mae,
             "ok" if mae is not None else "undefined", "")

    with open(os.path.join(RESULTS, "thermo_metrics.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset_id", "run_id", "method_name", "split_id",
                    "evaluation_subset", "metric_name", "target_scale", "value",
                    "n_requested", "n_success", "n_used", "metric_status",
                    "reason", "evidence_level", "comparison_set_id",
                    "schema_version", "data_version"])
        w.writerows(metric_rows)

    # ---- 覆盖率 ----
    with open(os.path.join(RESULTS, "thermo_coverage.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "n_requested", "n_success", "n_failed", "coverage",
                    "n_train", "n_val", "n_test"])
        w.writerow([RUN_ID, len(preds), len(ok), len(preds) - len(ok),
                    "%.6f" % (len(ok) / len(preds)),
                    sum(1 for x in ok if x["split"] == "train"),
                    sum(1 for x in ok if x["split"] == "val"),
                    sum(1 for x in ok if x["split"] == "test")])

    # ---- 校准后逐样本预测（契约 predictions，按子集分文件）----
    cols = ["dataset_id", "sample_id", "true_value", "predicted_value",
            "predicted_value_log10", "method_name", "target_scale_model",
            "model_checkpoint", "predicted_tx_rate", "thermo_mode",
            "calibration_id", "run_id", "split_id", "subset", "seed",
            "prediction_status", "error_reason", "predicted_value_normalized",
            "label_transform_id", "schema_version", "data_version"]
    for sub in SUBSETS:
        rows = [x for x in ok if x["split"] == sub and x["pred_log10"] is not None]
        path = os.path.join(PRED_DIR, "thermo_%s_calibrated.csv" % sub)
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f); w.writerow(cols)
            for x in rows:
                lg = x["pred_log10"]
                w.writerow([DATASET, x["sample_id"], "%.10g" % x["strength"],
                            "%.10g" % (10 ** lg), "%.10g" % lg, METHOD, "tool_raw",
                            "", "%.10g" % x["tx"], "scan", calib["calibration_id"],
                            RUN_ID, split_id, sub, "", "ok", "", "", "", SCHEMA,
                            data_version])

    print("[calib] %s: a=%.4f b=%.4f n_fit=%d" % (calib["calibration_id"], a, b, len(tr)))
    for row in metric_rows:
        print("  %-18s %-8s %-9s n=%-6d value=%s" % (row[4], row[5], row[6], row[10], row[7]))


if __name__ == "__main__":
    main()
