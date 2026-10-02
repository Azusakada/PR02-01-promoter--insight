from __future__ import annotations

import numpy as np

from .common import ids_hash, write_table


def ranks(x):
    order = np.argsort(x, kind="stable")
    result = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        result[order[i:j]] = (i + j - 1) / 2 + 1
        i = j
    return result


def compute_metrics(true, predicted):
    y, p = np.asarray(true, dtype=float), np.asarray(predicted, dtype=float)
    if not len(y):
        return {name: (None, "没有有效样本") for name in ["r2", "mae", "spearman"]}
    error = y - p
    result = {"mae": (float(np.mean(np.abs(error))), None)}
    total = float(np.sum((y - np.mean(y)) ** 2))
    result["r2"] = (None, "少于2条样本或真实标签恒定") if len(y) < 2 or total == 0 else (float(1 - np.sum(error ** 2) / total), None)
    ry, rp = ranks(y), ranks(p)
    if len(y) < 2 or np.ptp(ry) == 0 or np.ptp(rp) == 0:
        result["spearman"] = (None, "少于2条样本或真实/预测排序恒定")
    else:
        result["spearman"] = (float(np.corrcoef(ry, rp)[0, 1]), None)
    return result


def export_metrics(predictions, cfg, output):
    used = [r for r in predictions if r["prediction_status"] == "ok" and r["true_value"] is not None]
    n_success = sum(r["prediction_status"] == "ok" for r in predictions)
    subset = cfg["prediction"]["subset"]
    comparison = "cnn_" + subset + "_" + ids_hash([r["sample_id"] for r in used])[:16]
    rows = []
    for scale in ["log10", "raw"]:
        true = [np.log10(r["true_value"]) if scale == "log10" else r["true_value"] for r in used]
        pred = [r["predicted_value_log10"] if scale == "log10" else r["predicted_value"] for r in used]
        for metric, (value, reason) in compute_metrics(true, pred).items():
            rows.append({"dataset_id": predictions[0]["dataset_id"], "run_id": cfg["run_id"], "method_name": "cnn_1d",
                         "split_id": cfg["data"]["split_id"], "evaluation_subset": subset + "_all",
                         "metric_name": metric, "target_scale": scale, "value": value,
                         "n_requested": len(predictions), "n_success": n_success, "n_used": len(used),
                         "metric_status": "ok" if reason is None else "undefined", "reason": reason,
                         "evidence_level": cfg["evidence_level"], "comparison_set_id": comparison,
                         "schema_version": "2.0.0", "data_version": predictions[0]["data_version"]})
    write_table(output / "metrics_cnn.csv", rows)
    write_table(output / "common_eval_ids_cnn.tsv", [{"sample_id": r["sample_id"]} for r in used], ["sample_id"], delimiter="\t")
    write_table(output / "coverage_cnn.csv", [{"run_id": cfg["run_id"], "subset": subset, "n_requested": len(predictions),
                "n_success": n_success, "n_failed": len(predictions)-n_success, "n_used": len(used), "coverage": n_success/len(predictions)}])
    return rows
