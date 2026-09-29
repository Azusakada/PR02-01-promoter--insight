from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "01_Ecoli_strength"
FEATURE_PATH = Path(__file__).resolve().parent / "input" / "promoter_physicochemical.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "results"

DATA_PATH = DATA_DIR / "data_v1.tsv"
SPLIT_PATH = DATA_DIR / "split_manifest.tsv"
RUN_ID = "knn_ecoli_full_20260929_v1"
DATASET_ID = "course_ecoli50_strength"
SPLIT_ID = "ecoli50_random_20260928_v1"
SEED = 20260928
FEATURE_VERSION = "physicochemical8_v1"
METHOD_NAME = "knn_physchem_full"
PRIMARY_METRIC = "r2_log10"
PATIENCE = 3
K_GRID = [
    1,
    3,
    5,
    7,
    9,
    11,
    15,
    21,
    31,
    41,
    51,
    71,
    101,
    151,
    201,
    251,
    301,
    351,
    401,
    501,
    601,
    801,
    1001,
    1201,
    1501,
    2001,
    2501,
    3001,
    4001,
    5001,
]
FEATURE_COLUMNS = [
    "gc_content",
    "at_content",
    "gc_skew",
    "at_skew",
    "melting_temp",
    "bendability",
    "stacking_energy",
    "entropy",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def metric_row(y_true: np.ndarray, y_pred_log10: np.ndarray, subset: str, k: int, fit_scope: str) -> dict:
    true_raw = np.power(10.0, y_true)
    pred_raw = np.power(10.0, y_pred_log10)
    spearman = float(spearmanr(y_true, y_pred_log10).statistic)
    n = len(y_true)
    return {
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "split_id": SPLIT_ID,
        "evaluation_subset": subset,
        "extra_fit_scope": fit_scope,
        "extra_k": k,
        "metric_name": "r2",
        "target_scale": "log10",
        "value": float(r2_score(y_true, y_pred_log10)),
        "n_requested": n,
        "n_success": n,
        "n_used": n,
        "metric_status": "ok",
        "reason": "",
        "evidence_level": "final",
        "schema_version": "2.0.0",
        "data_version": "ecoli50_strength_v1",
    }, {
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "split_id": SPLIT_ID,
        "evaluation_subset": subset,
        "extra_fit_scope": fit_scope,
        "extra_k": k,
        "metric_name": "mae",
        "target_scale": "log10",
        "value": float(mean_absolute_error(y_true, y_pred_log10)),
        "n_requested": n,
        "n_success": n,
        "n_used": n,
        "metric_status": "ok",
        "reason": "",
        "evidence_level": "final",
        "schema_version": "2.0.0",
        "data_version": "ecoli50_strength_v1",
    }, {
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "split_id": SPLIT_ID,
        "evaluation_subset": subset,
        "extra_fit_scope": fit_scope,
        "extra_k": k,
        "metric_name": "spearman",
        "target_scale": "log10",
        "value": spearman,
        "n_requested": n,
        "n_success": n,
        "n_used": n,
        "metric_status": "ok",
        "reason": "",
        "evidence_level": "final",
        "schema_version": "2.0.0",
        "data_version": "ecoli50_strength_v1",
    }, {
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "split_id": SPLIT_ID,
        "evaluation_subset": subset,
        "extra_fit_scope": fit_scope,
        "extra_k": k,
        "metric_name": "r2",
        "target_scale": "raw",
        "value": float(r2_score(true_raw, pred_raw)),
        "n_requested": n,
        "n_success": n,
        "n_used": n,
        "metric_status": "ok",
        "reason": "",
        "evidence_level": "final",
        "schema_version": "2.0.0",
        "data_version": "ecoli50_strength_v1",
    }, {
        "dataset_id": DATASET_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "split_id": SPLIT_ID,
        "evaluation_subset": subset,
        "extra_fit_scope": fit_scope,
        "extra_k": k,
        "metric_name": "mae",
        "target_scale": "raw",
        "value": float(mean_absolute_error(true_raw, pred_raw)),
        "n_requested": n,
        "n_success": n,
        "n_used": n,
        "metric_status": "ok",
        "reason": "",
        "evidence_level": "final",
        "schema_version": "2.0.0",
        "data_version": "ecoli50_strength_v1",
    }


def prediction_frame(
    frame: pd.DataFrame,
    pred_log10: np.ndarray,
    subset: str,
    k: int,
    fit_scope: str,
) -> pd.DataFrame:
    pred_raw = np.power(10.0, pred_log10)
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "sample_id": frame["sample_id"].to_numpy(),
            "true_value": frame["strength"].to_numpy(dtype=float),
            "predicted_value": pred_raw,
            "extra_true_value_log10": frame["target_log10"].to_numpy(dtype=float),
            "predicted_value_log10": pred_log10,
            "method_name": METHOD_NAME,
            "target_scale_model": "log10",
            "model_checkpoint": "knn_ecoli_full_model.joblib",
            "predicted_tx_rate": "",
            "thermo_mode": "",
            "calibration_id": "",
            "extra_k": k,
            "extra_fit_scope": fit_scope,
            "run_id": RUN_ID,
            "split_id": SPLIT_ID,
            "subset": subset,
            "seed": SEED,
            "prediction_status": "ok",
            "error_reason": "",
            "predicted_value_normalized": "",
            "label_transform_id": "",
            "schema_version": "2.0.0",
            "data_version": "ecoli50_strength_v1",
        }
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    old_combined_predictions = OUTPUT_DIR / "predictions_knn_ecoli_full.csv"
    if old_combined_predictions.exists():
        old_combined_predictions.unlink()
    samples = pd.read_csv(DATA_PATH, sep="\t")
    splits = pd.read_csv(SPLIT_PATH, sep="\t")
    features = pd.read_csv(FEATURE_PATH)
    if list(features.columns) != FEATURE_COLUMNS or len(features) != len(samples):
        raise ValueError("physicochemical feature table does not align with data_v1")
    if features.isna().any().any() or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("physicochemical feature table contains invalid values")
    if not samples["sample_id"].is_unique or not splits["sample_id"].is_unique:
        raise ValueError("sample IDs must be unique")
    if set(splits["split"]) != {"train", "val", "test"}:
        raise ValueError("expected train/val/test split")
    if set(samples["sample_id"]) != set(splits["sample_id"]):
        raise ValueError("sample/split IDs do not match")
    split_by_id = splits.set_index("sample_id")["split"]
    samples = samples.copy()
    samples["split"] = samples["sample_id"].map(split_by_id)
    feature_by_id = features.astype(float).copy()
    feature_by_id.insert(0, "sample_id", samples["sample_id"].to_numpy())
    feature_by_id = feature_by_id.set_index("sample_id")

    train = samples.loc[samples["split"].eq("train")].copy()
    val = samples.loc[samples["split"].eq("val")].copy()
    test = samples.loc[samples["split"].eq("test")].copy()
    if (len(train), len(val), len(test)) != (8318, 1783, 1783):
        raise ValueError("unexpected split sizes")
    x_train_raw = feature_by_id.loc[train["sample_id"], FEATURE_COLUMNS].to_numpy()
    x_val_raw = feature_by_id.loc[val["sample_id"], FEATURE_COLUMNS].to_numpy()
    x_test_raw = feature_by_id.loc[test["sample_id"], FEATURE_COLUMNS].to_numpy()
    y_train = train["target_log10"].to_numpy(dtype=float)
    y_val = val["target_log10"].to_numpy(dtype=float)
    y_test = test["target_log10"].to_numpy(dtype=float)

    scaler = StandardScaler().fit(x_train_raw)
    x_train = scaler.transform(x_train_raw)
    x_val = scaler.transform(x_val_raw)
    x_test = scaler.transform(x_test_raw)

    search_rows: list[dict] = []
    best_score = -np.inf
    best_k: int | None = None
    no_improvement = 0
    stop_reason = "grid_exhausted"
    for round_index, k in enumerate(K_GRID, start=1):
        model = KNeighborsRegressor(
            n_neighbors=k,
            weights="distance",
            metric="euclidean",
            n_jobs=-1,
        )
        model.fit(x_train, y_train)
        pred_val = model.predict(x_val)
        score = float(r2_score(y_val, pred_val))
        mae = float(mean_absolute_error(y_val, pred_val))
        rho = float(spearmanr(y_val, pred_val).statistic)
        improved = score > best_score
        if improved:
            best_score = score
            best_k = k
            no_improvement = 0
        else:
            no_improvement += 1
        search_rows.append(
            {
                "round": round_index,
                "k": k,
                "weights": "distance",
                "metric": "euclidean",
                "val_r2_log10": score,
                "val_mae_log10": mae,
                "val_spearman_log10": rho,
                "improved": improved,
                "best_k_after_round": best_k,
                "best_val_r2_log10_after_round": best_score,
                "no_improvement_rounds": no_improvement,
                "patience": PATIENCE,
            }
        )
        if no_improvement >= PATIENCE:
            stop_reason = "early_stopping_patience_3"
            break
    if best_k is None:
        raise RuntimeError("hyperparameter search did not select a k")

    selected_model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    selected_model.fit(x_train, y_train)
    val_pred_log10 = selected_model.predict(x_val)

    combined_x = np.vstack([x_train_raw, x_val_raw])
    combined_y = np.concatenate([y_train, y_val])
    final_scaler = StandardScaler().fit(combined_x)
    final_model = KNeighborsRegressor(
        n_neighbors=best_k,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    final_model.fit(final_scaler.transform(combined_x), combined_y)
    test_pred_log10 = final_model.predict(final_scaler.transform(x_test_raw))

    metrics = pd.DataFrame(
        metric_row(y_val, val_pred_log10, "val", best_k, "train")
        + metric_row(y_test, test_pred_log10, "test", best_k, "train_plus_val")
    )
    metrics["comparison_set_id"] = "knn_ecoli_full_test"
    predictions_val = prediction_frame(val, val_pred_log10, "val", best_k, "train")
    predictions_test = prediction_frame(test, test_pred_log10, "test", best_k, "train_plus_val")
    coverage = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "run_id": RUN_ID,
                "method_name": METHOD_NAME,
                "split_id": SPLIT_ID,
                "subset": subset,
                "n_requested": len(frame),
                "n_success": len(frame),
                "n_failed": 0,
                "coverage": 1.0,
                "failure_reasons": "",
                "evidence_level": "final",
                "k": best_k,
                "schema_version": "2.0.0",
                "data_version": "ecoli50_strength_v1",
            }
            for subset, frame in (("val", val), ("test", test))
        ]
    )
    common_ids = pd.DataFrame(
        {
            "comparison_set_id": "knn_ecoli_full_test",
            "sample_id": test["sample_id"],
            "dataset_id": DATASET_ID,
            "split_id": SPLIT_ID,
            "subset": "test",
            "method_name": METHOD_NAME,
            "prediction_status": "ok",
        }
    )

    model_bundle = {
        "model": final_model,
        "scaler": final_scaler,
        "feature_columns": FEATURE_COLUMNS,
        "feature_version": FEATURE_VERSION,
        "method_name": METHOD_NAME,
        "dataset_id": DATASET_ID,
        "split_id": SPLIT_ID,
        "run_id": RUN_ID,
        "target_scale": "log10",
        "best_k": best_k,
        "fit_scope": "train_plus_val",
        "train_n": len(train),
        "val_n": len(val),
        "test_n": len(test),
        "evidence_level": "final",
    }
    joblib.dump(model_bundle, OUTPUT_DIR / "knn_ecoli_full_model.joblib")
    pd.DataFrame(search_rows).to_csv(OUTPUT_DIR / "hyperparameter_search.csv", index=False, lineterminator="\n")
    predictions_val.to_csv(OUTPUT_DIR / "predictions_knn_ecoli_full_val.csv", index=False, lineterminator="\n")
    predictions_test.to_csv(OUTPUT_DIR / "predictions_knn_ecoli_full_test.csv", index=False, lineterminator="\n")
    metrics.to_csv(OUTPUT_DIR / "metrics_knn_ecoli_full.csv", index=False, lineterminator="\n")
    coverage.to_csv(OUTPUT_DIR / "coverage_knn_ecoli_full.csv", index=False, lineterminator="\n")
    common_ids.to_csv(OUTPUT_DIR / "common_eval_ids_knn_ecoli_full.tsv", sep="\t", index=False, lineterminator="\n")

    config = {
        "schema_version": "2.0.0",
        "dataset_id": DATASET_ID,
        "data_version": "ecoli50_strength_v1",
        "split_id": SPLIT_ID,
        "run_id": RUN_ID,
        "method_name": METHOD_NAME,
        "evidence_level": "final",
        "feature_version": FEATURE_VERSION,
        "feature_columns": FEATURE_COLUMNS,
        "input_files": {
            "samples": str(DATA_PATH.relative_to(ROOT)).replace("\\", "/"),
            "splits": str(SPLIT_PATH.relative_to(ROOT)).replace("\\", "/"),
            "physicochemical": str(FEATURE_PATH.relative_to(ROOT)).replace("\\", "/"),
        },
        "input_sha256": {
            "samples": sha256_file(DATA_PATH),
            "splits": sha256_file(SPLIT_PATH),
            "physicochemical": sha256_file(FEATURE_PATH),
        },
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "target": "target_log10 = log10(strength)",
        "feature_scaling": "StandardScaler fitted on train for validation selection; refitted on train+val for final test model",
        "model": {"class": "sklearn.neighbors.KNeighborsRegressor", "weights": "distance", "metric": "euclidean", "n_jobs": -1},
        "k_grid": K_GRID,
        "patience": PATIENCE,
        "primary_metric": PRIMARY_METRIC,
        "search_rounds_run": len(search_rows),
        "search_stop_reason": stop_reason,
        "best_k": best_k,
        "best_val_r2_log10": best_score,
        "final_fit_scope": "train_plus_val",
        "code_path": "KNN/run_ecoli_knn_full.py",
        "code_commit_at_generation": git_head(),
        "python": platform.python_version(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": "python KNN/run_ecoli_knn_full.py",
    }
    (OUTPUT_DIR / "knn_ecoli_full_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    val_metric = metrics.loc[(metrics["evaluation_subset"] == "val") & (metrics["target_scale"] == "log10")]
    test_metric = metrics.loc[(metrics["evaluation_subset"] == "test") & (metrics["target_scale"] == "log10")]
    report = f"""# E. coli 完整 KNN 实验报告

## 实验设计

- 样本：11,884 条 E. coli 50 bp 启动子序列。
- 特征：8 类理化特征；StandardScaler 只在训练数据上拟合。
- 目标：`target_log10 = log10(strength)`。
- 固定划分：train={len(train):,}，val={len(val):,}，test={len(test):,}。
- 模型：距离加权欧氏距离 KNN 回归。
- 超参数：按 `k={K_GRID}` 顺序搜索，监控 val 的 log10 R²。
- 早停：连续 {PATIENCE} 轮无提升后停止；本次执行 {len(search_rows)} 轮，停止原因是 `{stop_reason}`。
- 最优 `k`：{best_k}。

## 结果

| 子集 | 拟合数据 | log10 R² | log10 MAE | log10 Spearman | raw R² | raw MAE | coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| val | train | {float(val_metric.loc[val_metric.metric_name.eq('r2'),'value'].iloc[0]):.6f} | {float(val_metric.loc[val_metric.metric_name.eq('mae'),'value'].iloc[0]):.6f} | {float(val_metric.loc[val_metric.metric_name.eq('spearman'),'value'].iloc[0]):.6f} | {float(metrics.loc[(metrics.evaluation_subset=='val')&(metrics.target_scale=='raw')&(metrics.metric_name=='r2'),'value'].iloc[0]):.6f} | {float(metrics.loc[(metrics.evaluation_subset=='val')&(metrics.target_scale=='raw')&(metrics.metric_name=='mae'),'value'].iloc[0]):.6f} | 1.0 |
| test | train + val | {float(test_metric.loc[test_metric.metric_name.eq('r2'),'value'].iloc[0]):.6f} | {float(test_metric.loc[test_metric.metric_name.eq('mae'),'value'].iloc[0]):.6f} | {float(test_metric.loc[test_metric.metric_name.eq('spearman'),'value'].iloc[0]):.6f} | {float(metrics.loc[(metrics.evaluation_subset=='test')&(metrics.target_scale=='raw')&(metrics.metric_name=='r2'),'value'].iloc[0]):.6f} | {float(metrics.loc[(metrics.evaluation_subset=='test')&(metrics.target_scale=='raw')&(metrics.metric_name=='mae'),'value'].iloc[0]):.6f} | 1.0 |

## 交付产物

- `knn_ecoli_full_config.json`：完整实验配置、输入哈希、搜索和早停记录。
- `hyperparameter_search.csv`：每轮 `k` 的验证集结果。
- `knn_ecoli_full_model.joblib`：使用 train+val 重拟合的最终模型和标准化器。
- `predictions_knn_ecoli_full_val.csv`、`predictions_knn_ecoli_full_test.csv`：完整 val/test 逐样本预测。
- `metrics_knn_ecoli_full.csv`：log10/raw 尺度的 R²、MAE、Spearman。
- `coverage_knn_ecoli_full.csv`：val/test 请求数、成功数和覆盖率。
- `common_eval_ids_knn_ecoli_full.tsv`：完整 test 评估样本 ID。

本实验是完整 E. coli KNN 回归基线；其它分类任务不并入本回归实验。
"""
    (OUTPUT_DIR / "knn_ecoli_full_report.md").write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "train": len(train),
                "val": len(val),
                "test": len(test),
                "search_rounds": len(search_rows),
                "stop_reason": stop_reason,
                "best_k": best_k,
                "test_r2_log10": float(test_metric.loc[test_metric.metric_name.eq("r2"), "value"].iloc[0]),
                "test_spearman_log10": float(test_metric.loc[test_metric.metric_name.eq("spearman"), "value"].iloc[0]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
