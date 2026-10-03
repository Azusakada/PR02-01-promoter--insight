# -*- coding: utf-8 -*-
"""
李玘航：k-mer + Ridge 主基线，可选 SVR。

规则：
- 只用冻结 split，不再自行切分后再横向比较
- train 拟合，val 选择超参，记录全部候选
- 测试集用 train 上选定的模型，不把 val 并回训练
- 标签：target_log10 = log10(strength)
"""
from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, SVR

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "data_snapshot"
FEAT_DIR = ROOT / "features"
OUT = ROOT / "results"

DATASET_ID = "course_ecoli50_strength"
DATA_VERSION = "ecoli50_strength_v1"
SPLIT_ID = "ecoli50_random_20260928_v1"
TRANSFORM_ID = "log10mm_59711a1c2217859e"
SEED = 20260928
RIDGE_RUN_ID = "ridge_kmer_20261003_v1"
SVR_RUN_ID = "svr_kmer_20261003_v1"
ALPHAS = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
LINEAR_SVR_CS = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
MIN_LOG10 = 0.9380190974762103
MAX_LOG10 = 5.914235601697235


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    samples = pd.read_csv(SNAPSHOT / "data_v1.tsv", sep="\t")
    splits = pd.read_csv(SNAPSHOT / "split_manifest.tsv", sep="\t")
    samples = samples.merge(splits[["sample_id", "split"]], on="sample_id", how="inner")
    if len(samples) != 11884:
        raise RuntimeError("merged table size changed")
    train = samples.loc[samples["split"].eq("train")].copy()
    val = samples.loc[samples["split"].eq("val")].copy()
    test = samples.loc[samples["split"].eq("test")].copy()
    return samples, train, val, test


def load_kmer(k: int) -> tuple[np.ndarray, np.ndarray]:
    blob = np.load(FEAT_DIR / f"kmer{k}.npz", allow_pickle=True)
    return blob["X"].astype(np.float64), blob["sample_ids"].astype(str)


def load_physchem() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(FEAT_DIR / "physchem8.tsv", sep="\t")
    cols = [c for c in df.columns if c != "sample_id"]
    return df[cols].to_numpy(dtype=np.float64), df["sample_id"].astype(str).to_numpy(), cols


def take(x_all: np.ndarray, ids_all: np.ndarray, wanted: pd.Series) -> np.ndarray:
    pos = {sid: i for i, sid in enumerate(ids_all)}
    return np.stack([x_all[pos[sid]] for sid in wanted.astype(str)], axis=0)


def metrics_block(y_true_log: np.ndarray, y_pred_log: np.ndarray, subset: str, method: str, run_id: str, extra: dict) -> list[dict]:
    y_true_raw = np.power(10.0, y_true_log)
    y_pred_raw = np.power(10.0, y_pred_log)
    rows = []
    pairs = [
        ("r2", "log10", float(r2_score(y_true_log, y_pred_log))),
        ("mae", "log10", float(mean_absolute_error(y_true_log, y_pred_log))),
        ("spearman", "log10", float(spearmanr(y_true_log, y_pred_log).statistic)),
        ("r2", "raw", float(r2_score(y_true_raw, y_pred_raw))),
        ("mae", "raw", float(mean_absolute_error(y_true_raw, y_pred_raw))),
    ]
    n = len(y_true_log)
    for name, scale, value in pairs:
        rows.append(
            {
                "dataset_id": DATASET_ID,
                "run_id": run_id,
                "method_name": method,
                "split_id": SPLIT_ID,
                "evaluation_subset": subset,
                "extra_fit_scope": extra.get("fit_scope", "train"),
                "extra_k": extra.get("k", ""),
                "extra_alpha": extra.get("alpha", ""),
                "extra_C": extra.get("C", ""),
                "metric_name": name,
                "target_scale": scale,
                "value": value,
                "n_requested": n,
                "n_success": n,
                "n_used": n,
                "metric_status": "ok",
                "reason": "",
                "evidence_level": "preliminary",
                "schema_version": "2.0.0",
                "data_version": DATA_VERSION,
            }
        )
    return rows


def prediction_frame(
    frame: pd.DataFrame,
    pred_log: np.ndarray,
    method: str,
    run_id: str,
    checkpoint: str,
    subset: str,
    extra: dict,
) -> pd.DataFrame:
    pred_log = np.asarray(pred_log, dtype=float)
    pred_raw = np.power(10.0, pred_log)
    pred_norm = (pred_log - MIN_LOG10) / (MAX_LOG10 - MIN_LOG10)
    return pd.DataFrame(
        {
            "dataset_id": DATASET_ID,
            "sample_id": frame["sample_id"].to_numpy(),
            "true_value": frame["strength"].to_numpy(dtype=float),
            "predicted_value": pred_raw,
            "predicted_value_log10": pred_log,
            "method_name": method,
            "target_scale_model": "log10",
            "model_checkpoint": checkpoint,
            "predicted_tx_rate": "",
            "thermo_mode": "",
            "calibration_id": "",
            "run_id": run_id,
            "split_id": SPLIT_ID,
            "subset": subset,
            "seed": SEED,
            "prediction_status": "ok",
            "error_reason": "",
            "predicted_value_normalized": pred_norm,
            "label_transform_id": TRANSFORM_ID,
            "schema_version": "2.0.0",
            "data_version": DATA_VERSION,
            "extra_true_value_log10": frame["target_log10"].to_numpy(dtype=float),
            "extra_k": extra.get("k", ""),
            "extra_alpha": extra.get("alpha", ""),
            "extra_C": extra.get("C", ""),
            "extra_fit_scope": extra.get("fit_scope", "train"),
        }
    )


def fit_ridge(x_tr, y_tr, x_va, alpha: float) -> tuple[Ridge, StandardScaler, np.ndarray]:
    scaler = StandardScaler().fit(x_tr)
    model = Ridge(alpha=alpha, random_state=SEED)
    model.fit(scaler.transform(x_tr), y_tr)
    pred = model.predict(scaler.transform(x_va))
    return model, scaler, pred


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples, train, val, test = load_frames()
    y_tr = train["target_log10"].to_numpy(dtype=float)
    y_va = val["target_log10"].to_numpy(dtype=float)
    y_te = test["target_log10"].to_numpy(dtype=float)
    phys_all, phys_ids, phys_cols = load_physchem()

    search_rows = []
    best = None
    feature_cache = {}
    for k in (3, 4, 5):
        x_all, ids_all = load_kmer(k)
        feature_cache[f"kmer{k}"] = (x_all, ids_all)
        x_tr = take(x_all, ids_all, train["sample_id"])
        x_va = take(x_all, ids_all, val["sample_id"])
        for alpha in ALPHAS:
            _, _, pred_va = fit_ridge(x_tr, y_tr, x_va, alpha)
            score = float(r2_score(y_va, pred_va))
            row = {
                "feature_set": f"kmer{k}",
                "k": k,
                "alpha": alpha,
                "n_features": int(x_tr.shape[1]),
                "val_r2_log10": score,
                "val_mae_log10": float(mean_absolute_error(y_va, pred_va)),
                "val_spearman_log10": float(spearmanr(y_va, pred_va).statistic),
            }
            search_rows.append(row)
            if best is None or score > best["val_r2_log10"]:
                best = dict(row)

    # 补充：最佳 k-mer 拼接 8 类理化特征，只作为对照，不自动覆盖主线
    k_main = int(best["k"])
    x_all, ids_all = feature_cache[f"kmer{k_main}"]
    x_tr = np.hstack([take(x_all, ids_all, train["sample_id"]), take(phys_all, phys_ids, train["sample_id"])])
    x_va = np.hstack([take(x_all, ids_all, val["sample_id"]), take(phys_all, phys_ids, val["sample_id"])])
    for alpha in ALPHAS:
        _, _, pred_va = fit_ridge(x_tr, y_tr, x_va, alpha)
        search_rows.append(
            {
                "feature_set": f"kmer{k_main}_physchem8",
                "k": k_main,
                "alpha": alpha,
                "n_features": int(x_tr.shape[1]),
                "val_r2_log10": float(r2_score(y_va, pred_va)),
                "val_mae_log10": float(mean_absolute_error(y_va, pred_va)),
                "val_spearman_log10": float(spearmanr(y_va, pred_va).statistic),
            }
        )

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(OUT / "ridge_search.csv", index=False, lineterminator="\n")

    # 主线：k-mer only 中 val R² 最高者
    kmer_only = search_df[search_df["feature_set"].str.fullmatch(r"kmer[345]")]
    winner = kmer_only.sort_values("val_r2_log10", ascending=False).iloc[0]
    k = int(winner["k"])
    alpha = float(winner["alpha"])
    method = f"kmer{k}_ridge"
    x_all, ids_all = feature_cache[f"kmer{k}"]
    x_tr = take(x_all, ids_all, train["sample_id"])
    x_va = take(x_all, ids_all, val["sample_id"])
    x_te = take(x_all, ids_all, test["sample_id"])
    model, scaler, pred_va = fit_ridge(x_tr, y_tr, x_va, alpha)
    pred_te = model.predict(scaler.transform(x_te))
    extra = {"k": k, "alpha": alpha, "fit_scope": "train"}
    ckpt = "ridge_model.joblib"
    joblib.dump(
        {
            "model": model,
            "scaler": scaler,
            "k": k,
            "alpha": alpha,
            "feature_set": f"kmer{k}",
            "method_name": method,
            "target_scale": "log10",
            "split_id": SPLIT_ID,
            "run_id": RIDGE_RUN_ID,
        },
        OUT / ckpt,
    )
    pred_ridge = pd.concat(
        [
            prediction_frame(val, pred_va, method, RIDGE_RUN_ID, ckpt, "val", extra),
            prediction_frame(test, pred_te, method, RIDGE_RUN_ID, ckpt, "test", extra),
        ],
        ignore_index=True,
    )
    pred_ridge.to_csv(OUT / "predictions_ridge.csv", index=False, lineterminator="\n")
    metrics = pd.DataFrame(
        metrics_block(y_va, pred_va, "val", method, RIDGE_RUN_ID, extra)
        + metrics_block(y_te, pred_te, "test", method, RIDGE_RUN_ID, extra)
    )
    metrics.to_csv(OUT / "ridge_metrics.csv", index=False, lineterminator="\n")

    # SVR：同一 split、同一最佳 k-mer、同一 log10 标签
    svr_search = []
    best_svr = None
    x_tr_s = scaler.transform(x_tr)
    x_va_s = scaler.transform(x_va)
    x_te_s = scaler.transform(x_te)
    for C in LINEAR_SVR_CS:
        svr = LinearSVR(C=C, random_state=SEED, max_iter=200000, dual=True, tol=1e-4)
        svr.fit(x_tr_s, y_tr)
        pred = svr.predict(x_va_s)
        score = float(r2_score(y_va, pred))
        row = {
            "model": "LinearSVR",
            "C": C,
            "kernel": "linear",
            "val_r2_log10": score,
            "val_mae_log10": float(mean_absolute_error(y_va, pred)),
            "val_spearman_log10": float(spearmanr(y_va, pred).statistic),
            "n_iter": int(getattr(svr, "n_iter_", -1)),
        }
        svr_search.append(row)
        if best_svr is None or score > best_svr["val_r2_log10"]:
            best_svr = dict(row)
            best_svr_model = svr
    # 再补一个 RBF SVR 默认 C=1，证明第二种核可跑；不扩大网格以免过久
    rbf = SVR(kernel="rbf", C=1.0, gamma="scale")
    rbf.fit(x_tr_s, y_tr)
    pred_rbf = rbf.predict(x_va_s)
    svr_search.append(
        {
            "model": "SVR",
            "C": 1.0,
            "kernel": "rbf",
            "val_r2_log10": float(r2_score(y_va, pred_rbf)),
            "val_mae_log10": float(mean_absolute_error(y_va, pred_rbf)),
            "val_spearman_log10": float(spearmanr(y_va, pred_rbf).statistic),
            "n_iter": -1,
        }
    )
    if svr_search[-1]["val_r2_log10"] > best_svr["val_r2_log10"]:
        best_svr = dict(svr_search[-1])
        best_svr_model = rbf
    pd.DataFrame(svr_search).to_csv(OUT / "svr_search.csv", index=False, lineterminator="\n")
    pred_va_svr = best_svr_model.predict(x_va_s)
    pred_te_svr = best_svr_model.predict(x_te_s)
    svr_method = f"kmer{k}_svr"
    svr_ckpt = "svr_model.joblib"
    extra_svr = {"k": k, "C": best_svr["C"], "fit_scope": "train", "alpha": ""}
    joblib.dump(
        {
            "model": best_svr_model,
            "scaler": scaler,
            "k": k,
            "C": best_svr["C"],
            "kernel": best_svr["kernel"],
            "method_name": svr_method,
            "target_scale": "log10",
            "split_id": SPLIT_ID,
            "run_id": SVR_RUN_ID,
        },
        OUT / svr_ckpt,
    )
    pred_svr = pd.concat(
        [
            prediction_frame(val, pred_va_svr, svr_method, SVR_RUN_ID, svr_ckpt, "val", extra_svr),
            prediction_frame(test, pred_te_svr, svr_method, SVR_RUN_ID, svr_ckpt, "test", extra_svr),
        ],
        ignore_index=True,
    )
    pred_svr.to_csv(OUT / "predictions_svr.csv", index=False, lineterminator="\n")
    svr_metrics = pd.DataFrame(
        metrics_block(y_va, pred_va_svr, "val", svr_method, SVR_RUN_ID, extra_svr)
        + metrics_block(y_te, pred_te_svr, "test", svr_method, SVR_RUN_ID, extra_svr)
    )
    svr_metrics.to_csv(OUT / "svr_metrics.csv", index=False, lineterminator="\n")

    coverage = pd.DataFrame(
        [
            {
                "dataset_id": DATASET_ID,
                "run_id": run_id,
                "method_name": mname,
                "split_id": SPLIT_ID,
                "subset": subset,
                "n_requested": n,
                "n_success": n,
                "n_failed": 0,
                "coverage": 1.0,
                "failure_reasons": "",
                "evidence_level": "preliminary",
                "schema_version": "2.0.0",
                "data_version": DATA_VERSION,
            }
            for run_id, mname in ((RIDGE_RUN_ID, method), (SVR_RUN_ID, svr_method))
            for subset, n in (("val", len(val)), ("test", len(test)))
        ]
    )
    coverage.to_csv(OUT / "coverage_summary.csv", index=False, lineterminator="\n")
    eval_ids = pd.concat(
        [
            pred_ridge[["sample_id", "dataset_id", "split_id", "subset", "method_name", "prediction_status"]].assign(
                comparison_set_id="ridge_kmer_val_test"
            ),
            pred_svr[["sample_id", "dataset_id", "split_id", "subset", "method_name", "prediction_status"]].assign(
                comparison_set_id="svr_kmer_val_test"
            ),
        ],
        ignore_index=True,
    )
    eval_ids = eval_ids[
        ["comparison_set_id", "sample_id", "dataset_id", "split_id", "subset", "method_name", "prediction_status"]
    ]
    eval_ids.to_csv(OUT / "common_eval_ids.tsv", sep="\t", index=False, lineterminator="\n")

    config = {
        "schema_version": "2.0.0",
        "dataset_id": DATASET_ID,
        "data_version": DATA_VERSION,
        "split_id": SPLIT_ID,
        "transform_id": TRANSFORM_ID,
        "owner": "李玘航",
        "ridge_run_id": RIDGE_RUN_ID,
        "svr_run_id": SVR_RUN_ID,
        "method_name": method,
        "svr_method_name": svr_method,
        "evidence_level": "preliminary",
        "target": "target_log10 = log10(strength)",
        "selection_rule": "maximize val log10 R2 among k-mer-only Ridge candidates; record all k x alpha",
        "selected": {"feature_set": f"kmer{k}", "k": k, "alpha": alpha, "val_r2_log10": float(winner["val_r2_log10"])},
        "svr_selected": best_svr,
        "alpha_grid": ALPHAS,
        "k_grid": [3, 4, 5],
        "final_fit_scope": "train_only",
        "do_not_refit_train_plus_val": True,
        "physchem_role": "supplementary control in ridge_search.csv, not the main method",
        "input_files": {
            "samples": "Ridge/data_snapshot/data_v1.tsv",
            "splits": "Ridge/data_snapshot/split_manifest.tsv",
            "label_transform": "Ridge/data_snapshot/label_transform.json",
            "features": "Ridge/features/feature_manifest.json",
        },
        "input_sha256": {
            "samples": sha256_file(SNAPSHOT / "data_v1.tsv"),
            "splits": sha256_file(SNAPSHOT / "split_manifest.tsv"),
        },
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "python": platform.python_version(),
        "generated_at_utc": datetime.now(timezone.utc).astimezone().isoformat(),
        "command": "python Ridge/src/prepare_frozen_inputs.py && python Ridge/src/build_feature_bundle.py && python Ridge/src/train_ridge_svr.py",
    }
    (OUT / "ridge_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    val_r2 = float(metrics.loc[(metrics.evaluation_subset.eq("val")) & (metrics.metric_name.eq("r2")) & (metrics.target_scale.eq("log10")), "value"].iloc[0])
    test_r2 = float(metrics.loc[(metrics.evaluation_subset.eq("test")) & (metrics.metric_name.eq("r2")) & (metrics.target_scale.eq("log10")), "value"].iloc[0])
    test_rho = float(metrics.loc[(metrics.evaluation_subset.eq("test")) & (metrics.metric_name.eq("spearman")) & (metrics.target_scale.eq("log10")), "value"].iloc[0])
    print(json.dumps({
        "ridge_method": method,
        "k": k,
        "alpha": alpha,
        "val_r2_log10": val_r2,
        "test_r2_log10": test_r2,
        "test_spearman_log10": test_rho,
        "svr": best_svr,
        "n_ridge_candidates": int(len(search_df)),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
