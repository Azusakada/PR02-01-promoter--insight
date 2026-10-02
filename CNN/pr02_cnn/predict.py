from __future__ import annotations

import csv
import copy
import math
from pathlib import Path

import torch

from .common import (ModelArtifact, PredictionBundle, artifact, config, fresh_dir, get_device, label,
                     relative, resolve, run_manifest, seed_everything, v, write_json, write_table)
from .data import load_bundle, one_hot
from .metrics import export_metrics
from .model import PromoterCNN


def load_model(model_artifact: ModelArtifact, device="cpu"):
    manifest = v.read_json(model_artifact.manifest)
    v.require(v.sha256(model_artifact.checkpoint) == manifest["checkpoint_sha256"], "checkpoint SHA256不符")
    v.require(v.sha256(model_artifact.config) == manifest["config_sha256"], "模型配置SHA256不符")
    path = resolve(manifest["label_transform"])
    v.require(v.sha256(path) == manifest["label_transform_sha256"], "标签变换SHA256不符")
    transform = v.read_json(path)
    checkpoint = torch.load(model_artifact.checkpoint, map_location="cpu", weights_only=True)
    meta = checkpoint["metadata"]
    for key in ["model_id", "method_name", "model_config", "sequence_length", "channels", "target_scale_model",
                "evidence_level", "data_version", "split_id", "samples_sha256", "splits_sha256", "train", "val", "seed"]:
        v.require(meta[key] == manifest[key], f"checkpoint与manifest的{key}不符")
    v.require(meta["transform"] == transform == manifest["transform"], "模型绑定的transform不一致")
    v.require(manifest["sequence_length"] == 50 and manifest["channels"] == list("ACGT"), "拒绝150bp或不同通道模型")
    model = PromoterCNN(meta["model_config"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(get_device(device)).eval()
    return model, manifest


def prediction_row(row, normalized, metadata, cfg, checkpoint_path, error=None):
    result = {"dataset_id": "course_ecoli50_strength", "sample_id": row["sample_id"],
              "true_value": row.get("strength"), "predicted_value": None, "predicted_value_log10": None,
              "method_name": "cnn_1d", "target_scale_model": "log10_minmax", "model_checkpoint": relative(checkpoint_path),
              "predicted_tx_rate": None, "thermo_mode": None, "calibration_id": None,
              "run_id": cfg["run_id"], "split_id": metadata["split_id"], "subset": cfg["prediction"]["subset"],
              "seed": metadata["seed"], "prediction_status": "failed", "error_reason": error,
              "predicted_value_normalized": None, "label_transform_id": metadata["transform"]["transform_id"],
              "schema_version": "2.0.0", "data_version": metadata["data_version"]}
    if error is None:
        try:
            log10, raw = label.restore_prediction(normalized, metadata["transform"])
            result.update(predicted_value=raw, predicted_value_log10=log10, predicted_value_normalized=normalized,
                          prediction_status="ok", error_reason=None)
        except (ValueError, OverflowError) as e:
            result["error_reason"] = f"inverse_transform: {e}"
    return result


def predict_rows(requests, model, metadata, cfg, checkpoint_path):
    """Preserve every request and its order; isolate malformed sequences and inference errors."""
    device = next(model.parameters()).device
    model.eval()
    results, valid = [None] * len(requests), []
    for index, row in enumerate(requests):
        try:
            valid.append((index, row, one_hot(row["sequence"])))
        except (ValueError, KeyError) as e:
            results[index] = prediction_row(row, None, metadata, cfg, checkpoint_path, f"encoding: {e}")
    with torch.no_grad():
        for start in range(0, len(valid), cfg["training"]["batch_size"]):
            batch = valid[start:start+cfg["training"]["batch_size"]]
            try:
                values = model(torch.stack([x for _, _, x in batch]).to(device)).cpu().tolist()
            except (RuntimeError, ValueError) as e:
                # Try each request independently; never silently drop the batch.
                for index, row, x in batch:
                    try:
                        value = float(model(x.unsqueeze(0).to(device)).item())
                        results[index] = prediction_row(row, value, metadata, cfg, checkpoint_path)
                    except (RuntimeError, ValueError) as individual:
                        results[index] = prediction_row(row, None, metadata, cfg, checkpoint_path, f"inference: {individual}")
            else:
                for (index, row, _), value in zip(batch, values):
                    results[index] = prediction_row(row, value, metadata, cfg, checkpoint_path)
    return results


def predict_sequences(input_table, model_artifact, run_config_path, output_dir) -> PredictionBundle:
    cfg = config(Path(run_config_path))
    seed_everything(cfg)
    model, metadata = load_model(model_artifact, cfg["training"]["device"])
    v.require(cfg["model"] == metadata["model_config"], "预测配置中的模型结构与保存模型不一致")
    v.require(cfg["training"]["seed"] == metadata["seed"], "预测配置seed与保存模型不一致")
    rows, groups, transform = load_bundle(cfg, resolve(metadata["label_transform"]))
    v.require(cfg["evidence_level"] == metadata["evidence_level"], "预测不能提升模型证据等级")
    v.require(metadata["samples_sha256"] == v.sha256(resolve(cfg["data"]["samples"])), "主表与模型绑定版本不一致")
    v.require(metadata["splits_sha256"] == v.sha256(resolve(cfg["data"]["splits"])), "划分文件与模型绑定版本不一致")
    subset = cfg["prediction"]["subset"]
    v.require(subset in {"train", "val", "test"}, "当前预测API支持固定数据的train/val/test")
    v.require(subset != "test" or cfg["prediction"].get("allow_test") is True, "test需在冻结方案后显式开启allow_test")
    expected = {r["sample_id"]: r for r in groups[subset]}
    delimiter = "\t" if Path(input_table).suffix.lower() == ".tsv" else ","
    with resolve(input_table).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        v.require(reader.fieldnames and {"sample_id", "sequence"} <= set(reader.fieldnames), "预测输入须含sample_id和sequence")
        requests = list(reader)
    ids = [r["sample_id"] for r in requests]
    v.require(len(ids) == len(set(ids)) and set(ids) == set(expected), "预测输入必须完整覆盖所声明subset，不能重复或包含未知ID")
    for request in requests:
        original = expected[request["sample_id"]]
        for field in ["dataset_id", "data_version", "schema_version"]:
            if field in request:
                v.require(request[field] == original[field], f"{request['sample_id']}: {field}不一致")
        # A malformed sequence gets a failed row; a valid but changed sequence is a version conflict.
        try:
            one_hot(request["sequence"])
        except ValueError:
            pass
        else:
            v.require(request["sequence"] == original["sequence"], f"{request['sample_id']}: 序列与冻结主表不符")
        if request.get("strength"):
            v.require(float(request["strength"]) == original["strength"], "输入真值与主表不一致")
        request["strength"] = original["strength"]
    output = resolve(output_dir)
    fresh_dir(output)
    predictions = predict_rows(requests, model, metadata, cfg, model_artifact.checkpoint)
    path = output / "predictions_cnn.csv"
    write_table(path, predictions)
    metrics = export_metrics(predictions, cfg, output)
    qc = output / "prediction_qc.json"
    write_json(qc, {"n_requested": len(requests), "n_success": sum(r["prediction_status"] == "ok" for r in predictions),
                   "failed_ids": [r["sample_id"] for r in predictions if r["prediction_status"] == "failed"],
                   "request_order_preserved": ids == [r["sample_id"] for r in predictions], "subset": subset,
                   "evidence_level": cfg["evidence_level"], "checkpoint_sha256": metadata["checkpoint_sha256"]})
    v.bundle_check(resolve(cfg["data"]["samples"]), resolve(cfg["data"]["splits"]),
                   predictions=path, transform=resolve(metadata["label_transform"]))
    v.load_table("metrics", output / "metrics_cnn.csv")
    artifacts = [artifact("predictions", path), artifact("metrics", output / "metrics_cnn.csv"),
                 artifact("prediction_qc", qc), artifact("coverage", output / "coverage_cnn.csv"),
                 artifact("used_ids", output / "common_eval_ids_cnn.tsv"), artifact("model_manifest", model_artifact.manifest)]
    failed = any(r["prediction_status"] == "failed" for r in predictions)
    manifest = run_manifest(cfg, output, metadata["data_version"], artifacts, fit=[], selection=[], evaluation=[subset],
                            status="partial" if failed else "success")
    return PredictionBundle(path, qc, manifest)
