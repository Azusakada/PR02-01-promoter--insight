from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml

from .common import (ROOT, ModelArtifact, artifact, config, fresh_dir, get_device, relative,
                     resolve, run_manifest, seed_everything, v, write_json, write_table)
from .data import PromoterDataset, load_bundle
from .model import PromoterCNN
from .predict import predict_sequences
from .train import fit_cnn, model_smoke


def prepared_inputs(cfg, folder):
    _, groups, transform = load_bundle(cfg)
    fresh_dir(folder)
    for subset in ["train", "val", "test"]:
        write_table(folder / f"{subset}.tsv", groups[subset], delimiter="\t")
    write_json(folder / "label_transform.json", transform)
    return groups


def train_command(cfg_path):
    cfg = config(cfg_path)
    v.require(cfg["prediction"]["subset"] == "val", "中期train命令只自动导出val预测")
    output = resolve(cfg["output_dir"])
    v.require(not output.exists() or not any(output.iterdir()), "训练输出目录非空；请使用新的run_id/output_dir")
    inputs = ROOT / "work" / "cnn_inputs" / cfg["run_id"]
    prepared_inputs(cfg, inputs)
    model = fit_cnn(inputs / "train.tsv", inputs / "val.tsv", inputs / "label_transform.json", cfg_path, output)
    predict_sequences(inputs / "val.tsv", model, cfg_path, output / "validation")
    print(f"Training and validation completed: {output}")
    return model


def smoke_command(cfg_path, output):
    cfg = config(cfg_path)
    cfg["evidence_level"] = "smoke"
    cfg["run_id"] = output.name
    seed_everything(cfg)
    rows, groups, transform = load_bundle(cfg)
    device = get_device(cfg["training"]["device"])
    fresh_dir(output)
    dataset = PromoterDataset(groups["train"][:4], transform)
    report = model_smoke(PromoterCNN(cfg["model"]), dataset, device, output)
    run_manifest(cfg, output, rows[0]["data_version"],
                 [artifact("smoke_report", output / "save_load_smoke_test.json"), artifact("smoke_checkpoint", output / "smoke_state_dict.pt")],
                 fit=["train"], selection=[], evaluation=[])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def main(argv=None):
    p = argparse.ArgumentParser(description="PR02-01 50bp CNN training, smoke testing and checkpoint inference")
    sub = p.add_subparsers(dest="command", required=True)
    for name in ["train", "smoke"]:
        cmd = sub.add_parser(name)
        cmd.add_argument("--config", type=Path, default=ROOT / "CNN/configs/cnn_run_config.yaml")
        if name == "smoke":
            cmd.add_argument("--output", type=Path, default=ROOT / "CNN/runs/cnn_smoke_20261002_v1")
    cmd = sub.add_parser("predict")
    cmd.add_argument("--model-dir", type=Path, required=True)
    cmd.add_argument("--input", type=Path, required=True)
    cmd.add_argument("--config", type=Path, required=True)
    cmd.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    try:
        if args.command == "train":
            train_command(resolve(args.config))
        elif args.command == "smoke":
            smoke_command(resolve(args.config), resolve(args.output))
        else:
            directory = resolve(args.model_dir)
            model = ModelArtifact(directory / "best_checkpoint.pt", directory / "model_manifest.json", directory / "cnn_run_config.yaml")
            result = predict_sequences(resolve(args.input), model, resolve(args.config), resolve(args.output))
            print(result.predictions)
    except (v.ContractError, ValueError, OSError, RuntimeError) as e:
        p.exit(2, f"CNN运行失败: {e}\n")
    return 0
