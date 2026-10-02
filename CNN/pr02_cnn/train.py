from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .common import (ModelArtifact, artifact, config, fresh_dir, git_info, relative, resolve,
                     run_manifest, seed_everything, get_device, v, write_json, write_table)
from .data import PromoterDataset, load_bundle, make_loader, subset_identity, verify_subset
from .model import PromoterCNN


def model_smoke(model, dataset, device, output: Path):
    """Backward and save/load use a cloned model, without touching real initial weights."""
    candidate = copy.deepcopy(model).to(device)
    x = dataset.x[:min(4, len(dataset))].to(device)
    y = dataset.y[:len(x)].to(device)
    candidate.train()
    optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001)
    before = [p.detach().clone() for p in candidate.parameters()]
    features = candidate.features(x)
    pred = candidate(x)
    loss = nn.MSELoss()(pred, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in candidate.parameters())
    optimizer.step()
    assert any(not torch.equal(a, b.detach()) for a, b in zip(before, candidate.parameters()))
    candidate.eval()
    with torch.no_grad():
        first, repeated = candidate(x), candidate(x)
    path = output / "smoke_state_dict.pt"
    torch.save(candidate.state_dict(), path)
    restored = copy.deepcopy(model).to(device)
    restored.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    restored.eval()
    with torch.no_grad():
        loaded = restored(x)
    assert torch.equal(first, repeated) and torch.equal(first, loaded)
    report = {"status": "passed", "evidence_level": "smoke", "performance_claim": False,
              "input_shape": list(x.shape), "feature_shape": list(features.shape), "output_shape": list(pred.shape),
              "channels": list("ACGT"), "forward": True, "finite_loss": bool(torch.isfinite(loss)),
              "backward_finite": True, "optimizer_updates_parameters": True,
              "save_load_equal": True, "repeat_prediction_equal": True,
              "max_abs_reload_difference": float((first-loaded).abs().max().item()), "device": str(device)}
    write_json(output / "save_load_smoke_test.json", report)
    return report


def epoch(model, loader, device, optimizer=None, gradient_clip=5):
    model.train(optimizer is not None)
    total, count = 0.0, 0
    context = torch.enable_grad() if optimizer else torch.no_grad()
    with context:
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            if optimizer:
                optimizer.zero_grad(set_to_none=True)
            predicted = model(x)
            loss = nn.functional.mse_loss(predicted, y)
            if not torch.isfinite(loss):
                raise RuntimeError("loss非有限，训练中止；不保存为成功运行")
            if optimizer:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
                optimizer.step()
            total += loss.item() * len(y)
            count += len(y)
    return total / count


def plot_training(history, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.6), layout="constrained")
    for key, color, name in [("train_mse_normalized", "#466d9e", "Train"), ("val_mse_normalized", "#bc6b36", "Validation")]:
        ax.plot([r["epoch"] for r in history], [r[key] for r in history], color=color, label=name, linewidth=1.8)
    ax.set(xlabel="Epoch", ylabel="MSE on train-fitted normalized log10 scale", title="E. coli 50 bp CNN training and validation")
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    fig.savefig(output / "training_curve.png", dpi=180)
    plt.close(fig)


def fit_cnn(train_table, val_table, transform_path, config_path, output_dir) -> ModelArtifact:
    """Concrete API v2.0 adapter. No test rows enter loaders or selection."""
    cfg = config(Path(config_path))
    rows, groups, transform = load_bundle(cfg, transform_path)
    train_rows = verify_subset(resolve(train_table), groups["train"], "train")
    val_rows = verify_subset(resolve(val_table), groups["val"], "val")
    # git_info verifies a real base commit; uncommitted source is snapshotted separately.
    git = git_info()
    seed_everything(cfg)
    device = get_device(cfg["training"]["device"])
    output = resolve(output_dir)
    fresh_dir(output)
    write_json(output / "label_transform.json", transform)
    write_table(output / "train_ids.tsv", [{"sample_id": r["sample_id"]} for r in train_rows], delimiter="\t")
    write_table(output / "val_ids.tsv", [{"sample_id": r["sample_id"]} for r in val_rows], delimiter="\t")
    actual_cfg = output / "cnn_run_config.yaml"
    actual_cfg.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    train_ds, val_ds = PromoterDataset(train_rows, transform), PromoterDataset(val_rows, transform)
    model = PromoterCNN(cfg["model"]).to(device)
    smoke = model_smoke(model, train_ds, device, output)
    # Smoke consumes RNG state (dropout); reset before the actual training loop.
    seed_everything(cfg)
    train_loader, val_loader = make_loader(train_ds, cfg, True), make_loader(val_ds, cfg)
    t = cfg["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    best, progress_best, best_epoch, wait = math.inf, math.inf, 0, 0
    history = []
    identity = {"schema_version": "2.0.0", "model_id": cfg["run_id"], "method_name": "cnn_1d",
                "sequence_length": 50, "channels": list("ACGT"), "model_config": cfg["model"],
                "data_version": rows[0]["data_version"], "split_id": cfg["data"]["split_id"],
                "samples_sha256": v.sha256(resolve(cfg["data"]["samples"])),
                "splits_sha256": v.sha256(resolve(cfg["data"]["splits"])),
                "transform": transform, "target_scale_model": "log10_minmax",
                "train": subset_identity(train_rows), "val": subset_identity(val_rows),
                "seed": t["seed"], "evidence_level": cfg["evidence_level"],
                "smoke_test": cfg["evidence_level"] == "smoke", "code_commit": git["code_commit"]}
    checkpoint = output / "best_checkpoint.pt"
    with (output / "training_log.jsonl").open("w", encoding="utf-8") as log:
        for number in range(1, t["max_epochs"] + 1):
            start = time.perf_counter()
            train_loss = epoch(model, train_loader, device, optimizer, t["gradient_clip"])
            val_loss = epoch(model, val_loader, device)
            if val_loss < best:
                best, best_epoch = val_loss, number
                torch.save({"state_dict": {k: value.detach().cpu() for k, value in model.state_dict().items()},
                            "metadata": identity, "best_epoch": number, "best_val_mse_normalized": best}, checkpoint)
            if val_loss < progress_best - t["min_delta"]:
                progress_best, wait = val_loss, 0
            else:
                wait += 1
            row = {"epoch": number, "train_mse_normalized": train_loss, "val_mse_normalized": val_loss,
                   "best_epoch": best_epoch, "seconds": time.perf_counter() - start}
            history.append(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(f"epoch={number:03d} train_mse={train_loss:.6f} val_mse={val_loss:.6f} best={best_epoch}", flush=True)
            if wait >= t["patience"]:
                break
    write_table(output / "training_history.csv", history)
    plot_training(history, output)
    identity.update({"checkpoint": relative(checkpoint), "checkpoint_sha256": v.sha256(checkpoint),
                     "label_transform": relative(output / "label_transform.json"),
                     "label_transform_sha256": v.sha256(output / "label_transform.json"),
                     "config": relative(actual_cfg), "config_sha256": v.sha256(actual_cfg),
                     "selection_rule": "minimum validation MSE; best checkpoint; no train+val refit",
                     "best_epoch": best_epoch, "best_val_mse_normalized": best, "epochs_completed": len(history),
                     "input_dtype": "float32", "input_shape": [None, 4, 50],
                     "position_mapping": list(range(50)), "coordinate_version": "ecoli50_local_v1",
                     "fit_subsets": ["train"], "selection_subsets": ["val"], "test_used": False,
                     "encoding": "A/C/G/T one-hot; same-padding stride-1; local provided orientation",
                     "smoke_validation": smoke})
    manifest = output / "model_manifest.json"
    write_json(manifest, identity)
    from .predict import load_model
    restored, _ = load_model(ModelArtifact(checkpoint, manifest, actual_cfg), str(device))
    restored.eval()
    # Compare a fresh reload of best to an independent second fresh reload.
    second, _ = load_model(ModelArtifact(checkpoint, manifest, actual_cfg), str(device))
    with torch.no_grad():
        x = val_ds.x[:4].to(device)
        assert torch.equal(restored(x), second(x))
    artifact_rows = [artifact("model_manifest", manifest), artifact("checkpoint", checkpoint),
                     artifact("config", actual_cfg), artifact("label_transform", output / "label_transform.json"),
                     artifact("training_history", output / "training_history.csv"), artifact("training_log", output / "training_log.jsonl"),
                     artifact("figure", output / "training_curve.png"), artifact("smoke_report", output / "save_load_smoke_test.json"),
                     artifact("train_ids", output / "train_ids.tsv"), artifact("val_ids", output / "val_ids.tsv"),
                     artifact("dataset", resolve(cfg["data"]["samples"])), artifact("splits", resolve(cfg["data"]["splits"]))]
    run_manifest(cfg, output, rows[0]["data_version"], artifact_rows, fit=["train"], selection=["val"], evaluation=["val"])
    return ModelArtifact(checkpoint, manifest, actual_cfg)
