"""Bounded, validation-only CNN search followed by predeclared seed repeats."""
from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import json
import statistics
from pathlib import Path

import yaml

from pr02_cnn.cli import train_command
from pr02_cnn.common import (ROOT, config, fresh_dir, relative, resolve, v,
                             write_json, write_table)
from pr02_cnn.model import PromoterCNN


def read_results(run, candidate, seed, stage):
    manifest = v.read_json(run / "model_manifest.json")
    cfg = config(run / "cnn_run_config.yaml")
    with (run / "validation/metrics_cnn.csv").open(encoding="utf-8", newline="") as f:
        metrics = list(csv.DictReader(f))
    with (run / "validation/coverage_cnn.csv").open(encoding="utf-8", newline="") as f:
        coverage = next(csv.DictReader(f))
    row = {"candidate": candidate, "stage": stage, "seed": seed,
           "run_id": cfg["run_id"], "run_dir": relative(run),
           "parameters": sum(p.numel() for p in PromoterCNN(cfg["model"]).parameters()),
           "best_epoch": manifest["best_epoch"], "epochs_completed": manifest["epochs_completed"],
           "val_mse_normalized": manifest["best_val_mse_normalized"],
           "n_requested": int(coverage["n_requested"]), "n_success": int(coverage["n_success"])}
    for m in metrics:
        v.require(m["metric_status"] == "ok", "本轮要求所有指标可计算")
        row[m["target_scale"] + "_" + m["metric_name"]] = float(m["value"])
    v.require(manifest["test_used"] is False and manifest["fit_subsets"] == ["train"]
              and manifest["selection_subsets"] == ["val"], "只允许train拟合和val选模")
    v.require(row["n_success"] == row["n_requested"] == 1783, "必须保留全部验证样本")
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-config", type=Path, default=ROOT / "CNN/configs/cnn_optimization_round1.yaml")
    args = p.parse_args()
    study_path = resolve(args.study_config)
    study = yaml.safe_load(study_path.read_text(encoding="utf-8"))
    base = config(resolve(study["base_config"]))
    output = resolve(study["output_dir"])
    v.require(study["selection_metric"] == "val_mse_normalized", "选择规则必须预先声明")
    names = [c["name"] for c in study["candidates"]]
    v.require(len(names) == len(set(names)) and all(n.replace('_','').isalnum() for n in names), "候选名称无效")
    seeds = [study["search_seed"], *study["stability_seeds"]]
    v.require(len(set(seeds)) == len(seeds) and all(type(s) is int and s >= 0 for s in seeds), "随机种子无效")
    v.require(1 <= study["stability_top_k"] <= len(names), "复跑数量无效")
    fresh_dir(output)
    plan = {"study": study, "study_config_sha256": v.sha256(study_path),
            "data_sha256": v.sha256(resolve(base["data"]["samples"])),
            "splits_sha256": v.sha256(resolve(base["data"]["splits"])),
            "transform_sha256": v.sha256(resolve(base["data"]["transform"])),
            "fit_subset": "train", "selection_subset": "val", "test_evaluated": False,
            "selection_rule": "Search seed ranks six configurations by minimum validation MSE. "
                "Repeat top two using both predeclared additional seeds; select configuration by "
                "mean validation MSE over all three seeds. Deliver its search-seed checkpoint, "
                "never the best-scoring seed. Scores are preliminary validation estimates.",
            "source_base_config_sha256": v.sha256(resolve(study["base_config"]))}
    # Freeze the whole candidate/seed schedule before looking at any new scores.
    write_json(output / "study_plan.json", plan)
    configs = {}
    for c in study["candidates"]:
        for seed in seeds:
            cfg = copy.deepcopy(base)
            cfg["model"].update(c["model"])
            cfg["training"].update(c["training"])
            cfg["training"].update(seed=seed, max_epochs=study["max_epochs"], patience=study["patience"])
            cfg["run_id"] = f"{study['study_id']}_{c['name']}_s{seed}"
            cfg["output_dir"] = relative(output / cfg["run_id"])
            cfg["prediction"].update(subset="val", allow_test=False)
            path = output / "configs" / f"{c['name']}_s{seed}.yaml"
            path.parent.mkdir(exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
            configs[c["name"], seed] = path

    trials = []
    def execute(name, seed, stage):
        cfg_path = configs[name, seed]
        cfg = config(cfg_path)
        print(f"START {stage}: {name}, seed={seed}", flush=True)
        logfile = output / f"{name}_s{seed}.log"
        with logfile.open("w", encoding="utf-8") as log, contextlib.redirect_stdout(log):
            train_command(cfg_path)
        row = read_results(resolve(cfg["output_dir"]), name, seed, stage)
        trials.append(row)
        write_table(output / "trials.csv", trials)
        print(f"DONE {name}, seed={seed}, best_epoch={row['best_epoch']}, "
              f"log10_R2={row['log10_r2']:.5f}, Spearman={row['log10_spearman']:.5f}, "
              f"MAE={row['log10_mae']:.5f}", flush=True)

    for name in names:
        execute(name, study["search_seed"], "search")
    ranked = sorted(trials, key=lambda r:(r["val_mse_normalized"], r["candidate"]))
    finalists = [r["candidate"] for r in ranked[:study["stability_top_k"]]]
    write_json(output / "search_selection.json", {"ranking":ranked, "finalists":finalists,
                                                 "selection_subset":"val", "test_evaluated":False})
    for name in finalists:
        for seed in study["stability_seeds"]:
            execute(name, seed, "stability")
    aggregate=[]
    metrics=["val_mse_normalized","log10_r2","log10_spearman","log10_mae","raw_r2"]
    for name in finalists:
        rows=[r for r in trials if r["candidate"]==name]
        item={"candidate":name,"n_seeds":len(rows),"seeds":[r["seed"] for r in rows]}
        for key in metrics:
            values=[r[key] for r in rows]
            item[key+"_mean"]=statistics.mean(values)
            item[key+"_std"]=statistics.stdev(values)
        aggregate.append(item)
    aggregate.sort(key=lambda r:(r["val_mse_normalized_mean"],r["candidate"]))
    selected=aggregate[0]["candidate"]
    canonical=next(r for r in trials if r["candidate"]==selected and r["seed"]==study["search_seed"])
    summary={"study_id":study["study_id"],"completed_runs":len(trials),"candidates":len(names),
             "finalists":aggregate,"selected_candidate":selected,"selected_run":canonical,
             "test_evaluated":False,"evidence_level":"preliminary",
             "selection_rule":plan["selection_rule"],"plan_sha256":v.sha256(output/"study_plan.json")}
    write_json(output / "summary.json", summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
