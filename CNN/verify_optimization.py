"""Independently verify an optimization round, reload its selected model and plot results."""
from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path

import numpy as np
import scipy.stats
from sklearn.metrics import mean_absolute_error, r2_score

from pr02_cnn.common import (ROOT, ModelArtifact, artifact, config, fresh_dir, relative,
                             resolve, v, write_json, write_table)
from pr02_cnn.predict import predict_sequences


def read_table(path, delimiter=","):
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study-dir", type=Path, default=ROOT / "CNN/runs/cnn_optimization_20261004_round1")
    args = p.parse_args()
    study = resolve(args.study_dir)
    plan = v.read_json(study / "study_plan.json")
    summary = v.read_json(study / "summary.json")
    v.require(v.sha256(study/"study_plan.json")==summary["plan_sha256"], "预声明计划哈希不符")
    trials = read_table(study / "trials.csv")
    base = config(resolve(plan["study"]["base_config"]))
    for name, key in [("samples", "data_sha256"), ("splits", "splits_sha256"), ("transform", "transform_sha256")]:
        v.require(v.sha256(resolve(base["data"][name])) == plan[key], "公共输入文件发生变化")
    splits = read_table(resolve(base["data"]["splits"]), "\t")
    subset_ids = {subset:{r["sample_id"] for r in splits if r["split"]==subset} for subset in ["train","val","test"]}
    samples = {r["sample_id"]:r for r in read_table(resolve(base["data"]["samples"]),"\t")}
    verified=[]
    for trial in trials:
        run = resolve(trial["run_dir"])
        meta = v.read_json(run / "model_manifest.json")
        v.run_check(run / "run_manifest.json", ROOT)
        v.run_check(run / "validation/run_manifest.json", ROOT)
        train_ids={r["sample_id"] for r in read_table(run/"train_ids.tsv","\t")}
        val_ids={r["sample_id"] for r in read_table(run/"val_ids.tsv","\t")}
        v.require(train_ids == subset_ids["train"] and val_ids == subset_ids["val"], "训练/验证ID不匹配")
        v.require(not train_ids & subset_ids["test"] and meta["test_used"] is False, "检测到test进入训练")
        rows=read_table(run/"validation/predictions_cnn.csv")
        v.require(len(rows)==1783 and {r["sample_id"] for r in rows}==subset_ids["val"], "验证样本缺失/重复")
        v.require(all(r["prediction_status"]=="ok" and r["subset"]=="val" for r in rows), "预测状态/子集不符")
        v.require(all(float(r["true_value"])==float(samples[r["sample_id"]]["strength"]) for r in rows), "真实标签被改变")
        y=np.log10([float(samples[r["sample_id"]]["strength"]) for r in rows])
        z=np.array([float(r["predicted_value_log10"]) for r in rows])
        transform=meta["transform"]
        span=transform["max_log10"]-transform["min_log10"]
        independently={"log10_r2":float(r2_score(y,z)),
                       "log10_mae":float(mean_absolute_error(y,z)),
                       "log10_spearman":float(scipy.stats.spearmanr(y,z).statistic),
                       "val_mse_normalized":float(np.mean(((y-z)/span)**2))}
        raw_y=np.array([float(samples[r["sample_id"]]["strength"]) for r in rows])
        raw_z=np.array([float(r["predicted_value"]) for r in rows])
        v.require(np.allclose(raw_z,np.power(10.,z),rtol=1e-12,atol=1e-12), "原始尺度反变换不符")
        independently.update(raw_r2=float(r2_score(raw_y,raw_z)),
                             raw_mae=float(mean_absolute_error(raw_y,raw_z)),
                             raw_spearman=float(scipy.stats.spearmanr(raw_y,raw_z).statistic))
        for key,value in independently.items():
            tolerance=1e-8 if key=="val_mse_normalized" else 1e-11
            v.require(math.isclose(value,float(trial[key]),abs_tol=tolerance,rel_tol=tolerance), "独立指标不符: "+key)
        v.require(all(math.isfinite(value) for value in independently.values()), "指标非有限")
        verified.append({"run_id":trial["run_id"],"n_val":len(rows),"metrics":independently})

    search=[r for r in trials if r["stage"]=="search"]
    top=[r["candidate"] for r in sorted(search,key=lambda r:(float(r["val_mse_normalized"]),r["candidate"]))[:plan["study"]["stability_top_k"]]]
    means={name:statistics.mean(float(r["val_mse_normalized"]) for r in trials if r["candidate"]==name) for name in top}
    independently_selected=min(means,key=lambda name:(means[name],name))
    v.require(independently_selected==summary["selected_candidate"], "配置选择与预声明规则不符")
    expected_seeds=set([plan["study"]["search_seed"],*plan["study"]["stability_seeds"]])
    for name in top:
        v.require({int(r["seed"]) for r in trials if r["candidate"]==name}==expected_seeds, "复跑种子不完整")
        item=next(r for r in summary["finalists"] if r["candidate"]==name)
        for metric in ["val_mse_normalized","log10_r2","log10_spearman","log10_mae","raw_r2"]:
            values=np.array([float(r[metric]) for r in trials if r["candidate"]==name])
            v.require(math.isclose(float(values.mean()),item[metric+"_mean"],abs_tol=1e-12), "种子均值不符")
            v.require(math.isclose(float(values.std(ddof=1)),item[metric+"_std"],abs_tol=1e-12), "种子标准差不符")
    expected_count=len(plan["study"]["candidates"])+len(top)*len(plan["study"]["stability_seeds"])
    v.require(len(trials)==summary["completed_runs"]==expected_count, "试验数不完整")
    selected=summary["selected_run"]
    v.require(selected["seed"]==plan["study"]["search_seed"], "不能选择最好分数的种子")
    run=resolve(selected["run_dir"])
    model=ModelArtifact(run/"best_checkpoint.pt",run/"model_manifest.json",run/"cnn_run_config.yaml")
    input_table=ROOT/"work/cnn_inputs"/selected["run_id"]/"val.tsv"
    reload_dir=run/"reload_validation"
    if reload_dir.exists():
        v.run_check(reload_dir/"run_manifest.json",ROOT)
        reload_predictions=reload_dir/"predictions_cnn.csv"
    else:
        reloaded=predict_sequences(input_table,model,model.config,reload_dir)
        reload_predictions=reloaded.predictions
    v.require(reload_predictions.read_bytes()==(run/"validation/predictions_cnn.csv").read_bytes(), "重载后预测不一致")
    config_path=ROOT/"CNN/configs/cnn_optimized_round1.yaml"
    if config_path.exists():
        v.require(config_path.read_bytes()==model.config.read_bytes(), "已有优化配置与选定模型不一致")
    else:
        config_path.write_bytes(model.config.read_bytes())
    verification={"status":"passed","completed_runs":len(trials),"independent_metrics":verified,
                  "train_ids":8318,"val_ids":1783,"test_evaluated":False,
                  "selection_rule_verified":True,"selected_reload_prediction_byte_equal":True,
                  "selected_config":artifact("config",config_path)}
    write_json(study/"independent_verification.json",verification)

    # Export a compact scientific figure, with the exact underlying source tables.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    history=read_table(run/"training_history.csv")
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout="constrained")
    epochs=[int(r["epoch"]) for r in history]
    axes[0].plot(epochs,[float(r["train_mse_normalized"]) for r in history],label="Train",color="#466d9e")
    axes[0].plot(epochs,[float(r["val_mse_normalized"]) for r in history],label="Validation",color="#bc6b36")
    axes[0].axvline(selected["best_epoch"],color="#777777",linestyle="--",label=f"Selected epoch {selected['best_epoch']}")
    axes[0].set(title="Selected CNN: training history",xlabel="Epoch",ylabel="Normalized log10 MSE")
    axes[0].legend(frameon=False)
    seed_source=[]
    for j,name in enumerate(top):
        rows=[r for r in trials if r["candidate"]==name]
        values=[float(r["log10_r2"]) for r in rows]
        axes[1].scatter(j+np.linspace(-.07,.07,len(values)),values,s=40,color=["#466d9e","#bc6b36"][j%2],alpha=.85)
        axes[1].errorbar(j+.24,statistics.mean(values),yerr=statistics.stdev(values),color="#333333",fmt="s",capsize=5)
        for r in rows:
            seed_source.append({"candidate":name,"seed":r["seed"],"val_log10_r2":r["log10_r2"]})
    axes[1].set(xticks=list(range(len(top))),xticklabels=top,ylabel="Validation log10 R2",title="Three predeclared seeds per finalist")
    for ax in axes:ax.grid(alpha=.15)
    figure=study/"optimization_summary.png"
    fig.savefig(figure,dpi=180);plt.close(fig)
    write_table(study/"seed_plot_source.csv",seed_source)
    write_json(study/"figure_manifest.json",{"figure":artifact("figure",figure),
          "history":artifact("source",run/"training_history.csv"),
          "seeds":artifact("source",study/"seed_plot_source.csv"),"visual_review":"pending"})
    print("Verified all trials, independent metrics, fixed IDs and selected-model reload.")
    print("Selected config:",config_path)
    print("Summary figure:",figure)


if __name__ == "__main__":
    main()
