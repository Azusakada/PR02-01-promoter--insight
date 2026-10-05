"""Generate validation diagnostics from predictions, keeping the exact source table."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pr02_cnn.common import ROOT, artifact, fresh_dir, resolve, v, write_json, write_table


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True, help="Explicit new run directory; existing analysis is never overwritten")
    args = p.parse_args()
    run = resolve(args.run_dir)
    output = run / "analysis"
    fresh_dir(output)
    model = v.read_json(run / "model_manifest.json")
    predictions = run / "validation/predictions_cnn.csv"
    rows = v.load_table("predictions", predictions)
    source = [{"sample_id": r["sample_id"], "true_log10": math.log10(r["true_value"]),
               "predicted_log10": r["predicted_value_log10"],
               "residual_log10": math.log10(r["true_value"]) - r["predicted_value_log10"]}
              for r in rows if r["prediction_status"] == "ok"]
    write_table(output / "validation_plot_source.tsv", source, delimiter="\t")
    y = np.array([r["true_log10"] for r in source])
    z = np.array([r["predicted_log10"] for r in source])
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.7), layout="constrained")
    axes[0].scatter(y, z, s=9, alpha=0.3, color="#456c97", edgecolors="none", rasterized=True)
    lo, hi = min(y.min(), z.min()) - .15, max(y.max(), z.max()) + .15
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", color="#6e6e6e", linewidth=1)
    axes[0].set(xlabel="Observed log10 strength", ylabel="Predicted log10 strength", title=f"Validation predictions  n={len(source):,}", xlim=(lo, hi), ylim=(lo, hi))
    axes[1].scatter(y, y-z, s=9, alpha=0.3, color="#b97743", edgecolors="none", rasterized=True)
    axes[1].axhline(0, color="#6e6e6e", linewidth=1, linestyle="--")
    axes[1].set(xlabel="Observed log10 strength", ylabel="Observed minus predicted log10", title="Validation residuals")
    for ax in axes:
        ax.grid(alpha=.12)
    path = output / "validation_diagnostics.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    write_json(output / "figure_manifest.json", {"title": "CNN validation predictions and residuals",
        "run_id": model["model_id"], "data_version": model["data_version"], "split_id": model["split_id"],
        "subset": "val", "target_scale": "log10", "n_samples": len(source), "annotation_status": "missing",
        "evidence_level": model["evidence_level"], "source": artifact("source_table", output / "validation_plot_source.tsv"),
        "predictions": artifact("predictions", predictions), "figure": artifact("figure", path),
        "visual_review": "pending", "caption": "All successful validation samples; no outliers removed. Residual = observed - predicted."})
    print(path)


if __name__ == "__main__":
    main()
