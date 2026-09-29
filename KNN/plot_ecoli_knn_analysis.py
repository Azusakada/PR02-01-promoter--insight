"""Plot the E. coli KNN k-search and final test diagnostics.

The k-search plots use validation predictions only.  The diagnostic plot uses
the held-out test predictions only and is never used for model selection.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).resolve().parent / "results"
FIGURES = EXPERIMENT / "figures"


def plot_k_search() -> None:
    search = pd.read_csv(EXPERIMENT / "hyperparameter_search.csv")
    search = search.sort_values("k").reset_index(drop=True)
    best = search.loc[search["val_r2_log10"].idxmax()]
    stopped = search.iloc[-1]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
    fig.suptitle("E. coli distance-weighted KNN: validation metrics across k", fontsize=15)

    axes[0].plot(search["k"], search["val_r2_log10"], marker="o", linewidth=1.7,
                 color="#1f4e79", label="Validation R² (log10 strength)")
    axes[0].axhline(0, color="#777777", linewidth=0.9, linestyle="--")
    axes[0].scatter([best["k"]], [best["val_r2_log10"]], color="#c00000", zorder=4,
                    label=f"Best k={int(best['k'])}")
    axes[0].axvline(best["k"], color="#c00000", linestyle=":", linewidth=1.1)
    axes[0].set_ylabel("R² (log10)")
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(search["k"], search["val_mae_log10"], marker="o", linewidth=1.7,
                 color="#2f7d32", label="Validation MAE (log10 strength)")
    axes[1].axvline(best["k"], color="#c00000", linestyle=":", linewidth=1.1)
    axes[1].set_ylabel("MAE (log10)")
    axes[1].legend(loc="best", frameon=False)

    axes[2].plot(search["k"], search["val_spearman_log10"], marker="o", linewidth=1.7,
                 color="#7030a0", label="Validation Spearman (log10 strength)")
    axes[2].axvline(best["k"], color="#c00000", linestyle=":", linewidth=1.1)
    axes[2].set_ylabel("Spearman")
    axes[2].set_xlabel("Number of neighbors k (log scale)")
    axes[2].legend(loc="best", frameon=False)

    for axis in axes:
        axis.set_xscale("log")
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    # The shaded interval is the three consecutive non-improving rounds that
    # caused the patience-based search to stop.
    for axis in axes:
        axis.axvspan(2001, stopped["k"], color="#f4b183", alpha=0.18)
    axes[0].annotate(
        "3 non-improving rounds\n(early stop at k=3001)",
        xy=(2501, float(search.loc[search["k"] == 2501, "val_r2_log10"].iloc[0])),
        xytext=(3300, -0.10),
        arrowprops={"arrowstyle": "->", "color": "#8b4513", "lw": 1.0},
        fontsize=9,
        color="#6b3410",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIGURES / "knn_validation_metrics_vs_k.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_test_diagnostics() -> None:
    pred = pd.read_csv(EXPERIMENT / "predictions_knn_ecoli_full_test.csv")
    true = pred["extra_true_value_log10"].to_numpy()
    estimate = pred["predicted_value_log10"].to_numpy()
    residual = estimate - true
    r2 = 1 - np.sum((true - estimate) ** 2) / np.sum((true - true.mean()) ** 2)
    mae = np.mean(np.abs(residual))

    lo = min(true.min(), estimate.min())
    hi = max(true.max(), estimate.max())
    padding = (hi - lo) * 0.04
    limits = (lo - padding, hi + padding)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.suptitle("E. coli final KNN (k=1501): held-out test diagnostics", fontsize=15)

    axes[0].scatter(true, estimate, s=13, alpha=0.35, color="#1f4e79", edgecolors="none")
    axes[0].plot(limits, limits, linestyle="--", color="#c00000", linewidth=1.2,
                 label="Ideal: predicted = true")
    axes[0].set_xlim(limits)
    axes[0].set_ylim(limits)
    axes[0].set_xlabel("True log10(strength)")
    axes[0].set_ylabel("Predicted log10(strength)")
    axes[0].set_title(f"Prediction agreement\nR²={r2:.4f}, MAE={mae:.4f}")
    axes[0].legend(loc="best", frameon=False)

    axes[1].scatter(true, residual, s=13, alpha=0.35, color="#7030a0", edgecolors="none")
    axes[1].axhline(0, linestyle="--", color="#c00000", linewidth=1.2)
    axes[1].set_xlabel("True log10(strength)")
    axes[1].set_ylabel("Residual (predicted - true)")
    axes[1].set_title("Residual pattern")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIGURES / "knn_test_diagnostics_k1501.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_k_search()
    plot_test_diagnostics()
    print(f"Generated figures in {FIGURES}")


if __name__ == "__main__":
    main()
