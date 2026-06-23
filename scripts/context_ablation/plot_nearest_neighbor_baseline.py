from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
NN_DIR = ROOT / "outputs" / "nearest_neighbor_baseline"
OUT_DIR = NN_DIR / "comparison"
FIG_DIR = OUT_DIR / "figures"

CONTEXT_ORDER = ["c5", "c12", "c18", "c25", "c26", "c33"]
METRIC_ORDER = ["eta", "tau_transfer", "ipr", "purity", "c_l1"]


def context_from_run(run_name: str) -> str:
    return run_name.removeprefix("nsf_h27_").removesuffix("_seed0")


def load_results() -> pd.DataFrame:
    rows = []
    for path in sorted(NN_DIR.glob("nsf_h27_*_seed0/nearest_neighbor_summary.csv")):
        run_name = path.parent.name
        df = pd.read_csv(path)
        df.insert(0, "run", run_name)
        df.insert(1, "context", context_from_run(run_name))
        rows.append(df)
    if not rows:
        raise FileNotFoundError(f"No nearest_neighbor_summary.csv files under {NN_DIR}")
    out = pd.concat(rows, ignore_index=True)
    out["context"] = pd.Categorical(out["context"], categories=CONTEXT_ORDER, ordered=True)
    out["metric"] = pd.Categorical(out["metric"], categories=METRIC_ORDER, ordered=True)
    return out.sort_values(["context", "metric"]).reset_index(drop=True)


def summarize_by_context(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("context", observed=False)
    summary = grouped.agg(
        mean_model_mae=("model_mae", "mean"),
        mean_nn_mae=("nearest_neighbor_mae", "mean"),
        mean_random_mae=("random_mae", "mean"),
        mean_model_reduction_vs_nn=("model_reduction_vs_nn", "mean"),
        mean_model_better_vs_nn=("model_better_fraction_vs_nn", "mean"),
        mean_nn_reduction_vs_random=("nn_reduction_vs_random", "mean"),
    ).reset_index()
    return summary


def plot_model_vs_nn_mae(summary: pd.DataFrame) -> None:
    x = np.arange(len(summary))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - width / 2, summary["mean_model_mae"], width, label="NSF model", color="#4C78A8")
    ax.bar(x + width / 2, summary["mean_nn_mae"], width, label="nearest neighbor", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["context"])
    ax.set_ylabel("mean MAE across 5 labels")
    ax.set_title("NSF vs nearest-neighbor baseline")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_vs_nn_mean_mae.png", dpi=180)
    plt.close(fig)


def plot_model_reduction_vs_nn(summary: pd.DataFrame) -> None:
    vals = summary["mean_model_reduction_vs_nn"].to_numpy(dtype=float) * 100.0
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in vals]
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(summary["context"], vals, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("model reduction vs NN (%)")
    ax.set_title("Positive means NSF beats nearest neighbor")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        va = "bottom" if v >= 0 else "top"
        offset = 3.0 if v >= 0 else -3.0
        ax.text(i, v + offset, f"{v:.1f}%", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_reduction_vs_nn.png", dpi=180)
    plt.close(fig)


def plot_model_better_fraction(summary: pd.DataFrame) -> None:
    vals = summary["mean_model_better_vs_nn"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(summary["context"], vals, color="#4C78A8")
    ax.axhline(0.5, color="black", lw=0.8, linestyle="--", label="50%")
    ax.set_ylim(0, 1)
    ax.set_ylabel("model better fraction vs NN")
    ax.set_title("Target-wise win rate against nearest neighbor")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_better_fraction_vs_nn.png", dpi=180)
    plt.close(fig)


def plot_nn_reduction_vs_random(summary: pd.DataFrame) -> None:
    vals = summary["mean_nn_reduction_vs_random"].to_numpy(dtype=float) * 100.0
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(summary["context"], vals, color="#54A24B")
    ax.set_ylim(0, 100)
    ax.set_ylabel("NN reduction vs random (%)")
    ax.set_title("Nearest neighbor is a much stronger baseline than random")
    ax.grid(axis="y", alpha=0.25)
    for i, v in enumerate(vals):
        ax.text(i, v + 1.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "nn_reduction_vs_random.png", dpi=180)
    plt.close(fig)


def plot_label_delta_heatmap(df: pd.DataFrame) -> None:
    mat = df.pivot(index="context", columns="metric", values="delta_model_minus_nn")
    mat = mat.loc[CONTEXT_ORDER, METRIC_ORDER]
    values = mat.to_numpy(dtype=float)
    lim = np.nanmax(np.abs(values))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.set_xticks(np.arange(len(METRIC_ORDER)))
    ax.set_xticklabels(METRIC_ORDER, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(CONTEXT_ORDER)))
    ax.set_yticklabels(CONTEXT_ORDER)
    ax.set_title("MAE delta: NSF model - nearest neighbor")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:.3f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="delta MAE")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "label_delta_model_minus_nn_heatmap.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_results()
    summary = summarize_by_context(df)
    df.to_csv(OUT_DIR / "nearest_neighbor_by_metric.csv", index=False)
    summary.to_csv(OUT_DIR / "nearest_neighbor_by_context.csv", index=False)

    plot_model_vs_nn_mae(summary)
    plot_model_reduction_vs_nn(summary)
    plot_model_better_fraction(summary)
    plot_nn_reduction_vs_random(summary)
    plot_label_delta_heatmap(df)

    print(f"[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()

