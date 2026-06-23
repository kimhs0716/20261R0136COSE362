from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
COMPARISON_DIR = ROOT / "outputs" / "model_performance" / "comparison"
FIGURE_DIR = COMPARISON_DIR / "figures"
SUMMARY_CSV = COMPARISON_DIR / "context_performance_summary.csv"
LABELS = ("eta", "tau_transfer", "ipr", "purity", "c_l1")


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"Missing comparison summary CSV: {SUMMARY_CSV}")

    df = pd.read_csv(SUMMARY_CSV).sort_values("context_dim").reset_index(drop=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plot_mean_reduction(df)
    plot_label_reduction(df)
    plot_label_mae(df)
    plot_nll_vs_reduction(df)
    plot_win_rate(df)
    print(f"[saved] {FIGURE_DIR}")


def plot_mean_reduction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    ax.bar(df["context"], df["mean_reduction_fraction"], color=highlight_c26(df))
    ax.set_ylim(0, max(0.85, df["mean_reduction_fraction"].max() + 0.04))
    ax.set_ylabel("mean reduction vs random")
    ax.set_title("Overall conditional MAE reduction")
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, df["mean_reduction_fraction"])
    fig.savefig(FIGURE_DIR / "mean_reduction_by_context.png", dpi=180)
    plt.close(fig)


def plot_label_reduction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    x = np.arange(len(df))
    width = 0.15
    for i, label in enumerate(LABELS):
        ax.bar(x + (i - 2) * width, df[f"{label}_reduction"], width=width, label=label)
    ax.set_xticks(x, df["context"])
    ax.set_ylim(0, 0.9)
    ax.set_ylabel("reduction vs random")
    ax.set_title("Label-wise error reduction")
    ax.legend(ncols=5, fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURE_DIR / "label_reduction_by_context.png", dpi=180)
    plt.close(fig)


def plot_label_mae(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(LABELS), figsize=(15, 3.8), constrained_layout=True)
    for ax, label in zip(axes, LABELS):
        ax.bar(df["context"], df[f"{label}_mae"], color=highlight_c26(df))
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(axis="y", alpha=0.25)
        if label == "tau_transfer":
            ax.set_ylabel("MAE")
    fig.suptitle("Model MAE by target label", y=1.05)
    fig.savefig(FIGURE_DIR / "label_mae_by_context.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_nll_vs_reduction(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    ax.scatter(df["best_val_nll"], df["mean_reduction_fraction"], s=80, color="#4c78a8")
    for _, row in df.iterrows():
        ax.annotate(row["context"], (row["best_val_nll"], row["mean_reduction_fraction"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("best validation NLL")
    ax.set_ylabel("mean reduction vs random")
    ax.set_title("Validation likelihood is not identical to simulator MAE")
    ax.grid(alpha=0.25)
    fig.savefig(FIGURE_DIR / "nll_vs_mean_reduction.png", dpi=180)
    plt.close(fig)


def plot_win_rate(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6), constrained_layout=True)
    ax.bar(df["context"], df["mean_model_better_fraction"], color=highlight_c26(df))
    ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="random tie line")
    ax.set_ylim(0.45, 0.9)
    ax.set_ylabel("model better fraction")
    ax.set_title("Target-wise win rate against random baseline")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    annotate_bars(ax, df["mean_model_better_fraction"])
    fig.savefig(FIGURE_DIR / "win_rate_by_context.png", dpi=180)
    plt.close(fig)


def highlight_c26(df: pd.DataFrame) -> list[str]:
    return ["#d95f02" if context == "c26" else "#4c78a8" for context in df["context"]]


def annotate_bars(ax, values) -> None:
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            patch.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )


if __name__ == "__main__":
    main()

