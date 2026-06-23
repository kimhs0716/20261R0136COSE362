from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "context_claims_summary"
FIG_DIR = OUT_DIR / "figures"

CONTEXT_ORDER = ["c5", "c12", "c18", "c25", "c26", "c33"]
CONTEXT_COLORS = {
    "c5": "#6b7280",
    "c12": "#4c78a8",
    "c18": "#f58518",
    "c25": "#b279a2",
    "c26": "#54a24b",
    "c33": "#72b7b2",
}


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["context"] = pd.Categorical(out["context"], categories=CONTEXT_ORDER, ordered=True)
    return out.sort_values("context").reset_index(drop=True)


def prepare_c3_columns(c3: pd.DataFrame) -> pd.DataFrame:
    out = c3.copy()
    out["fmo_percentile_mean"] = out["fmo_percentile_generated_mean"]
    out["fmo_percentile_std"] = out["fmo_percentile_generated_std"]
    out["c3_baseline_label"] = "generated baseline"
    return out


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=220, bbox_inches="tight")
    plt.close(fig)


def pct(x: float, digits: int = 1) -> str:
    return f"{100.0 * x:.{digits}f}%"


def setup_axes(ax: plt.Axes, *, grid_axis: str = "x") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis=grid_axis, color="#9ca3af", alpha=0.25, linewidth=0.8)


def plot_claim_verdict_dashboard(perf: pd.DataFrame, nn: pd.DataFrame, c3: pd.DataFrame) -> None:
    best_mae = ordered(perf).loc[ordered(perf)["mean_model_mae"].idxmin()]
    best_nll = ordered(perf).loc[ordered(perf)["best_val_nll"].idxmin()]
    best_vs_nn = ordered(nn).loc[ordered(nn)["mean_model_better_vs_nn"].idxmax()]
    best_c3 = ordered(c3).loc[ordered(c3)["fmo_percentile_mean"].idxmax()]

    cards = [
        {
            "claim": "A",
            "title": "Condition representation matters",
            "verdict": "ACCEPT",
            "main": f"best MAE: {best_mae['context']} ({best_mae['mean_model_mae']:.3f})",
            "sub": f"best NLL: {best_nll['context']} ({best_nll['best_val_nll']:.2f})",
            "color": "#1f9d55",
        },
        {
            "claim": "B",
            "title": "NSF beats random, not retrieval",
            "verdict": "ACCEPT",
            "main": "all contexts beat random",
            "sub": f"best NSF win vs NN: {best_vs_nn['context']} ({pct(best_vs_nn['mean_model_better_vs_nn'])})",
            "color": "#1f9d55",
        },
        {
            "claim": "C",
            "title": "FMO top-likelihood claim",
            "verdict": "REJECT",
            "main": f"best FMO percentile: {best_c3['context']} ({best_c3['fmo_percentile_mean']:.1f}%)",
            "sub": "criterion: >= 95%",
            "color": "#d62728",
        },
    ]

    fig, ax = plt.subplots(figsize=(11.8, 3.4))
    ax.axis("off")
    for i, card in enumerate(cards):
        x = i / 3.0 + 0.015
        w = 0.305
        ax.add_patch(
            Rectangle(
                (x, 0.12),
                w,
                0.76,
                transform=ax.transAxes,
                facecolor="#f9fafb",
                edgecolor="#d1d5db",
                linewidth=1.1,
            )
        )
        ax.text(x + 0.025, 0.78, f"Claim {card['claim']}", transform=ax.transAxes, fontsize=16, fontweight="bold")
        ax.text(x + 0.025, 0.62, card["title"], transform=ax.transAxes, fontsize=10.2, color="#111827")
        ax.text(
            x + 0.025,
            0.45,
            card["verdict"],
            transform=ax.transAxes,
            fontsize=13,
            color=card["color"],
            fontweight="bold",
        )
        ax.text(x + 0.025, 0.30, card["main"], transform=ax.transAxes, fontsize=10.0, color="#111827")
        ax.text(x + 0.025, 0.19, card["sub"], transform=ax.transAxes, fontsize=9.2, color="#4b5563")
    ax.text(0.015, 0.98, "Claim-level result summary", transform=ax.transAxes, fontsize=15, fontweight="bold", va="top")
    save(fig, "01_claim_verdict_dashboard.png")


def plot_context_design(perf: pd.DataFrame) -> None:
    dims = dict(zip(perf["context"], perf["context_dim"]))
    rows = [f"{c} ({dims[c]}D)" for c in CONTEXT_ORDER]
    cols = ["5 labels", "eigenvalues", "dyn summary", "pop trajectory"]
    matrix = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 1],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)), cols)
    ax.set_yticks(np.arange(len(rows)), rows)
    ax.set_title("Same target H, different condition blocks", fontsize=13)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                "yes" if matrix[i, j] > 0 else "-",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > 0 else "#6b7280",
                fontsize=10,
                fontweight="bold" if matrix[i, j] > 0 else "normal",
            )
    ax.set_xlabel("condition block")
    ax.set_ylabel("context")
    save(fig, "02_context_design_map.png")


def plot_ablation_ladder(perf: pd.DataFrame) -> None:
    df = ordered(perf).set_index("context")
    edges = [
        ("c5", "c12", "+ eigvals"),
        ("c5", "c18", "+ dyn summary"),
        ("c12", "c25", "+ dyn summary"),
        ("c5", "c26", "+ pop traj"),
        ("c26", "c33", "+ eigvals"),
    ]
    labels = []
    deltas = []
    for src, dst, label in edges:
        src_val = float(df.loc[src, "mean_reduction_fraction"]) * 100.0
        dst_val = float(df.loc[dst, "mean_reduction_fraction"]) * 100.0
        labels.append(f"{src} to {dst}\n{label}")
        deltas.append(dst_val - src_val)

    y = np.arange(len(edges))
    colors = ["#3f6f9f" if d >= 0 else "#b35c44" for d in deltas]
    fig, ax = plt.subplots(figsize=(9.4, 4.9))
    ax.axvline(0, color="#374151", linewidth=1.2)
    ax.barh(y, deltas, color=colors, alpha=0.86)
    for yi, delta in zip(y, deltas):
        if delta < 0:
            x_text = delta - 0.15
            ha = "right"
        else:
            x_text = delta + 0.12
            ha = "left"
        ax.text(
            x_text,
            yi,
            f"{delta:+.1f} pt",
            va="center",
            ha=ha,
            fontsize=9.5,
            color="#111827",
            fontweight="bold",
        )
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Delta MAE reduction (percentage points)")
    ax.set_title("Feature-block ablation")
    ax.set_xlim(-3.8, 6.7)
    setup_axes(ax)
    save(fig, "03_feature_block_ablation_ladder.png")


def plot_nll_mae_ranking(perf: pd.DataFrame) -> None:
    df = ordered(perf)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3), sharey=True)
    y = np.arange(len(df))

    axes[0].scatter(df["best_val_nll"], y, s=95, color=[CONTEXT_COLORS[c] for c in df["context"]])
    axes[0].set_xlabel("best validation NLL (lower is better)")
    axes[0].set_yticks(y, df["context"])
    axes[0].invert_yaxis()
    axes[0].set_title("Density fit")
    setup_axes(axes[0])
    best_nll_i = int(df["best_val_nll"].idxmin())
    axes[0].annotate(
        "best NLL",
        xy=(df.loc[best_nll_i, "best_val_nll"], best_nll_i),
        xytext=(df.loc[best_nll_i, "best_val_nll"] + 2.5, best_nll_i + 0.25),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )

    axes[1].scatter(df["mean_reduction_fraction"] * 100.0, y, s=95, color=[CONTEXT_COLORS[c] for c in df["context"]])
    axes[1].set_xlabel("mean MAE reduction vs random (%)")
    axes[1].set_title("Simulator-label reconstruction")
    setup_axes(axes[1])
    best_mae_i = int(df["mean_model_mae"].idxmin())
    axes[1].annotate(
        "best MAE",
        xy=(df.loc[best_mae_i, "mean_reduction_fraction"] * 100.0, best_mae_i),
        xytext=(df.loc[best_mae_i, "mean_reduction_fraction"] * 100.0 - 5.6, best_mae_i - 0.35),
        arrowprops={"arrowstyle": "->", "color": "#374151"},
        fontsize=9,
    )
    fig.suptitle("Validation likelihood and simulator MAE select different contexts", fontsize=14)
    save(fig, "04_nll_vs_mae_ranking.png")


def plot_label_reduction_heatmap(perf: pd.DataFrame) -> None:
    df = ordered(perf).set_index("context")
    labels = ["eta", "tau_transfer", "ipr", "purity", "c_l1"]
    mat = np.array([[df.loc[c, f"{label}_reduction"] for label in labels] for c in CONTEXT_ORDER], dtype=float)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    im = ax.imshow(mat * 100.0, cmap="YlGnBu", vmin=60, vmax=82, aspect="auto")
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(CONTEXT_ORDER)), CONTEXT_ORDER)
    ax.set_title("Per-label MAE reduction vs random baseline")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            value = mat[i, j] * 100.0
            color = "white" if value >= 76.0 else "#111827"
            ax.text(j, i, f"{value:.1f}%", ha="center", va="center", fontsize=9, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("MAE reduction")
    save(fig, "05_label_reduction_heatmap.png")


def plot_three_baseline_ladder(nn: pd.DataFrame) -> None:
    df = ordered(nn)
    y = np.arange(len(df))
    nsf = df["mean_model_mae"] / df["mean_random_mae"]
    nearest = df["mean_nn_mae"] / df["mean_random_mae"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvline(1.0, color="#374151", linestyle="--", linewidth=1.4, label="random")
    for yi, nval, mval in zip(y, nearest, nsf):
        ax.plot([nval, mval], [yi, yi], color="#9ca3af", linewidth=2.2, zorder=1)
    ax.scatter(nearest, y, color="#54a24b", label="nearest-neighbor", s=85, zorder=3)
    ax.scatter(nsf, y, color="#4c78a8", label="NSF", s=85, zorder=3)
    for yi, mval, nval in zip(y, nsf, nearest):
        ax.text(mval + 0.018, yi, f"{mval:.2f}", va="center", fontsize=8.5, color="#1f2937")
        if nval < 0.08:
            ax.text(nval + 0.018, yi, f"{nval:.2f}", va="center", ha="left", fontsize=8.5, color="#1f2937")
        else:
            ax.text(nval - 0.018, yi, f"{nval:.2f}", va="center", ha="right", fontsize=8.5, color="#1f2937")
    ax.set_yticks(y, df["context"])
    ax.invert_yaxis()
    ax.set_xlabel("mean MAE / random-baseline MAE (lower is better)")
    ax.set_title("NSF improves over random, but retrieval remains the stronger baseline")
    ax.set_xlim(0, 1.08)
    setup_axes(ax)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "06_three_baseline_ladder.png")


def plot_win_rate_vs_nn(nn: pd.DataFrame) -> None:
    df = ordered(nn)
    y = np.arange(len(df))
    values = df["mean_model_better_vs_nn"] * 100.0
    colors = ["#54a24b" if v >= 50.0 else "#f58518" for v in values]

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.barh(y, values, color=colors, alpha=0.9)
    ax.axvline(50.0, linestyle="--", color="#374151", linewidth=1.3)
    for yi, val in zip(y, values):
        ax.text(val + 1.0, yi, f"{val:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y, df["context"])
    ax.invert_yaxis()
    ax.set_xlim(0, 56)
    ax.set_xlabel("targets where NSF MAE < nearest-neighbor MAE (%)")
    ax.set_title("Even the best context does not consistently beat nearest-neighbor retrieval")
    setup_axes(ax)
    save(fig, "07_win_rate_vs_nearest_neighbor.png")


def plot_c3_seed_interval(c3_seed: pd.DataFrame) -> None:
    df = ordered(
        c3_seed.groupby(["context", "context_dim"], as_index=False)
        .agg(
            mean=("fmo_percentile_generated", "mean"),
            min=("fmo_percentile_generated", "min"),
            max=("fmo_percentile_generated", "max"),
        )
        .rename(columns={"mean": "fmo_percentile_mean"})
    )
    y = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.axvline(95.0, color="#d62728", linestyle="--", linewidth=1.4, label="top-5% criterion")
    for yi, row in df.iterrows():
        ax.plot([row["min"], row["max"]], [yi, yi], color="#9ca3af", linewidth=2.4, zorder=1)
        ax.scatter(row["fmo_percentile_mean"], yi, color=CONTEXT_COLORS[row["context"]], s=90, zorder=3)
        ax.text(row["fmo_percentile_mean"] + 1.0, yi, f"{row['fmo_percentile_mean']:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y, df["context"])
    ax.invert_yaxis()
    ax.set_xlim(-2, 100)
    ax.set_xlabel("FMO likelihood percentile among generated samples (%)")
    ax.set_title("C3 is rejected robustly across generated-sampling seeds")
    setup_axes(ax)
    ax.legend(loc="lower right", frameon=True)
    save(fig, "08_c3_seed_interval.png")


def load_c3_distribution(context: str, seed: int = 716) -> tuple[np.ndarray, float]:
    path = (
        ROOT
        / "outputs"
        / "c3_biology_likelihood"
        / f"nsf_h27_{context}_seed0"
        / f"seed_{seed}"
        / "c3_likelihood_scores.npz"
    )
    scores = np.load(path)
    generated = np.asarray(scores["generated_logp_under_c_fmo"], dtype=float)
    fmo_logp = float(np.asarray(scores["fmo_logp"]).ravel()[0])
    return generated, fmo_logp


def plot_c3_logp_distribution() -> None:
    contexts = ["c26", "c33"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    for ax, context in zip(axes, contexts):
        generated, fmo_logp = load_c3_distribution(context)
        ax.hist(generated, bins=70, density=True, color=CONTEXT_COLORS[context], alpha=0.72)
        ax.axvline(fmo_logp, color="#d62728", linewidth=2.0, label="FMO log-likelihood")
        percentile = float((generated <= fmo_logp).mean() * 100.0)
        ax.text(
            0.03,
            0.94,
            f"{context}: FMO percentile {percentile:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#d1d5db"},
        )
        ax.set_xlabel("")
        ax.set_title(f"{context} generated baseline")
        setup_axes(ax, grid_axis="y")
    axes[0].set_ylabel("density")
    axes[1].legend(loc="upper right", frameon=True)
    fig.suptitle("FMO lies in the low-likelihood tail of generated samples", fontsize=14)
    fig.supxlabel("log p(H | c_FMO)", y=0.02)
    fig.subplots_adjust(bottom=0.20, top=0.80, wspace=0.22)
    save(fig, "09_c3_logp_distribution_best_cases.png")


def plot_c3_baseline_comparison(c3: pd.DataFrame) -> None:
    df = ordered(c3)
    y = np.arange(len(df))
    dataset = df["fmo_percentile_dataset_mean"]
    generated = df["fmo_percentile_generated_mean"]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for yi, a, b in zip(y, dataset, generated):
        ax.plot([a, b], [yi, yi], color="#9ca3af", linewidth=2.2, zorder=1)
    ax.scatter(dataset, y, color="#4c78a8", label="dataset baseline", s=80, zorder=2)
    ax.scatter(generated, y, color="#f58518", label="generated baseline", s=80, zorder=3)
    ax.axvline(95.0, linestyle="--", color="#374151", linewidth=1.3, label="top-5% criterion")
    ax.set_yticks(y, df["context"])
    ax.invert_yaxis()
    ax.set_xlim(-2, 102)
    ax.set_xlabel("FMO likelihood percentile")
    ax.set_title("Dataset baseline can look positive, but generated baseline rejects C3")
    setup_axes(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=3, frameon=True)
    save(fig, "10_c3_dataset_vs_generated_baseline.png")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for old_png in FIG_DIR.glob("*.png"):
        old_png.unlink()

    perf = pd.read_csv(ROOT / "outputs" / "model_performance" / "comparison" / "context_performance_summary.csv")
    nn = pd.read_csv(ROOT / "outputs" / "nearest_neighbor_baseline" / "comparison" / "nearest_neighbor_by_context.csv")
    c3_raw = pd.read_csv(ROOT / "outputs" / "c3_biology_likelihood" / "c3_fmo_likelihood_summary.csv")
    c3 = prepare_c3_columns(c3_raw)
    c3_seed = pd.read_csv(ROOT / "outputs" / "c3_biology_likelihood" / "c3_fmo_likelihood_summary_by_seed.csv")

    plot_claim_verdict_dashboard(perf, nn, c3)
    plot_context_design(ordered(perf))
    plot_ablation_ladder(perf)
    plot_nll_mae_ranking(perf)
    plot_label_reduction_heatmap(perf)
    plot_three_baseline_ladder(nn)
    plot_win_rate_vs_nn(nn)
    plot_c3_seed_interval(c3_seed)
    plot_c3_logp_distribution()
    plot_c3_baseline_comparison(c3_raw)

    print(f"saved figures to: {FIG_DIR}")


if __name__ == "__main__":
    main()

