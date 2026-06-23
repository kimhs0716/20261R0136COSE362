from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "d_family_model_performance"
ASSIGNMENT_PATH = (
    ROOT
    / "data"
    / "clustered_from_clean"
    / "scalable_mode_clustering"
    / "csv"
    / "dynamic_family_assignments.csv"
)
MODEL_PERFORMANCE_DIR = ROOT / "outputs" / "model_performance"
NN_BASELINE_DIR = ROOT / "outputs" / "nearest_neighbor_baseline"

LABELS = ("eta", "tau_transfer", "ipr", "purity", "c_l1")
CONTEXT_ORDER = ("c5", "c12", "c18", "c25", "c26", "c33")
RANDOM_SEED = 716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze conditional NSF simulator-error performance by global dynamic "
            "family labels D000-D012. The D/S labels exist only for the 62k clean "
            "subset, so this script reports coverage explicitly."
        )
    )
    parser.add_argument("--assignments", type=Path, default=ASSIGNMENT_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_PERFORMANCE_DIR)
    parser.add_argument("--nn-dir", type=Path, default=NN_BASELINE_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--min-family-n", type=int, default=10)
    return parser.parse_args()


def infer_context(run_name: str) -> str:
    match = re.search(r"nsf_h27_(c\d+)_seed", run_name)
    if not match:
        raise ValueError(f"Cannot infer context from run directory: {run_name}")
    return match.group(1)


def context_sort_key(context: str) -> int:
    try:
        return CONTEXT_ORDER.index(context)
    except ValueError:
        return len(CONTEXT_ORDER)


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_assignments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    required = {"real_row", "priority_group", "structural_mode_id", "dynamic_family_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Assignment file is missing columns: {sorted(missing)}")
    return df.rename(
        columns={
            "real_row": "dataset_index",
            "priority_group": "priority_group",
            "structural_mode_id": "structural_mode",
            "dynamic_family_id": "dynamic_family",
        }
    )


def load_model_runs(model_dir: Path, assignments: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage_rows = []
    assign_cols = ["dataset_index", "priority_group", "structural_mode", "dynamic_family"]
    for run_dir in sorted(model_dir.glob("nsf_h27_c*_seed0"), key=lambda p: context_sort_key(infer_context(p.name))):
        csv_path = run_dir / "conditional_mae_samples.csv"
        if not csv_path.exists():
            continue
        context = infer_context(run_dir.name)
        df = pd.read_csv(csv_path)
        n_targets = int(df["target_id"].nunique())
        target_df = df[df["method"] == "model"][["target_id", "dataset_index", "target_eta"]].drop_duplicates("target_id")
        covered = target_df.merge(assignments[assign_cols], on="dataset_index", how="inner")
        coverage_rows.append(
            {
                "run_name": run_dir.name,
                "context": context,
                "n_targets": n_targets,
                "n_targets_with_dynamic_label": int(len(covered)),
                "coverage_fraction": float(len(covered) / max(n_targets, 1)),
                "min_covered_dataset_index": int(covered["dataset_index"].min()) if len(covered) else None,
                "max_covered_dataset_index": int(covered["dataset_index"].max()) if len(covered) else None,
            }
        )
        merged = df.merge(assignments[assign_cols], on="dataset_index", how="inner")
        merged["context"] = context
        merged["run_name"] = run_dir.name
        rows.append(merged)

    if not rows:
        raise FileNotFoundError(f"No conditional_mae_samples.csv files found under {model_dir}")
    return pd.concat(rows, ignore_index=True), pd.DataFrame(coverage_rows)


def load_nn_runs(nn_dir: Path, assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    assign_cols = ["dataset_index", "priority_group", "structural_mode", "dynamic_family"]
    for run_dir in sorted(nn_dir.glob("nsf_h27_c*_seed0"), key=lambda p: context_sort_key(infer_context(p.name))):
        csv_path = run_dir / "nearest_neighbor_samples.csv"
        if not csv_path.exists():
            continue
        context = infer_context(run_dir.name)
        df = pd.read_csv(csv_path)
        merged = df.merge(assignments[assign_cols], on="dataset_index", how="inner")
        merged["context"] = context
        merged["run_name"] = run_dir.name
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def summarize_by_dynamic_family(model_rows: pd.DataFrame, nn_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = [model_rows]
    if len(nn_rows):
        combined.append(nn_rows)
    all_rows = pd.concat(combined, ignore_index=True, sort=False)

    metric_rows = []
    profile_rows = []
    group_cols = ["context", "dynamic_family"]
    for (context, dynamic_family), group in all_rows.groupby(group_cols, sort=False):
        target_meta = group[["target_id", "dataset_index", "target_eta", "priority_group", "structural_mode"]].drop_duplicates(
            "target_id"
        )
        structural_top = (
            target_meta["structural_mode"].mode().iloc[0] if len(target_meta["structural_mode"].dropna()) else ""
        )
        priority_top = target_meta["priority_group"].mode().iloc[0] if len(target_meta["priority_group"].dropna()) else ""
        profile_rows.append(
            {
                "context": context,
                "dynamic_family": dynamic_family,
                "n_targets": int(target_meta["target_id"].nunique()),
                "target_eta_median": float(target_meta["target_eta"].median()),
                "target_eta_mean": float(target_meta["target_eta"].mean()),
                "dominant_structural_mode": structural_top,
                "dominant_priority_group": priority_top,
            }
        )

        for label in LABELS:
            err_col = f"abs_err_{label}"
            pivot = group.pivot_table(index="target_id", columns="method", values=err_col, aggfunc="mean")
            if "model" not in pivot.columns or "random_baseline" not in pivot.columns:
                continue
            model_err = pivot["model"].dropna()
            random_err = pivot["random_baseline"].dropna()
            common = model_err.index.intersection(random_err.index)
            if len(common) == 0:
                continue
            model_err = model_err.loc[common]
            random_err = random_err.loc[common]
            row = {
                "context": context,
                "dynamic_family": dynamic_family,
                "metric": label,
                "n_targets": int(len(common)),
                "model_mae": float(model_err.mean()),
                "random_mae": float(random_err.mean()),
                "delta_model_minus_random": float(model_err.mean() - random_err.mean()),
                "reduction_vs_random": float(1.0 - model_err.mean() / max(random_err.mean(), 1e-12)),
                "model_better_vs_random": float((model_err < random_err).mean()),
            }
            if "nearest_neighbor_baseline" in pivot.columns:
                nn_err = pivot["nearest_neighbor_baseline"].dropna()
                common_nn = common.intersection(nn_err.index)
                if len(common_nn):
                    row.update(
                        {
                            "nn_mae": float(nn_err.loc[common_nn].mean()),
                            "delta_model_minus_nn": float(model_err.loc[common_nn].mean() - nn_err.loc[common_nn].mean()),
                            "model_better_vs_nn": float((model_err.loc[common_nn] < nn_err.loc[common_nn]).mean()),
                        }
                    )
                else:
                    row.update({"nn_mae": np.nan, "delta_model_minus_nn": np.nan, "model_better_vs_nn": np.nan})
            else:
                row.update({"nn_mae": np.nan, "delta_model_minus_nn": np.nan, "model_better_vs_nn": np.nan})
            metric_rows.append(row)

    return pd.DataFrame(metric_rows), pd.DataFrame(profile_rows)


def make_overall_summary(metric_summary: pd.DataFrame, profile: pd.DataFrame) -> pd.DataFrame:
    agg = (
        metric_summary.groupby(["context", "dynamic_family"], as_index=False)
        .agg(
            n_targets=("n_targets", "max"),
            mean_reduction_vs_random=("reduction_vs_random", "mean"),
            mean_model_better_vs_random=("model_better_vs_random", "mean"),
            mean_delta_model_minus_nn=("delta_model_minus_nn", "mean"),
            mean_model_better_vs_nn=("model_better_vs_nn", "mean"),
        )
        .merge(profile, on=["context", "dynamic_family", "n_targets"], how="left")
    )
    agg["context_order"] = agg["context"].map(lambda c: context_sort_key(str(c)))
    return agg.sort_values(["context_order", "dynamic_family"]).drop(columns=["context_order"])


def heatmap(
    df: pd.DataFrame,
    value_col: str,
    title: str,
    out_path: Path,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    percent: bool = False,
) -> None:
    pivot = df.pivot(index="dynamic_family", columns="context", values=value_col)
    pivot = pivot.reindex(columns=[c for c in CONTEXT_ORDER if c in pivot.columns])
    families = list(pivot.index)
    contexts = list(pivot.columns)

    fig_w = max(8.5, 1.35 * len(contexts) + 3.0)
    fig_h = max(5.8, 0.45 * len(families) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(contexts)), labels=contexts)
    ax.set_yticks(np.arange(len(families)), labels=families)
    ax.set_title(title, pad=14)
    ax.set_xlabel("context policy")
    ax.set_ylabel("dynamic family")
    for i in range(len(families)):
        for j in range(len(contexts)):
            val = pivot.iloc[i, j]
            if pd.isna(val):
                text = "-"
            elif percent:
                text = f"{val * 100:.0f}%"
            else:
                text = f"{val:.2f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8, color="#111111")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label(value_col)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_family_profile(profile: pd.DataFrame, out_path: Path) -> None:
    # The target subset is the same for every context, so use the first available context.
    base_context = sorted(profile["context"].unique(), key=context_sort_key)[0]
    df = profile[profile["context"] == base_context].sort_values("dynamic_family")
    fig, ax1 = plt.subplots(figsize=(11.0, 5.8))
    x = np.arange(len(df))
    ax1.bar(x, df["n_targets"], color="#6B7A90", alpha=0.85)
    ax1.set_ylabel("covered validation targets")
    ax1.set_xticks(x, labels=df["dynamic_family"], rotation=45, ha="right")
    ax1.set_xlabel("dynamic family")
    ax2 = ax1.twinx()
    ax2.plot(x, df["target_eta_median"], color="#D95F02", marker="o", linewidth=2.0)
    ax2.set_ylabel("median target eta")
    ax1.set_title("D-family coverage and target efficiency in the covered validation subset")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(
    path: Path,
    coverage: pd.DataFrame,
    overall: pd.DataFrame,
    metric_summary: pd.DataFrame,
    figures: dict[str, Path],
) -> None:
    coverage_line = coverage.iloc[0]
    best_context = overall.groupby("context")["mean_reduction_vs_random"].mean().sort_values(ascending=False).index[0]
    weak_nn = overall.groupby("context")["mean_model_better_vs_nn"].mean().sort_values(ascending=False)

    md = []
    md.append("# D-family model-performance follow-up\n")
    md.append(
        "This analysis is restricted to validation targets that can be matched to the clean 62k subset "
        "with global dynamic family labels `D000`-`D012`. It should therefore be read as a follow-up "
        "on labeled targets, not as a replacement for the full held-out validation evaluation.\n"
    )
    md.append("## Coverage\n")
    md.append(
        f"- Among 1000 evaluation targets per context, "
        f"{int(coverage_line['n_targets_with_dynamic_label'])} targets "
        f"({coverage_line['coverage_fraction'] * 100:.1f}%).\n"
    )
    md.append("- Because coverage is partial, family-level results are directional diagnostics.\n")
    md.append("## Main Readout\n")
    md.append(
        f"- By mean MAE reduction across D-families, `{best_context}` gives the strongest reduction versus the random baseline. "
        "The nearest-neighbor baseline can still be stronger in some families, so this result is best interpreted as family-wise behavior rather than a strict generative-model win.\n"
    )
    md.append(
        f"- NSF win rate against nearest-neighbor by context: "
        + ", ".join(f"{ctx}: {val * 100:.1f}%" for ctx, val in weak_nn.items())
        + ".\n"
    )
    md.append("## Figures\n")
    for label, fig_path in figures.items():
        rel = fig_path.relative_to(path.parent).as_posix()
        md.append(f"![{label}]({rel})\n")
        if label == "reduction_vs_random":
            md.append(
                "This figure shows how much the NSF-generated Hamiltonians reduce simulator-label MAE versus a random Hamiltonian baseline within each dynamic family.\n"
            )
        elif label == "win_vs_nn":
            md.append(
                "This figure shows the fraction of targets for which NSF has lower absolute error than nearest-neighbor retrieval.\n"
            )
        elif label == "family_profile":
            md.append(
                "This figure summarizes the labeled target coverage and median eta for each dynamic family in the matched validation subset.\n"
            )
    md.append("## Output Tables\n")
    md.append("- `coverage_by_context.csv`: context蹂?D/S label coverage\n")
    md.append("- `d_family_metric_summary.csv`: label蹂?MAE, reduction, win-rate\n")
    md.append("- `d_family_overall_summary.csv`: family-level mean reduction and win-rate summary\n")
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    assignments = load_assignments(args.assignments)
    model_rows, coverage = load_model_runs(args.model_dir, assignments)
    nn_rows = load_nn_runs(args.nn_dir, assignments)
    metric_summary, profile = summarize_by_dynamic_family(model_rows, nn_rows)
    overall = make_overall_summary(metric_summary, profile)

    coverage.to_csv(args.out_dir / "coverage_by_context.csv", index=False)
    profile.to_csv(args.out_dir / "d_family_target_profile.csv", index=False)
    metric_summary.to_csv(args.out_dir / "d_family_metric_summary.csv", index=False)
    overall.to_csv(args.out_dir / "d_family_overall_summary.csv", index=False)

    figures = {
        "reduction_vs_random": figures_dir / "d_family_reduction_vs_random_heatmap.png",
        "win_vs_nn": figures_dir / "d_family_win_vs_nearest_neighbor_heatmap.png",
        "family_profile": figures_dir / "d_family_coverage_and_eta_profile.png",
    }
    heatmap(
        overall,
        "mean_reduction_vs_random",
        "Mean simulator-MAE reduction vs random baseline by D family",
        figures["reduction_vs_random"],
        cmap="YlGnBu",
        vmin=0.0,
        vmax=1.0,
        percent=True,
    )
    heatmap(
        overall,
        "mean_model_better_vs_nn",
        "Fraction of labels where NSF beats nearest-neighbor baseline",
        figures["win_vs_nn"],
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        percent=True,
    )
    plot_family_profile(profile, figures["family_profile"])

    write_markdown(args.out_dir / "d_family_model_performance_summary.md", coverage, overall, metric_summary, figures)
    write_json(
        args.out_dir / "manifest.json",
        {
            "assignment_path": args.assignments,
            "model_performance_dir": args.model_dir,
            "nearest_neighbor_dir": args.nn_dir,
            "coverage_note": "D/S labels are available only for the 62k clean subset; only covered validation targets are analyzed.",
            "coverage_by_context": coverage.to_dict(orient="records"),
            "outputs": {
                "coverage": "coverage_by_context.csv",
                "metric_summary": "d_family_metric_summary.csv",
                "overall_summary": "d_family_overall_summary.csv",
                "markdown": "d_family_model_performance_summary.md",
                "figures": {k: str(v.relative_to(args.out_dir)) for k, v in figures.items()},
            },
        },
    )

    print("[saved]", args.out_dir)
    print(coverage.to_string(index=False))
    print("\n[overall context means]")
    print(
        overall.groupby("context", as_index=False)
        .agg(
            mean_reduction_vs_random=("mean_reduction_vs_random", "mean"),
            mean_model_better_vs_nn=("mean_model_better_vs_nn", "mean"),
        )
        .sort_values("context", key=lambda s: s.map(context_sort_key))
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

