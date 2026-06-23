from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mutual_info_score,
    normalized_mutual_info_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "clustered_from_clean" / "scalable_mode_clustering"
OUT_DIR = ROOT / "outputs" / "cluster_relationship"

GLOBAL_ASSIGN_CSV = DATA_DIR / "csv" / "dynamic_family_assignments.csv"
DYNAMIC_SUMMARY_CSV = DATA_DIR / "csv" / "dynamic_family_summary.csv"
STRUCTURAL_SUMMARY_CSV = DATA_DIR / "csv" / "structural_mode_summary.csv"
GLOBAL_CROSS_TAB_CSV = DATA_DIR / "csv" / "dynamic_structural_cross_tab.csv"
STABILITY_CSV = DATA_DIR / "csv" / "mode_stability_summary.csv"
SUBMODE_SUMMARY_CSV = DATA_DIR / "csv" / "dynamic_structural_submode_summary.csv"

DYNAMIC_PROFILE_COLUMNS = [
    "eta10_median",
    "eta20_median",
    "eta50_median",
    "t80_median",
    "t90_median",
    "tau_transfer_est_median",
    "residence_sink34_0_10ps_median",
    "residence_detour567_0_10ps_median",
    "loss_50ps_median",
]

PRIORITY_COLUMNS = [
    "frac_real_fast_high",
    "frac_real_late_high",
    "frac_real_non_high",
    "frac_real_very_fast",
]


@dataclass(frozen=True)
class OutputPaths:
    csv: Path
    figures: Path
    json: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze global D000-D012 dynamic families and S000-S037 structural modes."
    )
    parser.add_argument("--seed", type=int, default=716)
    parser.add_argument("--permutations", type=int, default=100)
    return parser.parse_args()


def make_output_dirs() -> OutputPaths:
    paths = OutputPaths(
        csv=OUT_DIR / "csv",
        figures=OUT_DIR / "figures",
        json=OUT_DIR / "json",
    )
    for path in (paths.csv, paths.figures, paths.json):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def portable_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def one_hot_pipeline(cols: list[str]) -> Pipeline:
    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)
    return Pipeline(
        [
            ("onehot", ColumnTransformer([("cat", encoder, cols)], remainder="drop")),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )


def conditional_entropy(counts: np.ndarray, axis: int) -> float:
    counts = counts.astype(float)
    total = counts.sum()
    if total <= 0:
        return np.nan

    if axis == 0:
        group_sum = counts.sum(axis=0, keepdims=True)
        p_group = group_sum[0] / total
        p = np.divide(counts, group_sum, out=np.zeros_like(counts), where=group_sum > 0)
        h = entropy_columns(p)
    elif axis == 1:
        group_sum = counts.sum(axis=1, keepdims=True)
        p_group = group_sum[:, 0] / total
        p = np.divide(counts, group_sum, out=np.zeros_like(counts), where=group_sum > 0)
        h = entropy_rows(p)
    else:
        raise ValueError("axis must be 0 or 1")

    return float(np.sum(p_group * h))


def entropy_rows(p: np.ndarray) -> np.ndarray:
    logp = np.zeros_like(p)
    positive = p > 0
    logp[positive] = np.log2(p[positive])
    return -np.sum(p * logp, axis=1)


def entropy_columns(p: np.ndarray) -> np.ndarray:
    logp = np.zeros_like(p)
    positive = p > 0
    logp[positive] = np.log2(p[positive])
    return -np.sum(p * logp, axis=0)


def contingency_metrics(dynamic: np.ndarray, structural: np.ndarray) -> dict[str, float]:
    dynamic = np.asarray(dynamic).astype(str)
    structural = np.asarray(structural).astype(str)
    table = pd.crosstab(dynamic, structural)
    counts = table.to_numpy()
    total = counts.sum()
    if total <= 0:
        raise ValueError("empty contingency table")

    return {
        "n": int(total),
        "n_dynamic_families": int(table.shape[0]),
        "n_structural_modes": int(table.shape[1]),
        "purity_d_to_s": float(table.max(axis=1).sum() / total),
        "purity_s_to_d": float(table.max(axis=0).sum() / total),
        "nmi": float(normalized_mutual_info_score(dynamic, structural)),
        "ari": float(adjusted_rand_score(dynamic, structural)),
        "mutual_info": float(mutual_info_score(dynamic, structural)),
        "entropy_s_given_d": conditional_entropy(counts, axis=1),
        "entropy_d_given_s": conditional_entropy(counts, axis=0),
    }


def random_association_baseline(
    dynamic: np.ndarray,
    structural: np.ndarray,
    rng: np.random.Generator,
    n_perm: int,
) -> dict[str, float]:
    vals = []
    for _ in range(n_perm):
        vals.append(contingency_metrics(dynamic, rng.permutation(structural)))

    out = {}
    for key in ["purity_d_to_s", "purity_s_to_d", "nmi", "ari", "entropy_s_given_d", "entropy_d_given_s"]:
        arr = np.asarray([v[key] for v in vals], dtype=float)
        out[f"random_{key}_mean"] = float(np.mean(arr))
        out[f"random_{key}_p05"] = float(np.percentile(arr, 5))
        out[f"random_{key}_p95"] = float(np.percentile(arr, 95))
    return out


def analyze_d_global(paths: OutputPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_csv(DYNAMIC_SUMMARY_CSV)
    stability = pd.read_csv(STABILITY_CSV)

    summary.to_csv(paths.csv / "d_global_profile_summary.csv", index=False)
    stability.to_csv(paths.csv / "d_global_stability_summary.csv", index=False)

    plot_d_global_profile(summary, paths)
    plot_d_global_priority_composition(summary, paths)
    plot_d_global_size(summary, paths)
    plot_stability(stability, paths)
    return summary, stability


def plot_d_global_profile(summary: pd.DataFrame, paths: OutputPaths) -> None:
    data = summary.set_index("dynamic_family_id")[DYNAMIC_PROFILE_COLUMNS].copy()
    data = data.apply(lambda col: col.fillna(col.median()), axis=0)
    z = StandardScaler().fit_transform(data.to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    im = ax.imshow(z, aspect="auto", cmap="coolwarm", vmin=-2.5, vmax=2.5)
    ax.set_title("Global dynamic family profile: D000-D012")
    ax.set_xlabel("dynamics scalar")
    ax.set_ylabel("global dynamic family")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_xticklabels(data.columns, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_yticklabels(data.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015, label="z-score across D families")
    fig.tight_layout()
    fig.savefig(paths.figures / "d_global_profile_heatmap.png", dpi=180)
    plt.close(fig)


def plot_d_global_priority_composition(summary: pd.DataFrame, paths: OutputPaths) -> None:
    data = summary.set_index("dynamic_family_id")[PRIORITY_COLUMNS].copy()
    labels = {
        "frac_real_fast_high": "fast_high",
        "frac_real_late_high": "late_high",
        "frac_real_non_high": "non_high",
        "frac_real_very_fast": "very_fast",
    }

    fig, ax = plt.subplots(figsize=(10, 4.8))
    bottom = np.zeros(len(data))
    colors = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"]
    for col, color in zip(PRIORITY_COLUMNS, colors):
        vals = data[col].to_numpy(dtype=float)
        ax.bar(data.index, vals, bottom=bottom, label=labels[col], color=color)
        bottom += vals
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("fraction")
    ax.set_title("Priority-group composition by D global family")
    ax.tick_params(axis="x", labelrotation=30)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    fig.savefig(paths.figures / "d_global_priority_composition.png", dpi=180)
    plt.close(fig)


def plot_d_global_size(summary: pd.DataFrame, paths: OutputPaths) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    ax.bar(summary["dynamic_family_id"], summary["n"], color="#4C78A8")
    ax.set_ylabel("n")
    ax.set_title("Global dynamic family size")
    ax.tick_params(axis="x", labelrotation=30)
    fig.tight_layout()
    fig.savefig(paths.figures / "d_global_size.png", dpi=180)
    plt.close(fig)


def plot_stability(stability: pd.DataFrame, paths: OutputPaths) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    x = np.arange(len(stability))
    width = 0.38
    labels = stability["kind"] + " " + stability["k_a"].astype(str) + "-" + stability["k_b"].astype(str)
    ax.bar(x - width / 2, stability["ari"], width, label="ARI", color="#4C78A8")
    ax.bar(x + width / 2, stability["nmi"], width, label="NMI", color="#F58518")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("score")
    ax.set_title("Mode stability under kNN setting changes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(paths.figures / "d_s_global_stability.png", dpi=180)
    plt.close(fig)


def analyze_sd_global(args: argparse.Namespace, paths: OutputPaths) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(args.seed)
    df = pd.read_csv(GLOBAL_ASSIGN_CSV)
    df["dynamic_family_id"] = df["dynamic_family_id"].astype(str)
    df["structural_mode_id"] = df["structural_mode_id"].astype(str)

    rows = []
    dynamic = df["dynamic_family_id"].to_numpy()
    structural = df["structural_mode_id"].to_numpy()
    metrics = contingency_metrics(dynamic, structural)
    metrics.update(random_association_baseline(dynamic, structural, rng, args.permutations))
    rows.append({"scope": "overall", "group": "all", **metrics})

    group_rows = []
    for group, part in df.groupby("priority_group"):
        dynamic = part["dynamic_family_id"].astype(str).to_numpy()
        structural = part["structural_mode_id"].astype(str).to_numpy()
        metrics = contingency_metrics(dynamic, structural)
        metrics.update(random_association_baseline(dynamic, structural, rng, args.permutations))
        group_rows.append({"scope": "priority_group", "group": group, **metrics})

    overall = pd.DataFrame(rows)
    groupwise = pd.DataFrame(group_rows)
    for out in (overall, groupwise):
        out["nmi_minus_random_mean"] = out["nmi"] - out["random_nmi_mean"]
        out["purity_d_to_s_minus_random_mean"] = out["purity_d_to_s"] - out["random_purity_d_to_s_mean"]
        out["entropy_s_given_d_minus_random_mean"] = out["entropy_s_given_d"] - out["random_entropy_s_given_d_mean"]

    overall.to_csv(paths.csv / "sd_global_association_summary.csv", index=False)
    groupwise.to_csv(paths.csv / "sd_global_group_association_summary.csv", index=False)

    plot_sd_global_heatmap(df, paths)
    plot_sd_global_group_heatmaps(df, paths)
    return overall, groupwise


def plot_sd_global_heatmap(df: pd.DataFrame, paths: OutputPaths) -> None:
    table = pd.crosstab(df["dynamic_family_id"], df["structural_mode_id"])
    row_norm = table.div(table.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(row_norm.to_numpy(), aspect="auto", cmap="magma")
    ax.set_title("Global S-D heatmap: P(S_global | D_global)")
    ax.set_xlabel("S_global")
    ax.set_ylabel("D_global")
    ax.set_xticks(np.arange(len(row_norm.columns)))
    ax.set_xticklabels(row_norm.columns, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(row_norm.index)))
    ax.set_yticklabels(row_norm.index, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="row-normalized fraction")
    fig.tight_layout()
    fig.savefig(paths.figures / "sd_global_heatmap.png", dpi=180)
    plt.close(fig)


def plot_sd_global_group_heatmaps(df: pd.DataFrame, paths: OutputPaths) -> None:
    groups = sorted(df["priority_group"].unique())
    fig, axes = plt.subplots(len(groups), 1, figsize=(12, 2.4 * len(groups)))
    if len(groups) == 1:
        axes = [axes]

    for ax, group in zip(axes, groups):
        part = df[df["priority_group"] == group]
        table = pd.crosstab(part["dynamic_family_id"], part["structural_mode_id"])
        row_norm = table.div(table.sum(axis=1), axis=0)
        ax.imshow(row_norm.to_numpy(), aspect="auto", cmap="magma")
        ax.set_title(group)
        ax.set_ylabel("D_global")
        ax.set_yticks(np.arange(len(row_norm.index)))
        ax.set_yticklabels(row_norm.index, fontsize=7)
        ax.set_xticks([])

    axes[-1].set_xlabel("S_global")
    fig.tight_layout()
    fig.savefig(paths.figures / "sd_global_group_heatmaps.png", dpi=180)
    plt.close(fig)


def analyze_s_to_d_prediction(paths: OutputPaths) -> pd.DataFrame:
    df = pd.read_csv(GLOBAL_ASSIGN_CSV)
    df = df[["priority_group", "structural_mode_id", "dynamic_family_id"]].dropna()
    y = df["dynamic_family_id"].astype(str).to_numpy()
    labels = np.sort(np.unique(y))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    rows = []
    for name, cols in [
        ("priority_only", ["priority_group"]),
        ("priority_plus_S_global", ["priority_group", "structural_mode_id"]),
    ]:
        model = one_hot_pipeline(cols)
        X = df[cols]
        pred = cross_val_predict(model, X, y, cv=cv, method="predict")
        prob = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
        rows.append(
            {
                "model": name,
                "n": int(len(df)),
                "accuracy": float(accuracy_score(y, pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, average="macro")),
                "log_loss": float(log_loss(y, prob, labels=labels)),
            }
        )

    out = pd.DataFrame(rows)
    base = out[out["model"] == "priority_only"].iloc[0]
    plus = out[out["model"] == "priority_plus_S_global"].iloc[0]
    out["accuracy_delta_vs_priority_only"] = out["accuracy"] - float(base["accuracy"])
    out["balanced_accuracy_delta_vs_priority_only"] = out["balanced_accuracy"] - float(base["balanced_accuracy"])
    out["macro_f1_delta_vs_priority_only"] = out["macro_f1"] - float(base["macro_f1"])
    out["log_loss_reduction_vs_priority_only"] = 1.0 - out["log_loss"] / float(base["log_loss"])
    out.loc[out["model"] == "priority_only", [
        "accuracy_delta_vs_priority_only",
        "balanced_accuracy_delta_vs_priority_only",
        "macro_f1_delta_vs_priority_only",
        "log_loss_reduction_vs_priority_only",
    ]] = 0.0

    out.to_csv(paths.csv / "s_global_to_d_global_prediction.csv", index=False)
    plot_s_to_d_prediction(out, paths)
    return out


def plot_s_to_d_prediction(pred: pd.DataFrame, paths: OutputPaths) -> None:
    plus = pred[pred["model"] == "priority_plus_S_global"].iloc[0]
    metrics = [
        ("accuracy", plus["accuracy_delta_vs_priority_only"]),
        ("balanced accuracy", plus["balanced_accuracy_delta_vs_priority_only"]),
        ("macro F1", plus["macro_f1_delta_vs_priority_only"]),
        ("log-loss reduction", plus["log_loss_reduction_vs_priority_only"]),
    ]
    names = [m[0] for m in metrics]
    vals = np.asarray([m[1] for m in metrics], dtype=float) * 100.0

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    colors = ["#4C78A8" if v >= 0 else "#E45756" for v in vals]
    ax.bar(names, vals, color=colors)
    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_ylabel("change from priority-only baseline (percentage points or %)")
    ax.set_title("Does S_global add information for predicting D_global?")
    ax.tick_params(axis="x", labelrotation=20)
    for i, v in enumerate(vals):
        va = "bottom" if v >= 0 else "top"
        ax.text(i, v + (0.15 if v >= 0 else -0.15), f"{v:.2f}", ha="center", va=va, fontsize=8)
    fig.tight_layout()
    fig.savefig(paths.figures / "s_global_to_d_global_prediction.png", dpi=180)
    plt.close(fig)


def export_submode_summary(paths: OutputPaths) -> pd.DataFrame:
    submode = pd.read_csv(SUBMODE_SUMMARY_CSV)
    submode.to_csv(paths.csv / "d_global_s_global_submode_summary.csv", index=False)
    return submode


def main() -> None:
    args = parse_args()
    paths = make_output_dirs()

    d_summary, stability = analyze_d_global(paths)
    sd_overall, sd_groupwise = analyze_sd_global(args, paths)
    pred = analyze_s_to_d_prediction(paths)
    submode = export_submode_summary(paths)

    manifest = {
        "scope": "global_D000_D012_only",
        "inputs": {
            "global_assignments": portable_path(GLOBAL_ASSIGN_CSV),
            "dynamic_family_summary": portable_path(DYNAMIC_SUMMARY_CSV),
            "structural_mode_summary": portable_path(STRUCTURAL_SUMMARY_CSV),
            "dynamic_structural_cross_tab": portable_path(GLOBAL_CROSS_TAB_CSV),
            "mode_stability_summary": portable_path(STABILITY_CSV),
            "dynamic_structural_submode_summary": portable_path(SUBMODE_SUMMARY_CSV),
        },
        "outputs": {
            "d_global_profile_summary": portable_path(paths.csv / "d_global_profile_summary.csv"),
            "d_global_stability_summary": portable_path(paths.csv / "d_global_stability_summary.csv"),
            "sd_global_association_summary": portable_path(paths.csv / "sd_global_association_summary.csv"),
            "sd_global_group_association_summary": portable_path(paths.csv / "sd_global_group_association_summary.csv"),
            "s_global_to_d_global_prediction": portable_path(paths.csv / "s_global_to_d_global_prediction.csv"),
            "d_global_s_global_submode_summary": portable_path(paths.csv / "d_global_s_global_submode_summary.csv"),
        },
        "row_counts": {
            "d_global_families": int(len(d_summary)),
            "stability_rows": int(len(stability)),
            "sd_overall_rows": int(len(sd_overall)),
            "sd_groupwise_rows": int(len(sd_groupwise)),
            "prediction_rows": int(len(pred)),
            "submode_rows": int(len(submode)),
        },
    }
    (paths.json / "cluster_relationship_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"saved outputs under: {OUT_DIR}")


if __name__ == "__main__":
    main()

