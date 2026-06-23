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
OUT_DIR = ROOT / "outputs" / "supplementary_robustness"
ASSIGNMENTS = (
    ROOT
    / "data"
    / "clustered_from_clean"
    / "scalable_mode_clustering"
    / "csv"
    / "dynamic_family_assignments.csv"
)
MODEL_DIR = ROOT / "outputs" / "model_performance"
MECH_SCORES = ROOT / "outputs" / "model_performance" / "comparison" / "c2_mechanistic_signature_scores.csv"
BATH_SCORES = ROOT / "outputs" / "c2_bath_sensitivity" / "c2_bath_sensitivity_scores.csv"

LABELS = ("eta", "tau_transfer", "ipr", "purity", "c_l1")
CONTEXT_ORDER = ("c5", "c12", "c18", "c25", "c26", "c33")
ETA_HIGH = 0.95
ETA_NONHIGH = 0.85
RANDOM_SEED = 716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run supplementary robustness analyses for the final FMO report: "
            "C2 threshold sensitivity, C2 score AUC, and D-family error heterogeneity."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--d-permutations", type=int, default=2000)
    return parser.parse_args()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def context_sort_key(context: str) -> int:
    try:
        return CONTEXT_ORDER.index(context)
    except ValueError:
        return len(CONTEXT_ORDER)


def infer_context(run_name: str) -> str:
    match = re.search(r"nsf_h27_(c\d+)_seed", run_name)
    if not match:
        raise ValueError(f"Cannot infer context from run directory: {run_name}")
    return match.group(1)


def pass_stats(pass_mask: np.ndarray, eta: np.ndarray) -> dict[str, float | int]:
    high = eta >= ETA_HIGH
    non = eta < ETA_NONHIGH
    high_pass = int(pass_mask[high].sum())
    non_pass = int(pass_mask[non].sum())
    high_n = int(high.sum())
    non_n = int(non.sum())
    high_rate = high_pass / max(high_n, 1)
    non_rate = non_pass / max(non_n, 1)
    return {
        "high_n": high_n,
        "nonhigh_n": non_n,
        "high_pass": high_pass,
        "nonhigh_pass": non_pass,
        "high_pass_rate": float(high_rate),
        "nonhigh_pass_rate": float(non_rate),
        "pass_rate_delta": float(high_rate - non_rate),
        "enrichment_ratio": float(high_rate / max(non_rate, 1e-12)),
    }


def rank_auc_binary(y_true: np.ndarray, score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=float)
    valid = np.isfinite(score)
    y_true = y_true[valid]
    score = score[valid]
    n_pos = int(y_true.sum())
    n_neg = int((~y_true).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").to_numpy()
    pos_rank_sum = float(ranks[y_true].sum())
    return float((pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def percentile_score(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=float)


def run_c2_threshold_sweep(out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not MECH_SCORES.exists():
        raise FileNotFoundError(MECH_SCORES)
    if not BATH_SCORES.exists():
        raise FileNotFoundError(BATH_SCORES)

    mech = pd.read_csv(MECH_SCORES)
    bath = pd.read_csv(BATH_SCORES)
    q_values = np.round(np.arange(0.50, 0.951, 0.05), 2)
    rows = []

    single_metrics = [
        ("bath_resonance_original", "original_mechanistic_signature", mech, "bath_score"),
        ("source_sink_delocalization", "original_mechanistic_signature", mech, "deloc_score"),
        ("spectral_only", "bath_sensitivity_variants", bath, "spectral_only_score"),
        ("bath_x_coupling", "bath_sensitivity_variants", bath, "bath_x_coupling_score"),
        ("bath_x_path", "bath_sensitivity_variants", bath, "bath_x_path_score"),
        ("bath_x_coupling_x_path", "bath_sensitivity_variants", bath, "bath_x_coupling_x_path_score"),
    ]
    for metric, source, df, score_col in single_metrics:
        eta = df["eta"].to_numpy(float)
        score = df[score_col].to_numpy(float)
        for q in q_values:
            threshold = float(np.quantile(score, q))
            row = {
                "source": source,
                "metric": metric,
                "score_column": score_col,
                "quantile": float(q),
                "threshold": threshold,
            }
            row.update(pass_stats(score >= threshold, eta))
            rows.append(row)

    # Joint threshold: both bath_score and deloc_score must pass the same quantile threshold.
    eta = mech["eta"].to_numpy(float)
    bath_score = mech["bath_score"].to_numpy(float)
    deloc_score = mech["deloc_score"].to_numpy(float)
    for q in q_values:
        bath_thr = float(np.quantile(bath_score, q))
        deloc_thr = float(np.quantile(deloc_score, q))
        row = {
            "source": "original_mechanistic_signature",
            "metric": "strict_joint_original",
            "score_column": "bath_score+deloc_score",
            "quantile": float(q),
            "threshold": float("nan"),
            "bath_threshold": bath_thr,
            "deloc_threshold": deloc_thr,
        }
        row.update(pass_stats((bath_score >= bath_thr) & (deloc_score >= deloc_thr), eta))
        rows.append(row)

    sweep = pd.DataFrame(rows)
    sweep.to_csv(out_dir / "c2_threshold_sweep.csv", index=False)

    auc_rows = []
    y_mech = ((mech["eta"].to_numpy(float) >= ETA_HIGH) | (mech["eta"].to_numpy(float) < ETA_NONHIGH))
    label_mech = mech["eta"].to_numpy(float)[y_mech] >= ETA_HIGH
    joint_rank_score = np.minimum(percentile_score(mech["bath_score"].to_numpy(float)), percentile_score(mech["deloc_score"].to_numpy(float)))
    auc_specs = [
        ("bath_resonance_original", "original_mechanistic_signature", mech["bath_score"].to_numpy(float), y_mech, label_mech),
        ("source_sink_delocalization", "original_mechanistic_signature", mech["deloc_score"].to_numpy(float), y_mech, label_mech),
        ("strict_joint_rank_score", "original_mechanistic_signature", joint_rank_score, y_mech, label_mech),
    ]
    y_bath = ((bath["eta"].to_numpy(float) >= ETA_HIGH) | (bath["eta"].to_numpy(float) < ETA_NONHIGH))
    label_bath = bath["eta"].to_numpy(float)[y_bath] >= ETA_HIGH
    for metric in ("spectral_only", "bath_x_coupling", "bath_x_path", "bath_x_coupling_x_path"):
        auc_specs.append((metric, "bath_sensitivity_variants", bath[f"{metric}_score"].to_numpy(float), y_bath, label_bath))
    for metric, source, score, selector, labels in auc_specs:
        auc_rows.append(
            {
                "source": source,
                "metric": metric,
                "auc_high_eta_vs_nonhigh": rank_auc_binary(labels, score[selector]),
                "n_high": int(labels.sum()),
                "n_nonhigh": int((~labels).sum()),
            }
        )
    auc = pd.DataFrame(auc_rows)
    auc.to_csv(out_dir / "c2_score_auc.csv", index=False)
    return sweep, auc


def eta_squared(values: np.ndarray, groups: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups)
    valid = np.isfinite(values) & pd.notna(groups)
    values = values[valid]
    groups = groups[valid]
    if len(values) <= 1:
        return float("nan")
    grand = float(values.mean())
    ss_total = float(np.sum((values - grand) ** 2))
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for g in np.unique(groups):
        v = values[groups == g]
        ss_between += len(v) * (float(v.mean()) - grand) ** 2
    return float(ss_between / ss_total)


def run_d_family_heterogeneity(out_dir: Path, *, seed: int, permutations: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments = pd.read_csv(ASSIGNMENTS).rename(
        columns={
            "real_row": "dataset_index",
            "structural_mode_id": "structural_mode",
            "dynamic_family_id": "dynamic_family",
        }
    )
    assign_cols = ["dataset_index", "priority_group", "structural_mode", "dynamic_family"]
    rng = np.random.default_rng(seed)
    rows = []
    family_rows = []

    for run_dir in sorted(MODEL_DIR.glob("nsf_h27_c*_seed0"), key=lambda p: context_sort_key(infer_context(p.name))):
        csv_path = run_dir / "conditional_mae_samples.csv"
        if not csv_path.exists():
            continue
        context = infer_context(run_dir.name)
        df = pd.read_csv(csv_path)
        df = df[df["method"] == "model"].merge(assignments[assign_cols], on="dataset_index", how="inner")
        for label in LABELS:
            err = df[f"abs_err_{label}"].to_numpy(float)
            groups = df["dynamic_family"].to_numpy(str)
            obs = eta_squared(err, groups)
            perm_vals = np.empty(permutations, dtype=float)
            for b in range(permutations):
                perm_vals[b] = eta_squared(err, rng.permutation(groups))
            p_value = float((1 + np.sum(perm_vals >= obs)) / (permutations + 1))
            rows.append(
                {
                    "context": context,
                    "metric": label,
                    "n_targets": int(len(df)),
                    "eta_squared_dynamic_family": obs,
                    "permutation_p_value": p_value,
                    "permutations": int(permutations),
                    "perm_mean": float(perm_vals.mean()),
                    "perm_p95": float(np.percentile(perm_vals, 95)),
                }
            )

        for (family, label), part in df.melt(
            id_vars=["target_id", "dynamic_family"],
            value_vars=[f"abs_err_{x}" for x in LABELS],
            var_name="metric",
            value_name="abs_error",
        ).groupby(["dynamic_family", "metric"]):
            family_rows.append(
                {
                    "context": context,
                    "dynamic_family": family,
                    "metric": label.replace("abs_err_", ""),
                    "n_targets": int(part["target_id"].nunique()),
                    "model_abs_error_mean": float(part["abs_error"].mean()),
                    "model_abs_error_median": float(part["abs_error"].median()),
                }
            )

    heterogeneity = pd.DataFrame(rows)
    family_error = pd.DataFrame(family_rows)
    heterogeneity.to_csv(out_dir / "d_family_error_heterogeneity.csv", index=False)
    family_error.to_csv(out_dir / "d_family_model_error_by_metric.csv", index=False)
    return heterogeneity, family_error


def plot_c2_threshold_sweep(sweep: pd.DataFrame, out_path: Path) -> None:
    keep = [
        "bath_resonance_original",
        "source_sink_delocalization",
        "strict_joint_original",
        "bath_x_coupling_x_path",
    ]
    colors = {
        "bath_resonance_original": "#4E79A7",
        "source_sink_delocalization": "#59A14F",
        "strict_joint_original": "#E15759",
        "bath_x_coupling_x_path": "#B07AA1",
    }
    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for metric in keep:
        part = sweep[sweep["metric"] == metric].sort_values("quantile")
        ax.plot(
            part["quantile"] * 100,
            part["enrichment_ratio"],
            marker="o",
            linewidth=2.0,
            color=colors.get(metric),
            label=metric.replace("_", " "),
        )
    ax.axhline(1.0, color="#333333", linestyle="--", linewidth=1.2)
    ax.set_xlabel("signature threshold quantile (%)")
    ax.set_ylabel("high-eta / non-high pass-rate ratio")
    ax.set_title("C2 threshold robustness")
    ax.grid(alpha=0.25)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_c2_auc(auc: pd.DataFrame, out_path: Path) -> None:
    df = auc.sort_values("auc_high_eta_vs_nonhigh")
    labels = df["metric"].str.replace("_", " ", regex=False)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    colors = ["#C73E3A" if v < 0.5 else "#2A9D55" for v in df["auc_high_eta_vs_nonhigh"]]
    ax.barh(labels, df["auc_high_eta_vs_nonhigh"], color=colors, alpha=0.88)
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("ROC AUC for high-eta vs non-high")
    ax.set_title("Continuous C2 scores as high-efficiency discriminators")
    ax.grid(axis="x", alpha=0.25)
    for i, v in enumerate(df["auc_high_eta_vs_nonhigh"]):
        ax.text(v + (0.015 if v < 0.92 else -0.08), i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_d_heterogeneity(heterogeneity: pd.DataFrame, out_path: Path) -> None:
    pivot = heterogeneity.pivot(index="context", columns="metric", values="eta_squared_dynamic_family")
    pivot = pivot.reindex(index=[c for c in CONTEXT_ORDER if c in pivot.index], columns=list(LABELS))
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    im = ax.imshow(pivot.to_numpy(float), cmap="YlOrRd", aspect="auto", vmin=0.0)
    ax.set_xticks(np.arange(len(pivot.columns)), labels=pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)), labels=pivot.index)
    ax.set_title("How much model-error variance is explained by D-family labels?")
    ax.set_xlabel("simulator label")
    ax.set_ylabel("context policy")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("eta-squared")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, sweep: pd.DataFrame, auc: pd.DataFrame, heterogeneity: pd.DataFrame, figures: dict[str, Path]) -> None:
    q75 = sweep[np.isclose(sweep["quantile"], 0.75)]
    c2_key = q75[q75["metric"].isin(["bath_resonance_original", "source_sink_delocalization", "strict_joint_original", "bath_x_coupling_x_path"])]
    best_auc = auc.sort_values("auc_high_eta_vs_nonhigh", ascending=False).iloc[0]
    worst_auc = auc.sort_values("auc_high_eta_vs_nonhigh", ascending=True).iloc[0]
    strongest_d = heterogeneity.sort_values("eta_squared_dynamic_family", ascending=False).iloc[0]

    md = []
    md.append("# Supplementary robustness analysis\n")
    md.append("## C2 threshold sensitivity\n")
    md.append(
        "The pass/fail decision for C2 signatures depends on threshold choice. "
        "This sweep varies the threshold from the 50% to 95% quantile and checks whether "
        "the high-eta versus non-high pass-rate ratio remains stable.\n"
    )
    md.append("Main results at the 75% threshold are shown below.\n")
    md.append("\n| metric | high pass rate | non-high pass rate | enrichment |\n")
    md.append("|---|---:|---:|---:|\n")
    for _, row in c2_key.iterrows():
        md.append(
            f"| {row['metric']} | {row['high_pass_rate'] * 100:.1f}% | "
            f"{row['nonhigh_pass_rate'] * 100:.1f}% | {row['enrichment_ratio']:.2f}x |\n"
        )
    md.append(
        "\nBath resonance alone is not consistently enriched in high-eta samples, "
        "whereas source-sink delocalization and bath-coupling-path combined scores "
        "remain more clearly enriched across threshold choices.\n"
    )
    md.append(f"![C2 threshold sweep]({figures['c2_threshold'].relative_to(path.parent).as_posix()})\n")

    md.append("## C2 continuous-score discrimination\n")
    md.append(
        f"Continuous scores are evaluated by ROC AUC for high-eta versus non-high separation. "
        f"The strongest score is `{best_auc['metric']}` (AUC {best_auc['auc_high_eta_vs_nonhigh']:.3f}), "
        f"and the weakest score is `{worst_auc['metric']}` (AUC {worst_auc['auc_high_eta_vs_nonhigh']:.3f}).\n"
    )
    md.append(f"![C2 AUC]({figures['c2_auc'].relative_to(path.parent).as_posix()})\n")

    md.append("## D-family error heterogeneity\n")
    md.append(
        "To check whether D-family labels explain model difficulty, the script computes eta-squared, "
        "the fraction of model absolute-error variance explained by dynamic family. "
        "The p-value is estimated by a permutation test over shuffled D labels.\n"
    )
    md.append(
        f"The strongest effect appears for `{strongest_d['context']}` / `{strongest_d['metric']}`: "
        f"eta-squared {strongest_d['eta_squared_dynamic_family']:.3f}, permutation p-value "
        f"{strongest_d['permutation_p_value']:.4f}.\n"
    )
    md.append(f"![D heterogeneity]({figures['d_heterogeneity'].relative_to(path.parent).as_posix()})\n")
    md.append("## Output files\n")
    md.append("- `c2_threshold_sweep.csv`\n")
    md.append("- `c2_score_auc.csv`\n")
    md.append("- `d_family_error_heterogeneity.csv`\n")
    md.append("- `d_family_model_error_by_metric.csv`\n")
    path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    sweep, auc = run_c2_threshold_sweep(out_dir)
    heterogeneity, family_error = run_d_family_heterogeneity(out_dir, seed=args.seed, permutations=args.d_permutations)

    figures = {
        "c2_threshold": figures_dir / "c2_threshold_sweep_enrichment.png",
        "c2_auc": figures_dir / "c2_score_auc.png",
        "d_heterogeneity": figures_dir / "d_family_error_heterogeneity.png",
    }
    plot_c2_threshold_sweep(sweep, figures["c2_threshold"])
    plot_c2_auc(auc, figures["c2_auc"])
    plot_d_heterogeneity(heterogeneity, figures["d_heterogeneity"])
    write_summary(out_dir / "supplementary_robustness_summary.md", sweep, auc, heterogeneity, figures)
    write_json(
        out_dir / "manifest.json",
        {
            "seed": args.seed,
            "d_permutations": args.d_permutations,
            "inputs": {
                "mechanistic_scores": MECH_SCORES,
                "bath_sensitivity_scores": BATH_SCORES,
                "dynamic_assignments": ASSIGNMENTS,
                "model_performance_dir": MODEL_DIR,
            },
            "outputs": {
                "c2_threshold_sweep": "c2_threshold_sweep.csv",
                "c2_score_auc": "c2_score_auc.csv",
                "d_family_error_heterogeneity": "d_family_error_heterogeneity.csv",
                "d_family_model_error_by_metric": "d_family_model_error_by_metric.csv",
                "summary": "supplementary_robustness_summary.md",
                "figures": {k: str(v.relative_to(out_dir)) for k, v in figures.items()},
            },
        },
    )
    print("[saved]", out_dir)
    print("[C2 AUC]")
    print(auc.sort_values("auc_high_eta_vs_nonhigh", ascending=False).to_string(index=False))
    print("\n[D-family strongest eta-squared]")
    print(heterogeneity.sort_values("eta_squared_dynamic_family", ascending=False).head(10).to_string(index=False))


if __name__ == "__main__":
    main()

