#!/usr/bin/env python3
"""Audit H27 main-model selection with support-filtered inverse-design yield.

This is a post-hoc audit over existing generated H samples and exact simulator
validation rows. It does not rerun the simulator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ASSIGNMENTS = ROOT / "temp/h27_dynamic_distance_diversity_audit_drive/dynamic_distance_assignments.csv"
DEFAULT_SUCCESS_RECHECK = ROOT / "temp/h27_dynamic_distance_diversity_audit_drive/dynamic_distance_success_only_recheck.csv"
DEFAULT_SCALABLE = ROOT / "outputs/experiments/20260622_h27_scalable_reference_diversity_audit/csv"
DEFAULT_PREPARED = ROOT / "data/h27_context_ablation_140k_cnf_prepared.npz"
DEFAULT_OUT = ROOT / "outputs/experiments/20260623_h27_main_model_selection_audit"

TARGETS = ("fast_high", "late_high", "very_fast")
MODEL_RUNS = {
    "HTBAL_CNF_MIXPRIOR": {
        "run_label": "20260622_h27_cnf_mode_prior/full",
        "condition_set": "CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR",
        "generated_npz": ROOT / "temp/h27_drive_model_artifacts/CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR_generated_samples.npz",
        "key_prefix": "CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR_",
    },
    "FLOW_HTBRANCHPINNTRAJ": {
        "run_label": "20260622_h27_flow_htbranchpinntraj/full",
        "condition_set": "CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ",
        "generated_npz": ROOT / "temp/h27_drive_model_artifacts/CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_generated_samples.npz",
        "key_prefix": "CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_",
    },
}
SCALABLE_FILES = {
    "HTBAL_CNF_MIXPRIOR": DEFAULT_SCALABLE
    / "20260622_h27_cnf_mode_prior_full_cfast_orange3_htbal_cnf_mixprior_targetmatch_scalable_reference_assignments.csv",
    "FLOW_HTBRANCHPINNTRAJ": DEFAULT_SCALABLE
    / "20260622_h27_flow_htbranchpinntraj_full_cfast_orange3_flow_htbranchpinntraj_targetmatch_scalable_reference_assignments.csv",
}
NLL_ROWS = [
    {"model": "CNF baseline", "overall_nll": 18.6341, "fast_high_nll": 23.0731, "late_high_nll": 23.5359, "very_fast_nll": 20.9801},
    {"model": "CNF_WMODE", "overall_nll": 18.7602, "fast_high_nll": 23.3311, "late_high_nll": 23.6909, "very_fast_nll": 20.9303},
    {"model": "HTBAL_CNF_MIXPRIOR", "overall_nll": 19.3473, "fast_high_nll": 24.8228, "late_high_nll": 24.2164, "very_fast_nll": 21.9651},
    {"model": "FLOW_HTBRANCHPINNTRAJ", "overall_nll": 21.6917, "fast_high_nll": 31.3686, "late_high_nll": 26.0834, "very_fast_nll": 27.3314},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--success-recheck", type=Path, default=DEFAULT_SUCCESS_RECHECK)
    parser.add_argument("--prepared-npz", type=Path, default=DEFAULT_PREPARED)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    csv_dir = args.out_dir / "csv"
    report_dir = args.out_dir / "reports"
    json_dir = args.out_dir / "json"
    for directory in (csv_dir, report_dir, json_dir):
        directory.mkdir(parents=True, exist_ok=True)

    assignments = pd.read_csv(args.assignments)
    assignments["target_match"] = assignments["target_match"].map(boolish)
    calibration, train_norm, mu, sd, scale = load_hspace_reference(args.prepared_npz)
    per_sample = build_per_sample(assignments, train_norm, mu, sd, scale)
    per_target = summarize_per_target(per_sample, calibration)
    aggregate = summarize_aggregate(per_target)
    selection = build_selection_matrix(per_target, aggregate, args.success_recheck)
    nll = pd.DataFrame(NLL_ROWS)

    per_sample.to_csv(csv_dir / "support_filtered_per_sample.csv", index=False)
    per_target.to_csv(csv_dir / "support_filtered_yield_per_target.csv", index=False)
    aggregate.to_csv(csv_dir / "support_filtered_yield_aggregate.csv", index=False)
    selection.to_csv(csv_dir / "main_model_selection_matrix.csv", index=False)
    nll.to_csv(csv_dir / "nll_context_from_drive_metrics.csv", index=False)
    (json_dir / "manifest.json").write_text(
        json.dumps(
            {
                "assignments": str(args.assignments),
                "prepared_npz": str(args.prepared_npz),
                "out_dir": str(args.out_dir),
                "targets": list(TARGETS),
                "models": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in MODEL_RUNS.items()},
                "heldout_nearest_train_calibration": calibration,
                "outputs": {
                    "per_sample": str(csv_dir / "support_filtered_per_sample.csv"),
                    "per_target": str(csv_dir / "support_filtered_yield_per_target.csv"),
                    "aggregate": str(csv_dir / "support_filtered_yield_aggregate.csv"),
                    "selection_matrix": str(csv_dir / "main_model_selection_matrix.csv"),
                    "report": str(report_dir / "main_model_selection_audit_kr.md"),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_report(report_dir / "main_model_selection_audit_kr.md", calibration, per_target, aggregate, selection, nll)
    print(report_dir / "main_model_selection_audit_kr.md")
    return 0


def boolish(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def load_hspace_reference(path: Path):
    prepared = np.load(path)
    train_idx = np.asarray(prepared["split_train"], dtype=np.int64)
    heldout_idx = np.concatenate(
        [
            np.asarray(prepared["split_val"], dtype=np.int64),
            np.asarray(prepared["split_test"], dtype=np.int64),
        ]
    )[:4096]
    h_all = np.asarray(prepared["H_params_28"], dtype=np.float32)
    train_h = h_all[train_idx]
    heldout_h = h_all[heldout_idx]
    mu = train_h.mean(axis=0)
    sd = train_h.std(axis=0)
    sd[sd < 1e-6] = 1.0
    train_norm = ((train_h - mu) / sd).astype(np.float32)
    heldout_norm = ((heldout_h - mu) / sd).astype(np.float32)
    scale = float(np.sqrt(train_norm.shape[1]))
    heldout_d = nearest_normalized_distance(heldout_norm, train_norm, scale)
    calibration = {
        "heldout_n": int(len(heldout_d)),
        "heldout_p10": float(np.quantile(heldout_d, 0.10)),
        "heldout_p50": float(np.quantile(heldout_d, 0.50)),
        "heldout_p75": float(np.quantile(heldout_d, 0.75)),
        "heldout_p90": float(np.quantile(heldout_d, 0.90)),
        "heldout_p95": float(np.quantile(heldout_d, 0.95)),
    }
    return calibration, train_norm, mu, sd, scale


def build_per_sample(assignments: pd.DataFrame, train_norm: np.ndarray, mu: np.ndarray, sd: np.ndarray, scale: float) -> pd.DataFrame:
    rows = []
    for model, spec in MODEL_RUNS.items():
        generated = np.load(spec["generated_npz"])
        model_assign = assignments[
            (assignments["run_label"] == spec["run_label"])
            & (assignments["condition_set"] == spec["condition_set"])
            & (assignments["target"].isin(TARGETS))
        ].copy()
        for target in TARGETS:
            key = f"{spec['key_prefix']}{target}_H_vec28_trace_zero"
            h = np.asarray(generated[key], dtype=np.float32)
            h_norm = ((h - mu) / sd).astype(np.float32)
            d = nearest_normalized_distance(h_norm, train_norm, scale)
            target_assign = model_assign[model_assign["target"] == target].set_index("source_generated_index")
            scalable = load_scalable_rows(model, target)
            for generated_index in range(len(h)):
                row = {
                    "model": model,
                    "target": target,
                    "generated_index": generated_index,
                    "nearest_train_norm": float(d[generated_index]),
                    "validated": generated_index in target_assign.index,
                    "target_match": False,
                    "eta20": np.nan,
                    "t80": np.nan,
                    "dynamic_cluster": "",
                    "dynamic_family_id": "",
                    "nearest_reference_distance": np.nan,
                }
                if row["validated"]:
                    hit = target_assign.loc[generated_index]
                    row.update(
                        {
                            "target_match": boolish(hit["target_match"]),
                            "eta20": float(hit["eta20"]),
                            "t80": float(hit["t80"]),
                            "dynamic_cluster": str(hit["dynamic_cluster"]),
                        }
                    )
                if generated_index in scalable.index:
                    ref = scalable.loc[generated_index]
                    row["dynamic_family_id"] = str(ref["dynamic_family_id"])
                    row["nearest_reference_distance"] = float(ref["nearest_reference_distance"])
                rows.append(row)
    return pd.DataFrame(rows)


def load_scalable_rows(model: str, target: str) -> pd.DataFrame:
    path = SCALABLE_FILES[model]
    if not path.exists():
        return pd.DataFrame().rename_axis("generated_index")
    df = pd.read_csv(path)
    df = df[df["target"] == target].copy()
    if "generated_index" in df.columns:
        idx_col = "generated_index"
    elif "source_generated_index" in df.columns:
        idx_col = "source_generated_index"
    else:
        return pd.DataFrame().rename_axis("generated_index")
    return df.set_index(idx_col)


def summarize_per_target(per_sample: pd.DataFrame, calibration: dict[str, float]) -> pd.DataFrame:
    rows = []
    for (model, target), part in per_sample.groupby(["model", "target"], sort=False):
        n_budget = int(len(part))
        target_match = part["target_match"].astype(bool)
        support_p50 = part["nearest_train_norm"] <= calibration["heldout_p50"]
        support_p90 = part["nearest_train_norm"] <= calibration["heldout_p90"]
        support_p95 = part["nearest_train_norm"] <= calibration["heldout_p95"]
        copy_like = part["nearest_train_norm"] < calibration["heldout_p10"]
        target_support_p90 = target_match & support_p90
        target_support_p95 = target_match & support_p95
        family_counts = part.loc[target_support_p90, "dynamic_family_id"].replace("", np.nan).dropna().value_counts()
        rows.append(
            {
                "model": model,
                "target": target,
                "n_budget": n_budget,
                "n_validated": int(part["validated"].sum()),
                "n_target_match": int(target_match.sum()),
                "target_match_per_budget": float(target_match.mean()),
                "n_target_and_support_p90": int(target_support_p90.sum()),
                "target_and_support_p90_yield": float(target_support_p90.sum() / n_budget),
                "n_target_and_support_p95": int(target_support_p95.sum()),
                "target_and_support_p95_yield": float(target_support_p95.sum() / n_budget),
                "support_p50_fraction_all": float(support_p50.mean()),
                "support_p90_fraction_all": float(support_p90.mean()),
                "copy_like_fraction_all": float(copy_like.mean()),
                "nearest_train_norm_median": float(part["nearest_train_norm"].median()),
                "nearest_train_norm_p90": float(part["nearest_train_norm"].quantile(0.90)),
                "eta20_median_targetmatch": finite_median(part.loc[target_match, "eta20"]),
                "t80_median_targetmatch": finite_median(part.loc[target_match, "t80"]),
                "support_p90_dynamic_family_count": int(len(family_counts)),
                "support_p90_top_family_fraction": float(family_counts.iloc[0] / max(1, target_support_p90.sum()))
                if len(family_counts)
                else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["target_order"] = out["target"].map({target: i for i, target in enumerate(TARGETS)})
    return out.sort_values(["model", "target_order"]).drop(columns=["target_order"]).reset_index(drop=True)


def summarize_aggregate(per_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in per_target.groupby("model", sort=False):
        rows.append(
            {
                "model": model,
                "n_budget_total": int(part["n_budget"].sum()),
                "n_target_match_total": int(part["n_target_match"].sum()),
                "target_match_per_budget_mean": float(part["target_match_per_budget"].mean()),
                "target_match_per_budget_min": float(part["target_match_per_budget"].min()),
                "target_and_support_p90_yield_mean": float(part["target_and_support_p90_yield"].mean()),
                "target_and_support_p90_yield_min": float(part["target_and_support_p90_yield"].min()),
                "target_and_support_p95_yield_mean": float(part["target_and_support_p95_yield"].mean()),
                "support_p90_fraction_all_mean": float(part["support_p90_fraction_all"].mean()),
                "nearest_train_norm_p90_mean": float(part["nearest_train_norm_p90"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_selection_matrix(per_target: pd.DataFrame, aggregate: pd.DataFrame, success_recheck_path: Path) -> pd.DataFrame:
    rows = []
    mix = aggregate[aggregate["model"] == "HTBAL_CNF_MIXPRIOR"].iloc[0]
    flow = aggregate[aggregate["model"] == "FLOW_HTBRANCHPINNTRAJ"].iloc[0]
    rows.append(
        {
            "criterion": "exact-high target-match mean",
            "HTBAL_CNF_MIXPRIOR": mix["target_match_per_budget_mean"],
            "FLOW_HTBRANCHPINNTRAJ": flow["target_match_per_budget_mean"],
            "reading": "FLOW higher on mean; MIXPRIOR higher on late_high.",
        }
    )
    rows.append(
        {
            "criterion": "minimum target-match across exact-high targets",
            "HTBAL_CNF_MIXPRIOR": mix["target_match_per_budget_min"],
            "FLOW_HTBRANCHPINNTRAJ": flow["target_match_per_budget_min"],
            "reading": "MIXPRIOR has higher floor because FLOW drops on late_high.",
        }
    )
    rows.append(
        {
            "criterion": "support-p90 filtered target-match mean",
            "HTBAL_CNF_MIXPRIOR": mix["target_and_support_p90_yield_mean"],
            "FLOW_HTBRANCHPINNTRAJ": flow["target_and_support_p90_yield_mean"],
            "reading": "Counts target-match only when nearest train-H is within heldout p90.",
        }
    )
    rows.append(
        {
            "criterion": "support-p90 filtered target-match floor",
            "HTBAL_CNF_MIXPRIOR": mix["target_and_support_p90_yield_min"],
            "FLOW_HTBRANCHPINNTRAJ": flow["target_and_support_p90_yield_min"],
            "reading": "Lower value marks target where inverse-design yield is weakest after support filtering.",
        }
    )
    rows.append(
        {
            "criterion": "H support p90 distance mean",
            "HTBAL_CNF_MIXPRIOR": mix["nearest_train_norm_p90_mean"],
            "FLOW_HTBRANCHPINNTRAJ": flow["nearest_train_norm_p90_mean"],
            "reading": "Lower means fewer high-distance generated samples.",
        }
    )
    if success_recheck_path.exists():
        success = pd.read_csv(success_recheck_path)
        for target in TARGETS:
            vals = {}
            for model, spec in MODEL_RUNS.items():
                hit = success[
                    (success["run_label"] == spec["run_label"])
                    & (success["condition_set"] == spec["condition_set"])
                    & (success["target"] == target)
                ]
                vals[model] = float(hit["targetmatch_existing_cluster_entropy_norm"].iloc[0]) if len(hit) else np.nan
            rows.append(
                {
                    "criterion": f"success dynamic-cluster entropy {target}",
                    "HTBAL_CNF_MIXPRIOR": vals["HTBAL_CNF_MIXPRIOR"],
                    "FLOW_HTBRANCHPINNTRAJ": vals["FLOW_HTBRANCHPINNTRAJ"],
                    "reading": "Higher means success samples are less concentrated in the existing dynamic clusters.",
                }
            )
    return pd.DataFrame(rows)


def nearest_normalized_distance(query: np.ndarray, reference: np.ndarray, scale: float, batch_size: int = 128) -> np.ndarray:
    reference_sq = np.sum(reference * reference, axis=1)
    mins = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        d2 = np.sum(batch * batch, axis=1)[:, None] + reference_sq[None, :] - 2.0 * batch.dot(reference.T)
        d2 = np.maximum(d2, 0.0)
        mins.append(np.sqrt(d2.min(axis=1)) / scale)
    return np.concatenate(mins).astype(np.float32)


def finite_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    return float(vals.median()) if len(vals) else np.nan


def write_report(
    path: Path,
    calibration: dict[str, float],
    per_target: pd.DataFrame,
    aggregate: pd.DataFrame,
    selection: pd.DataFrame,
    nll: pd.DataFrame,
) -> None:
    lines = [
        "# H27 Main Model Selection Audit",
        "",
        "## Purpose",
        "",
        "This audit combines exact simulator target-match and H-space support checks for generated samples.",
        "It summarizes existing generated-H validation outputs and does not rerun the simulator.",
        "",
        "## Heldout H-space calibration",
        "",
        df_to_md(pd.DataFrame([calibration]), float_cols=list(calibration.keys())),
        "",
        "Distance definition: nearest-train Euclidean distance after 28D H normalization by train mean/std, divided by sqrt(28).",
        "",
        "## Density/NLL context",
        "",
        df_to_md(nll, float_cols=["overall_nll", "fast_high_nll", "late_high_nll", "very_fast_nll"]),
        "",
        "Lower NLL is better. In this table, the baseline CNF has better density fit, while FLOW is weaker by this criterion.",
        "",
        "## Support-filtered inverse-design yield by target",
        "",
        df_to_md(
            per_target[
                [
                    "model",
                    "target",
                    "n_budget",
                    "n_target_match",
                    "target_match_per_budget",
                    "n_target_and_support_p90",
                    "target_and_support_p90_yield",
                    "n_target_and_support_p95",
                    "target_and_support_p95_yield",
                    "support_p90_fraction_all",
                    "nearest_train_norm_p90",
                    "support_p90_dynamic_family_count",
                    "support_p90_top_family_fraction",
                ]
            ],
            float_cols=[
                "target_match_per_budget",
                "target_and_support_p90_yield",
                "target_and_support_p95_yield",
                "support_p90_fraction_all",
                "nearest_train_norm_p90",
                "support_p90_top_family_fraction",
            ],
        ),
        "",
        "## Aggregate",
        "",
        df_to_md(
            aggregate,
            float_cols=[
                "target_match_per_budget_mean",
                "target_match_per_budget_min",
                "target_and_support_p90_yield_mean",
                "target_and_support_p90_yield_min",
                "target_and_support_p95_yield_mean",
                "support_p90_fraction_all_mean",
                "nearest_train_norm_p90_mean",
            ],
        ),
        "",
        "## Selection matrix",
        "",
        df_to_md(
            selection,
            float_cols=["HTBAL_CNF_MIXPRIOR", "FLOW_HTBRANCHPINNTRAJ"],
        ),
        "",
        "## Decision note",
        "",
        "FLOW_HTBRANCHPINNTRAJ is stronger in exact-high mean target-match and fast_high/very_fast yield.",
        "However, MIXPRIOR is more conservative for late_high floor, NLL, and H-space support tail.",
        "If FLOW is used as the main model, the claim shifts from CNF mixture prior to trajectory-branch guided generation.",
        "With the current evidence, FLOW can be presented as the strongest yield candidate, but not as a universally superior model without additional robustness checks.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def df_to_md(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    out = df.copy()
    for col in float_cols or []:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    headers = [str(c) for c in out.columns]
    lines = [
        "| " + " | ".join(markdown_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in out.iterrows():
        lines.append("| " + " | ".join(markdown_cell(row[col]) for col in out.columns) + " |")
    return "\n".join(lines)


def markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())

