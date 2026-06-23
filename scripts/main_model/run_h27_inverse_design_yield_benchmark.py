#!/usr/bin/env python3
"""Build a local inverse-design yield benchmark from validated H27 samples.

This is a post-hoc benchmark over already generated and exact-simulator
validated samples. It does not train a model or rerun the simulator. The goal is
to quantify whether a model can produce valid target-matching Hamiltonian
candidates under a fixed generation/simulation budget.
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
DEFAULT_SCALABLE_DIR = ROOT / "outputs/experiments/20260622_h27_scalable_reference_diversity_audit/csv"
DEFAULT_PREPARED_NPZ = ROOT / "data/h27_context_ablation_140k_cnf_prepared.npz"
DEFAULT_OUT = ROOT / "outputs/experiments/20260623_h27_inverse_design_yield_benchmark"
EXACT_HIGH_TARGETS = ("fast_high", "late_high", "very_fast")
CORE_MODELS = {
    "CNF baseline": ("20260622_h27_cnf_mode_prior/full", "CFAST_ORANGE3_CNF"),
    "CNF_WMODE": ("20260622_h27_cnf_mode_prior/full", "CFAST_ORANGE3_CNF_WMODE"),
    "HTBAL_CNF_MIXPRIOR": ("20260622_h27_cnf_mode_prior/full", "CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR"),
    "FLOW_HTBRANCHPINNTRAJ": (
        "20260622_h27_flow_htbranchpinntraj/full",
        "CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ",
    ),
}
DEFAULT_GENERATED_H_NPZS = {
    "HTBAL_CNF_MIXPRIOR": ROOT
    / "temp/h27_drive_model_artifacts/CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR_generated_samples.npz",
    "FLOW_HTBRANCHPINNTRAJ": ROOT
    / "temp/h27_drive_model_artifacts/CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_generated_samples.npz",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument("--success-recheck", type=Path, default=DEFAULT_SUCCESS_RECHECK)
    parser.add_argument("--scalable-dir", type=Path, default=DEFAULT_SCALABLE_DIR)
    parser.add_argument("--prepared-npz", type=Path, default=DEFAULT_PREPARED_NPZ)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--targets", default=",".join(EXACT_HIGH_TARGETS))
    parser.add_argument("--budget", type=int, default=512, help="Nominal generated samples per target.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv_dir = args.out_dir / "csv"
    report_dir = args.out_dir / "reports"
    json_dir = args.out_dir / "json"
    for path in (csv_dir, report_dir, json_dir):
        path.mkdir(exist_ok=True)

    targets = tuple(x.strip() for x in args.targets.split(",") if x.strip())
    assignments = pd.read_csv(args.assignments)
    assignments["target_match"] = assignments["target_match"].map(boolish)
    assignments["model_key"] = assignments["run_label"] + " :: " + assignments["condition_set"]
    exact = assignments[assignments["target"].isin(targets)].copy()

    per_target = build_per_target(exact, args.budget)
    aggregate = build_aggregate(per_target)
    core = add_core_labels(per_target)
    core_aggregate = add_core_labels(aggregate)

    success_recheck = pd.read_csv(args.success_recheck) if args.success_recheck.exists() else pd.DataFrame()
    success_recheck = success_recheck[success_recheck["target"].isin(targets)].copy() if not success_recheck.empty else success_recheck
    success_recheck_core = add_core_labels(success_recheck) if not success_recheck.empty else success_recheck

    scalable_summary = build_scalable_summary(args.scalable_dir, targets)
    hspace_novelty = build_hspace_novelty(args.prepared_npz, DEFAULT_GENERATED_H_NPZS)

    per_target.to_csv(csv_dir / "inverse_design_yield_per_target_all_models.csv", index=False)
    aggregate.to_csv(csv_dir / "inverse_design_yield_aggregate_all_models.csv", index=False)
    core.to_csv(csv_dir / "inverse_design_yield_per_target_core_models.csv", index=False)
    core_aggregate.to_csv(csv_dir / "inverse_design_yield_aggregate_core_models.csv", index=False)
    if not success_recheck_core.empty:
        success_recheck_core.to_csv(csv_dir / "success_only_dynamic_cluster_core_models.csv", index=False)
    if not scalable_summary.empty:
        scalable_summary.to_csv(csv_dir / "targetmatch_scalable_reference_core_models.csv", index=False)
    if not hspace_novelty.empty:
        hspace_novelty.to_csv(csv_dir / "hspace_nearest_train_novelty_core_models.csv", index=False)

    manifest = {
        "assignments": str(args.assignments),
        "success_recheck": str(args.success_recheck),
        "scalable_dir": str(args.scalable_dir),
        "prepared_npz": str(args.prepared_npz),
        "generated_h_npzs": {k: str(v) for k, v in DEFAULT_GENERATED_H_NPZS.items()},
        "out_dir": str(args.out_dir),
        "targets": list(targets),
        "nominal_budget_per_target": int(args.budget),
        "outputs": {
            "per_target_all": str(csv_dir / "inverse_design_yield_per_target_all_models.csv"),
            "aggregate_all": str(csv_dir / "inverse_design_yield_aggregate_all_models.csv"),
            "per_target_core": str(csv_dir / "inverse_design_yield_per_target_core_models.csv"),
            "aggregate_core": str(csv_dir / "inverse_design_yield_aggregate_core_models.csv"),
            "hspace_novelty": str(csv_dir / "hspace_nearest_train_novelty_core_models.csv"),
            "report": str(report_dir / "inverse_design_yield_benchmark_kr.md"),
        },
    }
    (json_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(
        report_dir / "inverse_design_yield_benchmark_kr.md",
        core,
        core_aggregate,
        success_recheck_core,
        scalable_summary,
        hspace_novelty,
        args,
    )
    print(report_dir / "inverse_design_yield_benchmark_kr.md")
    return 0


def boolish(value) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def model_label(run_label: str, condition_set: str) -> str | None:
    key = (str(run_label), str(condition_set))
    for label, pair in CORE_MODELS.items():
        if key == pair:
            return label
    return None


def add_core_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    out["model"] = [model_label(r, c) for r, c in zip(out["run_label"], out["condition_set"])]
    out = out[out["model"].notna()].copy()
    if "target" in out.columns:
        out["target_order"] = out["target"].map({t: i for i, t in enumerate(EXACT_HIGH_TARGETS)})
        out = out.sort_values(["model", "target_order"]).drop(columns=["target_order"], errors="ignore")
    else:
        out = out.sort_values(["model"])
    return out


def build_per_target(df: pd.DataFrame, budget: int) -> pd.DataFrame:
    rows = []
    grouped = df.groupby(["run_label", "condition_set", "target"], dropna=False)
    for (run_label, condition_set, target), part in grouped:
        success = part[part["target_match"]].copy()
        n = int(len(part))
        n_success = int(len(success))
        row = {
            "run_label": run_label,
            "condition_set": condition_set,
            "target": target,
            "n_generated_or_validated": n,
            "n_target_match": n_success,
            "target_match_rate": n_success / max(1, n),
            "valid_designs_per_512": n_success / max(1, n) * float(budget),
        }
        for col in ("eta10", "eta20", "eta50", "t80"):
            row[f"{col}_median_all"] = finite_median(part[col])
            row[f"{col}_median_targetmatch"] = finite_median(success[col])
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["run_label", "condition_set", "target"]).reset_index(drop=True)


def build_aggregate(per_target: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = per_target.groupby(["run_label", "condition_set"], dropna=False)
    for (run_label, condition_set), part in grouped:
        rows.append(
            {
                "run_label": run_label,
                "condition_set": condition_set,
                "n_generated_or_validated": int(part["n_generated_or_validated"].sum()),
                "n_target_match": int(part["n_target_match"].sum()),
                "mean_target_match_rate": float(part["target_match_rate"].mean()),
                "min_target_match_rate": float(part["target_match_rate"].min()),
                "valid_designs_per_512_mean": float(part["valid_designs_per_512"].mean()),
                "eta20_median_targetmatch_mean": float(part["eta20_median_targetmatch"].mean()),
                "t80_median_targetmatch_mean": float(part["t80_median_targetmatch"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_target_match_rate", ascending=False).reset_index(drop=True)


def finite_median(series: pd.Series) -> float:
    vals = pd.to_numeric(series, errors="coerce").dropna()
    if vals.empty:
        return float("nan")
    return float(vals.median())


def build_scalable_summary(scalable_dir: Path, targets: tuple[str, ...]) -> pd.DataFrame:
    files = {
        "HTBAL_CNF_MIXPRIOR": scalable_dir
        / "20260622_h27_cnf_mode_prior_full_cfast_orange3_htbal_cnf_mixprior_targetmatch_scalable_reference_assignments.csv",
        "FLOW_HTBRANCHPINNTRAJ": scalable_dir
        / "20260622_h27_flow_htbranchpinntraj_full_cfast_orange3_flow_htbranchpinntraj_targetmatch_scalable_reference_assignments.csv",
    }
    rows = []
    for model, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[df["target"].isin(targets)].copy()
        for target, part in df.groupby("target"):
            counts = part["dynamic_family_id"].value_counts()
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "n_targetmatch_assigned": int(len(part)),
                    "dynamic_family_count": int(part["dynamic_family_id"].nunique()),
                    "top_dynamic_family_id": str(counts.index[0]) if len(counts) else "",
                    "top_dynamic_family_fraction": float(counts.iloc[0] / len(part)) if len(part) else float("nan"),
                    "nearest_reference_distance_median": finite_median(part["nearest_reference_distance"]),
                    "nearest_reference_distance_p90": float(pd.to_numeric(part["nearest_reference_distance"], errors="coerce").quantile(0.90)),
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    out["target_order"] = out["target"].map({t: i for i, t in enumerate(EXACT_HIGH_TARGETS)})
    return out.sort_values(["model", "target_order"]).drop(columns=["target_order"]).reset_index(drop=True)


def build_hspace_novelty(prepared_npz: Path, generated_h_npzs: dict[str, Path]) -> pd.DataFrame:
    if not prepared_npz.exists():
        return pd.DataFrame()
    available = {model: path for model, path in generated_h_npzs.items() if path.exists()}
    if not available:
        return pd.DataFrame()

    prepared = np.load(prepared_npz)
    train_idx = np.asarray(prepared["split_train"], dtype=np.int64)
    heldout_idx = np.concatenate(
        [
            np.asarray(prepared["split_val"], dtype=np.int64),
            np.asarray(prepared["split_test"], dtype=np.int64),
        ]
    )[:4096]
    train_h = np.asarray(prepared["H_params_28"], dtype=np.float32)[train_idx]
    heldout_h = np.asarray(prepared["H_params_28"], dtype=np.float32)[heldout_idx]
    mu = train_h.mean(axis=0)
    sd = train_h.std(axis=0)
    sd[sd < 1e-6] = 1.0
    train_norm = ((train_h - mu) / sd).astype(np.float32)
    heldout_norm = ((heldout_h - mu) / sd).astype(np.float32)
    scale = float(np.sqrt(train_norm.shape[1]))

    heldout_d = nearest_normalized_distance(heldout_norm, train_norm, scale)
    heldout_median = float(np.median(heldout_d))
    heldout_p10 = float(np.quantile(heldout_d, 0.10))
    heldout_p90 = float(np.quantile(heldout_d, 0.90))

    rows = []
    for model, path in available.items():
        generated = np.load(path)
        for key in generated.files:
            target = generated_target_from_key(key)
            if not target:
                continue
            h = np.asarray(generated[key], dtype=np.float32)
            h_norm = ((h - mu) / sd).astype(np.float32)
            d = nearest_normalized_distance(h_norm, train_norm, scale)
            rows.append(
                {
                    "model": model,
                    "target": target,
                    "n_generated": int(len(h)),
                    "nearest_train_norm_median": float(np.median(d)),
                    "nearest_train_norm_p10": float(np.quantile(d, 0.10)),
                    "nearest_train_norm_p90": float(np.quantile(d, 0.90)),
                    "fraction_below_heldout_p10": float((d < heldout_p10).mean()),
                    "fraction_below_heldout_median": float((d < heldout_median).mean()),
                    "heldout_n": int(len(heldout_d)),
                    "heldout_nn_norm_median": heldout_median,
                    "heldout_nn_norm_p10": heldout_p10,
                    "heldout_nn_norm_p90": heldout_p90,
                }
            )
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame(rows)
    target_order = {target: i for i, target in enumerate(
        [
            "fast_high",
            "fast_high_mixture",
            "late_high",
            "late_high_mixture",
            "very_fast",
            "very_fast_mixture",
            "non_high",
            "non_high_mixture",
        ]
    )}
    out["target_order"] = out["target"].map(target_order).fillna(999).astype(int)
    return out.sort_values(["model", "target_order"]).drop(columns=["target_order"]).reset_index(drop=True)


def nearest_normalized_distance(query: np.ndarray, reference: np.ndarray, scale: float, batch_size: int = 128) -> np.ndarray:
    reference_sq = np.sum(reference * reference, axis=1)
    mins = []
    for start in range(0, len(query), batch_size):
        batch = query[start : start + batch_size]
        d2 = np.sum(batch * batch, axis=1)[:, None] + reference_sq[None, :] - 2.0 * batch.dot(reference.T)
        d2 = np.maximum(d2, 0.0)
        mins.append(np.sqrt(d2.min(axis=1)) / scale)
    return np.concatenate(mins).astype(np.float32)


def generated_target_from_key(key: str) -> str | None:
    suffix = "_H_vec28_trace_zero"
    if not key.endswith(suffix):
        return None
    stem = key[: -len(suffix)]
    prefixes = [
        "CFAST_ORANGE3_HTBAL_CNF_MIXPRIOR_",
        "CFAST_ORANGE3_FLOW_HTBRANCHPINNTRAJ_",
    ]
    for prefix in prefixes:
        if stem.startswith(prefix):
            return stem[len(prefix) :]
    return None


def write_report(
    path: Path,
    core: pd.DataFrame,
    core_aggregate: pd.DataFrame,
    success_recheck: pd.DataFrame,
    scalable_summary: pd.DataFrame,
    hspace_novelty: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    mix = core_aggregate[core_aggregate["model"] == "HTBAL_CNF_MIXPRIOR"].iloc[0]
    cnf = core_aggregate[core_aggregate["model"] == "CNF baseline"].iloc[0]
    wmode = core_aggregate[core_aggregate["model"] == "CNF_WMODE"].iloc[0]
    flow = core_aggregate[core_aggregate["model"] == "FLOW_HTBRANCHPINNTRAJ"].iloc[0]
    lines = [
        "# H27 Inverse-Design Yield Benchmark",
        "",
        "## Purpose",
        "",
        "This benchmark compares generated samples that have already been validated by the exact simulator.",
        "The main question is not NLL, but how many Hamiltonian candidates satisfy the requested target condition.",
        "",
        "## Inputs",
        "",
        f"- assignments: `{args.assignments}`",
        f"- targets: `{', '.join(args.targets.split(','))}`",
        f"- nominal budget: `{args.budget}` generated samples per target",
        "- Note: this script summarizes existing generated/validated outputs; it does not require rerunning training.",
        "",
        "## Aggregate exact-high yield",
        "",
        df_to_md(
            core_aggregate[
                [
                    "model",
                    "n_generated_or_validated",
                    "n_target_match",
                    "mean_target_match_rate",
                    "min_target_match_rate",
                    "valid_designs_per_512_mean",
                    "eta20_median_targetmatch_mean",
                    "t80_median_targetmatch_mean",
                ]
            ],
            float_cols=[
                "mean_target_match_rate",
                "min_target_match_rate",
                "valid_designs_per_512_mean",
                "eta20_median_targetmatch_mean",
                "t80_median_targetmatch_mean",
            ],
        ),
        "",
        "Interpretation:",
        "",
        f"- `HTBAL_CNF_MIXPRIOR` improves mean target-match over CNF baseline by `{100*(mix['mean_target_match_rate']-cnf['mean_target_match_rate']):.2f}` percentage points.",
        f"- `HTBAL_CNF_MIXPRIOR` improves mean target-match over CNF_WMODE by `{100*(mix['mean_target_match_rate']-wmode['mean_target_match_rate']):.2f}` percentage points.",
        f"- `FLOW_HTBRANCHPINNTRAJ` improves aggregate mean target-match over `HTBAL_CNF_MIXPRIOR` by `{100*(flow['mean_target_match_rate']-mix['mean_target_match_rate']):.2f}` percentage points.",
        "- FLOW is stronger on aggregate yield, while MIXPRIOR remains useful as a conservative comparison model.",
        "",
        "## Per-target exact-high yield",
        "",
        df_to_md(
            core[
                [
                    "model",
                    "target",
                    "n_generated_or_validated",
                    "n_target_match",
                    "target_match_rate",
                    "valid_designs_per_512",
                    "eta20_median_targetmatch",
                    "t80_median_targetmatch",
                ]
            ],
            float_cols=[
                "target_match_rate",
                "valid_designs_per_512",
                "eta20_median_targetmatch",
                "t80_median_targetmatch",
            ],
        ),
        "",
        "## Successful-sample reference-family spread",
        "",
    ]
    if scalable_summary.empty:
        lines += ["scalable reference assignment CSV媛 ?놁뼱 ?앸왂?덈떎.", ""]
    else:
        lines += [
            df_to_md(
                scalable_summary,
                float_cols=[
                    "top_dynamic_family_fraction",
                    "nearest_reference_distance_median",
                    "nearest_reference_distance_p90",
                ],
            ),
            "",
        ]
    lines += [
        "## Dynamic-distance cluster caveat for successful samples",
        "",
    ]
    if success_recheck.empty:
        lines += ["success-only dynamic cluster recheck CSV is missing, so this section is skipped.", ""]
    else:
        cols = [
            "model",
            "target",
            "n_target_match",
            "target_match_rate",
            "targetmatch_largest_existing_cluster_fraction",
            "targetmatch_existing_cluster_entropy_norm",
            "targetmatch_cluster_counts",
        ]
        available_cols = [c for c in cols if c in success_recheck.columns]
        lines += [
            df_to_md(
                success_recheck[available_cols],
                float_cols=[
                    "target_match_rate",
                    "targetmatch_largest_existing_cluster_fraction",
                    "targetmatch_existing_cluster_entropy_norm",
                ],
            ),
            "",
        ]
    lines += [
        "## H-space nearest-train novelty/support",
        "",
    ]
    if hspace_novelty.empty:
        lines += ["generated H NPZ or prepared NPZ is missing, so this section is skipped.", ""]
    else:
        hspace_exact = hspace_novelty[hspace_novelty["target"].isin(args.targets.split(","))].copy()
        lines += [
            df_to_md(
                hspace_exact[
                    [
                        "model",
                        "target",
                        "n_generated",
                        "nearest_train_norm_median",
                        "nearest_train_norm_p10",
                        "nearest_train_norm_p90",
                        "fraction_below_heldout_p10",
                        "fraction_below_heldout_median",
                        "heldout_nn_norm_median",
                    ]
                ],
                float_cols=[
                    "nearest_train_norm_median",
                    "nearest_train_norm_p10",
                    "nearest_train_norm_p90",
                    "fraction_below_heldout_p10",
                    "fraction_below_heldout_median",
                    "heldout_nn_norm_median",
                ],
            ),
            "",
            "Interpretation: very low nearest-train distance may indicate train-copy behavior, while very high distance may indicate samples outside empirical support. This is only a support check, not a replacement for simulator target-match.",
            "",
        ]
    lines += [
        "## Conclusion",
        "",
        "`HTBAL_CNF_MIXPRIOR` improves exact-high inverse-design yield over CNF baseline and CNF_WMODE, and is stronger on late_high.",
        "`FLOW_HTBRANCHPINNTRAJ` is strongest on fast_high, very_fast, and aggregate yield.",
        "The evidence therefore supports a comparison rather than a single universal winner.",
        "If the report emphasizes CNF mixture prior, `HTBAL_CNF_MIXPRIOR` should be the main CNF model and FLOW should be shown as a comparison model.",
        "If final model choice is based purely on yield, FLOW is the strongest candidate, but robustness and top-K checks should still be reported.",
        "",
        "For a stronger inverse-design claim, top-K ranking, lambda/noise robustness, and target-conditioned candidate-budget scaling should be evaluated under the same budget.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def df_to_md(df: pd.DataFrame, float_cols: list[str] | None = None) -> str:
    out = df.copy()
    for col in float_cols or []:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.4f}")
    headers = [str(c) for c in out.columns]
    rows = []
    for _, row in out.iterrows():
        rows.append([markdown_cell(row[col]) for col in out.columns])
    lines = [
        "| " + " | ".join(markdown_cell(h) for h in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def markdown_cell(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


if __name__ == "__main__":
    raise SystemExit(main())

