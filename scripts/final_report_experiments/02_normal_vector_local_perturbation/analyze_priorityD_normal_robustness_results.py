#!/usr/bin/env python3
"""Analyze completed Priority D normal-vector robustness outputs."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = ROOT / "outputs/bridge_normal_robustness_priorityD_20260613"
DEFAULT_OUT = ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out = make_dirs(resolve(args.out_dir))
    detail = load_completed_details(resolve(args.results_root))
    if detail.empty:
        raise ValueError(f"No completed normal_robustness_detail.csv found under {resolve(args.results_root)}")
    radius_summary = summarize_radius(detail)
    lambda_range_summary = summarize_lambda_range(detail)
    pair_summary = summarize_pair(detail, radius_summary)
    claim_table = build_claim_table(pair_summary, lambda_range_summary)
    radius_summary.to_csv(out["csv"] / "normal_robustness_radius_summary.csv", index=False, encoding="utf-8-sig")
    lambda_range_summary.to_csv(out["csv"] / "normal_robustness_lambda_range_summary.csv", index=False, encoding="utf-8-sig")
    pair_summary.to_csv(out["csv"] / "normal_robustness_pair_summary.csv", index=False, encoding="utf-8-sig")
    claim_table.to_csv(out["csv"] / "normal_robustness_claim_evidence_caveat.csv", index=False, encoding="utf-8-sig")
    write_report(out["reports"] / "priorityD_normal_robustness_analysis.md", detail, radius_summary, lambda_range_summary, pair_summary, claim_table)
    print(f"detail rows: {len(detail)}")
    print(f"completed groups: {detail['result_group'].nunique()}")
    print(f"out: {out['root']}")
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {"root": root, "csv": root / "csv", "reports": root / "reports"}
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_completed_details(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("*/csv/normal_robustness_detail.csv")):
        df = pd.read_csv(path)
        df["result_group"] = path.parents[1].name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def finite(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def support_fraction(group: pd.DataFrame, levels: set[str]) -> float:
    vals = group["support_level"].astype(str)
    if len(vals) == 0:
        return math.nan
    return finite(vals.isin(levels).mean())


def summarize_radius(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    group_cols = ["result_group", "pair_id", "target_priority", "target_category", "group_name", "radius"]
    for keys, group in detail.groupby(group_cols, sort=False):
        rec = dict(zip(group_cols, keys))
        eta = group["eta20"].to_numpy(float)
        rec.update(
            {
                "n_jobs": int(len(group)),
                "n_solver_success": int(group["solver_success"].astype(bool).sum()),
                "n_alpha": int(group["alpha"].nunique()),
                "n_lambda": int(group["lambda_reorg"].nunique()),
                "n_directions": int(group["normal_direction_i"].nunique()),
                "eta20_mean": finite(np.nanmean(eta)),
                "eta20_std": finite(np.nanstd(eta)),
                "eta20_min": finite(np.nanmin(eta)),
                "eta20_max": finite(np.nanmax(eta)),
                "eta20_q10": finite(np.nanquantile(eta, 0.10)),
                "eta20_q50": finite(np.nanquantile(eta, 0.50)),
                "eta20_q90": finite(np.nanquantile(eta, 0.90)),
                "support_high_medium_fraction": support_fraction(group, {"high", "medium"}),
                "support_low_or_better_fraction": support_fraction(group, {"high", "medium", "low"}),
                "support_levels": ",".join(sorted({str(x) for x in group["support_level"].dropna().unique() if str(x)})),
                "nearest_support_distance_median": finite(np.nanmedian(group["nearest_support_distance"].to_numpy(float))),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize_pair(detail: pd.DataFrame, radius_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (result_group, pair_id), group in detail.groupby(["result_group", "pair_id"], sort=False):
        base = group[group["radius"].astype(float).eq(0.0)].copy()
        pert = group[group["radius"].astype(float).gt(0.0)].copy()
        supported_pert = pert[pert["support_level"].astype(str).isin(["high", "medium", "low"])].copy()
        rows.append(
            {
                "result_group": result_group,
                "pair_id": pair_id,
                "target_priority": int(group["target_priority"].iloc[0]),
                "target_category": group["target_category"].iloc[0],
                "group_name": group["group_name"].iloc[0],
                "n_jobs": int(len(group)),
                "n_solver_success": int(group["solver_success"].astype(bool).sum()),
                "baseline_eta20_mean": finite(base["eta20"].mean()),
                "baseline_eta20_min": finite(base["eta20"].min()),
                "baseline_eta20_max": finite(base["eta20"].max()),
                "perturbed_eta20_mean": finite(pert["eta20"].mean()),
                "perturbed_eta20_min": finite(pert["eta20"].min()),
                "perturbed_eta20_max": finite(pert["eta20"].max()),
                "supported_perturbed_eta20_mean": finite(supported_pert["eta20"].mean()),
                "supported_perturbed_eta20_min": finite(supported_pert["eta20"].min()),
                "supported_perturbed_eta20_max": finite(supported_pert["eta20"].max()),
                "delta_perturbed_mean_vs_baseline_mean": finite(pert["eta20"].mean() - base["eta20"].mean()),
                "delta_supported_perturbed_mean_vs_baseline_mean": finite(supported_pert["eta20"].mean() - base["eta20"].mean()),
                "perturbed_support_high_medium_fraction": support_fraction(pert, {"high", "medium"}),
                "perturbed_support_low_or_better_fraction": support_fraction(pert, {"high", "medium", "low"}),
                "max_radius_with_any_high_medium": max_radius_with_support(pert, {"high", "medium"}),
                "max_radius_with_any_low_or_better": max_radius_with_support(pert, {"high", "medium", "low"}),
            }
        )
    return pd.DataFrame(rows)


def summarize_lambda_range(detail: pd.DataFrame) -> pd.DataFrame:
    range_rows: list[dict[str, Any]] = []
    cols = ["result_group", "pair_id", "target_priority", "target_category", "group_name", "radius", "alpha", "normal_direction_i"]
    for keys, group in detail.groupby(cols, sort=False):
        eta = group["eta20"].to_numpy(float)
        range_rows.append(
            {
                **dict(zip(cols, keys)),
                "lambda_count": int(group["lambda_reorg"].nunique()),
                "eta20_range_across_lambda": finite(np.nanmax(eta) - np.nanmin(eta)),
                "eta20_best_lambda": finite(group.iloc[int(np.nanargmax(eta))]["lambda_reorg"]),
                "eta20_worst_lambda": finite(group.iloc[int(np.nanargmin(eta))]["lambda_reorg"]),
                "support_level": ",".join(sorted({str(x) for x in group["support_level"].dropna().unique() if str(x)})),
            }
        )
    ranges = pd.DataFrame(range_rows)
    rows: list[dict[str, Any]] = []
    summary_cols = ["result_group", "pair_id", "target_priority", "target_category", "group_name", "radius"]
    for keys, group in ranges.groupby(summary_cols, sort=False):
        vals = group["eta20_range_across_lambda"].to_numpy(float)
        rows.append(
            {
                **dict(zip(summary_cols, keys)),
                "n_alpha_direction": int(len(group)),
                "lambda_range_mean": finite(np.nanmean(vals)),
                "lambda_range_median": finite(np.nanmedian(vals)),
                "lambda_range_q90": finite(np.nanquantile(vals, 0.90)),
                "lambda_range_max": finite(np.nanmax(vals)),
                "best_lambda_values": ",".join(str(float(x)).rstrip("0").rstrip(".") for x in sorted(group["eta20_best_lambda"].dropna().unique())),
                "support_levels": ",".join(sorted({str(x) for x in group["support_level"].dropna().unique() if str(x)})),
            }
        )
    return pd.DataFrame(rows)


def max_radius_with_support(df: pd.DataFrame, levels: set[str]) -> float:
    sub = df[df["support_level"].astype(str).isin(levels)]
    if sub.empty:
        return math.nan
    return finite(sub["radius"].astype(float).max())


def build_claim_table(pair_summary: pd.DataFrame, lambda_range_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in pair_summary.iterrows():
        lr = lambda_range_summary[lambda_range_summary["pair_id"].eq(row["pair_id"])].copy()
        baseline_lr = lr[lr["radius"].astype(float).eq(0.0)]
        supported_lr = lr[lr["radius"].astype(float).eq(0.25)]
        baseline_range = finite(baseline_lr["lambda_range_median"].iloc[0]) if not baseline_lr.empty else math.nan
        supported_range = finite(supported_lr["lambda_range_median"].iloc[0]) if not supported_lr.empty else math.nan
        if int(row["target_priority"]) == 1:
            claim = "Straight-path lambda-dramatic behavior is locally sensitive: nearby supported perturbations often raise eta and change the response scale."
            evidence = (
                f"Baseline eta20 mean={row['baseline_eta20_mean']:.3f}; "
                f"perturbed mean={row['perturbed_eta20_mean']:.3f}; "
                f"supported perturbed mean={row['supported_perturbed_eta20_mean']:.3f}; "
                f"median lambda eta-range radius0={baseline_range:.3f}, radius0.25={supported_range:.3f}; "
                f"max radius with low-or-better support={row['max_radius_with_any_low_or_better']}."
            )
            caveat = (
                "This supports a path-sensitivity/local-neighborhood question more than a mechanism proof. "
                "Many larger-radius perturbations become sparse support; interpret them as off-path sensitivity probes. "
                "This still does not provide rho/eigenstate-resolved contribution."
            )
        elif int(row["target_priority"]) == 2 and str(row["target_category"]) == "high_high_stable_combined_cluster":
            claim = "High-high stable target can be evaluated as functional stability under local perturbation, but the two pairs differ in direction."
            delta = finite(row["delta_supported_perturbed_mean_vs_baseline_mean"])
            direction = "increases" if delta > 0 else "decreases"
            evidence = (
                f"Baseline eta20 mean={row['baseline_eta20_mean']:.3f}; "
                f"supported perturbed mean={row['supported_perturbed_eta20_mean']:.3f}; "
                f"supported perturbation {direction} mean eta by {delta:.3f}; "
                f"low-or-better support fraction={row['perturbed_support_low_or_better_fraction']:.3f}."
            )
            caveat = (
                "Functional stability should be judged by supported radius 0.25 first. "
                "Sparse larger-radius perturbations are sensitivity probes. "
                "A high supported mean does not prove eigenstate invariance."
            )
        elif int(row["target_priority"]) == 2 and str(row["target_category"]) == "low_high_transition_combined_cluster":
            claim = "Low-high transition targets are locally sensitive: supported perturbations tend to raise eta relative to the straight path."
            evidence = (
                f"Baseline eta20 mean={row['baseline_eta20_mean']:.3f}; "
                f"supported perturbed mean={row['supported_perturbed_eta20_mean']:.3f}; "
                f"supported gain={row['delta_supported_perturbed_mean_vs_baseline_mean']:.3f}; "
                f"max low-or-better support radius={row['max_radius_with_any_low_or_better']}."
            )
            caveat = (
                "This supports a local-neighborhood transition/sensitivity reading, not a completed mechanism. "
                "The support-limited fraction differs by pair, so compare pair-level support before generalizing."
            )
        else:
            claim = "Normal robustness result is available for this target."
            evidence = f"Perturbed eta20 mean={row['perturbed_eta20_mean']:.3f}."
            caveat = "Use target-specific Priority D question before mechanism wording."
        rows.append(
            {
                "result_group": row["result_group"],
                "pair_id": row["pair_id"],
                "target_priority": int(row["target_priority"]),
                "claim": claim,
                "evidence": evidence,
                "caveat": caveat,
            }
        )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    sub = df.copy()
    for col in sub.columns:
        sub[col] = sub[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(sub.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    detail: pd.DataFrame,
    radius_summary: pd.DataFrame,
    lambda_range_summary: pd.DataFrame,
    pair_summary: pd.DataFrame,
    claim_table: pd.DataFrame,
) -> None:
    lines = [
        "# Priority D Normal-Vector Robustness Analysis",
        "",
        "이 보고서는 현재 완료된 normal-vector robustness 결과만 분석한다. 아직 완료되지 않은 priority group은 포함하지 않는다.",
        "",
        "## Completed Scope",
        "",
        f"- completed result groups: `{detail['result_group'].nunique()}`",
        f"- detail rows: `{len(detail)}`",
        f"- solver successes: `{int(detail['solver_success'].astype(bool).sum())}`",
        "",
        "## Pair Summary",
        "",
        markdown_table(pair_summary.round(6)),
        "",
        "## Claim / Evidence / Caveat",
        "",
        markdown_table(claim_table),
        "",
        "## Lambda Range By Radius",
        "",
        markdown_table(lambda_range_summary.round(6)),
        "",
        "## Radius Summary Preview",
        "",
        markdown_table(radius_summary.head(40).round(6)),
        "",
        "## Interpretation Notes",
        "",
        "- Radius is measured in `tangent-fraction` units unless the run plan says otherwise.",
        "- Larger radii often move into sparse support, so they are sensitivity probes rather than evidence for sampled-design robustness.",
        "- For lambda-dramatic targets, this analysis can show whether eta/path response is locally fragile or persistent, but it still cannot name eigenstate-resolved contribution without `rho_t` or a rerun with contribution storage.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
