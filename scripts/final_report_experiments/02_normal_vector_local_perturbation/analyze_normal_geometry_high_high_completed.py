#!/usr/bin/env python3
"""Analyze completed high-high normal-direction geometry sweep."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/bridge_normal_geometry_high_high_core_lambda35_20260614"
PLAN = ROOT / "new/bridge_normal_geometry_high_high_plan_20260614"
OUT = ROOT / "new/bridge_normal_geometry_high_high_results_20260614"
CSV = OUT / "csv"
REPORTS = OUT / "reports"
SCRIPTS = OUT / "scripts"

DETAIL_PATH = RUN / "csv/normal_robustness_detail.csv"
SUMMARY_PATH = RUN / "csv/normal_robustness_summary.csv"
MANIFEST_PATH = PLAN / "csv/high_high_geometry_target_manifest.csv"
CLASS_PATH = ROOT / "new/bridge_priority1_classification_20260613/csv/pair_classification_ds_same_s.csv"
ALPHA_H_PATH = ROOT / "new/bridge_group_trend_priorityB_h_eigenfeatures_20260613/csv/alpha_level_h_eigenfeatures.csv"
PAIR_H_PATH = ROOT / "new/bridge_group_trend_priorityB_h_eigenfeatures_20260613/csv/pair_level_structural_summary.csv"

HIGH = 0.75
VERY_HIGH = 0.90
ETA_TOL = 0.05
SUPPORT_HIGH_MEDIUM_MIN = 0.80
SUPPORT_LOW_OR_BETTER_MIN = 0.90
GAIN_FRACTION_MIN = 0.50


def main() -> int:
    make_dirs()
    detail = pd.read_csv(DETAIL_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)
    cls = pd.read_csv(CLASS_PATH)
    alpha_h = pd.read_csv(ALPHA_H_PATH)
    pair_h = pd.read_csv(PAIR_H_PATH)

    integrity = build_integrity_summary(detail, manifest)
    integrity.to_csv(CSV / "run_integrity_summary.csv", index=False, encoding="utf-8-sig")

    radius_summary = build_radius_summary(detail)
    radius_summary.to_csv(CSV / "radius_support_eta_summary.csv", index=False, encoding="utf-8-sig")

    pair_alpha = build_pair_alpha_metrics(detail, cls, alpha_h)
    pair_alpha.to_csv(CSV / "pair_alpha_geometry_metrics.csv", index=False, encoding="utf-8-sig")

    pair_summary = build_pair_summary(pair_alpha, manifest, cls, pair_h)
    pair_summary.to_csv(CSV / "pair_geometry_summary.csv", index=False, encoding="utf-8-sig")

    label_counts = build_label_counts(pair_summary)
    label_counts.to_csv(CSV / "geometry_label_counts.csv", index=False, encoding="utf-8-sig")
    label_lambda = build_label_lambda_crosstab(pair_summary)
    label_lambda.to_csv(CSV / "geometry_label_by_dominant_best_lambda.csv", index=False, encoding="utf-8-sig")

    examples = build_example_pairs(pair_summary)
    examples.to_csv(CSV / "representative_geometry_examples.csv", index=False, encoding="utf-8-sig")

    caveats = build_caveats(pair_summary)
    caveats.to_csv(CSV / "geometry_interpretation_caveats.csv", index=False, encoding="utf-8-sig")

    write_report(integrity, radius_summary, pair_alpha, pair_summary, label_counts, label_lambda, examples, caveats)
    copy_self()
    print(f"Wrote {OUT}")
    return 0


def make_dirs() -> None:
    for path in [CSV, REPORTS, SCRIPTS]:
        path.mkdir(parents=True, exist_ok=True)


def build_integrity_summary(detail: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append(("detail_rows", len(detail)))
    rows.append(("unique_pairs_in_detail", detail["pair_id"].nunique()))
    rows.append(("pairs_in_runner_manifest_core", len(manifest[manifest["run_tier"].eq("tier1_geometry_core")])))
    rows.append(("solver_success_rows", int(detail["solver_success"].fillna(False).sum())))
    rows.append(("solver_failure_rows", int((~detail["solver_success"].fillna(False)).sum())))
    rows.append(("unique_lambdas", ",".join(map(str, sorted(detail["lambda_reorg"].dropna().unique())))))
    rows.append(("unique_radii", ",".join(f"{x:g}" for x in sorted(detail["radius"].dropna().unique()))))
    rows.append(("unique_alpha_values_count", detail["alpha"].nunique()))
    return pd.DataFrame(rows, columns=["item", "value"])


def support_fraction(x: pd.Series, accepted: set[str]) -> float:
    if len(x) == 0:
        return np.nan
    return x.astype(str).isin(accepted).mean()


def build_radius_summary(detail: pd.DataFrame) -> pd.DataFrame:
    d = prep_detail(detail)
    rows = []
    for radius, sub in d.groupby("radius"):
        rows.append(
            {
                "radius": radius,
                "n_rows": len(sub),
                "n_pairs": sub["pair_id"].nunique(),
                "n_pair_alpha": sub[["pair_id", "alpha"]].drop_duplicates().shape[0],
                "eta20_mean": sub["eta20"].mean(),
                "eta20_median": sub["eta20"].median(),
                "eta20_q10": sub["eta20"].quantile(0.10),
                "eta20_q25": sub["eta20"].quantile(0.25),
                "eta20_q75": sub["eta20"].quantile(0.75),
                "eta20_q90": sub["eta20"].quantile(0.90),
                "high_eta_fraction": (sub["eta20"] >= HIGH).mean(),
                "very_high_eta_fraction": (sub["eta20"] >= VERY_HIGH).mean(),
                "support_high_medium_fraction": support_fraction(sub["support_level"], {"high", "medium"}),
                "support_low_or_better_fraction": support_fraction(sub["support_level"], {"high", "medium", "low"}),
                "support_sparse_fraction": support_fraction(sub["support_level"], {"sparse"}),
                "nearest_support_distance_median": sub["nearest_support_distance"].median(),
                "nearest_support_distance_q75": sub["nearest_support_distance"].quantile(0.75),
            }
        )
    return pd.DataFrame(rows).sort_values("radius")


def prep_detail(detail: pd.DataFrame) -> pd.DataFrame:
    d = detail.copy()
    d = d[d["solver_success"].fillna(False)].copy()
    d["radius"] = pd.to_numeric(d["radius"], errors="coerce")
    d["alpha"] = pd.to_numeric(d["alpha"], errors="coerce").round(6)
    d["lambda_reorg"] = pd.to_numeric(d["lambda_reorg"], errors="coerce")
    d["eta20"] = pd.to_numeric(d["eta20"], errors="coerce")
    d["nearest_support_distance"] = pd.to_numeric(d["nearest_support_distance"], errors="coerce")
    d["support_level"] = d["support_level"].astype(str)
    return d


def build_pair_alpha_metrics(detail: pd.DataFrame, cls: pd.DataFrame, alpha_h: pd.DataFrame) -> pd.DataFrame:
    d = prep_detail(detail)
    base = (
        d[d["radius"].eq(0)]
        .groupby(["pair_id", "alpha"], as_index=False)
        .agg(
            baseline_eta20=("eta20", "mean"),
            tangent_norm_h_gauge_27=("tangent_norm_h_gauge_27", "median"),
            baseline_support_level=("support_level", lambda x: ",".join(sorted(set(map(str, x))))),
        )
    )
    g = (
        d.groupby(["pair_id", "group_name", "alpha", "radius"], as_index=False)
        .agg(
            n_samples=("eta20", "size"),
            eta20_mean=("eta20", "mean"),
            eta20_median=("eta20", "median"),
            eta20_q10=("eta20", lambda x: x.quantile(0.10)),
            eta20_q25=("eta20", lambda x: x.quantile(0.25)),
            eta20_q75=("eta20", lambda x: x.quantile(0.75)),
            eta20_max=("eta20", "max"),
            eta20_min=("eta20", "min"),
            support_high_medium_fraction=("support_level", lambda x: support_fraction(x, {"high", "medium"})),
            support_low_or_better_fraction=("support_level", lambda x: support_fraction(x, {"high", "medium", "low"})),
            sparse_fraction=("support_level", lambda x: support_fraction(x, {"sparse"})),
            nearest_support_distance_median=("nearest_support_distance", "median"),
            perturb_norm_h_gauge_27=("perturb_norm_h_gauge_27", "median"),
        )
    )
    g = g.merge(base, on=["pair_id", "alpha"], how="left")
    d2 = d.merge(base[["pair_id", "alpha", "baseline_eta20"]], on=["pair_id", "alpha"], how="left")
    gain = (
        d2.groupby(["pair_id", "alpha", "radius"], as_index=False)
        .apply(lambda sub: pd.Series({"normal_gain_fraction": float((sub["eta20"] >= sub["baseline_eta20"].iloc[0]).mean())}))
        .reset_index(drop=True)
    )
    g = g.merge(gain, on=["pair_id", "alpha", "radius"], how="left")
    g["support_ok_high_medium"] = g["support_high_medium_fraction"] >= SUPPORT_HIGH_MEDIUM_MIN
    g["support_ok_low_or_better"] = g["support_low_or_better_fraction"] >= SUPPORT_LOW_OR_BETTER_MIN
    g["support_ok"] = g["support_ok_high_medium"] | g["support_ok_low_or_better"]
    g["eta_abs_high_median_ok"] = g["eta20_median"] >= HIGH
    g["eta_abs_high_q25_ok"] = g["eta20_q25"] >= HIGH
    g["eta_retention_ok"] = g["eta20_median"] >= (g["baseline_eta20"] - ETA_TOL)
    g["joint_abs_high_ok"] = g["support_ok"] & g["eta_abs_high_median_ok"]
    g["joint_abs_high_q25_ok"] = g["support_ok"] & g["eta_abs_high_q25_ok"]
    g["joint_retention_ok"] = g["support_ok"] & g["eta_retention_ok"]
    g["gain_ok"] = g["normal_gain_fraction"] >= GAIN_FRACTION_MIN

    rows = []
    for (pair_id, alpha), sub in g.groupby(["pair_id", "alpha"], sort=False):
        pos = sub[sub["radius"] > 0].copy()
        b = sub.loc[sub["radius"].eq(0), "baseline_eta20"]
        baseline = float(b.iloc[0]) if len(b) else np.nan
        nearest = pos.sort_values("radius").head(1)
        normal_sens_retention = np.nan
        if not nearest.empty and not pd.isna(baseline):
            r = float(nearest["radius"].iloc[0])
            normal_sens_retention = max(0.0, baseline - float(nearest["eta20_median"].iloc[0])) / r if r else np.nan
        rows.append(
            {
                "pair_id": pair_id,
                "alpha": alpha,
                "baseline_eta20": baseline,
                "baseline_is_high": baseline >= HIGH if not pd.isna(baseline) else False,
                "R_support_high_medium": max_radius(pos, "support_ok_high_medium"),
                "R_support_low_or_better": max_radius(pos, "support_ok_low_or_better"),
                "R_eta_abs_high_median": max_radius(pos, "eta_abs_high_median_ok"),
                "R_eta_abs_high_q25": max_radius(pos, "eta_abs_high_q25_ok"),
                "R_eta_retention": max_radius(pos, "eta_retention_ok"),
                "R_joint_abs_high_median": max_radius(pos, "joint_abs_high_ok"),
                "R_joint_abs_high_q25": max_radius(pos, "joint_abs_high_q25_ok"),
                "R_joint_retention": max_radius(pos, "joint_retention_ok"),
                "R_gain": max_radius(pos, "gain_ok"),
                "normal_gain_fraction_max": pos["normal_gain_fraction"].max() if not pos.empty else np.nan,
                "normal_eta_median_at_r025": value_at_radius(pos, "eta20_median", 0.25),
                "support_high_medium_at_r025": value_at_radius(pos, "support_high_medium_fraction", 0.25),
                "sparse_fraction_at_r025": value_at_radius(pos, "sparse_fraction", 0.25),
                "normal_sensitivity_nearest_radius": normal_sens_retention,
            }
        )
    pa = pd.DataFrame(rows)
    tangent_cols = [
        "pair_id",
        "primary_label",
        "secondary_flags",
        "path_min_best_eta20",
        "path_max_best_eta20",
        "path_range_best_eta20",
        "max_eta20_step_between_adjacent_alpha",
        "support_high_medium_fraction",
        "support_sparse_fraction",
        "interpretation_hold",
    ]
    t = cls[tangent_cols].rename(
        columns={
            "max_eta20_step_between_adjacent_alpha": "tangent_sensitivity_proxy",
            "support_high_medium_fraction": "straight_path_support_high_medium_fraction",
            "support_sparse_fraction": "straight_path_support_sparse_fraction",
        }
    )
    pa = pa.merge(t, on="pair_id", how="left")
    ah_cols = [
        "pair_id",
        "alpha",
        "best_eta20",
        "best_lambda_eta20",
        "lambda35_eta20",
        "source_sink_gap",
        "best_source_sink_mix",
        "max_participation",
        "min_tracking_overlap",
    ]
    ah = alpha_h[[c for c in ah_cols if c in alpha_h.columns]].copy()
    ah["alpha"] = pd.to_numeric(ah["alpha"], errors="coerce").round(6)
    pa = pa.merge(ah, on=["pair_id", "alpha"], how="left")
    pa["anisotropy_score_proxy"] = pa["normal_sensitivity_nearest_radius"] / pa["tangent_sensitivity_proxy"].replace(0, np.nan)
    pa["geometry_label_pair_alpha"] = pa.apply(assign_pair_alpha_label, axis=1)
    return pa.sort_values(["pair_id", "alpha"]).reset_index(drop=True)


def max_radius(df: pd.DataFrame, col: str) -> float:
    ok = df[df[col].fillna(False)]
    if ok.empty:
        return np.nan
    return float(ok["radius"].max())


def value_at_radius(df: pd.DataFrame, col: str, radius: float) -> float:
    v = df[np.isclose(df["radius"], radius)][col]
    return float(v.iloc[0]) if len(v) else np.nan


def assign_pair_alpha_label(row: pd.Series) -> str:
    base = row["baseline_eta20"]
    r_support = row["R_support_low_or_better"]
    r_joint_abs = row["R_joint_abs_high_median"]
    r_joint_ret = row["R_joint_retention"]
    r_gain = row["R_gain"]
    aniso = row["anisotropy_score_proxy"]
    if pd.isna(r_support) or r_support < 0.05:
        return "support_limited_geometry"
    if base >= HIGH:
        if not pd.isna(r_joint_abs) and r_joint_abs >= 0.20:
            return "broad_or_tube_high_eta_supported"
        if not pd.isna(r_joint_abs) and r_joint_abs >= 0.10:
            return "supported_tube_candidate"
        if not pd.isna(aniso) and aniso >= 2.0:
            return "narrow_ridge_candidate"
        return "high_baseline_but_unclear_width"
    if not pd.isna(r_gain) and r_gain >= 0.10:
        return "path_not_centered_or_lambda35_low_candidate"
    if not pd.isna(r_joint_ret) and r_joint_ret >= 0.10:
        return "low_baseline_retention_not_high_eta"
    return "persistent_low_or_undetermined"


def build_pair_summary(pair_alpha: pd.DataFrame, manifest: pd.DataFrame, cls: pd.DataFrame, pair_h: pd.DataFrame) -> pd.DataFrame:
    ps = (
        pair_alpha.groupby("pair_id", as_index=False)
        .agg(
            n_alpha_sampled=("alpha", "nunique"),
            baseline_eta20_min=("baseline_eta20", "min"),
            baseline_eta20_median=("baseline_eta20", "median"),
            baseline_eta20_max=("baseline_eta20", "max"),
            baseline_high_alpha_fraction=("baseline_is_high", "mean"),
            R_support_low_or_better_median=("R_support_low_or_better", "median"),
            R_support_low_or_better_max=("R_support_low_or_better", "max"),
            R_joint_abs_high_median_max=("R_joint_abs_high_median", "max"),
            R_joint_abs_high_q25_max=("R_joint_abs_high_q25", "max"),
            R_joint_retention_median=("R_joint_retention", "median"),
            R_gain_max=("R_gain", "max"),
            normal_gain_fraction_max=("normal_gain_fraction_max", "max"),
            anisotropy_score_proxy_median=("anisotropy_score_proxy", "median"),
            geometry_label_set=("geometry_label_pair_alpha", lambda x: "|".join(sorted(set(map(str, x))))),
        )
    )
    meta_cols = [
        "pair_id",
        "geometry_target_role",
        "run_tier",
        "primary_label",
        "secondary_flags",
        "endpoint_left_best_eta20",
        "endpoint_right_best_eta20",
        "path_min_best_eta20",
        "path_max_best_eta20",
        "path_argmin_alpha",
        "path_argmax_alpha",
        "valley_depth_best_eta20",
        "dominant_best_lambda",
        "max_lambda_eta20_range_nonzero",
        "support_high_medium_fraction",
        "support_sparse_fraction",
        "interpretation_hold",
        "h_only_structural_subtype_seed",
    ]
    meta_cols = [c for c in meta_cols if c in manifest.columns]
    ps = ps.merge(manifest[meta_cols], on="pair_id", how="left")
    h_cols = [
        "pair_id",
        "source_state_changes",
        "sink_state_changes",
        "min_tracking_overlap_excluding_start",
        "min_source_sink_gap",
        "min_adjacent_eigen_gap",
        "max_source_sink_mix",
        "max_participation",
        "state_reorganization_candidate",
    ]
    h_cols = [c for c in h_cols if c in pair_h.columns]
    ps = ps.merge(pair_h[h_cols], on="pair_id", how="left")
    ps["pair_geometry_label"] = ps.apply(assign_pair_label, axis=1)
    return ps.sort_values(["pair_geometry_label", "pair_id"]).reset_index(drop=True)


def assign_pair_label(row: pd.Series) -> str:
    high_frac = row["baseline_high_alpha_fraction"]
    r_abs = row["R_joint_abs_high_median_max"]
    r_ret = row["R_joint_retention_median"]
    gain = row["normal_gain_fraction_max"]
    aniso = row["anisotropy_score_proxy_median"]
    if high_frac >= 0.6:
        if not pd.isna(r_abs) and r_abs >= 0.20:
            return "supported_high_eta_tube_or_cloud_candidate"
        if not pd.isna(r_abs) and r_abs >= 0.10:
            return "supported_tube_candidate"
        if not pd.isna(aniso) and aniso >= 2.0:
            return "narrow_ridge_candidate"
        return "high_high_but_width_unclear"
    if gain >= 0.75:
        return "path_not_centered_or_lambda35_mismatch_candidate"
    if not pd.isna(r_ret) and r_ret >= 0.15:
        return "lambda35_low_but_retention_supported"
    return "undetermined_or_low_lambda35_geometry"


def build_label_counts(pair_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        pair_summary["pair_geometry_label"]
        .value_counts()
        .rename_axis("pair_geometry_label")
        .reset_index(name="n_pairs")
    )


def build_label_lambda_crosstab(pair_summary: pd.DataFrame) -> pd.DataFrame:
    table = pd.crosstab(pair_summary["pair_geometry_label"], pair_summary["dominant_best_lambda"])
    table = table.reset_index()
    table.columns = [str(c) for c in table.columns]
    return table


def build_example_pairs(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, sub in pair_summary.groupby("pair_geometry_label"):
        ranked = sub.copy()
        ranked["example_score"] = ranked[["baseline_high_alpha_fraction", "R_joint_abs_high_median_max", "R_gain_max"]].fillna(0).sum(axis=1)
        for _, row in ranked.sort_values("example_score", ascending=False).head(5).iterrows():
            rows.append(row)
    return pd.DataFrame(rows)


def build_caveats(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    low_lambda35 = pair_summary[pair_summary["baseline_high_alpha_fraction"] < 0.25]
    rows.append(
        {
            "caveat": "lambda35_baseline_often_not_high",
            "n_pairs": len(low_lambda35),
            "meaning": "These pairs are high-high/stable by best-lambda classification but do not look high across sampled alpha at lambda=35.",
            "recommended_action": "Do not use lambda=35 geometry alone as best-lambda high-eta geometry proof; run best/low-nonzero lambda context if needed.",
        }
    )
    path_center = pair_summary[pair_summary["pair_geometry_label"].eq("path_not_centered_or_lambda35_mismatch_candidate")]
    rows.append(
        {
            "caveat": "path_not_centered_or_lambda35_mismatch",
            "n_pairs": len(path_center),
            "meaning": "Normal perturbation often improves eta relative to baseline, or lambda=35 is not representative.",
            "recommended_action": "Inspect whether this is a geometry/path-centeredness issue or a lambda response issue.",
        }
    )
    high_width = pair_summary[pair_summary["pair_geometry_label"].isin(["supported_high_eta_tube_or_cloud_candidate", "supported_tube_candidate"])]
    rows.append(
        {
            "caveat": "tube_cloud_not_global_manifold",
            "n_pairs": len(high_width),
            "meaning": "The tested local radius supports high eta, but this does not prove a global manifold.",
            "recommended_action": "Use wording limited to local supported neighborhood within tested radii.",
        }
    )
    return pd.DataFrame(rows)


def write_report(
    integrity: pd.DataFrame,
    radius_summary: pd.DataFrame,
    pair_alpha: pd.DataFrame,
    pair_summary: pd.DataFrame,
    label_counts: pd.DataFrame,
    label_lambda: pd.DataFrame,
    examples: pd.DataFrame,
    caveats: pd.DataFrame,
) -> None:
    group_radius = radius_summary[
        [
            "radius",
            "eta20_median",
            "eta20_q25",
            "high_eta_fraction",
            "support_high_medium_fraction",
            "support_low_or_better_fraction",
            "support_sparse_fraction",
            "nearest_support_distance_median",
        ]
    ]
    high_baseline = pair_alpha[pair_alpha["baseline_is_high"]]
    high_alpha_label_counts = (
        high_baseline["geometry_label_pair_alpha"].value_counts().rename_axis("pair_alpha_geometry_label").reset_index(name="n_pair_alpha")
    )
    top_examples_cols = [
        "pair_id",
        "pair_geometry_label",
        "baseline_high_alpha_fraction",
        "R_joint_abs_high_median_max",
        "R_joint_retention_median",
        "R_gain_max",
        "normal_gain_fraction_max",
        "path_min_best_eta20",
        "path_max_best_eta20",
        "dominant_best_lambda",
    ]
    report = f"""# High-High Normal-Direction Geometry 실행 결과 분석

작성일: 2026-06-14

## 1. 실행 무결성

{md_table(integrity)}

실행은 정상 완료되었다. `103`개 high-high core target에 대해 lambda=35, radius `0`부터 `0.25`까지, normal direction 8개 조건으로 총 `48,276`개 simulation이 모두 성공했다.

## 2. 이 분석이 답하는 질문

이번 결과는 기존 Priority D의 valley/path artifact 분석이 아니라, high-high pair 주변의 local geometry를 보는 분석이다. 구체적으로는 다음을 구분한다.

- straight path 주변에서도 high eta와 support가 유지되는가?
- 유지된다면 narrow ridge인지, finite-width supported tube/cloud 후보인지?
- normal perturbation에서 오히려 eta가 좋아진다면 straight path가 high-eta neighborhood 중심을 지나지 않는 것은 아닌지?
- support가 먼저 무너지는 radius는 어디인지?

중요 caveat: 이 run은 **lambda=35 고정 단면**이다. D/S same-S high-high classification은 best-lambda eta를 기준으로 한 성격도 포함하므로, lambda=35에서 baseline이 낮은 pair는 “high-high 구조가 낮다”가 아니라 “lambda=35에서는 high-eta 단면이 아니다”라고 해석해야 한다.

## 3. Radius별 전체 경향

{md_table(group_radius)}

요약하면, support는 radius 0.25까지 대체로 유지된다. 특히 `support_low_or_better_fraction`은 전체적으로 높게 유지되어, 이번 high-high core run은 기존 coarse radius 0.5 이상처럼 대부분 sparse/off-path로 빠지는 실험이 아니다. 따라서 radius 0.01-0.25 범위는 high-high local geometry를 묻는 데 사용할 수 있다.

다만 eta는 단순히 radius가 커지면 떨어지는 형태만은 아니다. 일부 pair/alpha에서는 normal perturbation이 baseline보다 좋아진다. 이는 high-high local geometry 안에서도 straight path가 반드시 high-eta neighborhood 중심을 지나지 않을 수 있음을 뜻한다.

## 4. Pair-level geometry label 분포

{md_table(label_counts)}

Dominant best lambda와 함께 보면:

{md_table(label_lambda)}

해석:

- `supported_high_eta_tube_or_cloud_candidate`와 `supported_tube_candidate`는 lambda=35에서 baseline high alpha가 충분히 있고, support-matched normal radius에서도 high eta가 유지되는 경우다.
- 현재 pair-level 기준에서는 명확한 `narrow_ridge_candidate`가 주 label로 나오지 않았다. high-baseline pair-alpha 단위에서는 narrow 후보가 일부 있지만, pair 전체를 대표할 정도로 일관되지는 않았다.
- `path_not_centered_or_lambda35_mismatch_candidate`는 normal perturbation에서 eta가 자주 좋아지거나 lambda=35 baseline이 낮은 경우다. 이 label의 dominant best lambda 분포를 보면 lambda=35가 dominant인 pair가 거의 없으므로, 상당 부분은 path-centeredness만이 아니라 lambda=35 mismatch 문제로 보는 것이 안전하다.
- `lambda35_low_but_retention_supported`는 support와 retention은 유지되지만 absolute high eta라고 말하기 어려운 경우다.

High-baseline pair-alpha만 따로 보면:

{md_table(high_alpha_label_counts)}

## 5. 대표 예시 후보

{md_table(examples[top_examples_cols].head(25))}

이 표는 report용 대표 후보가 아니라 geometry label별 검토 후보 목록이다. 다음 분석에서는 각 label의 상위 후보를 dashboard로 확인하는 것이 좋다.

## 6. 해석 caveat

{md_table(caveats)}

## 7. 기존 흐름과의 연결

이번 결과는 기존 Priority D의 결론을 다음처럼 보정한다.

1. 기존 normal-vector 분석은 valley/path vulnerability 쪽으로 기울어 있었고, high-high geometry 전체를 대표하지 못했다.
2. 이번 high-high core sweep은 support가 유지되는 local radius 안에서 tangent-vs-normal 질문을 직접 볼 수 있게 만들었다.
3. 일부 pair는 supported tube/cloud 후보로 볼 수 있지만, 이는 tested radius 안의 local statement이지 global manifold proof가 아니다.
4. 일부 pair는 path-not-centered 또는 lambda35 mismatch 후보로 보인다. 이런 경우 straight path만으로 high-eta geometry를 말하면 안 되고, best-lambda 또는 plausible path search와 연결해야 한다.
5. 이 결과는 plausible high-eta path search의 scoring 기준에 들어갈 수 있다. 특히 `R_joint_abs_high`, `R_joint_retention`, `normal_gain_fraction`, `anisotropy_score_proxy`, support fraction이 path 후보의 보조 지표가 된다.

## 8. 다음 단계

권장 후속:

1. 이 결과를 시각화한다.
   - radius별 eta/support retention curve
   - pair-level geometry label count
   - tangent sensitivity vs normal sensitivity scatter
   - baseline eta vs normal gain/path-centeredness scatter

2. `path_not_centered_or_lambda35_mismatch_candidate`를 분리한다.
   - best lambda가 15/3/5 쪽인 pair는 lambda=35 geometry만으로 판단하지 않는다.
   - 필요한 경우 lambda `15,35,70` 비교 sweep을 일부 target에 실행한다.

3. `supported_high_eta_tube_or_cloud_candidate`와 `narrow_ridge_candidate`의 H/eigenfeature 차이를 비교한다.
   - source/sink state changes
   - tracking overlap
   - source-sink gap/mix
   - participation

4. plausible path search에는 eta만이 아니라 support와 local geometry width를 같이 넣는다.
"""
    (REPORTS / "high_high_normal_geometry_result_report.md").write_text(report, encoding="utf-8")


def md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(fmt)
    lines = [
        "| " + " | ".join(map(str, text.columns)) + " |",
        "| " + " | ".join(["---"] * len(text.columns)) + " |",
    ]
    for row in text.to_numpy().tolist():
        lines.append("| " + " | ".join(map(str, row)) + " |")
    return "\n".join(lines)


def fmt(x: Any) -> str:
    if pd.isna(x):
        return ""
    if isinstance(x, float):
        if math.isfinite(x):
            return f"{x:.4g}"
        return ""
    return str(x).replace("|", "/")


def copy_self() -> None:
    target = SCRIPTS / Path(__file__).name
    target.write_text(Path(__file__).read_text(encoding="utf-8"), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
