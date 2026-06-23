from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("new/current_direction_candidate_selection_20260614")
CSV = ROOT / "csv"
REPORTS = ROOT / "reports"


CLASS_PATH = Path("new/bridge_priority1_classification_20260613/csv/pair_classification_ds_same_s.csv")
PROXY_PATH = Path("new/bridge_group_trend_priorityA_proxy_20260613/csv/pair_proxy_subtypes_ds_same_s.csv")
H_PATH = Path("new/bridge_group_trend_priorityB_h_eigenfeatures_20260613/csv/pair_level_structural_summary.csv")
CONTRIB_MANIFEST_PATH = Path("new/bridge_overnight_group_contribution_expansion_20260614/csv/group_contribution_target_manifest.csv")
CONTRIB_FEATURE_PATH = Path("new/bridge_overnight_group_contribution_analysis_20260614/csv/contribution_feature_summary.csv")
LOCAL_PATH = Path("new/bridge_overnight_group_contribution_analysis_20260614/csv/straight_vs_local_perturbation_summary.csv")
SECTION6_PAIRS_PATH = Path("new/target_audit_sections6_7_20260614/csv/section6_actual_unique_pairs.csv")


def has_flag(series: pd.Series, flag: str) -> pd.Series:
    return series.fillna("").astype(str).str.split("|").apply(lambda xs: flag in xs)


def norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    q05 = s.quantile(0.05)
    q95 = s.quantile(0.95)
    if not np.isfinite(q05) or not np.isfinite(q95) or math.isclose(q05, q95):
        return pd.Series(0.0, index=s.index)
    return ((s - q05) / (q95 - q05)).clip(0, 1).fillna(0)


def alpha_subset(row: pd.Series) -> str:
    vals = {0.0, 0.5, 1.0}
    for col in [
        "path_argmin_alpha",
        "path_argmax_alpha",
        "best_low_alpha",
        "best_high_alpha",
        "lambda35_low_alpha",
        "lambda35_high_alpha",
        "steepest_best_eta_step_alpha",
    ]:
        if col in row and pd.notna(row[col]):
            try:
                v = round(float(row[col]) / 0.05) * 0.05
                vals.add(round(max(0.0, min(1.0, v)), 2))
            except Exception:
                pass
    return ",".join(f"{v:g}" for v in sorted(vals))


def load() -> pd.DataFrame:
    cls = pd.read_csv(CLASS_PATH)
    proxy = pd.read_csv(PROXY_PATH)
    h = pd.read_csv(H_PATH)
    manifest = pd.read_csv(CONTRIB_MANIFEST_PATH)
    contrib = pd.read_csv(CONTRIB_FEATURE_PATH)
    local = pd.read_csv(LOCAL_PATH)

    local_agg = (
        local.groupby("pair_id")
        .agg(
            local_delta_eta_max=("delta_eta20_local_minus_straight", "max"),
            local_delta_eta_min=("delta_eta20_local_minus_straight", "min"),
            local_delta_eta_abs_max=("delta_eta20_local_minus_straight", lambda x: float(np.nanmax(np.abs(x)))),
            local_delta_trap_max=("delta_local_minus_straight_trap_mean_10_20ps", "max"),
            local_delta_source_min=("delta_local_minus_straight_source12_mean_10_20ps", "min"),
            local_rows=("pair_id", "size"),
        )
        .reset_index()
    )

    contrib_sel = contrib[
        [
            "pair_id",
            "group_name",
            "target_category",
            "lambda_contrast_alpha",
            "best_lambda",
            "worst_lambda",
            "eta20_best_minus_worst",
            "delta_best_minus_worst_trap_mean_10_20ps",
            "delta_best_minus_worst_source12_mean_10_20ps",
            "delta_best_minus_worst_eig_coherence_l1_mean_10_20ps",
            "lambda_response_pattern",
        ]
    ]

    manifest_sel = manifest[
        [
            "pair_id",
            "target_priority",
            "target_category",
            "group_name",
            "existing_curves_npz",
            "existing_curves_pair_index",
        ]
    ].drop_duplicates("pair_id")

    df = cls.merge(proxy, on=["pair_id", "primary_label", "secondary_flags", "interpretation_hold_flags"], how="left", suffixes=("", "_proxy"))
    df = df.merge(h, on="pair_id", how="left", suffixes=("", "_h"))
    df = df.merge(manifest_sel, on="pair_id", how="inner", suffixes=("", "_manifest"))
    df = df.merge(contrib_sel, on="pair_id", how="left", suffixes=("", "_contrib"))
    df = df.merge(local_agg, on="pair_id", how="left")

    if SECTION6_PAIRS_PATH.exists():
        section6 = set(pd.read_csv(SECTION6_PAIRS_PATH)["pair_id"].astype(str))
    else:
        section6 = set()
    df["already_section6_dense_run"] = df["pair_id"].isin(section6)

    df["flag_deep_valley"] = has_flag(df["secondary_flags"], "deep_valley")
    df["flag_lambda_dramatic"] = has_flag(df["secondary_flags"], "lambda_dramatic")
    df["flag_large_step"] = has_flag(df["secondary_flags"], "large_adjacent_alpha_step")
    df["flag_best_lambda_switching"] = has_flag(df["secondary_flags"], "best_lambda_switching")
    df["flag_extreme_low_high"] = has_flag(df["secondary_flags"], "extreme_low_high_endpoints")
    df["flag_low_high"] = has_flag(df["secondary_flags"], "low_high_endpoints")
    df["flag_high_high"] = has_flag(df["secondary_flags"], "high_high_endpoints")
    df["hold"] = df["interpretation_hold"].fillna(False).astype(bool)
    df["support_ok"] = pd.to_numeric(df["support_high_medium_fraction"], errors="coerce").fillna(0) >= 0.8
    df["quality_okish"] = pd.to_numeric(df["mean_endpoint_quality_norm_within_unit"], errors="coerce").fillna(0) >= 0.2
    df["alpha_subset_recommended"] = df.apply(alpha_subset, axis=1)
    return df


def score_and_select(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d["valley_score"] = (
        3.0 * norm(d["valley_depth_best_eta20"])
        + 1.5 * d["flag_deep_valley"].astype(float)
        + 1.2 * norm(d["local_delta_eta_max"])
        + 0.8 * d["support_ok"].astype(float)
        + 0.5 * d["flag_lambda_dramatic"].astype(float)
        - 1.0 * d["hold"].astype(float)
    )
    d["low_high_score"] = (
        2.0 * norm(d["endpoint_abs_gap_best_eta20"])
        + 1.5 * norm(d["path_range_best_eta20"])
        + 1.5 * norm(d["local_delta_eta_max"])
        + 0.8 * d["flag_extreme_low_high"].astype(float)
        + 0.6 * d["flag_large_step"].astype(float)
        + 0.5 * d["support_ok"].astype(float)
        - 0.8 * d["hold"].astype(float)
    )
    d["lambda_score"] = (
        2.0 * d["flag_lambda_dramatic"].astype(float)
        + 2.0 * norm(d["max_lambda_eta20_range_nonzero"])
        + 1.5 * norm(d["eta20_best_minus_worst"])
        + 1.0 * norm(d["local_delta_eta_max"])
        + 0.5 * d["flag_best_lambda_switching"].astype(float)
        - 0.8 * d["hold"].astype(float)
    )
    d["control_score"] = (
        (d["primary_label"].eq("high_high_stable")).astype(float)
        + norm(d["path_min_best_eta20"])
        + norm(-d["local_delta_eta_max"].fillna(0))
        + 0.5 * d["support_ok"].astype(float)
        - 0.6 * d["hold"].astype(float)
    )

    selections = []
    groups = [
        (
            "P1_valley_path_artifact_or_vulnerability",
            d[(d["primary_label"].eq("high_high_valley")) | d["flag_deep_valley"]],
            "valley_score",
            20,
        ),
        (
            "P2_low_high_path_not_centered_or_transition",
            d[d["primary_label"].isin(["low_high_transition", "low_high_mixed_transition"]) | d["flag_low_high"] | d["flag_extreme_low_high"]],
            "low_high_score",
            25,
        ),
        (
            "P3_lambda_dramatic_geometry_dynamics_link",
            d[d["flag_lambda_dramatic"]],
            "lambda_score",
            25,
        ),
        (
            "P4_functional_stability_contrast_control",
            d[d["primary_label"].eq("high_high_stable")],
            "control_score",
            10,
        ),
    ]
    for label, sub, score_col, n in groups:
        take = sub.sort_values(score_col, ascending=False).head(n).copy()
        take["selection_group"] = label
        take["selection_score"] = take[score_col]
        selections.append(take)
    selected_long = pd.concat(selections, ignore_index=True)

    selected_unique = (
        selected_long.sort_values("selection_score", ascending=False)
        .drop_duplicates("pair_id")
        .sort_values(["selection_group", "selection_score"], ascending=[True, False])
    )

    return selected_long, selected_unique


def write_outputs(selected_long: pd.DataFrame, selected_unique: pd.DataFrame) -> None:
    CSV.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    cols = [
        "selection_group",
        "selection_score",
        "pair_id",
        "primary_label",
        "secondary_flags",
        "proxy_path_shape_subtype",
        "proxy_lambda_subtype",
        "support_high_medium_fraction",
        "support_distance_median_zrms",
        "mean_endpoint_quality_norm_within_unit",
        "interpretation_hold_flags",
        "already_section6_dense_run",
        "path_min_best_eta20",
        "path_max_best_eta20",
        "path_range_best_eta20",
        "valley_depth_best_eta20",
        "max_eta20_step_between_adjacent_alpha",
        "dominant_best_lambda",
        "max_lambda_eta20_range_nonzero",
        "eta20_best_minus_worst",
        "local_delta_eta_max",
        "local_delta_eta_min",
        "local_delta_trap_max",
        "local_delta_source_min",
        "h_only_structural_subtype_seed",
        "source_state_changes",
        "sink_state_changes",
        "min_tracking_overlap_excluding_start",
        "min_source_sink_gap",
        "max_source_sink_mix",
        "max_participation",
        "alpha_subset_recommended",
        "existing_curves_npz",
        "existing_curves_pair_index",
    ]
    cols = [c for c in cols if c in selected_unique.columns]
    selected_long[cols].to_csv(CSV / "current_direction_candidate_rows_by_group.csv", index=False, encoding="utf-8")
    selected_unique[cols].to_csv(CSV / "current_direction_candidate_unique_pairs.csv", index=False, encoding="utf-8")

    manifest_cols = [
        "pair_id",
        "selection_group",
        "selection_score",
        "primary_label",
        "secondary_flags",
        "alpha_subset_recommended",
        "existing_curves_npz",
        "existing_curves_pair_index",
    ]
    manifest = selected_unique[[c for c in manifest_cols if c in selected_unique.columns]].copy()
    manifest["target_priority"] = manifest["selection_group"].str.extract(r"P(\d+)").fillna("9").astype(int)
    manifest["target_category"] = manifest["selection_group"]
    manifest["group_name"] = manifest["selection_group"]
    manifest["alpha_subset"] = manifest["alpha_subset_recommended"]
    manifest["lambda_subset"] = "15,35,70"
    # Reorder to match the normal-vector runner's expected shape while retaining provenance.
    first = [
        "target_priority",
        "target_category",
        "pair_id",
        "group_name",
        "primary_label",
        "secondary_flags",
        "selection_score",
        "alpha_subset",
        "lambda_subset",
        "existing_curves_npz",
        "existing_curves_pair_index",
    ]
    manifest = manifest[[c for c in first if c in manifest.columns]]
    manifest.to_csv(CSV / "current_direction_normal_vector_candidate_manifest.csv", index=False, encoding="utf-8")

    summary = (
        selected_unique.groupby("selection_group")
        .agg(
            unique_pairs=("pair_id", "nunique"),
            already_section6_dense=("already_section6_dense_run", "sum"),
            median_score=("selection_score", "median"),
            median_local_delta=("local_delta_eta_max", "median"),
            median_valley_depth=("valley_depth_best_eta20", "median"),
            median_lambda_range=("max_lambda_eta20_range_nonzero", "median"),
            support_ok_fraction=("support_ok", "mean"),
            hold_fraction=("hold", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(CSV / "candidate_group_summary.csv", index=False, encoding="utf-8")

    top_lines = []
    top_lines.append("# Current-Direction Normal-Vector Candidate Selection")
    top_lines.append("")
    top_lines.append("작성일: 2026-06-14")
    top_lines.append("")
    top_lines.append("## 목적")
    top_lines.append("")
    top_lines.append("High-high core 103개는 별도 normal geometry 확장 대상으로 이미 정해져 있으므로, 여기서는 그 밖의 현재 연구 방향에 중요한 추가 normal-vector 후보를 고른다. 전수 1303개를 모두 돌리기보다, valley/path artifact, low-high transition, lambda-dramatic dynamics-link, functional-stability contrast를 대표할 수 있는 pair를 우선 선정했다.")
    top_lines.append("")
    top_lines.append("## 선정 그룹")
    top_lines.append("")
    for _, row in summary.iterrows():
        top_lines.append(f"- `{row['selection_group']}`: {int(row['unique_pairs'])} unique pairs, 이미 Section 6 dense에 포함된 pair {int(row['already_section6_dense'])}개, support_ok_fraction {row['support_ok_fraction']:.2f}, hold_fraction {row['hold_fraction']:.2f}")
    top_lines.append("")
    top_lines.append("## 해석")
    top_lines.append("")
    top_lines.append("- P1은 기존 selected valley 검증을 전체 valley/deep-valley 방향으로 넓힐 때 가장 직접적인 후보군이다.")
    top_lines.append("- P2는 low-high transition에서 straight path가 낮은 단면을 지나는지, endpoint-driven 변화인지 확인하는 후보군이다.")
    top_lines.append("- P3은 lambda response가 큰 pair에서 local geometry와 dynamics response가 연결되는지 확인하는 후보군이다.")
    top_lines.append("- P4는 high-high core와 겹칠 수 있는 functional-stability contrast control이다. 새 핵심 실험 대상이라기보다 결과 해석 기준선으로 쓰는 것이 적절하다.")
    top_lines.append("")
    top_lines.append("## 주의")
    top_lines.append("")
    top_lines.append("- `current_direction_normal_vector_candidate_manifest.csv`는 runner 호환성을 위해 `alpha_subset`, `lambda_subset`, `existing_curves_npz`, `existing_curves_pair_index`를 포함한다.")
    top_lines.append("- `lambda_subset`은 기존 Section 6 dense 조건과 맞추기 위해 `15,35,70`으로 둔다.")
    top_lines.append("- 후보 선정은 mechanism proof가 아니라 후속 검증 대상 선정이다.")
    top_lines.append("- `already_section6_dense_run=True`인 pair는 이미 기존 dense validation에 포함되었으므로, 재실행보다 기존 결과 재분석을 우선한다.")
    top_lines.append("")
    top_lines.append("## Top Candidates")
    top_lines.append("")
    for group, sub in selected_unique.groupby("selection_group", sort=False):
        top_lines.append(f"### {group}")
        top_lines.append("")
        top_lines.append("| pair_id | score | primary | flags | local_delta_max | valley_depth | lambda_range | already_run |")
        top_lines.append("| --- | ---: | --- | --- | ---: | ---: | ---: | --- |")
        for _, r in sub.head(10).iterrows():
            top_lines.append(
                f"| {r['pair_id']} | {r['selection_score']:.3f} | {r['primary_label']} | {str(r['secondary_flags'])[:80]} | "
                f"{float(r.get('local_delta_eta_max', np.nan)):.3f} | {float(r.get('valley_depth_best_eta20', np.nan)):.3f} | "
                f"{float(r.get('max_lambda_eta20_range_nonzero', np.nan)):.3f} | {bool(r['already_section6_dense_run'])} |"
            )
        top_lines.append("")
    (REPORTS / "current_direction_candidate_selection.md").write_text("\n".join(top_lines), encoding="utf-8")


def main() -> None:
    df = load()
    selected_long, selected_unique = score_and_select(df)
    write_outputs(selected_long, selected_unique)
    print(f"wrote {ROOT}")
    print(selected_unique.groupby("selection_group")["pair_id"].nunique())


if __name__ == "__main__":
    main()
