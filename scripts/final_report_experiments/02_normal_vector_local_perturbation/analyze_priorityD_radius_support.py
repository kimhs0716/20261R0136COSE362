from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CSV_OUT = OUT / "csv"
REPORT_OUT = OUT / "reports"

PATHS = {
    "dashboard": ROOT / "new/bridge_priorityD_robustness_dashboard_20260613/priorityD_robustness_dashboard.html",
    "dashboard_readme": ROOT / "new/bridge_priorityD_robustness_dashboard_20260613/README.md",
    "html_dashboard": ROOT / "htmls/55_priorityD_robustness_dashboard.html",
    "integrated": ROOT / "new/bridge_priorityD_integrated_interpretation_20260613/csv/priorityD_pair_integrated_interpretation.csv",
    "changed": ROOT / "new/bridge_priorityD_integrated_interpretation_20260613/csv/what_changed_after_robustness.csv",
    "radius": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_radius_summary.csv",
    "pair": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_pair_summary.csv",
    "lambda_range": ROOT / "new/bridge_group_trend_priorityD_robustness_analysis_20260613/csv/normal_robustness_lambda_range_summary.csv",
    "detail_glob_root": ROOT / "outputs/bridge_normal_robustness_priorityD_20260613",
}


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, encoding="utf-8-sig")


def has_sparse(levels: Any) -> bool:
    return "sparse" in str(levels).lower()


def has_medium_or_high(levels: Any) -> bool:
    text = str(levels).lower()
    return "medium" in text or "high" in text


def local_support_class(row: pd.Series) -> str:
    high = float(row["r025_support_high_medium_fraction"])
    low = float(row["r025_support_low_or_better_fraction"])
    sparse = bool(row["r025_has_sparse"])
    if high >= 0.75 and not sparse:
        return "strong_local_support"
    if high >= 0.50 and low >= 0.80:
        return "usable_local_support_with_caveat"
    if low >= 0.50:
        return "weak_local_support_low_or_better_only"
    return "support_limited_at_radius_0.25"


def radius_role(radius: float, high: float, low: float, sparse: bool) -> str:
    if radius == 0:
        return "straight_path_baseline"
    if radius <= 0.25:
        if high >= 0.75 and not sparse:
            return "local_neighborhood_supported"
        if low >= 0.50:
            return "local_neighborhood_with_support_caveat"
        return "local_probe_support_limited"
    if sparse or high < 0.25:
        return "off_path_sensitivity_probe"
    return "extended_local_probe_with_strong_caveat"


def robust_scope(row: pd.Series) -> str:
    group = row["group_name"]
    local = row["local_support_class"]
    delta = float(row["delta_supported_perturbed_mean_vs_baseline_mean"])
    supported = float(row["supported_perturbed_eta20_mean"])
    baseline = float(row["baseline_eta20_mean"])

    if local == "strong_local_support" and supported >= 0.75 and baseline >= 0.75:
        return "can_say_functional_stability_local_radius_0.25"
    if local in {"strong_local_support", "usable_local_support_with_caveat"} and delta >= 0.20:
        return "can_say_local_perturbations_raise_eta_with_caveat"
    if local.startswith("support_limited"):
        return "sensitivity_only_support_limited"
    if "valley" in str(group):
        return "valley_not_robust_bottleneck_currently"
    return "local_sensitivity_with_caveat"


def support_rule(row: pd.Series) -> str:
    local = row["local_support_class"]
    max_low = float(row["max_radius_with_any_low_or_better"])
    if local == "strong_local_support":
        return "Use radius 0.25 as primary local-neighborhood evidence; larger radii still require support check."
    if local == "usable_local_support_with_caveat":
        return "Use radius 0.25 only with explicit sparse/support caveat; do not generalize to larger radii."
    if local == "weak_local_support_low_or_better_only":
        return "Use radius 0.25 as weak sensitivity evidence; avoid robustness wording."
    if max_low <= 0.25:
        return "Treat as support-limited at radius 0.25; use for sensitivity only, not robustness."
    return "Treat as support-limited; inspect pair-specific adaptive smaller radii before any robustness claim."


def recommended_grid(row: pd.Series) -> dict[str, Any]:
    local = row["local_support_class"]
    pair_id = row["pair_id"]
    group = row["group_name"]
    if local == "strong_local_support":
        grid = "0,0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40"
        stop = "stop or mark off-path when high/medium support fraction < 0.50 or low-or-better < 0.80"
        purpose = "estimate local radius before support breakdown"
    elif local == "usable_local_support_with_caveat":
        grid = "0,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.225,0.25"
        stop = "primary interpretation only up to the largest radius with high/medium >= 0.50 and low-or-better >= 0.80"
        purpose = "find where radius 0.25 begins to mix sparse/off-path samples"
    else:
        grid = "0,0.01,0.025,0.05,0.075,0.10,0.125,0.15,0.175,0.20,0.25"
        stop = "use adaptive pair-specific cutoff; robustness claim requires high/medium >= 0.50 and low-or-better >= 0.80"
        purpose = "recover a smaller support-matched local radius"

    if "valley" in str(group):
        purpose += "; run specifically at valley alpha and high-eta reference alpha"
    return {
        "pair_id": pair_id,
        "group_name": group,
        "local_support_class": local,
        "recommended_radius_grid": grid,
        "adaptive_stop_rule": stop,
        "recommended_use": purpose,
        "support_matched_subset_rule": "for interpretation, keep directions with support level high/medium first; report low-or-better separately; exclude sparse from robustness claims",
    }


def md_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:
    view = df[cols].copy()
    if max_rows:
        view = view.head(max_rows)
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = view[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        else:
            view[c] = view[c].astype(str)
    header = "| " + " | ".join(view.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
    rows = ["| " + " | ".join(str(v).replace("|", "\\|").replace("\n", " ") for v in row) + " |" for row in view.to_numpy()]
    return "\n".join([header, sep, *rows])


def main() -> None:
    CSV_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.mkdir(parents=True, exist_ok=True)

    radius = read_csv(PATHS["radius"])
    pair = read_csv(PATHS["pair"])
    lambda_range = read_csv(PATHS["lambda_range"])
    integrated = read_csv(PATHS["integrated"])

    detail_paths = sorted(PATHS["detail_glob_root"].glob("*/csv/normal_robustness_detail.csv")) if PATHS["detail_glob_root"].exists() else []

    radius_rows = []
    for rad, g in radius.groupby("radius"):
        jobs = float(g["n_jobs"].sum())
        radius_rows.append(
            {
                "radius": rad,
                "pair_count": g["pair_id"].nunique(),
                "n_jobs": int(jobs),
                "weighted_support_high_medium_fraction": (g["support_high_medium_fraction"] * g["n_jobs"]).sum() / jobs,
                "weighted_support_low_or_better_fraction": (g["support_low_or_better_fraction"] * g["n_jobs"]).sum() / jobs,
                "pairs_with_sparse": int(g["support_levels"].map(has_sparse).sum()),
                "pairs_with_high": int(g["support_levels"].astype(str).str.contains("high", case=False).sum()),
                "pairs_with_medium": int(g["support_levels"].astype(str).str.contains("medium", case=False).sum()),
                "pairs_with_low": int(g["support_levels"].astype(str).str.contains("low", case=False).sum()),
                "eta20_q50_mean": g["eta20_q50"].mean(),
                "eta20_q10_mean": g["eta20_q10"].mean(),
                "eta20_q90_mean": g["eta20_q90"].mean(),
                "nearest_support_distance_median_mean": g["nearest_support_distance_median"].mean(),
                "interpretation_role": radius_role(
                    float(rad),
                    (g["support_high_medium_fraction"] * g["n_jobs"]).sum() / jobs,
                    (g["support_low_or_better_fraction"] * g["n_jobs"]).sum() / jobs,
                    bool(g["support_levels"].map(has_sparse).sum() > 0),
                ),
            }
        )
    radius_global = pd.DataFrame(radius_rows)
    radius_global.to_csv(CSV_OUT / "radius_global_support_summary.csv", index=False, encoding="utf-8-sig")

    lambda_global = (
        lambda_range.groupby("radius")
        .agg(
            pair_count=("pair_id", "nunique"),
            lambda_range_median_mean=("lambda_range_median", "mean"),
            lambda_range_q90_mean=("lambda_range_q90", "mean"),
            lambda_range_max_mean=("lambda_range_max", "mean"),
            pairs_with_sparse=("support_levels", lambda x: sum(has_sparse(v) for v in x)),
            support_level_examples=("support_levels", lambda x: "; ".join(sorted(set(map(str, x))))),
        )
        .reset_index()
    )
    lambda_global.to_csv(CSV_OUT / "radius_lambda_range_summary.csv", index=False, encoding="utf-8-sig")

    r025 = radius[radius["radius"].eq(0.25)][
        [
            "pair_id",
            "eta20_q10",
            "eta20_q50",
            "eta20_q90",
            "support_high_medium_fraction",
            "support_low_or_better_fraction",
            "support_levels",
            "nearest_support_distance_median",
        ]
    ].rename(
        columns={
            "eta20_q10": "r025_eta20_q10",
            "eta20_q50": "r025_eta20_q50",
            "eta20_q90": "r025_eta20_q90",
            "support_high_medium_fraction": "r025_support_high_medium_fraction",
            "support_low_or_better_fraction": "r025_support_low_or_better_fraction",
            "support_levels": "r025_support_levels",
            "nearest_support_distance_median": "r025_nearest_support_distance_median",
        }
    )
    r05 = radius[radius["radius"].eq(0.5)][
        ["pair_id", "support_high_medium_fraction", "support_low_or_better_fraction", "support_levels", "eta20_q50"]
    ].rename(
        columns={
            "support_high_medium_fraction": "r05_support_high_medium_fraction",
            "support_low_or_better_fraction": "r05_support_low_or_better_fraction",
            "support_levels": "r05_support_levels",
            "eta20_q50": "r05_eta20_q50",
        }
    )
    caveat = pair.merge(r025, on="pair_id", how="left").merge(r05, on="pair_id", how="left")
    integrated_cols = [
        "pair_id",
        "primary_label",
        "secondary_flags",
        "current_conclusion",
        "do_not_claim",
        "lambda_range_change_label",
        "lambda_range_change_text",
    ]
    caveat = caveat.merge(integrated[[c for c in integrated_cols if c in integrated.columns]], on="pair_id", how="left")
    caveat["r025_has_sparse"] = caveat["r025_support_levels"].map(has_sparse)
    caveat["r05_has_sparse"] = caveat["r05_support_levels"].map(has_sparse)
    caveat["local_support_class"] = caveat.apply(local_support_class, axis=1)
    caveat["robustness_wording_scope"] = caveat.apply(robust_scope, axis=1)
    caveat["support_interpretation_rule"] = caveat.apply(support_rule, axis=1)
    caveat["radius_0p5_or_larger_role"] = caveat.apply(
        lambda row: "off_path_sensitivity_probe" if row["r05_has_sparse"] or row["r05_support_high_medium_fraction"] < 0.25 else "extended_probe_with_caveat",
        axis=1,
    )
    caveat_cols = [
        "pair_id",
        "target_priority",
        "group_name",
        "target_category",
        "baseline_eta20_mean",
        "supported_perturbed_eta20_mean",
        "delta_supported_perturbed_mean_vs_baseline_mean",
        "r025_eta20_q50",
        "r025_support_high_medium_fraction",
        "r025_support_low_or_better_fraction",
        "r025_support_levels",
        "r025_nearest_support_distance_median",
        "r025_has_sparse",
        "r05_support_high_medium_fraction",
        "r05_support_low_or_better_fraction",
        "r05_support_levels",
        "local_support_class",
        "robustness_wording_scope",
        "radius_0p5_or_larger_role",
        "support_interpretation_rule",
        "lambda_range_change_label",
        "lambda_range_change_text",
        "current_conclusion",
        "do_not_claim",
    ]
    caveat[caveat_cols].to_csv(CSV_OUT / "pair_level_support_caveat.csv", index=False, encoding="utf-8-sig")

    rec = pd.DataFrame([recommended_grid(row) for _, row in caveat.iterrows()])
    rec.to_csv(CSV_OUT / "recommended_radius_grid.csv", index=False, encoding="utf-8-sig")

    rules = pd.DataFrame(
        [
            {
                "rule_name": "primary_local_support",
                "criterion": "radius <= 0.25 and high/medium support fraction >= 0.75 with no sparse support level",
                "interpretation": "Can use as local-neighborhood evidence; still not mechanism proof.",
            },
            {
                "rule_name": "usable_with_caveat",
                "criterion": "radius 0.25 has high/medium >= 0.50 and low-or-better >= 0.80, or low-or-better >= 0.50",
                "interpretation": "Can discuss sensitivity or local eta trend with explicit support caveat.",
            },
            {
                "rule_name": "support_limited",
                "criterion": "radius 0.25 low-or-better < 0.50 or high/medium < 0.20",
                "interpretation": "Do not use robustness wording; use as sensitivity hint only.",
            },
            {
                "rule_name": "large_radius_sparse",
                "criterion": "radius >= 0.5 has sparse support or high/medium support fraction < 0.25",
                "interpretation": "Off-path sensitivity probe, not sampled-manifold robustness evidence.",
            },
            {
                "rule_name": "valley_vulnerability",
                "criterion": "valley/deep-valley conclusion requires valley-alpha support-matched perturbation, not pair-aggregated radius summary",
                "interpretation": "Current evidence can call straight-path vulnerability candidate, not robust bottleneck.",
            },
        ]
    )
    rules.to_csv(CSV_OUT / "radius_interpretation_rules.csv", index=False, encoding="utf-8-sig")

    write_report(radius_global, lambda_global, caveat, rec, detail_paths)


def write_report(
    radius_global: pd.DataFrame,
    lambda_global: pd.DataFrame,
    caveat: pd.DataFrame,
    rec: pd.DataFrame,
    detail_paths: list[Path],
) -> None:
    strong = caveat[caveat["local_support_class"].eq("strong_local_support")]
    caveated = caveat[caveat["local_support_class"].eq("usable_local_support_with_caveat")]
    weak = caveat[caveat["local_support_class"].eq("weak_local_support_low_or_better_only")]
    limited = caveat[caveat["local_support_class"].eq("support_limited_at_radius_0.25")]
    sparse025 = caveat[caveat["r025_has_sparse"]]
    radius05_offpath = caveat[caveat["radius_0p5_or_larger_role"].eq("off_path_sensitivity_probe")]

    detail_note = (
        f"Detail CSV found: {len(detail_paths)} file(s)."
        if detail_paths
        else "지정된 outputs glob에서 `normal_robustness_detail.csv`는 발견되지 않았다. 따라서 이번 재검토는 radius/pair/lambda summary CSV와 dashboard payload 기반으로 수행했다."
    )

    report = f"""# Priority D Radius / Support Review

작성일: 2026-06-13

## 목적

Priority D normal-vector robustness에서 사용한 radius grid `0, 0.25, 0.5, 1.0, 1.5`와 support 해석이 적절했는지 재검토했다. 새 simulation은 실행하지 않았고, 55번 dashboard와 원 CSV를 다시 대조했다.

## 입력 확인

- dashboard: `htmls/55_priorityD_robustness_dashboard.html`
- dashboard build folder: `new/bridge_priorityD_robustness_dashboard_20260613`
- radius summary: `normal_robustness_radius_summary.csv`
- pair summary: `normal_robustness_pair_summary.csv`
- lambda range summary: `normal_robustness_lambda_range_summary.csv`
- integrated interpretation: `priorityD_pair_integrated_interpretation.csv`
- detail availability: {detail_note}

## 결론 요약

1. **radius 0.25는 local-neighborhood probe로 쓸 수 있지만 pair별 support caveat가 필요하다.** 전체 job 가중 기준 high/medium support fraction은 0.581, low-or-better는 0.810이다. 하지만 13개 중 7개 pair에서 이미 sparse가 섞인다.
2. **radius 0.5 이상은 대부분 off-path sensitivity probe다.** radius 0.5에서는 모든 pair가 sparse를 포함하고, high/medium support fraction은 job 가중 0.008이다. radius 1.0과 1.5는 high/medium과 low-or-better가 모두 0이다.
3. **현재 큰-radius eta 상승은 robustness evidence가 아니다.** 0.5 이상에서 eta q50이 높아지는 패턴이 보여도 support가 거의 sparse이므로 sampled design manifold 안에서 high eta region이 넓다는 뜻으로 쓰면 안 된다.
4. **valley/deep-valley는 robust bottleneck이 아니라 straight-path vulnerability 후보로 유지해야 한다.** 현재 summary는 pair/radius aggregate라 valley alpha에서 support-matched perturbation이 valley를 지우는지 또는 유지하는지 직접 판정할 수 없다.
5. **후속 실험은 coarse large-radius sweep보다 0.25 이하 dense/adaptive grid가 우선이다.** 기본 grid는 `0, 0.05, 0.10, 0.15, 0.20, 0.25`이고, support-limited pair는 `0, 0.01, 0.025, 0.05, ...`처럼 더 촘촘히 시작해야 한다.

## Radius별 Support 분포

{md_table(radius_global, [
    "radius",
    "pair_count",
    "n_jobs",
    "weighted_support_high_medium_fraction",
    "weighted_support_low_or_better_fraction",
    "pairs_with_sparse",
    "eta20_q50_mean",
    "nearest_support_distance_median_mean",
    "interpretation_role",
])}

## Radius별 Lambda Eta-Range

{md_table(lambda_global, [
    "radius",
    "pair_count",
    "lambda_range_median_mean",
    "lambda_range_q90_mean",
    "lambda_range_max_mean",
    "pairs_with_sparse",
    "support_level_examples",
])}

해석: straight path(radius 0)의 lambda eta-range가 평균적으로 가장 크고, radius 0.25에서 줄어든다. 그러나 radius 0.5 이상에서 lambda range가 더 줄어드는 것은 support가 sparse로 이동했기 때문일 수 있으므로, lambda mechanism 완화로 강하게 말하지 않는다.

## Radius 0.25 Pair 분류

- strong local support: {len(strong)} pair(s): {", ".join(strong["pair_id"].tolist()) or "none"}
- usable with caveat: {len(caveated)} pair(s): {", ".join(caveated["pair_id"].tolist()) or "none"}
- weak low-or-better only: {len(weak)} pair(s): {", ".join(weak["pair_id"].tolist()) or "none"}
- support-limited at radius 0.25: {len(limited)} pair(s): {", ".join(limited["pair_id"].tolist()) or "none"}
- radius 0.25 already includes sparse: {len(sparse025)} pair(s): {", ".join(sparse025["pair_id"].tolist()) or "none"}
- radius 0.5 or larger off-path sensitivity: {len(radius05_offpath)} / {len(caveat)} pair(s)

{md_table(caveat.sort_values(["local_support_class", "pair_id"]), [
    "pair_id",
    "group_name",
    "baseline_eta20_mean",
    "supported_perturbed_eta20_mean",
    "delta_supported_perturbed_mean_vs_baseline_mean",
    "r025_support_high_medium_fraction",
    "r025_support_low_or_better_fraction",
    "r025_support_levels",
    "local_support_class",
    "robustness_wording_scope",
])}

## Robust vs Sensitivity Wording

쓸 수 있는 표현:

- `local perturbation raises eta`: radius 0.25에서 support가 충분하거나 caveat와 함께 볼 수 있는 pair에 한해 가능하다.
- `functional stability candidate`: straight path와 supported perturbation이 모두 high eta에 머무는 high-high pair에서 가능하다.
- `straight-path vulnerability candidate`: valley/deep-valley에서 supported perturbation이 eta를 올릴 때 가능하다.
- `path/local-neighborhood sensitivity`: sparse가 섞이거나 support가 제한된 경우 가장 안전한 표현이다.

피해야 할 표현:

- radius 0.5 이상 결과를 근거로 `wide robust manifold`, `robust bottleneck`, `sampled-manifold radius`라고 쓰지 않는다.
- pair-aggregate 결과만으로 valley alpha의 `bottleneck` 또는 `boundary`를 확정하지 않는다.
- lambda range 감소를 H eigenfeature 변화로 쓰지 않는다. H(alpha)는 lambda에 따라 변하지 않는다.

## Straight Path가 Local High-Eta Neighborhood의 중심인지 판단하려면

최소 기준:

1. radius 0.25 이하에서 support-matched subset(high/medium 우선, low-or-better 보조)을 따로 집계한다.
2. straight baseline eta가 local supported q50보다 낮은지뿐 아니라, q10/q90와 방향별 분포를 본다.
3. radius를 0.05 단위 이하로 촘촘히 두어, eta 상승이 즉시 나타나는지 또는 일정 radius 이후 나타나는지 확인한다.
4. support-limited pair는 adaptive cutoff를 먼저 찾고, cutoff 바깥은 robustness가 아니라 sensitivity로 둔다.

현재 데이터에서 straight path가 low-section일 가능성은 여러 pair에서 보이지만, local neighborhood의 중심/경계 여부는 아직 말할 수 없다.

## Valley / Deep-Valley 후속 기준

현재 valley/deep-valley pair에서 supported eta가 straight baseline보다 높아지는 것은 `straight-path vulnerability` 해석을 강화한다. 하지만 robust vulnerability인지 straight-path artifact인지는 다음 조건이 있어야 구분된다.

- valley alpha와 high-eta reference alpha를 분리해서 normal perturbation을 수행한다.
- 각 alpha에서 support-matched direction만 따로 집계한다.
- radius grid는 `0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.25`를 우선한다.
- valley alpha에서 support-matched q50/q90도 낮게 유지되면 vulnerability 후보가 강화된다.
- valley alpha에서 작은 radius만 줘도 eta가 회복되면 straight-path artifact 또는 under-sampled transition 후보로 낮춘다.

## 추천 Radius Grid

기본 grid:

- `0, 0.05, 0.10, 0.15, 0.20, 0.25`

support-limited / valley 정밀 grid:

- `0, 0.01, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25`

strong local support pair의 선택 확장:

- `0.30, 0.35, 0.40`까지만 선택적으로 확장하고, radius 0.5는 support breakdown 확인용으로만 둔다.

Adaptive stop rule:

- high/medium support fraction `< 0.50` 또는 low-or-better `< 0.80`가 되면 그 이후 radius는 robustness claim에서 제외한다.
- sparse-only radius는 off-path sensitivity probe로만 사용한다.

## 산출물

- `csv/radius_global_support_summary.csv`
- `csv/radius_lambda_range_summary.csv`
- `csv/pair_level_support_caveat.csv`
- `csv/recommended_radius_grid.csv`
- `csv/radius_interpretation_rules.csv`
"""
    (REPORT_OUT / "priorityD_radius_support_review.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
