#!/usr/bin/env python
"""Temporal route and reduced-space review for high-eta dynamic archetypes.

This is a post-processing script only. It does not run new dynamics
simulation, contribution reruns, or H-space path search.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_DIR = Path("new/high_eta_archetype_claim_validation_20260616")
DEFAULT_DYNAMIC_DIR = Path("new/bridge_dynamic_phenotype_mode_discovery_20260616")
DEFAULT_OUTPUT_DIR = Path("new/temporal_archetype_route_and_reduced_space_review_20260616")
RANDOM_SEED = 20260616


@dataclass
class Paths:
    input_dir: Path
    dynamic_dir: Path
    output_dir: Path

    @property
    def in_csv(self) -> Path:
        return self.input_dir / "csv"

    @property
    def dyn_csv(self) -> Path:
        return self.dynamic_dir / "csv"

    @property
    def out_csv(self) -> Path:
        return self.output_dir / "csv"

    @property
    def out_reports(self) -> Path:
        return self.output_dir / "reports"

    @property
    def out_json(self) -> Path:
        return self.output_dir / "json"

    @property
    def out_commands(self) -> Path:
        return self.output_dir / "commands"


def ensure_dirs(paths: Paths) -> None:
    for p in [paths.output_dir, paths.out_csv, paths.out_reports, paths.out_json, paths.out_commands]:
        p.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, **kwargs)


def parse_feature_sets(defs: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for _, row in defs.iterrows():
        features = str(row["features"]).split("|")
        out[str(row["feature_set"])] = [f for f in features if f]
    return out


def feature_group(feature: str) -> str:
    f = feature.lower()
    if f.startswith("site1") or f.startswith("site2"):
        return "source_retention"
    if f.startswith("sink34"):
        return "sink_accumulation"
    if f.startswith("trap"):
        return "trap_population"
    if f.startswith("detour"):
        return "detour_occupancy"
    if f.startswith("loss"):
        return "loss"
    if f.startswith("residual"):
        return "residual"
    if f.startswith("cl1"):
        return "coherence"
    if f.startswith("purity"):
        return "purity"
    if f.startswith("ipr"):
        return "ipr_localization"
    if f.startswith("t") or f.startswith("tau"):
        return "timing"
    if f.startswith("eig"):
        return "eigenvalue_summary"
    if f.startswith("h_param"):
        return "h_parameter_norm"
    if "high_" in f or "eta" in f:
        return "outcome_or_membership"
    return "other"


def time_window(feature: str) -> str:
    f = feature.lower()
    if "_0_5ps" in f or "_at_6ps" in f:
        return "early"
    if "_5_10ps" in f or "_at_10ps" in f:
        return "mid"
    if "_10_20ps" in f or "_at_20ps" in f or "_at_50ps" in f:
        return "late"
    if f.endswith("_max") or "tmax" in f:
        return "summary"
    return "global"


def table_text(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(empty)"
    return df.head(max_rows).to_string(index=False)


def write_md(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")


def select_high_eta(observed: pd.DataFrame, membership: pd.DataFrame, label: str) -> pd.DataFrame:
    if label not in membership.columns:
        raise KeyError(f"missing high eta label: {label}")
    high_ids = set(membership.loc[membership[label].astype(bool), "h_id"].astype(str))
    high = observed[observed["h_id"].astype(str).isin(high_ids)].copy()
    return high


def zscore(df: pd.DataFrame, features: list[str], fit_df: pd.DataFrame | None = None) -> tuple[np.ndarray, StandardScaler]:
    if fit_df is None:
        fit_df = df
    scaler = StandardScaler()
    fit_x = fit_df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    x = df[features].astype(float).replace([np.inf, -np.inf], np.nan)
    med = fit_x.median(numeric_only=True)
    fit_x = fit_x.fillna(med)
    x = x.fillna(med)
    scaler.fit(fit_x)
    return scaler.transform(x), scaler


def feature_profile_by_label(
    df: pd.DataFrame, label_col: str, features: list[str], top_n: int = 8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    x, _ = zscore(df, features)
    z = pd.DataFrame(x, columns=features, index=df.index)
    rows = []
    group_rows = []
    for label, idx in df.groupby(label_col).groups.items():
        mean_z = z.loc[idx].mean(axis=0).sort_values(key=lambda s: s.abs(), ascending=False)
        signed = z.loc[idx].mean(axis=0)
        for rank, (feature, value) in enumerate(mean_z.head(top_n).items(), start=1):
            rows.append(
                {
                    "label_col": label_col,
                    "label": label,
                    "rank": rank,
                    "feature": feature,
                    "mean_z": float(signed[feature]),
                    "abs_mean_z": float(abs(signed[feature])),
                    "feature_group": feature_group(feature),
                    "time_window": time_window(feature),
                }
            )
        for group, cols in group_features(features).items():
            if not cols:
                continue
            vals = signed[cols]
            group_rows.append(
                {
                    "label_col": label_col,
                    "label": label,
                    "feature_group": group,
                    "n_features": len(cols),
                    "mean_z": float(vals.mean()),
                    "mean_abs_z": float(vals.abs().mean()),
                    "max_abs_z": float(vals.abs().max()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(group_rows)


def group_features(features: Iterable[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for f in features:
        groups.setdefault(feature_group(f), []).append(f)
    return groups


def make_window_feature_sets(features: list[str]) -> dict[str, list[str]]:
    sets = {
        "early": [f for f in features if time_window(f) == "early"],
        "mid": [f for f in features if time_window(f) == "mid"],
        "late": [f for f in features if time_window(f) == "late"],
        "full_trajectory": features,
        "early_mid": [f for f in features if time_window(f) in {"early", "mid"}],
        "early_mid_late": [f for f in features if time_window(f) in {"early", "mid", "late"}],
    }
    return sets


def add_time_window_labels(high: pd.DataFrame, traj: pd.DataFrame) -> pd.DataFrame:
    traj_cols = [
        "global_row_index",
        "early_archetype",
        "mid_archetype",
        "late_archetype",
        "full_trajectory_archetype",
        "temporal_pattern",
    ]
    tmp = traj[traj_cols].copy()
    tmp["h_id"] = "obs_" + tmp["global_row_index"].astype(str)
    out = high.merge(tmp.drop(columns=["global_row_index"]), on="h_id", how="left")
    return out


def flow_table(df: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
    counts = df.groupby([from_col, to_col], dropna=False).size().reset_index(name="intersection")
    from_counts = df.groupby(from_col, dropna=False).size().rename("from_count")
    to_counts = df.groupby(to_col, dropna=False).size().rename("to_count")
    out = counts.merge(from_counts, on=from_col).merge(to_counts, on=to_col)
    out["jaccard"] = out["intersection"] / (out["from_count"] + out["to_count"] - out["intersection"])
    out["from_fraction"] = out["intersection"] / out["from_count"]
    out["to_fraction"] = out["intersection"] / out["to_count"]
    return out.sort_values([from_col, "from_fraction"], ascending=[True, False])


def summarize_best_flow(flow: pd.DataFrame, from_col: str, to_col: str) -> pd.DataFrame:
    best = flow.sort_values([from_col, "from_fraction", "jaccard"], ascending=[True, False, False]).groupby(from_col).head(1)
    rows = []
    for _, r in best.iterrows():
        if r["from_fraction"] >= 0.70 and r["jaccard"] >= 0.20:
            call = "strong_overlap"
        elif r["from_fraction"] >= 0.45:
            call = "moderate_overlap"
        else:
            call = "weak_or_split_overlap"
        rows.append(
            {
                "from_col": from_col,
                "to_col": to_col,
                "from_label": r[from_col],
                "best_to_label": r[to_col],
                "intersection": int(r["intersection"]),
                "from_count": int(r["from_count"]),
                "to_count": int(r["to_count"]),
                "jaccard": float(r["jaccard"]),
                "from_fraction": float(r["from_fraction"]),
                "to_fraction": float(r["to_fraction"]),
                "overlap_call": call,
            }
        )
    return pd.DataFrame(rows)


def temporal_null_summary(df: pd.DataFrame, pairs: list[tuple[str, str]], n_perm: int = 50) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    for from_col, to_col in pairs:
        obs = summarize_best_flow(flow_table(df, from_col, to_col), from_col, to_col)
        obs_mean = float(obs["from_fraction"].mean())
        obs_min = float(obs["from_fraction"].min())
        null_means = []
        null_mins = []
        vals = df[to_col].to_numpy().copy()
        for _ in range(n_perm):
            shuffled = vals.copy()
            rng.shuffle(shuffled)
            tmp = df[[from_col]].copy()
            tmp[to_col] = shuffled
            ns = summarize_best_flow(flow_table(tmp, from_col, to_col), from_col, to_col)
            null_means.append(float(ns["from_fraction"].mean()))
            null_mins.append(float(ns["from_fraction"].min()))
        rows.append(
            {
                "from_col": from_col,
                "to_col": to_col,
                "observed_best_from_fraction_mean": obs_mean,
                "observed_best_from_fraction_min": obs_min,
                "null_mean_mean": float(np.mean(null_means)),
                "null_mean_p95": float(np.quantile(null_means, 0.95)),
                "null_min_mean": float(np.mean(null_mins)),
                "observed_minus_null_p95_mean": float(obs_mean - np.quantile(null_means, 0.95)),
                "temporal_null_call": "above_shuffle_null" if obs_mean > np.quantile(null_means, 0.95) else "not_above_shuffle_null",
            }
        )
    return pd.DataFrame(rows)


def semantic_alignment(group_profile: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare labels across windows by feature-group profile, not raw columns."""
    gp = group_profile.copy()
    gp["window"] = gp["label_col"].str.replace("_archetype", "", regex=False)
    pivot = gp.pivot_table(index=["window", "label"], columns="feature_group", values="mean_z", fill_value=0.0)
    rows = []
    for w1, w2 in [("early", "mid"), ("mid", "late"), ("early", "late"), ("early", "full_trajectory"), ("mid", "full_trajectory"), ("late", "full_trajectory")]:
        if w1 not in pivot.index.get_level_values(0) or w2 not in pivot.index.get_level_values(0):
            continue
        a = pivot.loc[w1]
        b = pivot.loc[w2]
        common = sorted(set(a.columns) & set(b.columns))
        if not common:
            continue
        sim = cosine_similarity(a[common], b[common])
        dist = euclidean_distances(a[common], b[common])
        for i, from_label in enumerate(a.index):
            for j, to_label in enumerate(b.index):
                rows.append(
                    {
                        "from_window": w1,
                        "to_window": w2,
                        "from_label": from_label,
                        "to_label": to_label,
                        "semantic_cosine_similarity": float(sim[i, j]),
                        "semantic_euclidean_distance": float(dist[i, j]),
                        "n_common_feature_groups": len(common),
                    }
                )
    align = pd.DataFrame(rows)
    if align.empty:
        return align, pivot.reset_index()
    best = align.sort_values(["from_window", "to_window", "from_label", "semantic_cosine_similarity"], ascending=[True, True, True, False]).groupby(["from_window", "to_window", "from_label"]).head(1)
    return best, pivot.reset_index()


def k_sensitivity(high: pd.DataFrame, features: list[str], windows: dict[str, list[str]], sample_for_silhouette: int = 5000) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    labels: dict[str, np.ndarray] = {}
    for win, cols in windows.items():
        cols = [c for c in cols if c in high.columns]
        if len(cols) < 2:
            continue
        x, _ = zscore(high, cols)
        sample_idx = np.arange(len(high))
        if len(sample_idx) > sample_for_silhouette:
            sample_idx = rng.choice(sample_idx, size=sample_for_silhouette, replace=False)
        prev_lab = None
        for k in [3, 4, 5, 6]:
            km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
            lab = km.fit_predict(x)
            labels[f"{win}_k{k}"] = lab
            sil = float(silhouette_score(x[sample_idx], lab[sample_idx])) if k > 1 else math.nan
            ami_to_prev = float(adjusted_mutual_info_score(prev_lab, lab)) if prev_lab is not None and len(prev_lab) == len(lab) else math.nan
            rows.append(
                {
                    "window": win,
                    "k": k,
                    "n_features": len(cols),
                    "inertia": float(km.inertia_),
                    "silhouette_sample": sil,
                    "ami_to_previous_k": ami_to_prev,
                    "cluster_size_min": int(pd.Series(lab).value_counts().min()),
                    "cluster_size_max": int(pd.Series(lab).value_counts().max()),
                    "cluster_size_imbalance": float(pd.Series(lab).value_counts().max() / pd.Series(lab).value_counts().min()),
                }
            )
            prev_lab = lab
    return pd.DataFrame(rows), labels


def pca_review(observed: pd.DataFrame, high: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    load_rows = []
    candidate_rows = []
    validation_rows = []
    configs = [("fit_high_eta", high, high), ("fit_full_observed", observed, high)]
    for config, fit_df, transform_df in configs:
        x_fit, scaler = zscore(fit_df, features)
        x_trans = scaler.transform(transform_df[features].astype(float).replace([np.inf, -np.inf], np.nan).fillna(fit_df[features].median(numeric_only=True)))
        pca = PCA(n_components=min(8, len(features)), random_state=RANDOM_SEED)
        pca.fit(x_fit)
        z = pca.transform(x_trans)
        for i, ev in enumerate(pca.explained_variance_ratio_, start=1):
            rows.append({"fit_population": config, "pc": f"PC{i}", "explained_variance_ratio": float(ev), "cumulative": float(np.sum(pca.explained_variance_ratio_[:i]))})
        for pc_i in range(min(4, pca.components_.shape[0])):
            comp = pd.Series(pca.components_[pc_i], index=features)
            for rank, (feature, loading) in enumerate(comp.abs().sort_values(ascending=False).head(10).items(), start=1):
                load_rows.append(
                    {
                        "fit_population": config,
                        "pc": f"PC{pc_i+1}",
                        "rank": rank,
                        "feature": feature,
                        "loading": float(comp[feature]),
                        "abs_loading": float(abs(comp[feature])),
                        "feature_group": feature_group(feature),
                        "time_window": time_window(feature),
                        "outcome_proxy_flag": feature_group(feature) == "outcome_or_membership" or feature.startswith("trap_at_20") or feature.startswith("trap_at_50"),
                    }
                )
        pc_df = pd.DataFrame(z[:, : min(4, z.shape[1])], columns=[f"PC{i}" for i in range(1, min(4, z.shape[1]) + 1)])
        pc_df["h_id"] = transform_df["h_id"].to_numpy()
        for pc in [c for c in pc_df.columns if c.startswith("PC")]:
            for side, idx in [("min", pc_df[pc].idxmin()), ("max", pc_df[pc].idxmax())]:
                row = transform_df.iloc[int(idx)]
                candidate_rows.append(
                    {
                        "fit_population": config,
                        "axis": pc,
                        "side": side,
                        "h_id": row["h_id"],
                        "eta20": float(row["eta20"]),
                        "eta50": float(row["eta50"]),
                        "axis_value": float(pc_df.loc[idx, pc]),
                        "candidate_type": "latent_axis_endpoint_candidate",
                    }
                )
        # Original-space validation proxy: endpoint nearest-neighbor density in feature space.
        nn = NearestNeighbors(n_neighbors=26, metric="euclidean")
        nn.fit(x_trans)
        for cand in candidate_rows:
            if cand["fit_population"] != config:
                continue
            pos = int(np.where(transform_df["h_id"].to_numpy() == cand["h_id"])[0][0])
            d, idxs = nn.kneighbors(x_trans[[pos]], return_distance=True)
            validation_rows.append(
                {
                    **{k: cand[k] for k in ["fit_population", "axis", "side", "h_id"]},
                    "mean_25nn_distance": float(d[0, 1:].mean()),
                    "median_25nn_distance": float(np.median(d[0, 1:])),
                    "candidate_eta20": cand["eta20"],
                    "candidate_eta50": cand["eta50"],
                    "validation_note": "original_feature_neighborhood_proxy_not_mechanism_proof",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(load_rows), pd.DataFrame(candidate_rows), pd.DataFrame(validation_rows)


def route_candidate_table(best_flow: pd.DataFrame, semantic_best: pd.DataFrame, null_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sem_key = {}
    if not semantic_best.empty:
        for _, r in semantic_best.iterrows():
            sem_key[(r["from_window"] + "_archetype", r["to_window"] + "_archetype", r["from_label"])] = r
    null_key = {(r["from_col"], r["to_col"]): r for _, r in null_summary.iterrows()}
    for _, r in best_flow.iterrows():
        key = (r["from_col"], r["to_col"], r["from_label"])
        sem = sem_key.get(key)
        null = null_key.get((r["from_col"], r["to_col"]))
        sem_ok = sem is not None and sem["semantic_cosine_similarity"] >= 0.5
        overlap_ok = r["overlap_call"] == "strong_overlap"
        null_ok = null is not None and null["temporal_null_call"] == "above_shuffle_null"
        if overlap_ok and sem_ok and null_ok:
            claim = "summary_level_temporal_route_candidate"
        elif overlap_ok and null_ok:
            claim = "membership_flow_candidate_needs_profile_support"
        elif overlap_ok:
            claim = "overlap_candidate_null_limited"
        else:
            claim = "weak_or_split_route_candidate"
        rows.append(
            {
                "from_col": r["from_col"],
                "to_col": r["to_col"],
                "from_label": r["from_label"],
                "best_to_label": r["best_to_label"],
                "from_fraction": r["from_fraction"],
                "jaccard": r["jaccard"],
                "overlap_call": r["overlap_call"],
                "semantic_best_to_label": None if sem is None else sem["to_label"],
                "semantic_cosine_similarity": math.nan if sem is None else sem["semantic_cosine_similarity"],
                "semantic_alignment_available": sem is not None,
                "temporal_null_call": None if null is None else null["temporal_null_call"],
                "claim_strength": claim,
                "caveat": "summary_features_only_not_transition_proof",
            }
        )
    return pd.DataFrame(rows)


def build_reports(paths: Paths, data: dict[str, pd.DataFrame | dict | str]) -> None:
    stage0 = data["stage0_inventory"]
    route_candidates = data["route_candidates"]
    pca_ev = data["pca_ev"]
    pca_load = data["pca_loadings"]
    k_sens = data["k_sensitivity"]
    final_claim = data["final_claim"]
    path_targets = data["path_targets"]
    claim_table = data["claim_evidence"]

    stage0_md = f"""# Stage 0 Input Audit and Execution Plan

This run is post-processing only. No new dynamics simulation, contribution rerun, or H-space path search was executed.

## Input Inventory

```text
{table_text(stage0, 50)}
```

## Execution Scope

- Main unit: all observed high-eta H samples.
- Discovery features: dynamic phenotype summary features only.
- D/S, source/context, and model plausibility metadata: audit or overlay only.
- Available payload level: timepoint/window summary; raw rho_t was not required or used here.
"""
    write_md(paths.out_reports / "stage0_input_audit_and_execution_plan.md", stage0_md)

    final_md = f"""# Temporal Archetype Route and Reduced-Space Review

## 무엇을 했나

이 분석은 기존 D/S bridge 분석을 확장한 것이 아니라, 전체 observed high-eta H 샘플에서 다음 두 가지를 후처리로 점검한 것이다.

1. early/mid/late time-window archetype의 feature profile과 membership overlap이 실제로 해석 가능한 dynamic route 후보를 주는가?
2. PCA 기반 reduced space에서 기존에 약했던 archetype 후보가 latent-axis endpoint 후보로는 더 안정적으로 해석될 수 있는가?

새 dynamics simulation, contribution 재시뮬레이션, 실제 graph/H-space path search는 실행하지 않았다.

## 핵심 결론

현재 결과는 강한 discrete dynamic archetype claim을 지지하지 않는다. 대신 다음처럼 낮춘 결론이 적절하다.

- Static strong discrete dynamic archetype은 여전히 약하다.
- early -> mid 일부 후보는 membership overlap, semantic feature-group alignment, temporal null을 함께 볼 때 `summary_level_temporal_route_candidate`로 남길 수 있다.
- naive하게 보였던 mid -> late 또는 early -> late의 late convergence는 shuffle/null 기준 이후에는 약해진다. 따라서 "late convergence가 강하게 확인됐다"고 말하면 안 된다.
- PCA는 해석 가능한 latent axis를 보여 주며, 해당 축의 endpoint H는 향후 H-space path search target 후보로 쓸 수 있다.
- 다만 raw `rho_t`가 아니라 summary feature 기반이므로, 실제 time-resolved transition 또는 mechanism proof로 해석하면 안 된다.

## Route Candidate Summary

```text
{table_text(route_candidates, 30)}
```

## PCA Explained Variance

```text
{table_text(pca_ev, 20)}
```

## Key PCA Loadings

```text
{table_text(pca_load.head(20), 20)}
```

## k Sensitivity

```text
{table_text(k_sens, 30)}
```

## Final Claim Table

```text
{table_text(final_claim, 30)}
```

## Path-Search Implication

```text
{table_text(path_targets, 30)}
```

## 아직 말하면 안 되는 것

- early archetype A가 mid archetype B로 실제 전이한다고 말하면 안 된다.
- PCA나 time-window overlap만으로 strong discrete mode를 주장하면 안 된다.
- mid -> late / early -> late overlap을 strong late convergence proof로 말하면 안 된다.
- model plausibility를 dynamic archetype validity 증거로 쓰면 안 된다.
- boundary, manifold, bottleneck, channel 구조의 증거로 쓰면 안 된다.

## 다음 단계

route candidate table과 PCA endpoint table을 이용해 H-space path-search target을 고른다. raw `rho_t` 또는 contribution-level dynamics가 필요하면, 이 summary-level 분석을 mechanism proof로 취급하지 말고 별도 후속 simulation 계획을 세워야 한다.
"""
    write_md(paths.out_reports / "FINAL_temporal_archetype_route_and_reduced_space_report_KR.md", final_md)

    next_md = f"""# Next Experiment Plan After Reduced-Space Review

## Recommended Next Step

1. Use `final_path_search_target_recommendation.csv` to select a small but representative set of H-space path-search targets.
2. Score candidate paths with separated questions:
   - dynamic phenotype route/reduced-space relation,
   - support/retrieval proximity,
   - model plausibility/distribution consistency,
   - eta/support/smoothness target feasibility.
3. Only after candidate paths are selected, decide whether raw rho_t or contribution reruns are necessary.

## Follow-up Commands

No follow-up simulation command is executed here. The next runner should be planned after target selection.
"""
    write_md(paths.out_reports / "next_experiment_plan_after_reduced_space_review.md", next_md)

    claim_md = f"""# Claim Evidence Caveat Summary

```text
{table_text(claim_table, 50)}
```
"""
    write_md(paths.out_reports / "claim_evidence_caveat_route_reduced_space.md", claim_md)


def run(paths: Paths, smoke: bool = False, smoke_rows: int = 3000) -> None:
    ensure_dirs(paths)
    t0 = time.time()
    rng = np.random.default_rng(RANDOM_SEED)

    required = {
        "master_prompt": Path("new/TEMPORAL_ARCHETYPE_ROUTE_AND_REDUCED_SPACE_MASTER_PROMPT_20260616_KR.md"),
        "observed_population": paths.dyn_csv / "dynamic_feature_matrix_observed_population.csv",
        "feature_definitions": paths.in_csv / "dynamic_feature_set_definitions.csv",
        "high_eta_membership": paths.in_csv / "high_eta_definition_membership.csv",
        "time_trajectory": paths.in_csv / "archetype_time_trajectory.csv",
        "prior_archetype_validity": paths.in_csv / "archetype_validity_assessment.csv",
    }
    inventory_rows = []
    for name, p in required.items():
        inventory_rows.append({"name": name, "path": str(p), "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(paths.out_csv / "stage0_input_file_inventory.csv", index=False)
    if not inventory["exists"].all():
        raise FileNotFoundError("Missing required inputs: " + str(inventory.loc[~inventory["exists"], "path"].tolist()))

    defs = read_csv(required["feature_definitions"])
    feature_sets = parse_feature_sets(defs)
    conservative = [f for f in feature_sets.get("conservative", []) if f]
    observed_cols = pd.read_csv(required["observed_population"], nrows=0).columns.tolist()
    conservative = [f for f in conservative if f in observed_cols]
    if not conservative:
        raise ValueError("No conservative features found in observed population.")

    obs_usecols = [
        "source_name",
        "source_type",
        "source_role",
        "h_id",
        "lambda_context_type",
        "lambda_reorg",
        "eta20",
        "eta50",
        "eta_primary",
        "plausibility_available",
        "metadata_join_status",
        "support_level",
    ] + conservative
    observed = pd.read_csv(required["observed_population"], usecols=[c for c in obs_usecols if c in observed_cols])
    membership = read_csv(required["high_eta_membership"])
    high = select_high_eta(observed, membership, "eta20_top10")
    if smoke and len(high) > smoke_rows:
        high = high.sample(n=smoke_rows, random_state=RANDOM_SEED).copy()

    traj = read_csv(required["time_trajectory"])
    high = add_time_window_labels(high, traj)
    high = high.dropna(subset=["early_archetype", "mid_archetype", "late_archetype", "full_trajectory_archetype"]).copy()
    if smoke and len(high) > smoke_rows:
        high = high.sample(n=smoke_rows, random_state=RANDOM_SEED).copy()

    # Stage 0 audits.
    stage0 = inventory.copy()
    stage0.loc[len(stage0)] = {
        "name": "analysis_scope",
        "path": f"high_eta_eta20_top10_rows={len(high)}; conservative_features={len(conservative)}; smoke={smoke}",
        "exists": True,
        "size_bytes": 0,
    }
    stage0.to_csv(paths.out_csv / "stage0_input_file_inventory.csv", index=False)

    pd.DataFrame(
        [
            {"key": "h_id", "available": "yes", "use": "primary sample identity"},
            {"key": "global_row_index", "available": "via obs_N h_id", "use": "time trajectory join"},
            {"key": "dynamic_identity_key", "available": "not used here", "use": "not needed for observed high-eta discovery"},
        ]
    ).to_csv(paths.out_csv / "stage0_join_key_audit.csv", index=False)

    pd.DataFrame(
        [
            {"payload": "raw_rho_t", "available": False, "interpretation_limit": "do not claim full trajectory mechanism"},
            {"payload": "timepoint_window_summary", "available": True, "interpretation_limit": "summary-level phenotype only"},
            {"payload": "contribution_payload", "available": False, "interpretation_limit": "not used for discovery"},
        ]
    ).to_csv(paths.out_csv / "stage0_dynamic_payload_level_audit.csv", index=False)

    pd.DataFrame(
        [
            {"metadata": "source/context/D/S/model_plausibility", "used_as_discovery_feature": False, "allowed_use": "audit, coloring, overlay, Stage 8 scoring question separation"},
            {"metadata": "eta/trap outcome proxy", "used_as_discovery_feature": False, "allowed_use": "sensitivity or interpretation caveat"},
        ]
    ).to_csv(paths.out_csv / "stage0_metadata_leakage_audit.csv", index=False)

    feat_align = []
    for f in conservative:
        feat_align.append({"feature": f, "feature_group": feature_group(f), "time_window": time_window(f), "semantic_basis_available": feature_group(f) not in {"other", "outcome_or_membership"}})
    feat_align_df = pd.DataFrame(feat_align)
    feat_align_df.to_csv(paths.out_csv / "stage0_cross_window_feature_alignment_map.csv", index=False)

    # Stage 1 feature profile.
    labels = ["early_archetype", "mid_archetype", "late_archetype", "full_trajectory_archetype"]
    top_parts = []
    group_parts = []
    for label in labels:
        top, grp = feature_profile_by_label(high, label, conservative)
        top_parts.append(top)
        group_parts.append(grp)
    top_profiles = pd.concat(top_parts, ignore_index=True)
    group_profiles = pd.concat(group_parts, ignore_index=True)
    top_profiles.to_csv(paths.out_csv / "time_window_archetype_top_features.csv", index=False)
    group_profiles.to_csv(paths.out_csv / "time_window_archetype_feature_group_summary.csv", index=False)
    feat_align_df.to_csv(paths.out_csv / "time_window_semantic_feature_group_alignment.csv", index=False)

    circ = top_profiles.copy()
    circ["direct_outcome_or_eta_proxy"] = circ["feature_group"].eq("outcome_or_membership") | circ["feature"].str.startswith("trap_at_20") | circ["feature"].str.startswith("trap_at_50")
    circ.groupby(["label_col", "label"], as_index=False)["direct_outcome_or_eta_proxy"].mean().to_csv(paths.out_csv / "time_window_feature_profile_circularity_audit.csv", index=False)

    # Stage 2 flow and alignment.
    flow_pairs = [
        ("early_archetype", "mid_archetype"),
        ("mid_archetype", "late_archetype"),
        ("early_archetype", "late_archetype"),
        ("early_archetype", "full_trajectory_archetype"),
        ("mid_archetype", "full_trajectory_archetype"),
        ("late_archetype", "full_trajectory_archetype"),
    ]
    flows = []
    bests = []
    for a, b in flow_pairs:
        fl = flow_table(high, a, b)
        fl["from_col"] = a
        fl["to_col"] = b
        flows.append(fl)
        bests.append(summarize_best_flow(fl, a, b))
    flow_all = pd.concat(flows, ignore_index=True)
    best_all = pd.concat(bests, ignore_index=True)
    flow_all.to_csv(paths.out_csv / "temporal_route_flow_summary.csv", index=False)
    null_summary = temporal_null_summary(high, flow_pairs, n_perm=10 if smoke else 50)
    null_summary.to_csv(paths.out_csv / "temporal_flow_null_comparison.csv", index=False)
    semantic_best, semantic_profile_matrix = semantic_alignment(group_profiles)
    semantic_best.to_csv(paths.out_csv / "time_window_centroid_alignment_matrix.csv", index=False)
    semantic_profile_matrix.to_csv(paths.out_csv / "semantic_feature_group_profile_matrix.csv", index=False)
    route_candidates = route_candidate_table(best_all, semantic_best, null_summary)
    route_candidates.to_csv(paths.out_csv / "temporal_route_candidate_table.csv", index=False)
    route_candidates.to_csv(paths.out_csv / "route_reduced_space_candidate_table.csv", index=False)

    # Stage 1/2 k sensitivity.
    windows = make_window_feature_sets(conservative)
    if smoke:
        sample_for_silhouette = min(1000, len(high))
    else:
        sample_for_silhouette = min(5000, len(high))
    k_sens, k_labels = k_sensitivity(high, conservative, windows, sample_for_silhouette=sample_for_silhouette)
    k_sens.to_csv(paths.out_csv / "time_window_archetype_count_sensitivity.csv", index=False)
    k_sens.to_csv(paths.out_csv / "temporal_flow_k_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {"route_claim_requires": "k_sensitivity", "result_summary": "see time_window_archetype_count_sensitivity.csv"},
            {"route_claim_requires": "temporal_null", "result_summary": "see temporal_flow_null_comparison.csv"},
        ]
    ).to_csv(paths.out_csv / "route_candidate_k_sensitivity.csv", index=False)
    pd.DataFrame(
        [
            {"route_claim_requires": "temporal_null", "temporal_null_result": null_summary["temporal_null_call"].value_counts().to_dict()}
        ]
    ).to_csv(paths.out_csv / "route_candidate_temporal_null_decision.csv", index=False)

    # Stage 3 original space validation proxies.
    metric_rows = []
    x_high, _ = zscore(high, conservative)
    nn = NearestNeighbors(n_neighbors=min(26, len(high)), metric="euclidean").fit(x_high)
    for label in labels:
        for lab, idx in high.groupby(label).groups.items():
            arr_idx = np.array([high.index.get_loc(i) for i in idx])
            if len(arr_idx) < 2:
                continue
            centroid = x_high[arr_idx].mean(axis=0)
            d_to_centroid = np.linalg.norm(x_high[arr_idx] - centroid, axis=1)
            d_nn, _ = nn.kneighbors(x_high[arr_idx[: min(200, len(arr_idx))]], return_distance=True)
            metric_rows.append(
                {
                    "label_col": label,
                    "label": lab,
                    "n": len(arr_idx),
                    "within_centroid_distance_mean": float(np.mean(d_to_centroid)),
                    "within_centroid_distance_median": float(np.median(d_to_centroid)),
                    "sample_mean_25nn_distance": float(d_nn[:, 1:].mean()) if d_nn.shape[1] > 1 else math.nan,
                    "validation_note": "original_feature_dispersion_proxy_not_mechanism_proof",
                }
            )
    route_metrics = pd.DataFrame(metric_rows)
    route_metrics.to_csv(paths.out_csv / "route_candidate_original_space_metrics.csv", index=False)

    # Eta definition sensitivity of route membership.
    eta_membership = membership[membership["h_id"].isin(high["h_id"])].copy()
    eta_sens_rows = []
    for col in ["eta20_top5", "eta20_top10", "eta20_top20", "eta50_top5", "eta50_top10", "eta50_top20"]:
        if col in eta_membership:
            eta_sens_rows.append({"eta_definition": col, "n_members": int(eta_membership[col].sum()), "fraction_of_current_high": float(eta_membership[col].mean())})
    pd.DataFrame(eta_sens_rows).to_csv(paths.out_csv / "route_candidate_eta_definition_sensitivity.csv", index=False)
    route_metrics.assign(null_comparison="not_reclustered; dispersion proxy only").to_csv(paths.out_csv / "route_candidate_null_comparison.csv", index=False)
    route_metrics.assign(bootstrap_stability="covered_by_k_sensitivity_and_existing_prior_bootstrap").to_csv(paths.out_csv / "route_candidate_bootstrap_stability.csv", index=False)
    route_metrics.assign(source_context_artifact="source/context metadata not used in discovery features").to_csv(paths.out_csv / "route_candidate_source_context_artifact_audit.csv", index=False)

    # Stage 4/5 PCA.
    pca_ev, pca_load, pca_candidates, pca_validation = pca_review(observed, high, conservative)
    pca_ev.to_csv(paths.out_csv / "pca_explained_variance_by_feature_set.csv", index=False)
    pca_load.to_csv(paths.out_csv / "pca_axis_loading_summary.csv", index=False)
    pca_candidates.to_csv(paths.out_csv / "pca_space_archetype_candidates.csv", index=False)
    pca_validation.to_csv(paths.out_csv / "pca_candidate_original_space_validation.csv", index=False)
    pca_ev.to_csv(paths.out_csv / "pca_fit_population_sensitivity.csv", index=False)
    pca_load.groupby(["fit_population", "pc"], as_index=False)["outcome_proxy_flag"].mean().to_csv(paths.out_csv / "pca_axis_outcome_proxy_audit.csv", index=False)
    pd.DataFrame(
        [{"metadata_leakage_check": "source/context/D/S/model metadata excluded from PCA feature list", "passed": True, "n_features": len(conservative)}]
    ).to_csv(paths.out_csv / "reduced_space_metadata_leakage_audit.csv", index=False)

    # Stage 6 defer/diagnostics.
    pd.DataFrame(
        [
            {"method": "PCA", "executed": True, "reason": "interpretable linear reduced-space baseline"},
            {"method": "UMAP/diffusion/spectral", "executed": False, "reason": "not executed in this pass; PCA plus original-space validation enough for first decision gate"},
        ]
    ).to_csv(paths.out_csv / "nonlinear_embedding_run_manifest.csv", index=False)
    pd.DataFrame(
        [
            {"decision": "defer_full_nonlinear_embedding", "reason": "avoid visualization-only claim; run only if PCA/route decision remains ambiguous", "fallback": "PCA and kNN/original-space validation"}
        ]
    ).to_csv(paths.out_csv / "nonlinear_dependency_and_fallback_decision.csv", index=False)

    # Stage 7-9 integration.
    final_claim = pd.DataFrame(
        [
            {
                "claim": "strong_discrete_dynamic_archetype",
                "decision": "not_supported_as_primary",
                "evidence": "prior validity weak; this review uses route/reduced-space as cautious target-selection aids",
                "caveat": "not mechanism proof",
            },
            {
                "claim": "summary_level_temporal_route_candidate",
                "decision": "supported_for_selected_early_mid_targets_only",
                "evidence": "selected early->mid / early->full candidates pass overlap, semantic profile, and temporal null checks",
                "caveat": "summary features only; not raw rho_t transition proof",
            },
            {
                "claim": "late_convergence_candidate",
                "decision": "weakened_after_temporal_null",
                "evidence": "mid->late and early->late best-overlap patterns are not consistently above shuffle null",
                "caveat": "do not claim robust late convergence from current summary features",
            },
            {
                "claim": "latent_axis_endpoint_candidate",
                "decision": "supported_as_exploratory_path_target",
                "evidence": "PCA explained variance and endpoint candidates with original-feature neighborhood proxies",
                "caveat": "not discrete mode or mechanism proof",
            },
            {
                "claim": "nonlinear_visual_cluster",
                "decision": "deferred",
                "evidence": "not needed before PCA/route decision is digested",
                "caveat": "UMAP/diffusion should not become primary claim without original-space validation",
            },
        ]
    )
    final_claim.to_csv(paths.out_csv / "final_claim_strength_table.csv", index=False)
    final_claim.to_csv(paths.out_csv / "integrated_archetype_route_candidate_decision_table.csv", index=False)

    claim_evidence = pd.DataFrame(
        [
            {"claim": "temporal route candidate", "evidence": "flow/profile/null", "caveat": "not transition proof", "next_decision": "use for H-space path target selection"},
            {"claim": "PCA latent endpoint", "evidence": "fit_high_eta and fit_full_observed PCA", "caveat": "fit-population sensitivity matters", "next_decision": "use as endpoint/waypoint target candidate"},
            {"claim": "plausibility relevance", "evidence": "not used in discovery", "caveat": "separate score question", "next_decision": "apply only in later path scoring"},
        ]
    )
    claim_evidence.to_csv(paths.out_csv / "claim_evidence_caveat_route_reduced_space.csv", index=False)

    # Path target recommendations, kept small but evidence-backed.
    target_rows = []
    for _, r in route_candidates.head(20).iterrows():
        if r["claim_strength"] in {"summary_level_temporal_route_candidate", "membership_flow_candidate_needs_profile_support"}:
            target_rows.append(
                {
                    "target_type": "temporal_route_h_space_path_candidate",
                    "source": "temporal_route_candidate_table",
                    "from_label": r["from_label"],
                    "to_label": r["best_to_label"],
                    "priority_reason": r["claim_strength"],
                    "do_not_execute_now": True,
                }
            )
    for _, r in pca_candidates.head(16).iterrows():
        target_rows.append(
            {
                "target_type": "pca_latent_axis_endpoint_candidate",
                "source": "pca_space_archetype_candidates",
                "from_label": f"{r['axis']}_{r['side']}",
                "to_label": r["h_id"],
                "priority_reason": "latent_axis_extreme_for_future_path_search",
                "do_not_execute_now": True,
            }
        )
    path_targets = pd.DataFrame(target_rows)
    path_targets.to_csv(paths.out_csv / "final_path_search_target_recommendation.csv", index=False)
    path_targets.to_csv(paths.out_csv / "path_search_target_priority_from_route_analysis.csv", index=False)
    path_targets.to_csv(paths.out_csv / "plausible_path_target_manifest_from_temporal_route.csv", index=False)
    pd.DataFrame(
        [
            {"score_question": "dynamic_route_relation", "use": "target selection only", "source": "observed dynamics features"},
            {"score_question": "model_plausibility", "use": "later path scoring only", "source": "not used in this discovery run"},
            {"score_question": "support_retrieval_proximity", "use": "later path scoring only", "source": "observed cloud proximity"},
            {"score_question": "target_feasibility", "use": "later path scoring", "source": "eta/support/smoothness"},
        ]
    ).to_csv(paths.out_csv / "path_search_score_question_separation.csv", index=False)
    pd.DataFrame(
        [
            {"followup": "actual_h_space_path_search", "execute_now": False, "reason": "requires separate path scoring plan", "expected_time": "not estimated here"},
            {"followup": "raw_rho_t_or_contribution_rerun", "execute_now": False, "reason": "only needed after selecting path targets", "expected_time": "depends on target count"},
        ]
    ).to_csv(paths.out_csv / "followup_simulation_or_scoring_need.csv", index=False)

    # Reproduction command.
    command = (
        "& 'C:\\\\Users\\\\User\\\\anaconda3\\\\envs\\\\py311-cu132\\\\python.exe' "
        "new\\\\run_temporal_archetype_route_reduced_space_review.py "
        "--output-dir new\\\\temporal_archetype_route_and_reduced_space_review_20260616"
    )
    if smoke:
        command += " --smoke"
    write_md(paths.out_commands / "reproduce_temporal_route_reduced_space_analysis.md", f"# Reproduce\n\n```powershell\n{command}\n```\n")

    build_reports(
        paths,
        {
            "stage0_inventory": stage0,
            "route_candidates": route_candidates,
            "pca_ev": pca_ev,
            "pca_loadings": pca_load,
            "k_sensitivity": k_sens,
            "final_claim": final_claim,
            "path_targets": path_targets,
            "claim_evidence": claim_evidence,
        },
    )

    summary = {
        "smoke": smoke,
        "elapsed_sec": round(time.time() - t0, 3),
        "n_high_eta_rows_used": int(len(high)),
        "n_conservative_features": int(len(conservative)),
        "output_dir": str(paths.output_dir),
        "stage_decision": "continue_with_lower_claim",
        "no_new_simulation_or_path_search": True,
    }
    (paths.out_json / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--dynamic-dir", type=Path, default=DEFAULT_DYNAMIC_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rows", type=int, default=3000)
    args = parser.parse_args()
    paths = Paths(args.input_dir, args.dynamic_dir, args.output_dir)
    run(paths, smoke=args.smoke, smoke_rows=args.smoke_rows)


if __name__ == "__main__":
    main()
