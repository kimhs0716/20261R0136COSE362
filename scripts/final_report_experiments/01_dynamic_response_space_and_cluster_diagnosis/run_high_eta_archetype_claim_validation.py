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
from scipy.optimize import nnls
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "new" / "bridge_dynamic_phenotype_mode_discovery_20260616" / "csv"
OBSERVED_CSV = SOURCE_DIR / "dynamic_feature_matrix_observed_population.csv"
OVERLAY_CSV = SOURCE_DIR / "dynamic_feature_matrix_path_overlay.csv"
ETA_INVENTORY_CSV = SOURCE_DIR / "eta_definition_inventory.csv"
FEATURE_ABLATION_CSV = SOURCE_DIR / "feature_family_ablation_stability.csv"
SOURCE_AUDIT_CSV = SOURCE_DIR / "source_time_lambda_audit.csv"
PATH_TARGET_CSV = SOURCE_DIR / "path_search_target_pairs_by_dynamic_mode.csv"
DEFAULT_OUTPUT_DIR = ROOT / "new" / "high_eta_archetype_claim_validation_20260616"


META_COLS = {
    "source_name",
    "source_type",
    "source_role",
    "source_file_id",
    "source_row",
    "row_index_within_source",
    "h_id",
    "dynamic_identity_key",
    "h_identity_key",
    "lambda_context_type",
    "lambda_reorg",
    "pair_id",
    "alpha",
    "solver_success",
    "metadata_join_status",
    "plausibility_available",
    "support_level",
    "eta_label",
    "left_family",
    "right_family",
    "left_medoid_real_row",
    "right_medoid_real_row",
    "solver_error",
    "candidate_rank",
    "group_name",
    "waypoint_source_alpha",
    "waypoint_source_lambda",
    "waypoint_source_radius",
    "waypoint_source_direction",
    "target_priority",
    "target_category",
    "feature_set",
    "radius",
    "radius_mode",
    "normal_direction_i",
}

ETA_COLS = {"eta5", "eta10", "eta20", "eta50", "eta_final", "eta_primary"}
DIRECT_LEAKAGE_COLS = {"eta5", "eta10", "eta20", "eta50", "eta_final", "eta_primary", "trap_at_20ps", "trap_at_50ps"}
NEAR_OUTCOME_COLS = {"trap_at_10ps", "t80", "t90", "tau_transfer"}
EARLY_MARKERS = ("_at_6ps", "mean_0_5ps")
MID_MARKERS = ("_at_10ps", "_at_20ps", "mean_5_10ps", "mean_10_20ps")
LATE_MARKERS = ("_at_50ps", "tau_transfer", "t80", "t90")


@dataclass
class RunContext:
    out: Path
    csv: Path
    reports: Path
    commands: Path
    logs: Path
    json_dir: Path
    smoke: bool
    rng: np.random.Generator
    started_at: float


def ensure_dirs(out: Path) -> RunContext:
    csv = out / "csv"
    reports = out / "reports"
    commands = out / "commands"
    logs = out / "logs"
    json_dir = out / "json"
    for p in [out, csv, reports, commands, logs, json_dir]:
        p.mkdir(parents=True, exist_ok=True)
    return RunContext(out, csv, reports, commands, logs, json_dir, False, np.random.default_rng(0), time.time())


def log(ctx: RunContext, message: str) -> None:
    elapsed = time.time() - ctx.started_at
    line = f"[{elapsed:8.1f}s] {message}"
    print(line, flush=True)
    with (ctx.logs / "progress.log").open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_csv(path: Path, nrows: int | None = None, usecols: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, usecols=usecols, low_memory=False)


def numeric_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c in META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols


def classify_feature(col: str) -> tuple[str, str]:
    if col in ETA_COLS:
        return "outcome_eta", "exclude_direct_outcome"
    if col in DIRECT_LEAKAGE_COLS:
        return "near_or_direct_outcome", "exclude_main_use_sensitivity"
    if col in NEAR_OUTCOME_COLS:
        return "timing_or_near_outcome", "sensitivity"
    if "cl1_" in col:
        return "coherence", "main"
    if "purity_" in col:
        return "purity", "main"
    if "ipr_" in col:
        return "ipr", "main"
    if col.startswith(("site", "sink", "detour", "loss", "residual")):
        return "population", "main"
    if col.startswith(("eig_", "h_param", "H_fro")):
        return "structural_proxy", "sensitivity"
    if col.startswith("t"):
        return "timing", "sensitivity"
    return "other_numeric", "sensitivity"


def build_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    nums = numeric_cols(df)
    quality = []
    rows = []
    for c in nums:
        family, tier = classify_feature(c)
        std = float(pd.to_numeric(df[c], errors="coerce").std(skipna=True))
        miss = float(pd.to_numeric(df[c], errors="coerce").isna().mean())
        rows.append({"feature": c, "family": family, "tier": tier, "std": std, "missing_fraction": miss})
        if miss < 0.2 and std >= 1e-8:
            quality.append(c)
    conservative = [c for c in quality if classify_feature(c)[1] == "main"]
    conservative = [c for c in conservative if c not in DIRECT_LEAKAGE_COLS]
    expanded = [c for c in quality if c not in ETA_COLS]
    expanded = [c for c in expanded if c not in {"eta_primary", "eta_final"}]
    early = [c for c in conservative if any(m in c for m in EARLY_MARKERS)]
    mid = [c for c in conservative if any(m in c for m in MID_MARKERS)]
    late = [c for c in expanded if any(m in c for m in LATE_MARKERS)]
    return {
        "rows": rows,
        "conservative": conservative,
        "expanded": expanded,
        "early": early,
        "mid": mid,
        "late": late,
        "full_trajectory": conservative,
    }


def high_eta_definitions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    defs = {}
    rows = []
    for col in ["eta20", "eta50"]:
        if col not in df.columns:
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        valid = vals.dropna()
        for pct in [5, 10, 20]:
            thr = float(valid.quantile(1 - pct / 100.0))
            name = f"{col}_top{pct}"
            mask = vals >= thr
            defs[name] = mask.fillna(False)
            rows.append(
                {
                    "definition": name,
                    "eta_column": col,
                    "top_percent": pct,
                    "threshold": thr,
                    "n_valid_eta": int(valid.shape[0]),
                    "n_members": int(mask.sum()),
                    "member_fraction": float(mask.mean()),
                }
            )
    names = list(defs)
    overlap = []
    for a in names:
        for b in names:
            ma, mb = defs[a], defs[b]
            inter = int((ma & mb).sum())
            union = int((ma | mb).sum())
            overlap.append(
                {
                    "definition_a": a,
                    "definition_b": b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else np.nan,
                    "a_in_b_fraction": inter / int(ma.sum()) if int(ma.sum()) else np.nan,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(overlap), defs


def prepare_scaled(df: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, StandardScaler, pd.Series]:
    use = df[features].apply(pd.to_numeric, errors="coerce")
    med = use.median(axis=0, skipna=True)
    use = use.fillna(med)
    scaler = StandardScaler()
    x = scaler.fit_transform(use)
    return x.astype(np.float32), scaler, med


def transform_scaled(df: pd.DataFrame, features: list[str], scaler: StandardScaler, med: pd.Series) -> np.ndarray:
    use = df.reindex(columns=features).apply(pd.to_numeric, errors="coerce").fillna(med)
    return scaler.transform(use).astype(np.float32)


def sample_indices(mask: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    idx = np.flatnonzero(mask)
    if idx.size <= n:
        return idx
    return rng.choice(idx, size=n, replace=False)


def pairwise_distance_summary(x: np.ndarray, rng: np.random.Generator, max_n: int = 2000) -> dict[str, float]:
    if x.shape[0] <= 1:
        return {"pairwise_mean": np.nan, "pairwise_median": np.nan}
    if x.shape[0] > max_n:
        x = x[rng.choice(x.shape[0], size=max_n, replace=False)]
    d = pairwise_distances(x, metric="euclidean")
    tri = d[np.triu_indices_from(d, k=1)]
    return {"pairwise_mean": float(np.nanmean(tri)), "pairwise_median": float(np.nanmedian(tri))}


def heterogeneity_metrics(
    df: pd.DataFrame,
    x: np.ndarray,
    defs: dict[str, pd.Series],
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    eta20 = pd.to_numeric(df.get("eta20"), errors="coerce") if "eta20" in df else pd.Series(np.nan, index=df.index)
    low_mid = (eta20 < eta20.quantile(0.5)).fillna(False).to_numpy()
    all_mask = np.ones(x.shape[0], dtype=bool)
    for name, mask_s in defs.items():
        mask = mask_s.to_numpy()
        n = int(mask.sum())
        controls = {
            "high_definition": mask,
            "row_count_random_observed": np.isin(np.arange(x.shape[0]), sample_indices(all_mask, n, rng)),
            "row_count_low_mid_control": np.isin(np.arange(x.shape[0]), sample_indices(low_mid, n, rng)),
        }
        for label, cmask in controls.items():
            xs = x[cmask]
            if xs.size == 0:
                continue
            var_mean = float(np.nanmean(np.nanvar(xs, axis=0)))
            dist = pairwise_distance_summary(xs, rng)
            rows.append(
                {
                    "definition": name,
                    "group": label,
                    "n_rows": int(cmask.sum()),
                    "mean_feature_variance": var_mean,
                    **dist,
                }
            )
    return pd.DataFrame(rows)


def pca_loadings(x: np.ndarray, features: list[str], out_n: int = 5) -> tuple[pd.DataFrame, pd.DataFrame, PCA]:
    pca = PCA(n_components=min(out_n, x.shape[1], x.shape[0]), random_state=0)
    z = pca.fit_transform(x)
    ev = pd.DataFrame(
        {
            "pc": [f"PC{i+1}" for i in range(len(pca.explained_variance_ratio_))],
            "explained_variance_ratio": pca.explained_variance_ratio_,
        }
    )
    rows = []
    for i, comp in enumerate(pca.components_):
        order = np.argsort(np.abs(comp))[::-1][:15]
        for rank, j in enumerate(order, start=1):
            rows.append({"pc": f"PC{i+1}", "rank": rank, "feature": features[j], "loading": float(comp[j])})
    return ev, pd.DataFrame(rows), pca


def farthest_anchors(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    center = x.mean(axis=0, keepdims=True)
    first = int(np.argmax(np.linalg.norm(x - center, axis=1)))
    anchors = [first]
    min_dist = np.linalg.norm(x - x[first], axis=1)
    for _ in range(1, k):
        nxt = int(np.argmax(min_dist))
        anchors.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(x - x[nxt], axis=1))
    return np.array(anchors, dtype=int)


def nearest_anchor_assignment(x: np.ndarray, anchor_x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = pairwise_distances(x, anchor_x)
    labels = np.argmin(d, axis=1)
    mins = np.min(d, axis=1)
    return labels, mins


def reconstruction_errors(x: np.ndarray, anchor_x: np.ndarray, max_rows: int, rng: np.random.Generator) -> pd.DataFrame:
    idx = np.arange(x.shape[0])
    if x.shape[0] > max_rows:
        idx = rng.choice(x.shape[0], size=max_rows, replace=False)
    a = anchor_x.T
    rows = []
    for i in idx:
        coeff, _ = nnls(a, x[i])
        s = coeff.sum()
        if s > 0:
            coeff = coeff / s
        recon = coeff @ anchor_x
        err = float(np.linalg.norm(x[i] - recon))
        rows.append({"sample_index_within_subset": int(i), "reconstruction_error": err, "max_coeff": float(coeff.max() if coeff.size else np.nan)})
    return pd.DataFrame(rows)


def make_archetypes(
    df: pd.DataFrame,
    x: np.ndarray,
    features: list[str],
    mask: np.ndarray,
    k: int,
    rng: np.random.Generator,
    label_prefix: str,
    max_recon_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    subset_idx = np.flatnonzero(mask)
    xs = x[subset_idx]
    anchor_local = farthest_anchors(xs, k, rng)
    anchor_global = subset_idx[anchor_local]
    anchor_x = x[anchor_global]
    labels_local, dist = nearest_anchor_assignment(xs, anchor_x)
    labels = np.full(x.shape[0], -1, dtype=int)
    labels[subset_idx] = labels_local
    inv = []
    for a_i, g in enumerate(anchor_global):
        row = df.iloc[int(g)]
        inv.append(
            {
                "archetype_label": f"{label_prefix}_A{a_i}",
                "method": "farthest_point_anchor",
                "global_row_index": int(g),
                "h_id": row.get("h_id", ""),
                "eta20": row.get("eta20", np.nan),
                "eta50": row.get("eta50", np.nan),
                "nearest_member_count": int((labels_local == a_i).sum()),
                "mean_nearest_distance": float(np.mean(dist[labels_local == a_i])) if (labels_local == a_i).sum() else np.nan,
            }
        )
    profile_rows = []
    for a_i in range(k):
        members = xs[labels_local == a_i]
        if members.size == 0:
            continue
        mean_z = members.mean(axis=0)
        order = np.argsort(np.abs(mean_z))[::-1][:20]
        for rank, j in enumerate(order, start=1):
            profile_rows.append(
                {
                    "archetype_label": f"{label_prefix}_A{a_i}",
                    "rank": rank,
                    "feature": features[j],
                    "mean_z": float(mean_z[j]),
                }
            )
    recon = reconstruction_errors(xs, anchor_x, max_recon_rows, rng)
    recon["definition"] = label_prefix
    return pd.DataFrame(inv), pd.DataFrame(profile_rows), recon, labels, anchor_x


def kmeans_stability(x: np.ndarray, mask: np.ndarray, rng: np.random.Generator, k_values: Iterable[int]) -> pd.DataFrame:
    idx = np.flatnonzero(mask)
    if idx.size > 12000:
        idx = rng.choice(idx, size=12000, replace=False)
    xs = x[idx]
    rows = []
    for k in k_values:
        if xs.shape[0] <= k:
            continue
        km = KMeans(n_clusters=k, n_init=10, random_state=17)
        labels = km.fit_predict(xs)
        sil_n = min(5000, xs.shape[0])
        sil_idx = rng.choice(xs.shape[0], size=sil_n, replace=False) if xs.shape[0] > sil_n else np.arange(xs.shape[0])
        sil = silhouette_score(xs[sil_idx], labels[sil_idx]) if len(np.unique(labels[sil_idx])) > 1 else np.nan
        idx1 = rng.choice(xs.shape[0], size=xs.shape[0] // 2, replace=False)
        idx2 = rng.choice(xs.shape[0], size=xs.shape[0] // 2, replace=False)
        l1 = KMeans(n_clusters=k, n_init=5, random_state=31).fit_predict(xs[idx1])
        l2_full = KMeans(n_clusters=k, n_init=5, random_state=37).fit(xs[idx2]).predict(xs[idx1])
        ami = adjusted_mutual_info_score(l1, l2_full)
        rows.append({"k": k, "n_rows_sampled": int(xs.shape[0]), "silhouette_sample": float(sil), "split_half_ami_proxy": float(ami)})
    return pd.DataFrame(rows)


def nmf_candidate(x: np.ndarray, mask: np.ndarray, features: list[str], k: int, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.flatnonzero(mask)
    if idx.size > 20000:
        idx = rng.choice(idx, size=20000, replace=False)
    xs = x[idx]
    # Shift to nonnegative for NMF; this is a diagnostic, not the primary claim.
    xs_nonneg = xs - xs.min(axis=0, keepdims=True)
    model = NMF(n_components=k, init="nndsvda", random_state=23, max_iter=500)
    w = model.fit_transform(xs_nonneg)
    comp = model.components_
    rows = []
    for a_i, weights in enumerate(w.T):
        top_local = int(np.argmax(weights))
        global_i = int(idx[top_local])
        rows.append(
            {
                "archetype_label": f"nmf_A{a_i}",
                "method": "nmf_component_high_weight_representative",
                "global_row_index": global_i,
                "component_weight_max": float(weights[top_local]),
            }
        )
    prof = []
    for a_i, c in enumerate(comp):
        order = np.argsort(c)[::-1][:20]
        for rank, j in enumerate(order, start=1):
            prof.append({"archetype_label": f"nmf_A{a_i}", "rank": rank, "feature": features[j], "component_weight": float(c[j])})
    return pd.DataFrame(rows), pd.DataFrame(prof)


def time_window_archetypes(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    defs: dict[str, pd.Series],
    main_def: str,
    k: int,
    rng: np.random.Generator,
    max_recon_rows: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mask = defs[main_def].to_numpy()
    inv_all = []
    traj = pd.DataFrame({"global_row_index": np.flatnonzero(mask)})
    stability = []
    for window in ["early", "mid", "late", "full_trajectory"]:
        feats = feature_sets.get(window, [])
        if len(feats) < 2:
            continue
        xw, _, _ = prepare_scaled(df, feats)
        inv, prof, recon, labels, _ = make_archetypes(df, xw, feats, mask, k, rng, f"{window}_{main_def}", max_recon_rows)
        inv["time_window"] = window
        prof["time_window"] = window
        recon["time_window"] = window
        inv_all.append(inv)
        subset_labels = labels[mask]
        traj[f"{window}_archetype"] = [f"{window}_{main_def}_A{int(v)}" if v >= 0 else "" for v in subset_labels]
        traj[f"{window}_archetype_id"] = subset_labels
        stability.append(
            {
                "time_window": window,
                "n_features": len(feats),
                "reconstruction_error_mean": float(recon["reconstruction_error"].mean()) if not recon.empty else np.nan,
                "reconstruction_error_median": float(recon["reconstruction_error"].median()) if not recon.empty else np.nan,
            }
        )
    if all(c in traj.columns for c in ["early_archetype", "mid_archetype", "late_archetype"]):
        traj["temporal_pattern"] = traj["early_archetype"] + " -> " + traj["mid_archetype"] + " -> " + traj["late_archetype"]
        traj["temporal_transition_count"] = np.nan
        traj["temporal_transition_note"] = "window_specific_labels_are_independent_not_directly_comparable"
    return pd.concat(inv_all, ignore_index=True) if inv_all else pd.DataFrame(), traj, pd.DataFrame(stability)


def threshold_stability(
    df: pd.DataFrame,
    x: np.ndarray,
    defs: dict[str, pd.Series],
    anchor_x: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []
    for name, mask_s in defs.items():
        idx = np.flatnonzero(mask_s.to_numpy())
        if idx.size == 0:
            continue
        labels, dist = nearest_anchor_assignment(x[idx], anchor_x)
        counts = pd.Series(labels).value_counts(normalize=True)
        entropy = -float(np.sum(counts.to_numpy() * np.log2(counts.to_numpy() + 1e-12)))
        rows.append(
            {
                "definition": name,
                "n_members": int(idx.size),
                "assignment_entropy": entropy,
                "mean_distance_to_anchor": float(np.mean(dist)),
                "dominant_anchor_fraction": float(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def feature_ablation_stability(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    defs: dict[str, pd.Series],
    main_def: str,
    full_labels: np.ndarray,
    rng: np.random.Generator,
    k: int,
) -> pd.DataFrame:
    mask = defs[main_def].to_numpy()
    rows = []
    for name in ["early", "mid", "late", "conservative", "expanded"]:
        feats = feature_sets.get(name, [])
        if len(feats) < 2:
            continue
        x, _, _ = prepare_scaled(df, feats)
        idx = np.flatnonzero(mask)
        if idx.size > 16000:
            idx = rng.choice(idx, size=16000, replace=False)
        km = KMeans(n_clusters=k, n_init=10, random_state=41)
        labels = km.fit_predict(x[idx])
        main = full_labels[idx]
        main = main[main >= 0]
        labels = labels[: main.shape[0]]
        ami = adjusted_mutual_info_score(main, labels) if main.size else np.nan
        rows.append({"feature_set": name, "n_features": len(feats), "n_rows": int(idx.size), "ami_vs_full_archetype": float(ami)})
    return pd.DataFrame(rows)


def neighborhood_support(x: np.ndarray, mask: np.ndarray, labels: np.ndarray, k_nn: int = 25) -> pd.DataFrame:
    idx = np.flatnonzero(mask)
    xs = x[idx]
    labs = labels[idx]
    nn = NearestNeighbors(n_neighbors=min(k_nn + 1, xs.shape[0]), metric="euclidean")
    nn.fit(xs)
    neigh = nn.kneighbors(xs, return_distance=False)[:, 1:]
    rows = []
    for lab in sorted(set(labs.tolist())):
        if lab < 0:
            continue
        local = np.flatnonzero(labs == lab)
        if local.size == 0:
            continue
        same_frac = []
        for i in local[: min(local.size, 5000)]:
            same_frac.append(float(np.mean(labs[neigh[i]] == lab)))
        rows.append({"archetype_id": int(lab), "n_members": int(local.size), "mean_neighbor_same_archetype_fraction": float(np.mean(same_frac))})
    return pd.DataFrame(rows)


def source_context_balance(df: pd.DataFrame, mask: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    rows = []
    for col in ["source_name", "source_type", "source_role", "lambda_context_type", "metadata_join_status"]:
        if col not in df.columns:
            continue
        for lab in sorted(set(labels[mask].tolist())):
            if lab < 0:
                continue
            m = mask & (labels == lab)
            vc = df.loc[m, col].fillna("missing").value_counts(normalize=True)
            rows.append(
                {
                    "metadata_column": col,
                    "archetype_id": int(lab),
                    "dominant_value": vc.index[0] if not vc.empty else "",
                    "dominant_fraction": float(vc.iloc[0]) if not vc.empty else np.nan,
                    "n_rows": int(m.sum()),
                }
            )
    return pd.DataFrame(rows)


def lambda_response_overlay(
    overlay: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    med: pd.Series,
    anchor_x: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lambda_available = overlay["lambda_reorg"].notna() if "lambda_reorg" in overlay else pd.Series(False, index=overlay.index)
    source_rows = []
    for source, g in overlay.groupby("source_name", dropna=False):
        source_rows.append(
            {
                "source_name": source,
                "n_rows": int(len(g)),
                "lambda_nonnull_fraction": float(g["lambda_reorg"].notna().mean()) if "lambda_reorg" in g else 0.0,
                "n_unique_lambda": int(g["lambda_reorg"].nunique(dropna=True)) if "lambda_reorg" in g else 0,
                "allowed_for_lambda_response": bool(g["lambda_reorg"].nunique(dropna=True) > 1) if "lambda_reorg" in g else False,
            }
        )
    use = overlay.loc[lambda_available].copy()
    if use.empty:
        return pd.DataFrame(source_rows), pd.DataFrame(), pd.DataFrame()
    x = transform_scaled(use, features, scaler, med)
    labels, dist = nearest_anchor_assignment(x, anchor_x)
    use["nearest_archetype_id"] = labels
    use["distance_to_archetype"] = dist
    key_cols = [c for c in ["source_name", "pair_id", "alpha", "h_identity_key"] if c in use.columns]
    grouped_rows = []
    for key, g in use.groupby(key_cols, dropna=False):
        if g["lambda_reorg"].nunique(dropna=True) <= 1:
            continue
        eta = pd.to_numeric(g["eta20"], errors="coerce")
        if eta.notna().sum() == 0:
            continue
        best_i = eta.idxmax()
        worst_i = eta.idxmin()
        grouped_rows.append(
            {
                **{c: v for c, v in zip(key_cols, key if isinstance(key, tuple) else (key,))},
                "nearest_archetype_id_mode": int(g["nearest_archetype_id"].mode().iloc[0]),
                "n_lambda": int(g["lambda_reorg"].nunique(dropna=True)),
                "best_lambda": float(g.loc[best_i, "lambda_reorg"]),
                "best_eta20": float(eta.loc[best_i]),
                "worst_lambda": float(g.loc[worst_i, "lambda_reorg"]),
                "worst_eta20": float(eta.loc[worst_i]),
                "eta20_range": float(eta.max() - eta.min()),
                "low_nonzero_best_lambda": bool(float(g.loc[best_i, "lambda_reorg"]) in {3.0, 5.0, 15.0}),
                "lambda_dramatic_proxy": bool((eta.max() - eta.min()) >= 0.15),
            }
        )
    detail = pd.DataFrame(grouped_rows)
    if detail.empty:
        summary = pd.DataFrame()
    else:
        summary = (
            detail.groupby("nearest_archetype_id_mode")
            .agg(
                n_h_points=("eta20_range", "size"),
                eta20_range_mean=("eta20_range", "mean"),
                lambda_dramatic_fraction=("lambda_dramatic_proxy", "mean"),
                low_nonzero_best_lambda_fraction=("low_nonzero_best_lambda", "mean"),
            )
            .reset_index()
        )
    return pd.DataFrame(source_rows), detail, summary


def path_behavior_overlay(
    overlay: pd.DataFrame,
    features: list[str],
    scaler: StandardScaler,
    med: pd.Series,
    anchor_x: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "pair_id" not in overlay.columns:
        return pd.DataFrame(), pd.DataFrame()
    bridge = overlay[overlay["source_name"].astype(str).str.contains("bridge_same_s", na=False)].copy()
    if bridge.empty:
        return pd.DataFrame(), pd.DataFrame()
    x = transform_scaled(bridge, features, scaler, med)
    labels, dist = nearest_anchor_assignment(x, anchor_x)
    bridge["nearest_archetype_id"] = labels
    bridge["distance_to_archetype"] = dist
    rows = []
    for pair, g in bridge.groupby("pair_id", dropna=False):
        eta = pd.to_numeric(g["eta20"], errors="coerce")
        arch = sorted(g["nearest_archetype_id"].dropna().astype(int).unique().tolist())
        rows.append(
            {
                "pair_id": pair,
                "n_rows": int(len(g)),
                "n_alpha": int(g["alpha"].nunique(dropna=True)) if "alpha" in g else np.nan,
                "n_lambda": int(g["lambda_reorg"].nunique(dropna=True)) if "lambda_reorg" in g else np.nan,
                "archetype_ids_seen": "|".join(map(str, arch)),
                "n_archetypes_seen": len(arch),
                "path_archetype_relation": "same_archetype_path_candidate" if len(arch) <= 1 else "cross_archetype_transition_candidate",
                "eta20_min": float(eta.min()),
                "eta20_max": float(eta.max()),
                "eta20_range": float(eta.max() - eta.min()),
                "mean_distance_to_archetype": float(g["distance_to_archetype"].mean()),
            }
        )
    pair_summary = pd.DataFrame(rows)
    target = pair_summary.sort_values(["n_archetypes_seen", "eta20_range"], ascending=[False, False]).head(200).copy()
    return pair_summary, target


def write_markdown(path: Path, title: str, sections: list[tuple[str, str]]) -> None:
    lines = [f"# {title}", ""]
    for heading, body in sections:
        lines.extend([f"## {heading}", "", body.strip(), ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def table_text(df: pd.DataFrame, max_rows: int = 40) -> str:
    if df is None or df.empty:
        return "No rows."
    shown = df.head(max_rows)
    suffix = "" if len(df) <= max_rows else f"\n\n... ({len(df) - max_rows} more rows omitted)"
    return "```text\n" + shown.to_string(index=False) + "\n```" + suffix


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-rows", type=int, default=12000)
    ap.add_argument("--n-archetypes", type=int, default=4)
    ap.add_argument("--bootstrap-rounds", type=int, default=8)
    ap.add_argument("--max-recon-rows", type=int, default=6000)
    ap.add_argument("--random-state", type=int, default=20260616)
    args = ap.parse_args()

    ctx = ensure_dirs(args.output_dir)
    ctx.smoke = args.smoke
    ctx.rng = np.random.default_rng(args.random_state)
    (ctx.logs / "progress.log").write_text("", encoding="utf-8")
    log(ctx, f"start high-eta archetype claim validation smoke={args.smoke}")

    nrows = args.smoke_rows if args.smoke else None
    inventory = []
    for p in [OBSERVED_CSV, OVERLAY_CSV, ETA_INVENTORY_CSV, FEATURE_ABLATION_CSV, SOURCE_AUDIT_CSV, PATH_TARGET_CSV]:
        inventory.append({"path": str(p.relative_to(ROOT)), "exists": p.exists(), "size_mb": p.stat().st_size / 1e6 if p.exists() else np.nan})
    pd.DataFrame(inventory).to_csv(ctx.csv / "stage0_input_inventory.csv", index=False)

    log(ctx, "read observed feature matrix")
    obs = read_csv(OBSERVED_CSV, nrows=nrows)
    if args.smoke and obs.shape[0] > args.smoke_rows:
        obs = obs.iloc[: args.smoke_rows].copy()
    log(ctx, f"observed rows={len(obs):,} cols={len(obs.columns):,}")

    feature_sets = build_feature_sets(obs)
    pd.DataFrame(feature_sets["rows"]).to_csv(ctx.csv / "stage0_feature_column_inventory.csv", index=False)
    pd.DataFrame(
        [
            {"feature_set": k, "n_features": len(v), "features": "|".join(v)}
            for k, v in feature_sets.items()
            if k != "rows"
        ]
    ).to_csv(ctx.csv / "dynamic_feature_set_definitions.csv", index=False)

    defs_df, overlap_df, defs = high_eta_definitions(obs)
    defs_df.to_csv(ctx.csv / "stage0_eta_threshold_inventory.csv", index=False)
    defs_df.to_csv(ctx.csv / "high_eta_definition_membership_summary.csv", index=False)
    overlap_df.to_csv(ctx.csv / "high_eta_definition_overlap.csv", index=False)
    membership = pd.DataFrame({"global_row_index": np.arange(len(obs)), "h_id": obs.get("h_id", pd.Series([""] * len(obs)))})
    for name, mask in defs.items():
        membership[name] = mask.to_numpy()
    membership.to_csv(ctx.csv / "high_eta_definition_membership.csv", index=False)

    source_balance_rows = []
    for name, mask in defs.items():
        for col in ["source_name", "source_type", "lambda_context_type", "metadata_join_status"]:
            if col not in obs:
                continue
            vc = obs.loc[mask, col].fillna("missing").value_counts(normalize=False)
            total = int(mask.sum())
            for val, count in vc.head(20).items():
                source_balance_rows.append({"definition": name, "metadata_column": col, "value": val, "n_rows": int(count), "fraction": count / total if total else np.nan})
    pd.DataFrame(source_balance_rows).to_csv(ctx.csv / "high_eta_definition_source_balance.csv", index=False)

    decision_rows = [
        {"stage": "stage_-1", "decision": "run_now", "reason": "Post-processing only; no new dynamics simulation.", "guardrail": "Observed high-eta archetype discovery is primary."},
        {"stage": "stage_0", "decision": "run_now", "reason": "Required observed feature matrix exists.", "guardrail": "Smoke/full row counts are recorded."},
    ]
    pd.DataFrame(decision_rows).to_csv(ctx.csv / "stage_decision_gate_table.csv", index=False)

    main_features = feature_sets["conservative"]
    if len(main_features) < 3:
        raise RuntimeError("Not enough conservative dynamic features.")
    log(ctx, f"prepare scaled conservative matrix n_features={len(main_features)}")
    x, scaler, med = prepare_scaled(obs, main_features)
    pd.DataFrame(
        {
            "feature": main_features,
            "mean": scaler.mean_,
            "scale": scaler.scale_,
            "median_fill": [float(med[f]) for f in main_features],
        }
    ).to_csv(ctx.csv / "dynamic_feature_quality_audit.csv", index=False)
    obs[["h_id"] + [c for c in ["eta20", "eta50", "source_name", "lambda_context_type"] if c in obs.columns]].to_csv(
        ctx.csv / "dynamic_feature_matrix_high_eta_ready.csv", index=False
    )

    log(ctx, "stage 3 heterogeneity metrics")
    het = heterogeneity_metrics(obs, x, defs, ctx.rng)
    het.to_csv(ctx.csv / "high_eta_heterogeneity_metrics.csv", index=False)
    ev, loadings, pca = pca_loadings(x, main_features)
    ev.to_csv(ctx.csv / "pca_explained_variance_high_eta_ready.csv", index=False)
    loadings.to_csv(ctx.csv / "high_eta_variation_axis_loadings.csv", index=False)
    strat = source_balance_rows.copy()
    pd.DataFrame(strat).to_csv(ctx.csv / "high_eta_stratified_heterogeneity_check.csv", index=False)

    main_def = "eta20_top10" if "eta20_top10" in defs else next(iter(defs))
    main_mask = defs[main_def].to_numpy()
    log(ctx, f"stage 4 archetype candidates main_def={main_def} n={int(main_mask.sum()):,}")
    inv, profiles, recon, labels, anchor_x = make_archetypes(
        obs,
        x,
        main_features,
        main_mask,
        args.n_archetypes,
        ctx.rng,
        main_def,
        args.max_recon_rows if not args.smoke else min(1000, args.max_recon_rows),
    )
    inv.to_csv(ctx.csv / "archetype_candidate_inventory.csv", index=False)
    profiles.to_csv(ctx.csv / "archetype_feature_profiles.csv", index=False)
    recon.to_csv(ctx.csv / "archetype_reconstruction_error.csv", index=False)
    obs_reps = inv[["archetype_label", "global_row_index", "h_id", "eta20", "eta50", "nearest_member_count", "mean_nearest_distance"]].copy()
    obs_reps.to_csv(ctx.csv / "archetype_representative_h.csv", index=False)

    try:
        nmf_inv, nmf_prof = nmf_candidate(x, main_mask, main_features, args.n_archetypes, ctx.rng)
        nmf_inv.to_csv(ctx.csv / "nmf_archetype_candidate_inventory.csv", index=False)
        nmf_prof.to_csv(ctx.csv / "nmf_archetype_feature_profiles.csv", index=False)
    except Exception as exc:  # keep fallback robust
        (ctx.reports / "nmf_fallback_note.md").write_text(f"# NMF Fallback Note\n\nNMF diagnostic failed and was not used for the main conclusion.\n\n`{exc}`\n", encoding="utf-8")

    source_context_balance(obs, main_mask, labels).to_csv(ctx.csv / "archetype_source_family_balance.csv", index=False)
    log(ctx, "time-window archetype candidates")
    tw_inv, tw_traj, tw_stab = time_window_archetypes(
        obs,
        feature_sets,
        defs,
        main_def,
        args.n_archetypes,
        ctx.rng,
        args.max_recon_rows if not args.smoke else min(1000, args.max_recon_rows),
    )
    tw_inv.to_csv(ctx.csv / "time_window_archetype_candidate_inventory.csv", index=False)
    tw_traj.to_csv(ctx.csv / "archetype_time_trajectory.csv", index=False)
    tw_stab.to_csv(ctx.csv / "time_window_archetype_stability.csv", index=False)

    log(ctx, "stage 5 stability and validity")
    threshold_stability(obs, x, defs, anchor_x, ctx.rng).to_csv(ctx.csv / "archetype_threshold_stability.csv", index=False)
    feature_ablation_stability(obs, feature_sets, defs, main_def, labels, ctx.rng, args.n_archetypes).to_csv(
        ctx.csv / "archetype_feature_ablation_stability.csv", index=False
    )
    kstab = kmeans_stability(x, main_mask, ctx.rng, range(2, 7))
    kstab.to_csv(ctx.csv / "spectrum_vs_cluster_metrics.csv", index=False)
    neigh = neighborhood_support(x, main_mask, labels)
    neigh.to_csv(ctx.csv / "high_eta_neighbor_graph_connectivity.csv", index=False)
    # Bootstrap proxy: fit anchors on split halves and compare nearest assignments on a common sample.
    boot_rows = []
    idx_main = np.flatnonzero(main_mask)
    common_n = min(4000 if not args.smoke else 800, idx_main.size)
    common = ctx.rng.choice(idx_main, size=common_n, replace=False)
    for b in range(args.bootstrap_rounds if not args.smoke else min(3, args.bootstrap_rounds)):
        half = ctx.rng.choice(idx_main, size=max(args.n_archetypes + 1, idx_main.size // 2), replace=False)
        local_anchor = half[farthest_anchors(x[half], args.n_archetypes, ctx.rng)]
        labs_a, _ = nearest_anchor_assignment(x[common], x[local_anchor])
        labs_main, _ = nearest_anchor_assignment(x[common], anchor_x)
        boot_rows.append({"bootstrap_round": b, "ami_vs_main_anchor_assignment": float(adjusted_mutual_info_score(labs_main, labs_a))})
    pd.DataFrame(boot_rows).to_csv(ctx.csv / "archetype_bootstrap_stability.csv", index=False)

    validity_rows = []
    thresh = pd.read_csv(ctx.csv / "archetype_threshold_stability.csv")
    ablation = pd.read_csv(ctx.csv / "archetype_feature_ablation_stability.csv")
    boot = pd.DataFrame(boot_rows)
    recon_mean = float(recon["reconstruction_error"].mean()) if not recon.empty else np.nan
    for row in inv.itertuples(index=False):
        aid = int(str(row.archetype_label).split("_A")[-1])
        nrow = neigh[neigh["archetype_id"] == aid]
        validity_rows.append(
            {
                "archetype_label": row.archetype_label,
                "threshold_dominant_fraction_mean": float(thresh["dominant_anchor_fraction"].mean()) if "dominant_anchor_fraction" in thresh else np.nan,
                "feature_ablation_ami_mean": float(ablation["ami_vs_full_archetype"].mean()) if "ami_vs_full_archetype" in ablation else np.nan,
                "bootstrap_ami_mean": float(boot["ami_vs_main_anchor_assignment"].mean()) if not boot.empty else np.nan,
                "neighborhood_support": float(nrow["mean_neighbor_same_archetype_fraction"].iloc[0]) if not nrow.empty else np.nan,
                "reconstruction_error_mean_overall": recon_mean,
                "validity_call": "candidate_needs_caveat",
            }
        )
    validity = pd.DataFrame(validity_rows)
    if not validity.empty:
        validity["validity_call"] = np.where(
            (validity["bootstrap_ami_mean"].fillna(0) >= 0.35)
            & (validity["neighborhood_support"].fillna(0) >= 0.35),
            "dynamic_archetype_candidate",
            "exploratory_extreme_candidate",
        )
    validity.to_csv(ctx.csv / "archetype_validity_assessment.csv", index=False)
    pd.DataFrame(
        [
            {
                "decision_axis": "static_vs_temporal",
                "time_window_reconstruction_mean": float(tw_stab["reconstruction_error_mean"].mean()) if not tw_stab.empty else np.nan,
                "full_reconstruction_mean": recon_mean,
                "temporal_assignment_pattern_count": int(tw_traj["temporal_pattern"].nunique()) if "temporal_pattern" in tw_traj else np.nan,
                "temporal_transition_fraction": np.nan,
                "decision": "evaluate_window_specific_patterns_as_auxiliary" if not tw_traj.empty else "static_only_due_missing_time_windows",
                "caveat": "time-window archetype labels are independently defined; do not interpret label changes as direct state transitions without alignment",
            }
        ]
    ).to_csv(ctx.csv / "static_vs_temporal_archetype_decision.csv", index=False)

    log(ctx, "stage 7/8 overlay analysis")
    overlay_nrows = args.smoke_rows * 2 if args.smoke else None
    overlay = read_csv(OVERLAY_CSV, nrows=overlay_nrows)
    source_avail, lambda_detail, lambda_summary = lambda_response_overlay(overlay, main_features, scaler, med, anchor_x)
    source_avail.to_csv(ctx.csv / "lambda_source_availability_by_archetype.csv", index=False)
    lambda_detail.to_csv(ctx.csv / "archetype_lambda_dramatic_cases.csv", index=False)
    lambda_summary.to_csv(ctx.csv / "archetype_lambda_response_summary.csv", index=False)
    lambda_summary.to_csv(ctx.csv / "time_window_archetype_lambda_response_summary.csv", index=False)
    pair_summary, path_targets = path_behavior_overlay(overlay, main_features, scaler, med, anchor_x)
    pair_summary.to_csv(ctx.csv / "bridge_path_archetype_transition_summary.csv", index=False)
    path_targets.to_csv(ctx.csv / "path_search_targets_by_archetype_claim.csv", index=False)
    path_targets.to_csv(ctx.csv / "next_plausible_path_target_manifest.csv", index=False)
    path_targets.to_csv(ctx.csv / "temporal_archetype_path_target_manifest.csv", index=False)

    claim_rows = []
    high_rows = het[(het["definition"] == main_def) & (het["group"] == "high_definition")]
    rand_rows = het[(het["definition"] == main_def) & (het["group"] == "row_count_random_observed")]
    hetero_ratio = np.nan
    if not high_rows.empty and not rand_rows.empty:
        hetero_ratio = float(high_rows["mean_feature_variance"].iloc[0] / rand_rows["mean_feature_variance"].iloc[0])
    validity_call = "exploratory_extreme_candidate"
    if not validity.empty and (validity["validity_call"] == "dynamic_archetype_candidate").mean() >= 0.5:
        validity_call = "dynamic_archetype_candidate"
    mode_claim = "discrete_mode_not_supported_as_primary"
    if not kstab.empty and kstab["silhouette_sample"].max() >= 0.35 and validity_call == "dynamic_archetype_candidate":
        mode_claim = "weakly_separated_dynamic_subtype_candidate"
    claim_rows.extend(
        [
            {
                "claim": "high_eta_heterogeneous",
                "decision": bool(hetero_ratio >= 0.9) if not math.isnan(hetero_ratio) else False,
                "evidence": f"variance_ratio_high_vs_random={hetero_ratio:.3f}" if not math.isnan(hetero_ratio) else "not_available",
                "caveat": "variance alone is not mechanism proof",
            },
            {
                "claim": "dynamic_archetype_candidate",
                "decision": validity_call,
                "evidence": "threshold/bootstrap/neighborhood validity table",
                "caveat": "candidate, not causal mechanism",
            },
            {
                "claim": "discrete_dynamic_mode",
                "decision": mode_claim,
                "evidence": f"max_silhouette={kstab['silhouette_sample'].max():.3f}" if not kstab.empty else "not_available",
                "caveat": "do not call discrete mode without stronger density/stability evidence",
            },
            {
                "claim": "lambda_response_connection",
                "decision": "overlay_only_association" if not lambda_summary.empty else "not_available",
                "evidence": "lambda-bearing overlay projected into observed archetype space",
                "caveat": "observed discovery source itself is not best-lambda evidence",
            },
        ]
    )
    claim_df = pd.DataFrame(claim_rows)
    claim_df.to_csv(ctx.csv / "claim_strength_decision_table.csv", index=False)
    claim_df.to_csv(ctx.csv / "claim_evidence_caveat_ledger.csv", index=False)

    pd.DataFrame(
        [
            {
                "target_type": "same_archetype_path",
                "selection_rule": "pairs with n_archetypes_seen <= 1 and high eta/support/plausibility to be checked in later path search",
                "current_manifest": "path_search_targets_by_archetype_claim.csv",
            },
            {
                "target_type": "cross_archetype_transition_path",
                "selection_rule": "pairs with multiple archetypes seen along historical bridge overlay",
                "current_manifest": "path_search_targets_by_archetype_claim.csv",
            },
            {
                "target_type": "temporal_archetype_transition_path",
                "selection_rule": "high-eta samples whose early/mid/late archetype memberships differ",
                "current_manifest": "temporal_archetype_path_target_manifest.csv",
            },
        ]
    ).to_csv(ctx.csv / "archetype_structural_followup_targets.csv", index=False)
    pd.DataFrame(
        [
            {
                "need_type": "rho_t_or_contribution_rerun",
                "decision": "defer_with_command",
                "reason": "Current stage uses stored timepoint/window summary; rerun only if temporal archetype decision changes path target selection.",
            }
        ]
    ).to_csv(ctx.csv / "contribution_rerun_target_need_by_archetype.csv", index=False)

    # Reports
    guard_body = (
        "Primary decision: discover dynamic archetype candidates from the full observed high-eta population, "
        "then decide whether they are useful for path-search target selection. Existing D/S bridge results are "
        "historical overlays only and are not used to define the discovery space."
    )
    write_markdown(ctx.reports / "research_question_guardrail.md", "Research Question Guardrail", [("Decision", guard_body)])
    write_markdown(
        ctx.reports / "stage0_input_audit.md",
        "Stage 0 Input Audit",
        [
            ("Scope", f"Observed rows read: {len(obs):,}. Smoke mode: {args.smoke}. Conservative features: {len(main_features)}."),
            ("Execution", "No new dynamics simulation was required. Full run uses existing observed and overlay matrices."),
        ],
    )
    write_markdown(
        ctx.reports / "stage1_high_eta_definition_sensitivity.md",
        "Stage 1 High-Eta Definition Sensitivity",
        [("Summary", table_text(defs_df))],
    )
    write_markdown(
        ctx.reports / "stage2_dynamic_feature_space_and_leakage_guard.md",
        "Stage 2 Dynamic Feature Space and Leakage Guard",
        [
            ("Summary", "Main conclusions use conservative dynamics features and exclude direct eta/final-trap leakage."),
            ("Feature Counts", table_text(pd.DataFrame([{"feature_set": k, "n_features": len(v)} for k, v in feature_sets.items() if k != "rows"]))),
        ],
    )
    write_markdown(
        ctx.reports / "stage3_high_eta_heterogeneity_report.md",
        "Stage 3 High-Eta Heterogeneity",
        [("Summary", table_text(het))],
    )
    write_markdown(
        ctx.reports / "stage4_archetype_candidate_extraction.md",
        "Stage 4 Archetype Candidate Extraction",
        [
            ("Summary", table_text(inv)),
            ("Caveat", "Archetype candidates are farthest-point dynamic endpoints, not mechanism proof. NMF diagnostics are auxiliary when available."),
        ],
    )
    write_markdown(
        ctx.reports / "stage5_archetype_stability_report.md",
        "Stage 5 Archetype Stability and Validity",
        [
            ("Validity", table_text(validity)),
            ("Temporal", "Time-window labels are coarse because this pass uses stored summary timepoints, not continuous rho_t."),
        ],
    )
    write_markdown(
        ctx.reports / "stage6_spectrum_vs_discrete_mode_decision.md",
        "Stage 6 Spectrum vs Discrete Mode Decision",
        [
            ("Cluster Metrics", table_text(kstab)),
            ("Decision", "Discrete mode claim remains conservative. Use dynamic archetype or temporal transition language unless stronger density/stability evidence appears."),
        ],
    )
    write_markdown(
        ctx.reports / "stage7_lambda_response_connection.md",
        "Stage 7 Lambda Response Connection",
        [
            ("Availability", table_text(source_avail)),
            ("Summary", table_text(lambda_summary)),
            ("Caveat", "Lambda response is an overlay association and does not prove observed archetype validity."),
        ],
    )
    write_markdown(
        ctx.reports / "stage8_bridge_path_connection.md",
        "Stage 8 Path Behavior and Historical Bridge Overlay",
        [
            ("Summary", table_text(pair_summary, max_rows=30)),
            ("Caveat", "Historical D/S bridge paths are overlays only; dynamic archetype discovery is not based on D/S labels."),
        ],
    )
    write_markdown(
        ctx.reports / "stage9_structural_eigenstate_followup_plan.md",
        "Stage 9 Structural/Eigenstate Follow-Up Plan",
        [
            ("Decision", "No new rho_t or dynamics simulation is launched here. Rerun targets should be selected only if archetype/temporal labels change path-search priorities."),
        ],
    )
    write_markdown(
        ctx.reports / "final_high_eta_archetype_claim_report.md",
        "Final High-Eta Archetype Claim Report",
        [
            ("What Was Done", f"Used observed population rows={len(obs):,}; high-eta discovery used {main_def}; main features={len(main_features)}."),
            ("Claim Decisions", table_text(claim_df)),
            ("Interpretation", "The safe claim should be chosen from the claim table. Do not assert causal mechanisms, manifolds, or discrete modes unless the evidence columns support it."),
        ],
    )
    write_markdown(
        ctx.reports / "next_path_search_plan_from_archetype_analysis.md",
        "Next Path Search Plan From Archetype Analysis",
        [
            ("Targets", "Use same-archetype, cross-archetype, temporal-transition, high-eta-but-low-plausibility, and plausible-but-low-eta target categories. The current manifests prioritize candidates; they do not prove path connectivity."),
            ("Files", "`csv/next_plausible_path_target_manifest.csv` and `csv/temporal_archetype_path_target_manifest.csv`."),
        ],
    )
    write_markdown(
        ctx.reports / "adaptive_execution_log.md",
        "Adaptive Execution Log",
        [
            ("Decisions", "Ran post-processing only. No candidate row reduction for full mode. Smoke mode is for code validation only."),
            ("Long Runs", "No new dynamics simulation was run. If future rho_t/contribution runs are needed, use stage9 target files."),
        ],
    )
    pd.DataFrame(
        [
            {"stage": "all", "decision": "run_now_postprocessing", "reason": "Existing feature matrices were sufficient for first-pass claim validation.", "caveat": "No new dynamics simulation."}
        ]
    ).to_csv(ctx.csv / "adaptive_execution_log.csv", index=False)

    repro = f"""# Reproduce high-eta archetype claim validation

Smoke:

```powershell
& C:\\Users\\User\\anaconda3\\envs\\py311-cu132\\python.exe new\\run_high_eta_archetype_claim_validation.py --smoke --output-dir {args.output_dir}
```

Full:

```powershell
& C:\\Users\\User\\anaconda3\\envs\\py311-cu132\\python.exe new\\run_high_eta_archetype_claim_validation.py --output-dir {args.output_dir}
```
"""
    (ctx.commands / "reproduce_high_eta_archetype_claim_validation.md").write_text(repro, encoding="utf-8")

    summary = {
        "smoke": args.smoke,
        "elapsed_seconds": time.time() - ctx.started_at,
        "observed_rows": int(len(obs)),
        "main_definition": main_def,
        "main_features": len(main_features),
        "n_archetypes": args.n_archetypes,
        "output_dir": str(args.output_dir),
    }
    (ctx.json_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log(ctx, f"done elapsed={summary['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
