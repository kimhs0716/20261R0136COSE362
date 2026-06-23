from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_mutual_info_score, davies_bouldin_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PYTHON_ENV = os.environ.get("PYTHON", sys.executable)


DEFAULT_INPUTS = {
    "observed_npz": ROOT / "FMO_H27_context_ablation" / "data" / "merged_h27_140k.npz",
    "bridge_detail": ROOT / "outputs" / "bridge_noise_selected_ds_same_s_20260613" / "csv" / "bridge_noise_sweep_detail.csv",
    "bridge_best_by_alpha": ROOT / "outputs" / "bridge_noise_selected_ds_same_s_20260613" / "csv" / "bridge_best_by_alpha.csv",
    "bridge_support": ROOT / "outputs" / "bridge_noise_selected_ds_same_s_20260613" / "csv" / "bridge_empirical_support_profile.csv",
    "bridge_pair_summary": ROOT / "outputs" / "bridge_noise_selected_ds_same_s_20260613" / "csv" / "bridge_pair_summary.csv",
    "piecewise_detail": ROOT / "outputs" / "bridge_overnight_piecewise_waypoint_20260614" / "csv" / "piecewise_waypoint_detail.csv",
    "piecewise_time": ROOT / "outputs" / "bridge_overnight_piecewise_waypoint_20260614" / "csv" / "piecewise_waypoint_time_window_summary.csv",
    "dense_detail": ROOT / "outputs" / "bridge_overnight_dense_radius_20260614" / "csv" / "normal_robustness_detail.csv",
    "dense_summary": ROOT / "outputs" / "bridge_overnight_dense_radius_20260614" / "csv" / "normal_robustness_summary.csv",
    "contribution_detail": ROOT / "outputs" / "bridge_overnight_contribution_group_full_20260614" / "csv" / "contribution_job_metadata.csv",
    "contribution_time": ROOT / "outputs" / "bridge_overnight_contribution_group_full_20260614" / "csv" / "time_window_contribution_summary.csv",
}


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


class Logger:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("w", encoding="utf-8")

    def write(self, msg: str) -> None:
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def ensure_dirs(out_dir: Path) -> Dict[str, Path]:
    dirs = {
        "root": out_dir,
        "csv": out_dir / "csv",
        "reports": out_dir / "reports",
        "commands": out_dir / "commands",
        "logs": out_dir / "logs",
        "json": out_dir / "json",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def file_info(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"path": str(path), "exists": False, "size_mb": np.nan, "mtime": ""}
    return {
        "path": str(path),
        "exists": True,
        "size_mb": round(path.stat().st_size / (1024**2), 3),
        "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(path.stat().st_mtime)),
    }


def nearest_time_index(times: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(times, dtype=float) - target)))


def curve_at(curves: np.ndarray, times: np.ndarray, target: float) -> np.ndarray:
    return curves[:, nearest_time_index(times, target)]


def time_to_threshold(curves: np.ndarray, times: np.ndarray, threshold: float) -> np.ndarray:
    reached = curves >= threshold
    any_reached = reached.any(axis=1)
    idx = reached.argmax(axis=1)
    out = np.full(curves.shape[0], np.nan, dtype=np.float32)
    out[any_reached] = times[idx[any_reached]]
    return out


def window_mean(curves: np.ndarray, times: np.ndarray, lo: float, hi: float) -> np.ndarray:
    mask = (times >= lo) & (times <= hi)
    if not mask.any():
        idx = nearest_time_index(times, (lo + hi) / 2.0)
        return curves[:, idx]
    return np.nanmean(curves[:, mask], axis=1)


def add_time_features_from_curves(df: pd.DataFrame, times: np.ndarray, eta_t=None, cl1_t=None, purity_t=None, ipr_t=None, path_t=None) -> pd.DataFrame:
    times = np.asarray(times, dtype=np.float32)
    if eta_t is not None:
        eta_t = np.asarray(eta_t)
        df["eta5"] = curve_at(eta_t, times, 5.0)
        df["eta10"] = curve_at(eta_t, times, 10.0)
        df["eta20"] = curve_at(eta_t, times, 20.0)
        df["eta50"] = curve_at(eta_t, times, 50.0)
        df["eta_final"] = eta_t[:, -1]
        df["t25"] = time_to_threshold(eta_t, times, 0.25)
        df["t50"] = time_to_threshold(eta_t, times, 0.50)
        df["t80"] = time_to_threshold(eta_t, times, 0.80)
        df["t90"] = time_to_threshold(eta_t, times, 0.90)
        df["eta_primary"] = df["eta20"].astype(float)
    if cl1_t is not None:
        cl1_t = np.asarray(cl1_t)
        df["cl1_mean_0_5ps"] = window_mean(cl1_t, times, 0, 5)
        df["cl1_mean_5_10ps"] = window_mean(cl1_t, times, 5, 10)
        df["cl1_mean_10_20ps"] = window_mean(cl1_t, times, 10, 20)
        df["cl1_max"] = np.nanmax(cl1_t, axis=1)
    if purity_t is not None:
        purity_t = np.asarray(purity_t)
        df["purity_mean_0_5ps"] = window_mean(purity_t, times, 0, 5)
        df["purity_mean_5_10ps"] = window_mean(purity_t, times, 5, 10)
        df["purity_mean_10_20ps"] = window_mean(purity_t, times, 10, 20)
        df["purity_max"] = np.nanmax(purity_t, axis=1)
    if ipr_t is not None:
        ipr_t = np.asarray(ipr_t)
        df["ipr_mean_0_5ps"] = window_mean(ipr_t, times, 0, 5)
        df["ipr_mean_5_10ps"] = window_mean(ipr_t, times, 5, 10)
        df["ipr_mean_10_20ps"] = window_mean(ipr_t, times, 10, 20)
        df["ipr_max"] = np.nanmax(ipr_t, axis=1)
    if path_t is not None:
        path_t = np.asarray(path_t)
        if path_t.ndim == 3:
            for target in [6, 10, 20, 50]:
                idx = nearest_time_index(times, float(target))
                for g in range(min(path_t.shape[2], 7)):
                    df[f"path_g{g}_at_{target}ps"] = path_t[:, idx, g]
    return df


def observed_features(npz_path: Path, logger: Logger) -> Tuple[pd.DataFrame, pd.DataFrame]:
    logger.write(f"loading observed npz: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    n = z["H_params"].shape[0]
    times = z["times"].astype(np.float32)
    df = pd.DataFrame({
        "source_name": "merged_h27_140k",
        "source_type": "observed_population",
        "source_role": "observed_population_discovery",
        "source_file_id": z["source_file_id"].astype(int),
        "source_row": z["source_row"].astype(int),
        "row_index_within_source": np.arange(n, dtype=np.int32),
        "h_id": [f"obs_{i}" for i in range(n)],
        "dynamic_identity_key": [f"obs_{i}_lambda_single" for i in range(n)],
        "h_identity_key": [f"obs_{i}" for i in range(n)],
        "lambda_context_type": "single_lambda_observed_sample",
        "lambda_reorg": z["lambda_reorg"].astype(np.float32),
        "pair_id": "",
        "alpha": np.nan,
        "solver_success": True,
        "metadata_join_status": "observed_no_ds_family_join",
        "plausibility_available": False,
        "support_level": "",
    })
    df["eta_label"] = z["eta"].astype(np.float32)
    pop_t = z["pop_t"].astype(np.float32)
    # Project convention from prior docs: 7 exciton sites + trap + loss.
    trap_idx = 7 if pop_t.shape[2] > 7 else pop_t.shape[2] - 1
    loss_idx = 8 if pop_t.shape[2] > 8 else None
    trap_t = pop_t[:, :, trap_idx]
    df = add_time_features_from_curves(
        df,
        times=times,
        eta_t=trap_t,
        cl1_t=z.get("cl1_t"),
        purity_t=z.get("purity_t"),
        ipr_t=z.get("ipr_t"),
    )
    # Align with existing CSV feature names where possible.
    for target in [6, 10, 20, 50]:
        idx = nearest_time_index(times, float(target))
        df[f"site1_at_{target}ps"] = pop_t[:, idx, 0]
        if pop_t.shape[2] > 1:
            df[f"site2_at_{target}ps"] = pop_t[:, idx, 1]
        if pop_t.shape[2] > 3:
            df[f"sink34_at_{target}ps"] = pop_t[:, idx, 2] + pop_t[:, idx, 3]
        if pop_t.shape[2] > 6:
            df[f"detour567_at_{target}ps"] = pop_t[:, idx, 4] + pop_t[:, idx, 5] + pop_t[:, idx, 6]
        df[f"trap_at_{target}ps"] = pop_t[:, idx, trap_idx]
        if loss_idx is not None:
            df[f"loss_at_{target}ps"] = pop_t[:, idx, loss_idx]
        df[f"residual_at_{target}ps"] = 1.0 - np.sum(pop_t[:, idx, :], axis=1)
    df["tau_transfer"] = z["tau_transfer"].astype(np.float32)
    eigs = z["eigs"].astype(np.float32)
    df["eig_min"] = np.nanmin(eigs, axis=1)
    df["eig_max"] = np.nanmax(eigs, axis=1)
    df["eig_spread"] = df["eig_max"] - df["eig_min"]
    h_params = z["H_params"].astype(np.float32)
    df["h_param_norm"] = np.linalg.norm(h_params, axis=1)
    df["eta_primary"] = df["eta20"].astype(float)

    audit = []
    eta_diff = np.abs(df["eta_label"].to_numpy() - df["eta50"].to_numpy())
    audit.append({
        "source_name": "merged_h27_140k",
        "n_rows": n,
        "times": "|".join(map(lambda x: f"{x:g}", times)),
        "eta_label_vs_trap_eta50_median_abs_diff": float(np.nanmedian(eta_diff)),
        "eta_label_vs_trap_eta50_p95_abs_diff": float(np.nanquantile(eta_diff, 0.95)),
        "eta_label_mean": float(np.nanmean(df["eta_label"])),
        "eta20_recomputed_mean": float(np.nanmean(df["eta20"])),
        "eta50_recomputed_mean": float(np.nanmean(df["eta50"])),
        "lambda_context_type": "single_lambda_observed_sample",
        "source_file_count": int(pd.Series(z["source_file_id"]).nunique()),
    })
    return df, pd.DataFrame(audit)


def read_detail_csv(path: Path, source_name: str, source_type: str, source_role: str, lambda_context_default: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source_name"] = source_name
    df["source_type"] = source_type
    df["source_role"] = source_role
    df["row_index_within_source"] = np.arange(len(df), dtype=np.int32)
    if "pair_id" not in df.columns:
        df["pair_id"] = ""
    if "alpha" not in df.columns:
        df["alpha"] = np.nan
    if "lambda_reorg" not in df.columns:
        df["lambda_reorg"] = np.nan
        df["lambda_context_type"] = "lambda_unknown_or_not_applicable"
    else:
        df["lambda_context_type"] = lambda_context_default
    if "solver_success" not in df.columns:
        df["solver_success"] = True
    if "support_level" not in df.columns:
        df["support_level"] = ""
    if "eta20" in df.columns:
        df["eta_primary"] = df["eta20"]
    elif "eta50" in df.columns:
        df["eta_primary"] = df["eta50"]
    else:
        df["eta_primary"] = np.nan
    if "h_identity_key" not in df.columns:
        alpha_str = df["alpha"].map(lambda x: "na" if pd.isna(x) else f"{float(x):.4g}")
        df["h_identity_key"] = df["source_name"].astype(str) + "::" + df["pair_id"].astype(str) + "::a" + alpha_str.astype(str)
        if "radius" in df.columns:
            df["h_identity_key"] += "::r" + df["radius"].map(lambda x: "na" if pd.isna(x) else f"{float(x):.4g}").astype(str)
        if "normal_direction_i" in df.columns:
            df["h_identity_key"] += "::d" + df["normal_direction_i"].astype(str)
    if "dynamic_identity_key" not in df.columns:
        lam_str = df["lambda_reorg"].map(lambda x: "na" if pd.isna(x) else f"{float(x):.4g}")
        df["dynamic_identity_key"] = df["h_identity_key"].astype(str) + "::l" + lam_str.astype(str)
    if "h_id" not in df.columns:
        df["h_id"] = df["h_identity_key"]
    if "metadata_join_status" not in df.columns:
        df["metadata_join_status"] = np.where(df["pair_id"].astype(str).str.contains("-S", regex=False), "pair_ds_label_available", "pair_metadata_partial")
    if "plausibility_available" not in df.columns:
        df["plausibility_available"] = False
    return df


def source_audit_rows(name: str, path: Path, df: Optional[pd.DataFrame], source_type: str, role: str, notes: str = "") -> Dict[str, object]:
    info = file_info(path)
    if role == "observed_population_discovery":
        coverage_scope = "full_observed_population_available_for_mode_definition"
    elif name == "bridge_same_s_detail":
        coverage_scope = "full_existing_ds_same_s_bridge_detail_not_global_population"
    else:
        coverage_scope = "available_prior_experiment_overlay_not_global_population"
    row = {
        "source_name": name,
        "source_type": source_type,
        "source_role": role,
        "input_path": str(path),
        "exists": info["exists"],
        "size_mb": info["size_mb"],
        "n_rows": 0 if df is None else len(df),
        "columns": "" if df is None else "|".join(df.columns[:80]),
        "lambda_context_types": "" if df is None or "lambda_context_type" not in df.columns else "|".join(sorted(map(str, df["lambda_context_type"].dropna().unique()))),
        "eta_columns": "" if df is None else "|".join([c for c in ["eta5", "eta10", "eta20", "eta50", "eta_label", "eta_primary"] if c in df.columns]),
        "metadata_join_status_values": "" if df is None or "metadata_join_status" not in df.columns else "|".join(sorted(map(str, df["metadata_join_status"].dropna().unique()))[:10]),
        "allowed_use": "baseline mode discovery" if role == "observed_population_discovery" else "path overlay / path-specific transition only",
        "coverage_scope": coverage_scope,
        "notes": notes,
    }
    return row


FEATURE_CANDIDATES = [
    # Performance/timing labels. Some are excluded for no-eta ablations later.
    "eta5", "eta10", "eta20", "eta50", "eta_label", "eta_primary", "t25", "t50", "t80", "t90", "tau_transfer",
    # Population/path summary.
    "site1_at_6ps", "site2_at_6ps", "sink34_at_6ps", "detour567_at_6ps", "trap_at_6ps", "loss_at_6ps", "residual_at_6ps",
    "site1_at_10ps", "site2_at_10ps", "sink34_at_10ps", "detour567_at_10ps", "trap_at_10ps", "loss_at_10ps", "residual_at_10ps",
    "site1_at_20ps", "site2_at_20ps", "sink34_at_20ps", "detour567_at_20ps", "trap_at_20ps", "loss_at_20ps", "residual_at_20ps",
    "site1_at_50ps", "site2_at_50ps", "sink34_at_50ps", "detour567_at_50ps", "trap_at_50ps", "loss_at_50ps", "residual_at_50ps",
    # Curve summaries.
    "cl1_mean_0_5ps", "cl1_mean_5_10ps", "cl1_mean_10_20ps", "cl1_max",
    "purity_mean_0_5ps", "purity_mean_5_10ps", "purity_mean_10_20ps", "purity_max",
    "ipr_mean_0_5ps", "ipr_mean_5_10ps", "ipr_mean_10_20ps", "ipr_max",
    # Structural quick features if available.
    "eig_min", "eig_max", "eig_spread", "h_param_norm",
    # Support/proximity if available.
    "nearest_support_distance", "nearest_non_endpoint_distance_zrms", "nearest_distance_zrms", "knn_mean_distance_zrms",
]


def existing_features(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in FEATURE_CANDIDATES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            if df[c].notna().sum() > 0:
                cols.append(c)
    return cols


def feature_groups(cols: Sequence[str]) -> Dict[str, List[str]]:
    eta_like = [c for c in cols if c.startswith("eta") or c in {"t25", "t50", "t80", "t90", "tau_transfer"}]
    final_trap_like = [c for c in cols if c.startswith("trap_at_50") or c in {"eta50", "eta_label"}]
    population = [c for c in cols if any(tok in c for tok in ["site", "sink", "detour", "trap", "loss", "residual"])]
    coherence = [c for c in cols if c.startswith("cl1") or c.startswith("purity") or c.startswith("ipr")]
    structural = [c for c in cols if c.startswith("eig_") or c.startswith("h_param") or "distance" in c or "support" in c]
    no_eta = [c for c in cols if c not in set(eta_like + final_trap_like)]
    return {
        "full": list(cols),
        "no_eta_no_final_trap": no_eta,
        "population_only": population,
        "coherence_ipr_only": coherence,
        "timing_only": [c for c in eta_like if c.startswith("t") or c == "tau_transfer"],
        "structural_support_only": structural,
    }


def prepare_matrix(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[np.ndarray, SimpleImputer, StandardScaler, List[str]]:
    usable = []
    for c in cols:
        if c not in df.columns or not pd.api.types.is_numeric_dtype(df[c]) or df[c].notna().sum() <= 5:
            continue
        vals = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        std = float(np.nanstd(vals.to_numpy(dtype=np.float64)))
        # Near-constant columns create artificial huge z-scores when path/probe
        # sources use a slightly different convention, and they do not help define
        # observed-population dynamic modes.
        if not np.isfinite(std) or std < 1e-6:
            continue
        usable.append(c)
    if not usable:
        raise ValueError("no usable feature columns")
    x = df[usable].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    imputer = SimpleImputer(strategy="median")
    x_imp = imputer.fit_transform(x)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_imp)
    return x_scaled.astype(np.float32), imputer, scaler, usable


def transform_matrix(df: pd.DataFrame, cols: Sequence[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    x = df[list(cols)].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    return scaler.transform(imputer.transform(x)).astype(np.float32)


def choose_k_and_cluster(x: np.ndarray, logger: Logger, max_k: int = 8, sample_for_metrics: int = 12000) -> Tuple[np.ndarray, pd.DataFrame, int]:
    rng = np.random.default_rng(20260616)
    if len(x) > sample_for_metrics:
        metric_idx = rng.choice(len(x), size=sample_for_metrics, replace=False)
    else:
        metric_idx = np.arange(len(x))
    rows = []
    labels_by_k = {}
    for k in range(2, max_k + 1):
        logger.write(f"MiniBatchKMeans k={k}")
        km = MiniBatchKMeans(n_clusters=k, random_state=20260616, batch_size=8192, n_init=10, max_iter=300)
        labels = km.fit_predict(x)
        labels_by_k[k] = labels
        sil = np.nan
        db = np.nan
        try:
            if len(np.unique(labels[metric_idx])) > 1:
                sil = float(silhouette_score(x[metric_idx], labels[metric_idx]))
                db = float(davies_bouldin_score(x[metric_idx], labels[metric_idx]))
        except Exception as exc:
            logger.write(f"metric failed for k={k}: {exc}")
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette_metric_sample": sil, "davies_bouldin_metric_sample": db})
    metrics = pd.DataFrame(rows)
    valid = metrics.dropna(subset=["silhouette_metric_sample"])
    if len(valid):
        # Prefer interpretable modest k when silhouette gains are small.
        best_k = int(valid.sort_values(["silhouette_metric_sample", "k"], ascending=[False, True]).iloc[0]["k"])
    else:
        best_k = 4
    return labels_by_k[best_k], metrics, best_k


def high_eta_membership(df: pd.DataFrame, out_csv: Path, logger: Logger) -> pd.DataFrame:
    rows = []
    df = df.copy()
    df["high_top5_source"] = False
    df["high_top10_source"] = False
    df["high_top20_source"] = False
    for src, g in df.groupby("source_name", dropna=False):
        vals = pd.to_numeric(g["eta_primary"], errors="coerce")
        valid = vals.notna()
        if valid.sum() < 10:
            rows.append({"source_name": src, "n_valid_eta": int(valid.sum()), "status": "too_few_valid_eta"})
            continue
        qs = {5: float(vals[valid].quantile(0.95)), 10: float(vals[valid].quantile(0.90)), 20: float(vals[valid].quantile(0.80))}
        for pct, q in qs.items():
            df.loc[g.index, f"high_top{pct}_source"] = vals >= q
            rows.append({"source_name": src, "definition": f"top{pct}_source", "threshold": q, "n_valid_eta": int(valid.sum()), "n_members": int((vals >= q).sum()), "status": "ok"})
    inv = pd.DataFrame(rows)
    inv.to_csv(out_csv, index=False, encoding="utf-8-sig")
    logger.write(f"wrote eta definition inventory: {out_csv}")
    # Row-level class.
    df["high_membership_count"] = df[["high_top5_source", "high_top10_source", "high_top20_source"]].sum(axis=1)
    df["eta_membership_label"] = np.select(
        [
            df["high_top5_source"],
            df["high_top10_source"],
            df["high_top20_source"],
        ],
        ["core_high_top5", "high_top10", "borderline_high_top20"],
        default="low_or_mid_reference",
    )
    return df


def lambda_response_summary(df: pd.DataFrame) -> pd.DataFrame:
    lam_df = df[(df["lambda_context_type"] == "lambda_sweep_path_point") & df["lambda_reorg"].notna() & df["eta_primary"].notna()].copy()
    if lam_df.empty:
        return pd.DataFrame()
    rows = []
    for h, g in lam_df.groupby(["source_name", "h_identity_key"], dropna=False):
        nonzero = g[g["lambda_reorg"].astype(float) > 0]
        use = nonzero if len(nonzero) else g
        vals = use["eta_primary"].astype(float)
        if vals.empty:
            continue
        best_idx = vals.idxmax()
        rows.append({
            "source_name": h[0],
            "h_identity_key": h[1],
            "n_lambda": int(g["lambda_reorg"].nunique()),
            "lambda_values": "|".join(map(lambda x: f"{float(x):g}", sorted(g["lambda_reorg"].dropna().unique()))),
            "eta20_lambda_min": float(vals.min()),
            "eta20_lambda_max": float(vals.max()),
            "eta20_lambda_range": float(vals.max() - vals.min()),
            "best_nonzero_lambda": float(g.loc[best_idx, "lambda_reorg"]),
            "best_nonzero_eta20": float(g.loc[best_idx, "eta_primary"]),
            "lambda35_eta20": float(g.loc[g["lambda_reorg"].astype(float).eq(35), "eta_primary"].iloc[0]) if (g["lambda_reorg"].astype(float).eq(35)).any() else np.nan,
            "low_nonzero_lambda_best": bool(float(g.loc[best_idx, "lambda_reorg"]) in {3.0, 5.0, 15.0}),
        })
    out = pd.DataFrame(rows)
    if len(out):
        q_range = out["eta20_lambda_range"].quantile(0.90)
        out["lambda_dramatic_dynamic_mode_candidate"] = out["eta20_lambda_range"] >= q_range
        out["lambda_response_strength"] = pd.cut(
            out["eta20_lambda_range"],
            bins=[-np.inf, 0.05, 0.15, 0.35, np.inf],
            labels=["weak_absolute", "moderate_absolute", "strong_absolute", "very_strong_absolute"],
        ).astype(str)
    return out


def mode_profiles(df: pd.DataFrame, label_col: str, cols: Sequence[str]) -> pd.DataFrame:
    rows = []
    for label, g in df.groupby(label_col, dropna=False):
        row = {"mode_label": label, "n_rows": len(g), "eta20_mean": g["eta20"].mean() if "eta20" in g else np.nan, "eta_primary_mean": g["eta_primary"].mean()}
        for c in cols:
            if c in g.columns and pd.api.types.is_numeric_dtype(g[c]):
                row[f"{c}_mean"] = g[c].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def write_markdown(path: Path, title: str, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "# " + title + "\n\n" + "\n".join(lines).rstrip() + "\n"
    path.write_text(text, encoding="utf-8")


def add_decision(decisions: List[Dict[str, object]], step: str, decision: str, reason: str, impact: str) -> None:
    decisions.append({"step": step, "decision": decision, "reason": reason, "impact": impact})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(ROOT / "new" / "bridge_dynamic_phenotype_mode_discovery_20260616"))
    ap.add_argument("--skip-umap", action="store_true")
    ap.add_argument("--max-umap-rows", type=int, default=160000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    dirs = ensure_dirs(out_dir)
    logger = Logger(dirs["logs"] / "progress.log")
    t0 = time.time()
    decisions: List[Dict[str, object]] = []
    adaptive_log: List[Dict[str, object]] = []

    try:
        logger.write("dynamic phenotype mode discovery started")
        add_decision(decisions, "initial", "run_now", "All required inputs are existing outputs; no new dynamics simulation needed.", "Use full available candidate rows from observed and path/detail sources.")

        file_rows = []
        for name, p in DEFAULT_INPUTS.items():
            row = file_info(p)
            row["input_name"] = name
            file_rows.append(row)
        pd.DataFrame(file_rows).to_csv(dirs["csv"] / "input_file_inventory.csv", index=False, encoding="utf-8-sig")

        observed_df, observed_audit = observed_features(DEFAULT_INPUTS["observed_npz"], logger)
        source_audit = [
            source_audit_rows("merged_h27_140k", DEFAULT_INPUTS["observed_npz"], observed_df, "observed_population", "observed_population_discovery")
        ]

        path_frames: List[pd.DataFrame] = []
        csv_specs = [
            ("bridge_same_s_detail", "linear_bridge_same_s", "constructed_path_overlay", "lambda_sweep_path_point", DEFAULT_INPUTS["bridge_detail"]),
            ("piecewise_waypoint_detail", "piecewise_waypoint", "constructed_path_overlay", "lambda_unknown_or_not_applicable", DEFAULT_INPUTS["piecewise_detail"]),
            ("dense_radius_detail", "dense_normal_radius", "perturbation_path_overlay", "lambda_unknown_or_not_applicable", DEFAULT_INPUTS["dense_detail"]),
            ("contribution_group_full_detail", "contribution_group_full", "perturbation_path_overlay", "lambda_sweep_path_point", DEFAULT_INPUTS["contribution_detail"]),
        ]
        for name, stype, role, lctx, path in csv_specs:
            if path.exists():
                logger.write(f"reading detail csv {name}: {path}")
                df = read_detail_csv(path, name, stype, role, lctx)
                path_frames.append(df)
                source_audit.append(source_audit_rows(name, path, df, stype, role))
            else:
                source_audit.append(source_audit_rows(name, path, None, stype, role, "missing"))

        source_audit_df = pd.DataFrame(source_audit)
        source_audit_df.to_csv(dirs["csv"] / "source_time_lambda_audit.csv", index=False, encoding="utf-8-sig")
        source_audit_df[["source_name", "source_type", "source_role", "allowed_use", "coverage_scope", "notes"]].to_csv(dirs["csv"] / "source_role_and_allowed_use.csv", index=False, encoding="utf-8-sig")
        observed_audit.to_csv(dirs["csv"] / "eta_time_convention_audit.csv", index=False, encoding="utf-8-sig")

        sf_dist = observed_df.groupby(["source_file_id", "lambda_reorg"], dropna=False).size().reset_index(name="n_rows")
        sf_summary = observed_df.groupby("source_file_id").agg(
            n_rows=("source_file_id", "size"),
            eta_primary_mean=("eta_primary", "mean"),
            eta_primary_p90=("eta_primary", lambda s: s.quantile(0.90)),
            lambda_min=("lambda_reorg", "min"),
            lambda_max=("lambda_reorg", "max"),
            lambda_nunique=("lambda_reorg", "nunique"),
        ).reset_index()
        sf_dist.to_csv(dirs["csv"] / "source_file_lambda_distribution_audit.csv", index=False, encoding="utf-8-sig")
        sf_summary.to_csv(dirs["csv"] / "source_file_lambda_summary.csv", index=False, encoding="utf-8-sig")

        metadata_rows = []
        for df in [observed_df] + path_frames:
            metadata_rows.append({
                "source_name": df["source_name"].iloc[0],
                "n_rows": len(df),
                "has_pair_id": bool("pair_id" in df.columns and df["pair_id"].astype(str).str.len().gt(0).any()),
                "has_ds_pair_label": bool("pair_id" in df.columns and df["pair_id"].astype(str).str.contains("-S", regex=False).any()),
                "has_support_level": bool("support_level" in df.columns and df["support_level"].astype(str).str.len().gt(0).any()),
                "has_plausibility": bool("plausibility_available" in df.columns and df["plausibility_available"].astype(bool).any()),
                "metadata_join_status_values": "|".join(sorted(map(str, df.get("metadata_join_status", pd.Series(dtype=str)).dropna().unique()))[:10]),
            })
        pd.DataFrame(metadata_rows).to_csv(dirs["csv"] / "metadata_join_availability_audit.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
            "baseline": "standard_fmo",
            "available": False,
            "status": "not_found_as_machine_readable_file_in_configured_inputs",
            "note": "Docs mention standard FMO eta values, but this run did not find a local machine-readable standard FMO trajectory in the required input set. Treat as anchor unavailable for this execution.",
        }]).to_csv(dirs["csv"] / "standard_fmo_anchor_availability.csv", index=False, encoding="utf-8-sig")

        stop_reasons = []
        if observed_audit["eta_label_vs_trap_eta50_median_abs_diff"].iloc[0] > 1e-3:
            stop_reasons.append("observed eta_label differs from recomputed trap eta50; keep eta_label and recomputed eta columns separate")
        if not path_frames:
            stop_reasons.append("no path overlay inputs available")
        go_lines = [
            "## Go/Revise Decision",
            "",
            "- Observed population baseline is available.",
            f"- Path overlay sources available: {len(path_frames)}.",
            "- Source roles are separated; constructed path points are not used to define observed population modes.",
            "- High-eta thresholds will be source-stratified.",
            "- Lambda response is restricted to lambda_sweep_path_point sources.",
        ]
        if stop_reasons:
            go_lines += ["", "### Revise caveats"] + [f"- {r}" for r in stop_reasons]
        else:
            go_lines += ["", "### Decision", "- Proceed to full feature extraction and mode discovery."]
        write_markdown(dirs["reports"] / "mode_discovery_input_go_stop.md", "Mode Discovery Input Go/Stop", go_lines)

        logger.write("combining feature rows")
        all_df = pd.concat([observed_df] + path_frames, ignore_index=True, sort=False)
        all_df["source_name"] = all_df["source_name"].fillna("unknown")
        all_df = high_eta_membership(all_df, dirs["csv"] / "eta_definition_inventory.csv", logger)
        all_df.to_csv(dirs["csv"] / "dynamic_feature_matrix_h_lambda.csv", index=False, encoding="utf-8-sig")
        observed_df = all_df[all_df["source_role"] == "observed_population_discovery"].copy()
        path_df = all_df[all_df["source_role"] != "observed_population_discovery"].copy()
        observed_df.to_csv(dirs["csv"] / "dynamic_feature_matrix_observed_population.csv", index=False, encoding="utf-8-sig")
        path_df.to_csv(dirs["csv"] / "dynamic_feature_matrix_path_overlay.csv", index=False, encoding="utf-8-sig")

        missing_rows = []
        for c in sorted(set(FEATURE_CANDIDATES).intersection(all_df.columns)):
            missing_rows.append({"feature": c, "missing_fraction": float(all_df[c].isna().mean()), "n_nonmissing": int(all_df[c].notna().sum())})
        pd.DataFrame(missing_rows).to_csv(dirs["csv"] / "missing_and_quality_flags.csv", index=False, encoding="utf-8-sig")

        feature_cols = existing_features(observed_df)
        groups = feature_groups(feature_cols)
        feature_dict_rows = []
        for group, cols in groups.items():
            for c in cols:
                feature_dict_rows.append({"feature": c, "feature_set": group, "used_for": "observed_population_baseline" if c in observed_df.columns else "unavailable"})
        pd.DataFrame(feature_dict_rows).to_csv(dirs["csv"] / "dynamic_feature_dictionary.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{"source_name": s, "n_rows": len(g), "feature_count": len(existing_features(g))} for s, g in all_df.groupby("source_name")]).to_csv(dirs["csv"] / "dynamic_feature_source_inventory.csv", index=False, encoding="utf-8-sig")

        logger.write("lambda response summary")
        lam_summary = lambda_response_summary(all_df)
        lam_summary.to_csv(dirs["csv"] / "lambda_response_h_level_summary.csv", index=False, encoding="utf-8-sig")

        # Eta membership overlap.
        defs = ["high_top5_source", "high_top10_source", "high_top20_source"]
        overlap = []
        for a in defs:
            for b in defs:
                n_a = int(all_df[a].sum())
                n_b = int(all_df[b].sum())
                n_ab = int((all_df[a] & all_df[b]).sum())
                overlap.append({"definition_a": a, "definition_b": b, "n_a": n_a, "n_b": n_b, "n_intersection": n_ab, "jaccard": n_ab / max(1, n_a + n_b - n_ab)})
        pd.DataFrame(overlap).to_csv(dirs["csv"] / "high_set_overlap_matrix.csv", index=False, encoding="utf-8-sig")
        all_df[["source_name", "source_type", "source_role", "h_identity_key", "dynamic_identity_key", "lambda_context_type", "eta_primary", "high_top5_source", "high_top10_source", "high_top20_source", "eta_membership_label"]].to_csv(dirs["csv"] / "eta_membership_by_definition.csv", index=False, encoding="utf-8-sig")
        all_df.groupby(["source_name", "eta_membership_label"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "core_conditional_borderline_high_groups.csv", index=False, encoding="utf-8-sig")

        # Main observed dynamic space.
        main_cols = groups["no_eta_no_final_trap"]
        if len(main_cols) < 5:
            main_cols = groups["full"]
            adaptive_log.append({"stage": "feature_set", "change": "fallback_to_full_features", "reason": "no_eta feature set had too few columns", "research_question_connection": "Need enough observed-population features for dynamic space shape."})
        add_decision(decisions, "feature_matrix", "run_now", f"Using {len(main_cols)} observed-population main features excluding direct eta/final-trap leakage where possible.", "Preserves all observed rows.")
        logger.write(f"preparing observed feature matrix with {len(main_cols)} columns and {len(observed_df)} rows")
        x_obs, imputer, scaler, used_cols = prepare_matrix(observed_df, main_cols)

        pca = PCA(n_components=min(10, x_obs.shape[1]), random_state=20260616)
        pca_x = pca.fit_transform(x_obs)
        pca_cols = [f"pc{i+1}" for i in range(pca_x.shape[1])]
        pca_df = observed_df[["source_name", "source_file_id", "source_row", "h_identity_key", "dynamic_identity_key", "eta_primary", "eta_membership_label"]].copy()
        for i, c in enumerate(pca_cols):
            pca_df[c] = pca_x[:, i]
        pca_df.to_csv(dirs["csv"] / "dynamic_embedding_pca.csv", index=False, encoding="utf-8-sig")

        logger.write("nearest-neighbor density metrics on observed PCA space")
        nn = NearestNeighbors(n_neighbors=11, algorithm="auto")
        nn.fit(pca_x[:, : min(6, pca_x.shape[1])])
        dist, _ = nn.kneighbors(pca_x[:, : min(6, pca_x.shape[1])])
        pca_df["nn10_mean_distance_pca"] = dist[:, 1:].mean(axis=1)
        pca_df[["h_identity_key", "nn10_mean_distance_pca"]].to_csv(dirs["csv"] / "observed_nearest_neighbor_density.csv", index=False, encoding="utf-8-sig")

        logger.write("clustering observed population")
        labels, k_metrics, best_k = choose_k_and_cluster(x_obs, logger)
        observed_df["dynamic_mode_label"] = [f"mode_{int(i)}" for i in labels]
        pca_df["dynamic_mode_label"] = observed_df["dynamic_mode_label"].values
        k_metrics["selected_k"] = best_k
        k_metrics.to_csv(dirs["csv"] / "cluster_stability_metrics.csv", index=False, encoding="utf-8-sig")

        # Feature-family ablation stability.
        ablation_rows = []
        for name, cols in groups.items():
            if name in {"full", "no_eta_no_final_trap"}:
                continue
            if len(cols) < 2:
                continue
            try:
                x_g, _, _, used_g = prepare_matrix(observed_df, cols)
                lab_g, metrics_g, k_g = choose_k_and_cluster(x_g, logger, max_k=6, sample_for_metrics=8000)
                ablation_rows.append({
                    "feature_family": name,
                    "n_features": len(used_g),
                    "selected_k": k_g,
                    "ami_vs_main_mode": float(adjusted_mutual_info_score(labels, lab_g)),
                    "best_silhouette_metric_sample": float(metrics_g["silhouette_metric_sample"].max(skipna=True)),
                })
            except Exception as exc:
                ablation_rows.append({"feature_family": name, "n_features": len(cols), "selected_k": np.nan, "ami_vs_main_mode": np.nan, "best_silhouette_metric_sample": np.nan, "error": str(exc)})
        ablation_df = pd.DataFrame(ablation_rows)
        ablation_df.to_csv(dirs["csv"] / "feature_family_ablation_stability.csv", index=False, encoding="utf-8-sig")

        # Null/sanity checks.
        logger.write("null and source-effect checks")
        rng = np.random.default_rng(20260616)
        high = observed_df["high_top10_source"].astype(int).to_numpy()
        shuffled = rng.permutation(high)
        mode_num = labels
        null_rows = []
        for name, y in [("actual_high_top10", high), ("shuffled_high_top10", shuffled), ("source_file_id", observed_df["source_file_id"].astype(int).to_numpy())]:
            try:
                ami = float(adjusted_mutual_info_score(mode_num, y))
            except Exception:
                ami = np.nan
            null_rows.append({"comparison": name, "adjusted_mutual_info_vs_mode": ami})
        null_df = pd.DataFrame(null_rows)
        null_df.to_csv(dirs["csv"] / "dynamic_space_null_check.csv", index=False, encoding="utf-8-sig")

        # Optional UMAP full observed. This is allowed to be skipped only when it blocks; PCA remains full.
        if not args.skip_umap:
            try:
                logger.write("running UMAP on full observed population")
                import umap

                reducer = umap.UMAP(n_neighbors=30, min_dist=0.08, n_components=2, metric="euclidean", random_state=20260616, low_memory=True)
                umap_x = reducer.fit_transform(x_obs)
                umap_df = observed_df[["source_name", "source_file_id", "source_row", "h_identity_key", "eta_primary", "eta_membership_label", "dynamic_mode_label"]].copy()
                umap_df["umap1"] = umap_x[:, 0]
                umap_df["umap2"] = umap_x[:, 1]
                umap_df.to_csv(dirs["csv"] / "dynamic_embedding_umap.csv", index=False, encoding="utf-8-sig")
            except Exception as exc:
                logger.write(f"UMAP failed or skipped by runtime issue: {exc}")
                adaptive_log.append({"stage": "umap", "change": "umap_unavailable", "reason": str(exc), "research_question_connection": "PCA/full clustering retained; UMAP is auxiliary visualization only."})
                pd.DataFrame([{"status": "umap_failed", "error": str(exc)}]).to_csv(dirs["csv"] / "dynamic_embedding_umap.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([{"status": "umap_skipped_by_arg"}]).to_csv(dirs["csv"] / "dynamic_embedding_umap.csv", index=False, encoding="utf-8-sig")

        # Path overlay on PCA and nearest mode center.
        if len(path_df):
            common_path_cols = [c for c in used_cols if c in path_df.columns]
            # Add missing columns as NaN so imputer can use observed medians.
            path_tmp = path_df.copy()
            for c in used_cols:
                if c not in path_tmp.columns:
                    path_tmp[c] = np.nan
            x_path = transform_matrix(path_tmp, used_cols, imputer, scaler)
            path_z_abs_max = np.nanmax(np.abs(x_path), axis=1)
            path_z_abs_gt5_frac = np.nanmean(np.abs(x_path) > 5.0, axis=1)
            path_z_abs_gt10_frac = np.nanmean(np.abs(x_path) > 10.0, axis=1)
            path_pca = pca.transform(x_path)
            km_final = MiniBatchKMeans(n_clusters=best_k, random_state=20260616, batch_size=8192, n_init=10, max_iter=300).fit(x_obs)
            path_labels = km_final.predict(x_path)
            path_overlay = path_df[["source_name", "source_type", "source_role", "pair_id", "alpha", "lambda_reorg", "h_identity_key", "dynamic_identity_key", "lambda_context_type", "eta_primary", "eta_membership_label", "support_level"]].copy()
            for i, c in enumerate(pca_cols[: min(5, len(pca_cols))]):
                path_overlay[c] = path_pca[:, i]
            path_overlay["nearest_observed_dynamic_mode_label"] = [f"mode_{int(i)}" for i in path_labels]
            path_overlay["path_overlay_feature_missing_fraction"] = path_tmp[used_cols].isna().mean(axis=1).to_numpy()
            path_overlay["path_overlay_max_abs_observed_z"] = path_z_abs_max
            path_overlay["path_overlay_feature_abs_z_gt5_fraction"] = path_z_abs_gt5_frac
            path_overlay["path_overlay_feature_abs_z_gt10_fraction"] = path_z_abs_gt10_frac
            path_overlay["path_overlay_feature_distribution_caveat"] = np.select(
                [
                    path_overlay["path_overlay_feature_abs_z_gt10_fraction"] > 0.05,
                    path_overlay["path_overlay_feature_abs_z_gt5_fraction"] > 0.10,
                    path_overlay["path_overlay_max_abs_observed_z"] > 20.0,
                ],
                [
                    "strong_feature_distribution_shift",
                    "moderate_feature_distribution_shift",
                    "single_feature_extreme_shift",
                ],
                default="within_observed_feature_scale_or_mild_shift",
            )
            path_overlay.to_csv(dirs["csv"] / "path_overlay_on_dynamic_embedding.csv", index=False, encoding="utf-8-sig")
            path_overlay.groupby(["source_name", "path_overlay_feature_distribution_caveat"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "path_overlay_feature_distribution_shift_summary.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(dirs["csv"] / "path_overlay_on_dynamic_embedding.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame().to_csv(dirs["csv"] / "path_overlay_feature_distribution_shift_summary.csv", index=False, encoding="utf-8-sig")

        # Mode profiles and shape decision.
        profile_cols = used_cols[:25]
        profiles = mode_profiles(observed_df, "dynamic_mode_label", profile_cols)
        profiles.to_csv(dirs["csv"] / "dynamic_mode_profile.csv", index=False, encoding="utf-8-sig")
        observed_df[["source_name", "source_file_id", "source_row", "h_identity_key", "dynamic_identity_key", "eta_primary", "eta_membership_label", "dynamic_mode_label"]].to_csv(dirs["csv"] / "dynamic_mode_assignment.csv", index=False, encoding="utf-8-sig")

        shape_metrics = []
        explained = pca.explained_variance_ratio_
        max_sil = float(k_metrics["silhouette_metric_sample"].max(skipna=True))
        db_at_best = float(k_metrics.loc[k_metrics["k"].eq(best_k), "davies_bouldin_metric_sample"].iloc[0])
        pc1_eta_corr = float(np.corrcoef(pca_x[:, 0], observed_df["eta_primary"].astype(float).to_numpy())[0, 1])
        source_ami = float(null_df.loc[null_df["comparison"].eq("source_file_id"), "adjusted_mutual_info_vs_mode"].iloc[0])
        actual_high_ami = float(null_df.loc[null_df["comparison"].eq("actual_high_top10"), "adjusted_mutual_info_vs_mode"].iloc[0])
        shuffled_high_ami = float(null_df.loc[null_df["comparison"].eq("shuffled_high_top10"), "adjusted_mutual_info_vs_mode"].iloc[0])
        if max_sil >= 0.25 and source_ami < 0.35:
            shape_label = "cluster_like_candidate"
        elif abs(pc1_eta_corr) >= 0.45:
            shape_label = "spectrum_like_eta_axis_candidate"
        elif max_sil < 0.12:
            shape_label = "mixed_or_weakly_structured"
        else:
            shape_label = "mixed_cluster_spectrum_candidate"
        if source_ami >= 0.35:
            shape_label += "_with_source_effect_caveat"
        shape_metrics.append({
            "observed_n": len(observed_df),
            "feature_set": "no_eta_no_final_trap" if set(used_cols) == set(groups["no_eta_no_final_trap"]) else "fallback",
            "n_features": len(used_cols),
            "pca_explained_var_pc1": float(explained[0]),
            "pca_explained_var_pc1_pc2": float(explained[:2].sum()),
            "selected_k": best_k,
            "max_silhouette_metric_sample": max_sil,
            "davies_bouldin_at_selected_k": db_at_best,
            "pc1_eta_primary_corr": pc1_eta_corr,
            "mode_vs_source_file_ami": source_ami,
            "mode_vs_actual_high_top10_ami": actual_high_ami,
            "mode_vs_shuffled_high_top10_ami": shuffled_high_ami,
            "shape_decision": shape_label,
        })
        shape_df = pd.DataFrame(shape_metrics)
        shape_df.to_csv(dirs["csv"] / "dynamic_space_shape_metrics.csv", index=False, encoding="utf-8-sig")

        # Archetype/extreme candidates.
        logger.write("archetype/extreme candidate extraction")
        arch_parts = []
        base = observed_df[["source_name", "source_file_id", "source_row", "h_identity_key", "eta_primary", "eta_membership_label", "dynamic_mode_label"]].copy()
        for pc in ["pc1", "pc2", "pc3"]:
            if pc in pca_df.columns:
                temp = pd.concat([base, pca_df[[pc]]], axis=1)
                for direction, asc in [("low", True), ("high", False)]:
                    cand = temp.sort_values(pc, ascending=asc).head(100).copy()
                    cand["extreme_definition"] = f"{pc}_{direction}"
                    cand["extreme_score"] = cand[pc]
                    arch_parts.append(cand)
        for feat in used_cols[:20]:
            temp = base.copy()
            temp[feat] = observed_df[feat]
            for direction, asc in [("low", True), ("high", False)]:
                cand = temp.sort_values(feat, ascending=asc).head(50).copy()
                cand["extreme_definition"] = f"{feat}_{direction}"
                cand["extreme_score"] = cand[feat]
                arch_parts.append(cand)
        archetypes = pd.concat(arch_parts, ignore_index=True, sort=False)
        overlap_counts = archetypes.groupby("h_identity_key").agg(
            n_extreme_definitions=("extreme_definition", "nunique"),
            definitions=("extreme_definition", lambda s: "|".join(sorted(set(map(str, s)))[:20])),
            eta_primary=("eta_primary", "first"),
            eta_membership_label=("eta_membership_label", "first"),
            dynamic_mode_label=("dynamic_mode_label", "first"),
            source_file_id=("source_file_id", "first"),
            source_row=("source_row", "first"),
        ).reset_index().sort_values(["n_extreme_definitions", "eta_primary"], ascending=[False, False])
        overlap_counts.to_csv(dirs["csv"] / "dynamic_archetype_candidates.csv", index=False, encoding="utf-8-sig")
        archetypes.groupby("extreme_definition").size().reset_index(name="n_candidates").to_csv(dirs["csv"] / "dynamic_extreme_definition_overlap.csv", index=False, encoding="utf-8-sig")
        overlap_counts.assign(
            caveat=np.where(overlap_counts["n_extreme_definitions"] >= 3, "repeated_extreme_candidate", "single_definition_extreme_only")
        ).to_csv(dirs["csv"] / "archetype_quality_and_caveat.csv", index=False, encoding="utf-8-sig")

        # Cross-tabs.
        observed_df.groupby(["dynamic_mode_label", "eta_membership_label"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "dynamic_mode_eta_crosstab.csv", index=False, encoding="utf-8-sig")
        observed_df.groupby(["dynamic_mode_label", "source_file_id"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "dynamic_mode_structural_crosstab.csv", index=False, encoding="utf-8-sig")
        if len(path_df):
            path_overlay = pd.read_csv(dirs["csv"] / "path_overlay_on_dynamic_embedding.csv")
            path_overlay.groupby(["nearest_observed_dynamic_mode_label", "source_name"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "dynamic_mode_ds_family_crosstab.csv", index=False, encoding="utf-8-sig")
            path_overlay.groupby(["nearest_observed_dynamic_mode_label", "support_level"]).size().reset_index(name="n_rows").to_csv(dirs["csv"] / "dynamic_mode_plausibility_support_summary.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame().to_csv(dirs["csv"] / "dynamic_mode_ds_family_crosstab.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame().to_csv(dirs["csv"] / "dynamic_mode_plausibility_support_summary.csv", index=False, encoding="utf-8-sig")

        eig_cols = [c for c in ["eig_min", "eig_max", "eig_spread", "h_param_norm"] if c in observed_df.columns]
        if eig_cols:
            mode_profiles(observed_df, "dynamic_mode_label", eig_cols).to_csv(dirs["csv"] / "dynamic_mode_eigenstate_summary.csv", index=False, encoding="utf-8-sig")
        else:
            pd.DataFrame([{"status": "no_observed_eigen_or_h_summary_columns"}]).to_csv(dirs["csv"] / "dynamic_mode_eigenstate_summary.csv", index=False, encoding="utf-8-sig")

        # Path search target recommendations.
        logger.write("path search target recommendation")
        targets = []
        if len(path_df):
            path_overlay = pd.read_csv(dirs["csv"] / "path_overlay_on_dynamic_embedding.csv", low_memory=False)
            path_by_pair = path_overlay.groupby("pair_id").agg(
                n_rows=("pair_id", "size"),
                n_nearest_modes=("nearest_observed_dynamic_mode_label", "nunique"),
                nearest_modes=("nearest_observed_dynamic_mode_label", lambda s: "|".join(sorted(set(map(str, s)))[:8])),
                eta_primary_min=("eta_primary", "min"),
                eta_primary_max=("eta_primary", "max"),
                eta_primary_mean=("eta_primary", "mean"),
                source_names=("source_name", lambda s: "|".join(sorted(set(map(str, s)))[:8])),
                missing_feature_median=("path_overlay_feature_missing_fraction", "median"),
                z_gt5_fraction_median=("path_overlay_feature_abs_z_gt5_fraction", "median"),
                z_gt10_fraction_median=("path_overlay_feature_abs_z_gt10_fraction", "median"),
                max_abs_z_median=("path_overlay_max_abs_observed_z", "median"),
                distribution_caveats=("path_overlay_feature_distribution_caveat", lambda s: "|".join(sorted(set(map(str, s)))[:8])),
            ).reset_index()
            ood = (path_by_pair["z_gt10_fraction_median"] > 0.05) | (path_by_pair["z_gt5_fraction_median"] > 0.10) | (path_by_pair["max_abs_z_median"] > 20.0)
            path_by_pair["target_type"] = np.select(
                [
                    ood,
                    path_by_pair["n_nearest_modes"].le(1) & path_by_pair["eta_primary_min"].ge(path_by_pair["eta_primary_min"].quantile(0.75)),
                    path_by_pair["n_nearest_modes"].gt(1),
                    (path_by_pair["eta_primary_max"] - path_by_pair["eta_primary_min"]).ge((path_by_pair["eta_primary_max"] - path_by_pair["eta_primary_min"]).quantile(0.90)),
                ],
                ["overlay_needs_feature_alignment_before_mode_claim", "same_mode_high_eta_connection_candidate", "different_mode_transition_candidate", "eta_range_transition_candidate"],
                default="path_overlay_reference_candidate",
            )
            path_by_pair["recommendation_strength"] = np.select(
                [
                    ood,
                    path_by_pair["missing_feature_median"] > 0.35,
                ],
                [
                    "feature_distribution_shift_caveat",
                    "feature_missingness_caveat",
                ],
                default="usable_overlay",
            )
            path_by_pair.to_csv(dirs["csv"] / "path_search_target_pairs_by_dynamic_mode.csv", index=False, encoding="utf-8-sig")
            usable_targets = path_by_pair[path_by_pair["recommendation_strength"].eq("usable_overlay")]
            target_source = usable_targets if len(usable_targets) else path_by_pair
            targets = target_source.sort_values(["n_nearest_modes", "eta_primary_mean"], ascending=[False, False]).head(30).to_dict("records")
        else:
            pd.DataFrame().to_csv(dirs["csv"] / "path_search_target_pairs_by_dynamic_mode.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame([{
                "criterion": "mode_to_path_search_usefulness",
                "decision": "useful_only_after_feature_alignment_check",
                "caveat": "If path overlay rows show large observed-feature z-score shifts, nearest observed mode labels are provisional and should not define mode-transition claims.",
            }]).to_csv(dirs["csv"] / "mode_to_path_search_usefulness.csv", index=False, encoding="utf-8-sig")

        # Claim ledger.
        claim_rows = [
            {
                "claim_id": "C1",
                "claim": f"Observed dynamic space is provisionally classified as {shape_label}.",
                "evidence": "dynamic_space_shape_metrics.csv; cluster_stability_metrics.csv; dynamic_space_null_check.csv",
                "caveat": "Feature choice, source_file effect, and eta/time conventions still constrain interpretation.",
                "allowed_claim": "dynamic feature space shape candidate",
                "forbidden_claim": "natural law mode count or causal mechanism proof",
            },
            {
                "claim_id": "C2",
                "claim": "Path-derived bridge/piecewise/dense/contribution points are overlays, not mode-defining population samples.",
                "evidence": "source_role_and_allowed_use.csv; path_overlay_on_dynamic_embedding.csv; path_overlay_feature_distribution_shift_summary.csv",
                "caveat": "Overlay feature mismatch or feature distribution shift can be substantial; nearest observed mode labels are provisional when z-score shift is high.",
                "allowed_claim": "path points can be compared to observed dynamic regions after feature alignment checks",
                "forbidden_claim": "constructed path points create or define observed modes, or definitely belong to a mode despite feature distribution shift",
            },
            {
                "claim_id": "C3",
                "claim": "Lambda response should be interpreted only for lambda_sweep_path_point sources.",
                "evidence": "lambda_response_h_level_summary.csv; source_time_lambda_audit.csv",
                "caveat": "Observed population rows have single lambda_reorg values, not same-H lambda sweeps.",
                "allowed_claim": "conditional lambda response for same-H/path sweep sources",
                "forbidden_claim": "single-lambda observed rows show lambda response",
            },
            {
                "claim_id": "C4",
                "claim": "Path-search target recommendations can be derived from mode overlay, but are prioritization tools rather than mechanism proof.",
                "evidence": "path_search_target_pairs_by_dynamic_mode.csv; mode_to_path_search_usefulness.csv",
                "caveat": "Targets with feature_distribution_shift_caveat should be used for feature-alignment follow-up first, not direct same-mode/different-mode claims.",
                "allowed_claim": "recommended follow-up target classes or alignment checks",
                "forbidden_claim": "mode connectivity has been proven",
            },
        ]
        pd.DataFrame(claim_rows).to_csv(dirs["csv"] / "claim_evidence_caveat_ledger.csv", index=False, encoding="utf-8-sig")

        pd.DataFrame(decisions).to_csv(dirs["reports"] / "execution_strategy_decisions.md", index=False)
        pd.DataFrame(adaptive_log).to_csv(dirs["csv"] / "adaptive_plan_change_log.csv", index=False, encoding="utf-8-sig")

        # Reports.
        elapsed = time.time() - t0
        shape = shape_metrics[0]
        report_lines = [
            "## What Was Run",
            "",
            f"- Observed population rows: {len(observed_df):,}",
            f"- Path/constructed overlay rows: {len(path_df):,}",
            f"- Main observed feature count: {len(used_cols)}",
            f"- Selected observed-mode k: {best_k}",
            f"- Shape decision: `{shape_label}`",
            f"- Runtime seconds: {elapsed:.1f}",
            "",
            "## Main Interpretation",
            "",
            "- The observed population was used as the only mode-defining baseline.",
            "- Constructed bridge, piecewise, dense-radius, and contribution points were projected as overlays.",
            "- Therefore the population-level picture is based on the full observed 140k sample, while path/probe conclusions are limited to the existing generated path/probe datasets.",
            "- High-eta was source-stratified; single-lambda observed rows were not treated as lambda-response evidence.",
            "- The result should be used to choose next path-search targets, not to claim causal mechanisms.",
            "",
            "## Key Quantitative Checks",
            "",
            f"- PCA PC1 explained variance: {shape['pca_explained_var_pc1']:.3f}",
            f"- PCA PC1+PC2 explained variance: {shape['pca_explained_var_pc1_pc2']:.3f}",
            f"- Best silhouette metric sample: {shape['max_silhouette_metric_sample']:.3f}",
            f"- PC1 vs eta_primary correlation: {shape['pc1_eta_primary_corr']:.3f}",
            f"- Mode vs source_file AMI: {shape['mode_vs_source_file_ami']:.3f}",
            f"- Mode vs actual high-top10 AMI: {shape['mode_vs_actual_high_top10_ami']:.3f}",
            f"- Mode vs shuffled high-top10 AMI: {shape['mode_vs_shuffled_high_top10_ami']:.3f}",
            "",
            "## Path Search Usefulness",
            "",
            "- Same-mode and different-mode target candidates are listed in `csv/path_search_target_pairs_by_dynamic_mode.csv`.",
            "- If overlay missing-feature fractions or observed-feature z-score shifts are high, those rows should be treated as provisional.",
            "- Recommended next step is to use the target list as a prioritization table for plausible path search, not as proof of connectivity.",
        ]
        if targets:
            report_lines += ["", "## Top Target Examples", ""]
            for t in targets[:10]:
                report_lines.append(f"- `{t.get('pair_id')}`: {t.get('target_type')} / modes={t.get('nearest_modes')} / eta range={t.get('eta_primary_min'):.3f}-{t.get('eta_primary_max'):.3f}")
        write_markdown(dirs["reports"] / "dynamic_phenotype_mode_discovery_report.md", "Dynamic Phenotype Mode Discovery Report", report_lines)

        write_markdown(
            dirs["reports"] / "dynamic_space_shape_decision.md",
            "Dynamic Space Shape Decision",
            [
                f"- Decision: `{shape_label}`",
                f"- Selected k: {best_k}",
                f"- Best silhouette metric sample: {max_sil:.3f}",
                f"- Source-file AMI: {source_ami:.3f}",
                f"- PC1 eta correlation: {pc1_eta_corr:.3f}",
                "",
                "This is a provisional dynamic-space shape label. It is not a mechanism proof.",
            ],
        )
        write_markdown(
            dirs["reports"] / "eta_definition_sensitivity_report.md",
            "Eta Definition Sensitivity",
            [
                "- High eta membership was computed per source_name, not pooled across observed and constructed sources.",
                "- top5/top10/top20 thresholds are listed in `csv/eta_definition_inventory.csv`.",
                "- Lambda=35/best/robust interpretations are restricted to lambda sweep path-point sources.",
            ],
        )
        write_markdown(
            dirs["reports"] / "dynamic_archetype_report.md",
            "Dynamic Archetype Candidate Report",
            [
                "- Archetype candidates were extracted from repeated PCA/feature extremes.",
                "- Candidates recurring across multiple definitions are stronger follow-up targets.",
                "- These are not mechanism classes; they are endpoint candidates for path-search questions.",
            ],
        )
        write_markdown(
            dirs["reports"] / "dynamic_structural_association_report.md",
            "Dynamic Structural Association Report",
            [
                "- Observed rows do not carry D/S labels, so D/S cross-tab is only available for path overlay rows.",
                "- D/S and path/probe results should not be generalized to the full observed population unless the source audit confirms coverage.",
                "- When path overlay feature distribution shift is high, nearest observed mode labels should be treated as provisional rather than direct mode membership.",
                "- Observed source-file cross-tabs are recorded as a batch/source-effect caveat.",
                "- H/eigen quick summaries are limited to available observed eigs/H norm columns.",
            ],
        )
        write_markdown(
            dirs["reports"] / "path_search_target_recommendation.md",
            "Path Search Target Recommendation",
            [
                "- Use `csv/path_search_target_pairs_by_dynamic_mode.csv` to choose same-mode and mode-transition candidates.",
                "- Do not treat target type as proof of connectivity.",
                "- First check `recommendation_strength`; feature-distribution-shift rows need alignment review before mode claims.",
                "- Next plausible path experiments should compare eta, support, model plausibility, smoothness, and dynamic-mode consistency.",
            ],
        )
        write_markdown(
            dirs["reports"] / "claim_evidence_caveat_summary.md",
            "Claim Evidence Caveat Summary",
            ["See `csv/claim_evidence_caveat_ledger.csv` for the machine-readable ledger."],
        )
        write_markdown(
            dirs["reports"] / "adaptive_plan_updates.md",
            "Adaptive Plan Updates",
            [
                "- Used existing detail CSVs for bridge/dense/contribution rather than reopening large NPZ payloads where equivalent summary columns existed.",
                "- This preserved full candidate rows while reducing unnecessary memory risk.",
                "- UMAP is auxiliary. PCA/full clustering and target tables remain the primary outputs.",
            ],
        )
        write_markdown(
            dirs["reports"] / "big_picture_and_overinterpretation_check.md",
            "Big Picture and Overinterpretation Check",
            [
                "- The goal is to refine path-search targets, not to complete a dynamic taxonomy.",
                "- The full-picture baseline is the observed 140k population. Existing bridge/path/probe outputs are overlays or prior experiment scopes, not replacements for the full population.",
                "- Dynamic modes are observed patterns, not causes.",
                "- Constructed path overlays should not define population modes.",
                "- Boundary/manifold/bottleneck/channel claims remain forbidden without additional support/geometry evidence.",
            ],
        )
        write_markdown(
            dirs["commands"] / "reproduce_dynamic_mode_discovery.md",
            "Reproduce Dynamic Mode Discovery",
            [
                "```powershell",
                f"& '{PYTHON_ENV}' new/run_dynamic_phenotype_mode_discovery.py --out-dir {out_dir}",
                "```",
            ],
        )
        write_markdown(
            dirs["reports"] / "path_search_go_no_go_decision.md",
            "Path Search Go/No-Go Decision",
            [
                "- Go: use mode overlay as a prioritization layer for plausible path search.",
                "- Caveat: if a candidate has high feature mismatch or source-specific instability, treat as provisional.",
                "- Do not stop path search if dynamic modes are weak; fallback to bridge/contribution/normal-vector target lists.",
            ],
        )

        pd.DataFrame([{
            "elapsed_seconds": elapsed,
            "observed_rows": len(observed_df),
            "path_overlay_rows": len(path_df),
            "shape_decision": shape_label,
            "selected_k": best_k,
        }]).to_json(dirs["json"] / "run_summary.json", orient="records", indent=2)
        logger.write(f"completed in {elapsed:.1f} seconds")
    finally:
        logger.close()


if __name__ == "__main__":
    main()
