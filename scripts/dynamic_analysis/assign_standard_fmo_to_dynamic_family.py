#!/usr/bin/env python3
"""Assign the standard FMO Hamiltonian to the existing global D families.

The existing D000-D012 labels were built from trajectory features, not from raw
Hamiltonian coordinates. This script therefore reuses the same dynamic feature
definition from ``fmo_research_clean/scripts/run_scalable_mode_clustering.py``:

  eta(t), d eta/dt, path-group trajectories, and a small set of dynamic metrics.

It then compares the stored standard-FMO trajectory against each D-family
centroid/medoid in that same scaled feature space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dynamic_diversity_audit_lambda35"
RAW_PATH = ROOT / "fmo_research_clean" / "outputs" / "pilot_sampling" / "pilot62000_t50_schema_v2_20260603_merged" / "pilot_raw.npz"
PREPARED_PATH = ROOT / "FMO_H27_context_ablation" / "data" / "clustered_from_clean" / "source_npz" / "prepared_flow_pilot_data.npz"
ASSIGN_PATH = ROOT / "FMO_H27_context_ablation" / "data" / "clustered_from_clean" / "scalable_mode_clustering" / "csv" / "dynamic_family_assignments.csv"
SUMMARY_PATH = ROOT / "FMO_H27_context_ablation" / "data" / "clustered_from_clean" / "scalable_mode_clustering" / "csv" / "dynamic_family_summary.csv"

DOWNSAMPLE_STRIDE = 4


def interp_many(times: np.ndarray, values: np.ndarray, query: float) -> np.ndarray:
    return np.asarray([np.interp(query, times, row) for row in values], dtype=np.float32)


def interp_one(times: np.ndarray, values: np.ndarray, query: float) -> float:
    return float(np.interp(query, times, values))


def dynamic_metrics(eta: np.ndarray, path: np.ndarray, times: np.ndarray, names: list[str], raw: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    cols: list[np.ndarray] = []
    col_names: list[str] = []

    for q in [10.0, 20.0, 50.0]:
        cols.append(interp_many(times, eta, q)[:, None])
        col_names.append(f"eta{int(q)}")

    for key in ["t80", "t90"]:
        cols.append(np.asarray(raw[key], dtype=np.float32)[:, None])
        col_names.append(key)

    final = np.maximum(eta[:, -1], 1e-7)
    tau = times[-1] - np.trapezoid(eta / final[:, None], times, axis=1)
    cols.append(tau[:, None].astype(np.float32))
    col_names.append("tau_transfer_est")

    mask10 = times <= 10.0
    width10 = max(float(times[mask10][-1] - times[mask10][0]), 1e-6)
    for group in ["sink34", "detour567"]:
        gi = names.index(group)
        residence = np.trapezoid(path[:, mask10, gi], times[mask10], axis=1) / width10
        cols.append(residence[:, None].astype(np.float32))
        col_names.append(f"residence_{group}_0_10ps")

    for group in ["trap", "loss", "residual", "sink34", "detour567"]:
        gi = names.index(group)
        cols.append(path[:, -1, gi][:, None].astype(np.float32))
        col_names.append(f"{group}_50ps")

    return np.concatenate(cols, axis=1).astype(np.float32), col_names


def dynamic_metrics_one(eta: np.ndarray, path: np.ndarray, times: np.ndarray, names: list[str], raw: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    eta2 = eta[None, :].astype(np.float32)
    path2 = path[None, :, :].astype(np.float32)
    wrapped_raw = {
        "t80": np.asarray([float(raw["standard_fmo_t80"])], dtype=np.float32),
        "t90": np.asarray([float(raw["standard_fmo_t90"])], dtype=np.float32),
    }
    metrics, metric_names = dynamic_metrics(eta2, path2, times, names, wrapped_raw)
    return metrics[0], metric_names


def robust_center_scale(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = np.nanmedian(x, axis=0).astype(np.float32)
    q75 = np.nanpercentile(x, 75, axis=0)
    q25 = np.nanpercentile(x, 25, axis=0)
    scale = (q75 - q25).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    scale = np.where(np.isfinite(scale) & (scale > 1e-7), scale, std)
    scale = np.where(np.isfinite(scale) & (scale > 1e-7), scale, 1.0).astype(np.float32)
    return center, scale


def apply_scale(x: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    scaled = ((x - center) / scale).astype(np.float32)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def build_dynamic_features(raw: dict[str, np.ndarray], train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], tuple[np.ndarray, np.ndarray]]:
    eta = np.asarray(raw["eta_t_dense"], dtype=np.float32)
    path = np.asarray(raw["path_t_dense"], dtype=np.float32)
    times = np.asarray(raw["times_dense"], dtype=np.float32)
    names = [str(x) for x in raw["path_group_names"]]

    stride = max(1, int(DOWNSAMPLE_STRIDE))
    ds = np.arange(0, len(times), stride)
    if ds[-1] != len(times) - 1:
        ds = np.append(ds, len(times) - 1)

    eta_ds = eta[:, ds]
    deta = np.gradient(eta, times, axis=1)[:, ds]
    path_ds = path[:, ds, :].reshape(len(eta), -1)
    metrics, metric_names = dynamic_metrics(eta, path, times, names, raw)
    x_raw = np.concatenate([eta_ds, deta, path_ds, metrics], axis=1).astype(np.float32)

    center, scale = robust_center_scale(x_raw[train_idx])
    x = apply_scale(x_raw, center, scale)
    return x, times, ds, metric_names, (center, scale)


def build_standard_fmo_feature(raw: dict[str, np.ndarray], times: np.ndarray, ds: np.ndarray, metric_names_ref: list[str], scaler: tuple[np.ndarray, np.ndarray]) -> tuple[np.ndarray, dict[str, float]]:
    eta = np.asarray(raw["standard_fmo_eta_t_dense"], dtype=np.float32)
    path = np.asarray(raw["standard_fmo_path_t_dense"], dtype=np.float32)
    names = [str(x) for x in raw["path_group_names"]]

    eta_ds = eta[ds]
    deta = np.gradient(eta, times)[ds]
    path_ds = path[ds, :].reshape(-1)
    metrics, metric_names = dynamic_metrics_one(eta, path, times, names, raw)
    if metric_names != metric_names_ref:
        raise RuntimeError(f"metric name mismatch: {metric_names} != {metric_names_ref}")

    x_raw = np.concatenate([eta_ds, deta, path_ds, metrics], axis=0).astype(np.float32)
    center, scale = scaler
    x = apply_scale(x_raw[None, :], center, scale)[0]

    metric_map = {name: float(value) for name, value in zip(metric_names, metrics)}
    metric_map.update(
        {
            "eta_final": float(raw["standard_fmo_eta_final"]),
            "t25": float(raw["standard_fmo_t25"]),
            "t50": float(raw["standard_fmo_t50"]),
            "t80": float(raw["standard_fmo_t80"]),
            "t90": float(raw["standard_fmo_t90"]),
        }
    )
    return x, metric_map


def assign_to_families(x: np.ndarray, labels: np.ndarray, x_fmo: np.ndarray, summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for family in sorted(pd.unique(labels)):
        idx = np.flatnonzero(labels == family)
        xf = x[idx]
        centroid = np.nanmean(xf, axis=0)
        centroid_distance = float(np.linalg.norm(x_fmo - centroid))
        row_distances = np.linalg.norm(xf - x_fmo[None, :], axis=1)
        medoid_pos = int(np.argmin(row_distances))
        medoid_distance = float(row_distances[medoid_pos])
        medoid_real_row = int(idx[medoid_pos])
        rows.append(
            {
                "dynamic_family_id": str(family),
                "centroid_distance": centroid_distance,
                "nearest_member_distance": medoid_distance,
                "nearest_member_real_row": medoid_real_row,
                "n": int(len(idx)),
            }
        )

    out = pd.DataFrame(rows).sort_values("centroid_distance").reset_index(drop=True)
    out["centroid_rank"] = np.arange(1, len(out) + 1)
    out = out.merge(summary, on="dynamic_family_id", how="left", suffixes=("", "_summary"))
    return out


def main() -> int:
    raw_npz = np.load(RAW_PATH, allow_pickle=True)
    raw = {key: raw_npz[key] for key in raw_npz.files}
    prepared_npz = np.load(PREPARED_PATH, allow_pickle=True)
    train_idx = np.asarray(prepared_npz["split_train"], dtype=int)
    assignments = pd.read_csv(ASSIGN_PATH)
    summary = pd.read_csv(SUMMARY_PATH)

    if not bool(raw["standard_fmo_available"]):
        raise RuntimeError(f"stored standard FMO trajectory unavailable: {raw['standard_fmo_error']}")

    x, times, ds, metric_names, scaler = build_dynamic_features(raw, train_idx)

    labels = np.empty(len(x), dtype=object)
    labels[:] = None
    for _, row in assignments.iterrows():
        labels[int(row["real_row"])] = str(row["dynamic_family_id"])
    if np.any(labels == None):  # noqa: E711
        missing = int(np.sum(labels == None))  # noqa: E711
        raise RuntimeError(f"{missing} rows have no D-family label")

    x_fmo, fmo_metrics = build_standard_fmo_feature(raw, times, ds, metric_names, scaler)
    family_rank = assign_to_families(x, labels, x_fmo, summary)

    tables = OUT / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    family_path = tables / "standard_fmo_dynamic_family_assignment.csv"
    metrics_path = tables / "standard_fmo_dynamic_metrics.csv"
    manifest_path = OUT / "standard_fmo_dynamic_family_assignment_manifest.json"

    family_rank.to_csv(family_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([fmo_metrics]).to_csv(metrics_path, index=False, encoding="utf-8-sig")
    manifest_path.write_text(
        json.dumps(
            {
                "raw_path": str(RAW_PATH.relative_to(ROOT)),
                "prepared_path": str(PREPARED_PATH.relative_to(ROOT)),
                "assignments_path": str(ASSIGN_PATH.relative_to(ROOT)),
                "summary_path": str(SUMMARY_PATH.relative_to(ROOT)),
                "feature_dim": int(x.shape[1]),
                "downsample_stride": int(DOWNSAMPLE_STRIDE),
                "nearest_family_by_centroid": str(family_rank.iloc[0]["dynamic_family_id"]),
                "nearest_family_by_centroid_distance": float(family_rank.iloc[0]["centroid_distance"]),
                "nearest_family_by_member": str(family_rank.sort_values("nearest_member_distance").iloc[0]["dynamic_family_id"]),
                "nearest_member_distance": float(family_rank.sort_values("nearest_member_distance").iloc[0]["nearest_member_distance"]),
                "outputs": {
                    "family_assignment": str(family_path.relative_to(ROOT)),
                    "fmo_metrics": str(metrics_path.relative_to(ROOT)),
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print("standard FMO dynamic-family assignment")
    print(f"  feature_dim: {x.shape[1]}")
    print(f"  nearest by centroid: {family_rank.iloc[0]['dynamic_family_id']}  distance={family_rank.iloc[0]['centroid_distance']:.4f}")
    by_member = family_rank.sort_values("nearest_member_distance").iloc[0]
    print(f"  nearest by member:   {by_member['dynamic_family_id']}  distance={by_member['nearest_member_distance']:.4f}")
    print(f"  saved: {family_path}")
    print(f"  saved: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
