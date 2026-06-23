#!/usr/bin/env python3
"""Train H27 CNF variants with c_l1-free dynamic-distance surrogate targets.

This is a thin wrapper around ``train_h27_cnf_mode_prior.py``.  The flow model,
generation format, and validation path stay the same, but the internal
HTBAL/branch mode targets are rebuilt from the c_l1-free dynamic-distance
reference:

    eta_t + path_t + pop_t[:, :7]

No c_l1, PCA shortcut, or static-H distance is used to define the internal
mode labels.  The trajectory surrogate still predicts population trajectories,
but its optional dynamic-class head now receives these corrected labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (ROOT, SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_h27_cnf_mode_prior as cnf
import train_h27_dynz_pinntraj_flow as pinnmod
from fmo_hamiltonian.trajectory_features import compute_eta_t, compute_path_groups


DEFAULT_REFERENCE_ASSIGNMENTS = Path("outputs/scalable_mode_clustering_20260604/csv/dynamic_family_assignments.csv")
DEFAULT_REFERENCE_FEATURE_CACHE = Path(
    "outputs/scalable_mode_clustering_20260604/npz/scalable_dynamic_reference_features_t101_eta_path_pop.npz"
)

_ORIGINAL_PARSE_ARGS = cnf.parse_args


def parse_args(argv: list[str] | None = None):
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--c-l1-free-reference-assignments", type=Path, default=DEFAULT_REFERENCE_ASSIGNMENTS)
    extra_parser.add_argument("--c-l1-free-reference-feature-cache", type=Path, default=DEFAULT_REFERENCE_FEATURE_CACHE)
    extra_parser.add_argument(
        "--c-l1-free-mode-column",
        choices=["dynamic_family_id", "structural_mode_id"],
        default="dynamic_family_id",
        help="Reference assignment column used as the internal mode label.",
    )
    extra_parser.add_argument(
        "--c-l1-free-components",
        default="eta,path,pop",
        help="Comma-separated c_l1-free distance components. Default matches the scalable diversity audit.",
    )
    extra_parser.add_argument(
        "--c-l1-free-assignment-chunk-size",
        type=int,
        default=16384,
        help="Rows per chunk when assigning 140k prepared rows to reference mode prototypes.",
    )
    extra_parser.add_argument(
        "--c-l1-free-use-sample-index-direct-labels",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Optionally override prototype assignment when prepared sample_index matches the reference CSV. "
            "Default is false because the 140k prepared rows and 62k reference rows may not share row provenance."
        ),
    )
    extra, remaining = extra_parser.parse_known_args(argv)
    args = _ORIGINAL_PARSE_ARGS(remaining)
    for key, value in vars(extra).items():
        setattr(args, key, value)
    args.htbal_mode_source = "c_l1_free_reference"
    return args


def main(argv: list[str] | None = None) -> int:
    cnf.parse_args = parse_args
    cnf.ddpm.build_htbal_mode_guidance = build_c_l1_free_reference_mode_guidance
    return cnf.main(argv)


def build_c_l1_free_reference_mode_guidance(context: dict[str, Any], args) -> dict[str, Any]:
    """Build internal mode guidance from c_l1-free dynamic reference families."""

    reference_assignments = resolve_existing(Path(args.c_l1_free_reference_assignments))
    reference_cache = resolve_existing(Path(args.c_l1_free_reference_feature_cache))
    components = [x.strip().lower() for x in str(args.c_l1_free_components).split(",") if x.strip()]
    if components != ["eta", "path", "pop"]:
        raise ValueError("This wrapper currently expects --c-l1-free-components eta,path,pop")

    ref = load_reference_feature_cache(reference_cache, components)
    ref_df = pd.read_csv(reference_assignments)
    ref_df = merge_reference_frame(ref_df, ref)

    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    times = np.asarray(context["raw"]["times"], dtype=np.float32)
    priority = np.asarray(context["priority_group"]).astype(str)
    train_idx = np.asarray(context["train_idx"], dtype=np.int64)
    sample_index = np.asarray(context["raw"]["sample_index"], dtype=np.int64)

    row_feature = make_sparse_c_l1_free_features(pop, components)
    ref_sparse_feature = downsample_reference_features(ref, times, components)
    pop_feat = pinnmod.numpy_pop_features(pop, times).astype(np.float32)

    target_groups = [x.strip() for x in str(args.target_guidance_groups).split(",") if x.strip()]
    mode_col = str(args.c_l1_free_mode_column)
    q_low = float(np.clip(args.htbal_feature_q_low, 0.0, 0.49))
    q_high = float(np.clip(args.htbal_feature_q_high, 0.51, 1.0))
    alpha = float(np.clip(args.htbal_prior_alpha, 0.0, 1.0))
    rho_min = float(max(0.0, args.htbal_prior_min))

    row_group_id = np.full(len(priority), -1, dtype=np.int64)
    row_mode_id = np.full(len(priority), -1, dtype=np.int64)
    corrected_labels = np.array([f"{g}:unguided" for g in priority], dtype=object)

    groups: list[str] = []
    by_group: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []
    chunk_size = max(1024, int(args.c_l1_free_assignment_chunk_size))

    for group in target_groups:
        ref_mask = ref_df["priority_group"].astype(str).eq(group).to_numpy()
        row_mask = priority == group
        group_train = train_idx[priority[train_idx] == group]
        if not np.any(ref_mask) or not np.any(row_mask) or len(group_train) < 2:
            continue

        group_ref = ref_df.loc[ref_mask].reset_index(drop=True)
        group_ref_x = ref_sparse_feature[ref_mask]
        mode_names = sorted(group_ref[mode_col].astype(str).unique().tolist())
        if len(mode_names) < 2:
            continue
        mode_to_id = {name: i for i, name in enumerate(mode_names)}

        proto_distance_features = []
        ref_counts = []
        for mode in mode_names:
            m = group_ref[mode_col].astype(str).eq(mode).to_numpy()
            ref_counts.append(int(m.sum()))
            proto_distance_features.append(np.nanmedian(group_ref_x[m], axis=0).astype(np.float32))
        proto_distance_features = np.stack(proto_distance_features).astype(np.float32)

        group_rows = np.flatnonzero(row_mask)
        assigned_local = assign_to_prototypes(
            row_feature[group_rows],
            proto_distance_features,
            chunk_size=chunk_size,
        )

        direct_hits = 0
        if bool(getattr(args, "c_l1_free_use_sample_index_direct_labels", False)):
            direct_map = dict(zip(group_ref["sample_index"].astype(int), group_ref[mode_col].astype(str)))
            for j, row_id in enumerate(group_rows):
                direct_mode = direct_map.get(int(sample_index[row_id]))
                if direct_mode in mode_to_id:
                    assigned_local[j] = int(mode_to_id[direct_mode])
                    direct_hits += 1

        gid = len(groups)
        groups.append(group)
        row_group_id[row_mask] = gid
        row_mode_id[group_rows] = assigned_local.astype(np.int64)
        corrected_labels[group_rows] = np.array([f"{group}:{mode_names[int(i)]}" for i in assigned_local], dtype=object)

        group_feat_train = pop_feat[group_train]
        feat_mu = np.nanmean(group_feat_train, axis=0).astype(np.float32)
        feat_sd = np.nanstd(group_feat_train, axis=0).astype(np.float32)
        feat_sd = np.where(feat_sd < 1e-6, 1.0, feat_sd).astype(np.float32)
        feat_low = np.nanquantile(group_feat_train, q_low, axis=0).astype(np.float32)
        feat_high = np.nanquantile(group_feat_train, q_high, axis=0).astype(np.float32)

        proto_pop: list[np.ndarray] = []
        proto_feat: list[np.ndarray] = []
        train_counts: list[int] = []
        for local_id, mode_name in enumerate(mode_names):
            rows_for_mode = group_train[row_mode_id[group_train] == local_id]
            if len(rows_for_mode) == 0:
                rows_for_mode = group_train
            train_counts.append(int(len(rows_for_mode)))
            p = np.nanmedian(pop[rows_for_mode], axis=0).astype(np.float32)
            p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
            proto_pop.append(p)
            proto_feat.append(np.nanmedian(pop_feat[rows_for_mode], axis=0).astype(np.float32))

        proto_pop_arr = np.stack(proto_pop).astype(np.float32)
        proto_feat_arr = np.stack(proto_feat).astype(np.float32)
        proto_z = ((proto_feat_arr - feat_mu[None, :]) / feat_sd[None, :]).astype(np.float32)

        ref_prior = np.asarray(ref_counts, dtype=np.float32)
        ref_prior = ref_prior / max(float(ref_prior.sum()), 1e-8)
        uniform = np.full_like(ref_prior, 1.0 / len(ref_prior), dtype=np.float32)
        pi_star = (1.0 - alpha) * ref_prior + alpha * uniform
        if rho_min > 0.0:
            pi_star = np.maximum(pi_star, rho_min).astype(np.float32)
            pi_star = pi_star / max(float(pi_star.sum()), 1e-8)

        by_group[group] = {
            "group_id": gid,
            "mode_source": "c_l1_free_reference",
            "mode_column": mode_col,
            "mode_names": mode_names,
            "proto_pop": proto_pop_arr,
            "proto_feat": proto_feat_arr,
            "proto_z": proto_z,
            "feat_mu": feat_mu,
            "feat_sd": feat_sd,
            "feat_low": feat_low,
            "feat_high": feat_high,
            "ref_prior": ref_prior.astype(np.float32),
            "pi_star": pi_star.astype(np.float32),
            "counts": ref_counts,
        }

        assigned_group_train = row_mode_id[group_train]
        for local_id, mode_name in enumerate(mode_names):
            summary_rows.append(
                {
                    "mode_source": "c_l1_free_reference",
                    "target_group": group,
                    "mode_column": mode_col,
                    "mode": mode_name,
                    "n_reference": int(ref_counts[local_id]),
                    "n_train_assigned": int(np.sum(assigned_group_train == local_id)),
                    "reference_fraction": float(ref_prior[local_id]),
                    "smoothed_internal_prior": float(pi_star[local_id]),
                    "direct_reference_label_rows": int(direct_hits),
                    "assignment_distance_components": ",".join(components),
                }
            )

    if not groups:
        raise RuntimeError(
            "No c_l1-free internal mode groups were built. "
            "Check reference assignments/cache paths and --target-guidance-groups."
        )

    context["dynamic_label"] = corrected_labels.astype(str)
    context["c_l1_free_dynamic_label"] = corrected_labels.astype(str)
    context["htbal_mode_guidance_summary"] = pd.DataFrame(summary_rows)
    context["c_l1_free_mode_guidance_meta"] = {
        "reference_assignments": str(reference_assignments),
        "reference_feature_cache": str(reference_cache),
        "mode_column": mode_col,
        "components": components,
        "prepared_times": times.tolist(),
        "reference_times": np.asarray(ref["times"], dtype=float).tolist(),
        "n_rows": int(len(priority)),
        "n_guided_rows": int(np.sum(row_mode_id >= 0)),
    }
    return {
        "groups": groups,
        "by_group": by_group,
        "row_group_id": row_group_id,
        "row_mode_id": row_mode_id,
        "summary": context["htbal_mode_guidance_summary"],
    }


def resolve_existing(path: Path) -> Path:
    if path.exists():
        return path
    if path.suffix.lower() == ".npz":
        manifest = path.with_name(path.stem + "_manifest.json")
        if manifest.exists():
            return manifest
    alt = ROOT / path
    if alt.exists():
        return alt
    if alt.suffix.lower() == ".npz":
        manifest = alt.with_name(alt.stem + "_manifest.json")
        if manifest.exists():
            return manifest
    raise FileNotFoundError(path)


def load_reference_feature_cache(path: Path, components: list[str]) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        return load_reference_feature_cache_manifest(path, components)
    z = np.load(path, allow_pickle=True)
    required = {"features", "times", "components", "sample_index", "priority_group", "dynamic_family_id", "structural_mode_id"}
    missing = sorted(required - set(z.files))
    if missing:
        raise ValueError(f"reference feature cache is missing keys: {missing}")
    cache_components = [str(x) for x in np.asarray(z["components"]).tolist()]
    if cache_components != components:
        raise ValueError(f"reference feature cache components={cache_components}, expected={components}")
    return {
        "features": np.asarray(z["features"], dtype=np.float32),
        "times": np.asarray(z["times"], dtype=np.float32),
        "components": cache_components,
        "real_row": np.asarray(z["real_row"], dtype=np.int64) if "real_row" in z else None,
        "sample_index": np.asarray(z["sample_index"], dtype=np.int64),
        "priority_group": np.asarray(z["priority_group"]).astype(str) if "priority_group" in z else None,
        "dynamic_family_id": np.asarray(z["dynamic_family_id"]).astype(str) if "dynamic_family_id" in z else None,
        "structural_mode_id": np.asarray(z["structural_mode_id"]).astype(str) if "structural_mode_id" in z else None,
    }


def load_reference_feature_cache_manifest(path: Path, components: list[str]) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cache_components = [str(x) for x in manifest.get("components", [])]
    if cache_components != components:
        raise ValueError(f"reference feature cache components={cache_components}, expected={components}")
    feature_parts = []
    real_row_parts = []
    sample_parts = []
    priority_parts = []
    dynamic_parts = []
    structural_parts = []
    for shard_name in manifest.get("shards", []):
        shard = np.load(path.parent / shard_name, allow_pickle=True)
        feature_parts.append(np.asarray(shard["features"], dtype=np.float32))
        real_row_parts.append(np.asarray(shard["real_row"], dtype=np.int64) if "real_row" in shard else None)
        sample_parts.append(np.asarray(shard["sample_index"], dtype=np.int64))
        if "priority_group" in shard:
            priority_parts.append(np.asarray(shard["priority_group"]).astype(str))
        if "dynamic_family_id" in shard:
            dynamic_parts.append(np.asarray(shard["dynamic_family_id"]).astype(str))
        if "structural_mode_id" in shard:
            structural_parts.append(np.asarray(shard["structural_mode_id"]).astype(str))
    if not feature_parts:
        raise RuntimeError(f"reference feature cache manifest has no shards: {path}")
    real_rows = None
    if all(part is not None for part in real_row_parts):
        real_rows = np.concatenate([part for part in real_row_parts if part is not None]).astype(np.int64)
    return {
        "features": np.concatenate(feature_parts, axis=0).astype(np.float32),
        "times": np.asarray(manifest["times"], dtype=np.float32),
        "components": cache_components,
        "real_row": real_rows,
        "sample_index": np.concatenate(sample_parts).astype(np.int64),
        "priority_group": np.concatenate(priority_parts).astype(str) if priority_parts else None,
        "dynamic_family_id": np.concatenate(dynamic_parts).astype(str) if dynamic_parts else None,
        "structural_mode_id": np.concatenate(structural_parts).astype(str) if structural_parts else None,
    }


def merge_reference_frame(ref_df: pd.DataFrame, ref: dict[str, Any]) -> pd.DataFrame:
    ref_df = ref_df.copy()
    if ref.get("real_row") is not None and "real_row" in ref_df.columns:
        key = "real_row"
        wanted = pd.Index(np.asarray(ref["real_row"], dtype=np.int64), name=key)
    else:
        key = "sample_index"
        wanted = pd.Index(np.asarray(ref["sample_index"], dtype=np.int64), name=key)
    if ref_df[key].duplicated().any():
        raise RuntimeError(f"reference assignments contain duplicate {key}; cannot align feature cache safely")
    out = ref_df.set_index(key).reindex(wanted).reset_index()
    missing = out["priority_group"].isna() if "priority_group" in out else np.ones(len(out), dtype=bool)
    if bool(np.asarray(missing).any()):
        raise RuntimeError(
            f"reference assignments do not cover all feature-cache rows by {key}: "
            f"missing={int(np.asarray(missing).sum())}"
        )
    for col in ("priority_group", "dynamic_family_id", "structural_mode_id"):
        cache_values = ref.get(col)
        if cache_values is None or col not in out:
            continue
        mismatch = out[col].astype(str).to_numpy() != np.asarray(cache_values).astype(str)
        if bool(mismatch.any()):
            print(f"warning: {col} mismatches between CSV and cache: {int(mismatch.sum())}", flush=True)
    return out.reset_index(drop=True)


def make_sparse_c_l1_free_features(pop: np.ndarray, components: list[str]) -> np.ndarray:
    parts = []
    if "eta" in components:
        eta = compute_eta_t(pop).astype(np.float32)
        parts.append(eta.reshape(len(pop), -1) / math.sqrt(max(1, eta.shape[1])))
    if "path" in components:
        path_t, _names = compute_path_groups(pop)
        parts.append(path_t.reshape(len(pop), -1) / math.sqrt(max(1, path_t.shape[1] * path_t.shape[2])))
    if "pop" in components:
        pop_site = pop[:, :, :7].astype(np.float32)
        parts.append(pop_site.reshape(len(pop), -1) / math.sqrt(max(1, pop_site.shape[1] * pop_site.shape[2])))
    return np.concatenate(parts, axis=1).astype(np.float32)


def downsample_reference_features(ref: dict[str, Any], prepared_times: np.ndarray, components: list[str]) -> np.ndarray:
    features = np.asarray(ref["features"], dtype=np.float32)
    ref_times = np.asarray(ref["times"], dtype=np.float32)
    prepared_times = np.asarray(prepared_times, dtype=np.float32)
    t_idx = np.array([int(np.argmin(np.abs(ref_times - t))) for t in prepared_times], dtype=np.int64)
    n_ref = features.shape[0]
    n_time = len(ref_times)
    offset = 0
    parts = []

    if "eta" in components:
        dense = features[:, offset : offset + n_time] * math.sqrt(n_time)
        offset += n_time
        eta = dense[:, t_idx]
        parts.append(eta / math.sqrt(max(1, len(t_idx))))
    if "path" in components:
        width = n_time * 7
        dense = features[:, offset : offset + width] * math.sqrt(width)
        offset += width
        path = dense.reshape(n_ref, n_time, 7)[:, t_idx, :]
        parts.append(path.reshape(n_ref, -1) / math.sqrt(max(1, len(t_idx) * 7)))
    if "pop" in components:
        width = n_time * 7
        dense = features[:, offset : offset + width] * math.sqrt(width)
        offset += width
        pop = dense.reshape(n_ref, n_time, 7)[:, t_idx, :]
        parts.append(pop.reshape(n_ref, -1) / math.sqrt(max(1, len(t_idx) * 7)))
    if offset != features.shape[1]:
        raise ValueError(f"unexpected feature width: consumed={offset}, available={features.shape[1]}")
    return np.concatenate(parts, axis=1).astype(np.float32)


def assign_to_prototypes(x: np.ndarray, prototypes: np.ndarray, chunk_size: int) -> np.ndarray:
    out = np.zeros(len(x), dtype=np.int64)
    proto_norm = np.einsum("ij,ij->i", prototypes, prototypes)
    for start in range(0, len(x), chunk_size):
        stop = min(len(x), start + chunk_size)
        part = x[start:stop]
        d2 = np.einsum("ij,ij->i", part, part)[:, None] + proto_norm[None, :] - 2.0 * part @ prototypes.T
        np.maximum(d2, 0.0, out=d2)
        out[start:stop] = np.argmin(d2, axis=1).astype(np.int64)
    return out


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

