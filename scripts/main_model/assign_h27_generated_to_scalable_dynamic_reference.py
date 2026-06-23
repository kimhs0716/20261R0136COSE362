#!/usr/bin/env python3
"""Assign generated H27 trajectories to the scalable dynamic-family reference.

This is the c_l1-free diversity audit for generated samples.  It does not
cluster generated samples by themselves.  Instead, it assigns each generated
trajectory to the nearest row in the existing 62k real dynamic reference, then
reports coverage over the reference dynamic family IDs and structural mode IDs.
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_RAW = Path("outputs/pilot_sampling/pilot62000_t50_schema_v2_20260603_merged/pilot_raw.npz")
DEFAULT_REF_ASSIGNMENTS = Path("outputs/scalable_mode_clustering_20260604/csv/dynamic_family_assignments.csv")
DEFAULT_OUT = Path("outputs/experiments/20260622_h27_scalable_reference_diversity_audit")
BASE_TARGETS = ("fast_high", "very_fast", "late_high", "non_high")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--detail", type=Path, required=True, help="Validation detail CSV, or dynamic_distance_assignments.csv.")
    p.add_argument("--trajectories", type=Path, required=True, help="Validator --save-trajectories NPZ.")
    p.add_argument("--condition-set", default="", help="Optional condition_set filter.")
    p.add_argument("--run-label", default="", help="Optional run_label filter.")
    p.add_argument("--target-regex", default="", help="Optional pandas regex filter on target.")
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--reference-assignments", type=Path, default=DEFAULT_REF_ASSIGNMENTS)
    p.add_argument(
        "--reference-feature-cache",
        type=Path,
        default=Path("outputs/scalable_mode_clustering_20260604/npz/scalable_dynamic_reference_features_t101_eta_path_pop.npz"),
        help="Optional precomputed c_l1-free reference features. Used when present, so the 1.8GB pilot_raw.npz is not required.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--components", default="eta,path,pop", help="Comma-separated eta,path,pop.")
    p.add_argument("--success-only", action="store_true", help="Keep only simulation_success rows if that column exists.")
    p.add_argument("--target-match-only", action="store_true", help="Keep only target_match=True rows.")
    p.add_argument("--max-reference-per-priority", type=int, default=-1, help="-1 uses all 62k reference rows.")
    p.add_argument("--generated-chunk-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=20260622)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name in ("csv", "json", "reports"):
        (args.out_dir / name).mkdir(exist_ok=True)

    detail = load_detail(args)
    generated = load_generated_features(args.trajectories, detail, args.components)
    reference = load_reference_features(args, generated["times"], generated["components"])
    assigned = assign_nearest_reference(detail, generated, reference, args.generated_chunk_size)
    summary = summarize(assigned, reference["reference_assignments"])
    family = summarize_breakdown(assigned, "dynamic_family_id")
    structural = summarize_breakdown(assigned, "structural_mode_id")

    stem = make_stem(args)
    assigned_path = args.out_dir / "csv" / f"{stem}_scalable_reference_assignments.csv"
    summary_path = args.out_dir / "csv" / f"{stem}_scalable_reference_summary.csv"
    family_path = args.out_dir / "csv" / f"{stem}_dynamic_family_breakdown.csv"
    structural_path = args.out_dir / "csv" / f"{stem}_structural_mode_breakdown.csv"
    report_path = args.out_dir / "reports" / f"{stem}_scalable_reference_diversity_report_kr.md"
    manifest_path = args.out_dir / "json" / f"{stem}_manifest.json"

    assigned.to_csv(assigned_path, index=False)
    summary.to_csv(summary_path, index=False)
    family.to_csv(family_path, index=False)
    structural.to_csv(structural_path, index=False)
    write_report(report_path, args, summary, family, structural)
    manifest = {
        "created_by": "scripts/assign_h27_generated_to_scalable_dynamic_reference.py",
        "detail": str(args.detail),
        "trajectories": str(args.trajectories),
        "raw": str(args.raw),
        "reference_assignments": str(args.reference_assignments),
        "components": generated["components"],
        "n_generated_assigned": int(len(assigned)),
        "outputs": {
            "assignments": str(assigned_path),
            "summary": str(summary_path),
            "dynamic_family_breakdown": str(family_path),
            "structural_mode_breakdown": str(structural_path),
            "report": str(report_path),
        },
    }
    manifest_path.write_text(json.dumps(clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"assigned: {assigned_path}")
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")
    return 0


def load_detail(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_csv(args.detail)
    if args.condition_set:
        df = df[df["condition_set"].astype(str).eq(args.condition_set)]
    if args.run_label:
        if "run_label" in df:
            df = df[df["run_label"].astype(str).eq(args.run_label)]
        else:
            df = df.copy()
            df["run_label"] = args.run_label
    if args.target_regex:
        df = df[df["target"].astype(str).str.contains(args.target_regex, regex=True, na=False)]
    if "simulation_success" in df and args.success_only:
        df = df[df["simulation_success"].map(boolish)]
    if args.target_match_only:
        df = df[df["target_match"].map(boolish)]
    if "base_target" not in df:
        df = df.copy()
        df["base_target"] = df["target"].astype(str).map(parse_base_target)
    if "simulation_success" not in df:
        df = df.copy()
        df["simulation_success"] = True
    required = {"condition_set", "target", "source_generated_index"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"detail is missing required columns: {missing}")
    if df.empty:
        raise RuntimeError("detail filter left zero rows")
    return df.reset_index(drop=True)


def load_generated_features(path: Path, detail: pd.DataFrame, components_text: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    components = [x.strip().lower() for x in components_text.split(",") if x.strip()]
    rows = []
    features = []
    common_times: np.ndarray | None = None
    missing = 0
    for _, row in detail.iterrows():
        label = f"{row['condition_set']}_{row['target']}_{int(row['source_generated_index']):04d}"
        time_key = f"{label}_times"
        eta_key = f"{label}_eta_t"
        path_key = f"{label}_path_t"
        pop_key = f"{label}_pop_t"
        if time_key not in z or eta_key not in z or path_key not in z:
            missing += 1
            continue
        if "pop" in components and pop_key not in z:
            components = [x for x in components if x != "pop"]
        times = np.asarray(z[time_key], dtype=np.float32)
        if common_times is None:
            common_times = times
        elif len(times) != len(common_times) or not np.allclose(times, common_times):
            raise RuntimeError("generated trajectory NPZ contains non-common time grids; add interpolation first")
        eta_t = np.asarray(z[eta_key], dtype=np.float32)
        path_t = np.asarray(z[path_key], dtype=np.float32)
        pop_t = np.asarray(z[pop_key], dtype=np.float32)[:, :7] if "pop" in components else None
        features.append(make_feature(eta_t, path_t, pop_t, components))
        rows.append(row.to_dict())
    if missing:
        print(f"warning: missing generated trajectories for {missing} detail rows", flush=True)
    if not rows or common_times is None:
        raise RuntimeError("no generated trajectory rows loaded")
    return {
        "rows": pd.DataFrame(rows),
        "features": np.stack(features).astype(np.float32),
        "times": common_times,
        "components": components,
    }


def load_reference_features(args: argparse.Namespace, times: np.ndarray, components: list[str]) -> dict[str, Any]:
    ref_df = pd.read_csv(args.reference_assignments)
    cache_path = resolve_reference_feature_cache(args.reference_feature_cache)
    if cache_path is not None:
        cached = load_reference_feature_cache(cache_path, ref_df, times, components)
        if cached is not None:
            ref_df, features = cached
            ref_df, features = maybe_subsample_reference(ref_df, features, args.max_reference_per_priority, args.seed)
            print(
                f"loaded scalable reference feature cache: n={len(ref_df)} feature_dim={features.shape[1]} "
                f"priority_groups={sorted(ref_df['priority_group'].astype(str).unique())}",
                flush=True,
            )
            return {"features": features, "reference_assignments": ref_df}
        print(f"warning: ignored incompatible reference feature cache: {cache_path}", flush=True)
    if not args.raw.exists():
        raise FileNotFoundError(
            f"missing raw reference: {args.raw}\n"
            f"also missing/invalid feature cache: {args.reference_feature_cache}\n"
            "Use a cache created from eta_t/path_t/pop_t[:7], or upload pilot_raw.npz."
        )
    raw = np.load(args.raw, allow_pickle=True)
    raw_times = np.asarray(raw["times_dense"] if "times_dense" in raw else raw["times"], dtype=np.float32)
    ref_df, keep = maybe_subsample_reference_rows(ref_df, args.max_reference_per_priority, args.seed)
    keep = np.ones(len(ref_df), dtype=bool)
    rows = ref_df["real_row"].to_numpy(int)
    eta = np.asarray(raw["eta_t_dense"] if "eta_t_dense" in raw else raw["eta_t"], dtype=np.float32)[rows]
    path_t = np.asarray(raw["path_t_dense"] if "path_t_dense" in raw else raw["path_t"], dtype=np.float32)[rows]
    pop_t = np.asarray(raw["pop_t_dense"] if "pop_t_dense" in raw else raw["pop_t"], dtype=np.float32)[rows, :, :7]
    ref_features = []
    for i in range(len(rows)):
        eta_i = interp_vector(raw_times, eta[i], times)
        path_i = interp_matrix(raw_times, path_t[i], times)
        pop_i = interp_matrix(raw_times, pop_t[i], times) if "pop" in components else None
        ref_features.append(make_feature(eta_i, path_i, pop_i, components))
    features = np.stack(ref_features).astype(np.float32)
    print(
        f"loaded scalable reference: n={len(ref_df)} feature_dim={features.shape[1]} "
        f"priority_groups={sorted(ref_df['priority_group'].astype(str).unique())}",
        flush=True,
    )
    return {"features": features, "reference_assignments": ref_df}


def resolve_reference_feature_cache(path: Path) -> Path | None:
    if path.exists():
        return path
    manifest = path.with_name(path.stem + "_manifest.json")
    if manifest.exists():
        return manifest
    return None


def load_reference_feature_cache(
    path: Path,
    ref_df: pd.DataFrame,
    times: np.ndarray,
    components: list[str],
) -> tuple[pd.DataFrame, np.ndarray] | None:
    if path.suffix.lower() == ".json":
        return load_reference_feature_cache_manifest(path, ref_df, times, components)
    z = np.load(path, allow_pickle=True)
    if "features" not in z or "times" not in z or "components" not in z:
        return None
    cache_times = np.asarray(z["times"], dtype=np.float32)
    if len(cache_times) != len(times) or not np.allclose(cache_times, times):
        return None
    cache_components = [str(x) for x in np.asarray(z["components"]).tolist()]
    if cache_components != list(components):
        return None
    features = np.asarray(z["features"], dtype=np.float32)
    if "real_row" in z:
        real_rows = np.asarray(z["real_row"], dtype=int)
        ref_df = ref_df.set_index("real_row").loc[real_rows].reset_index()
    if len(ref_df) != len(features):
        return None
    return ref_df.reset_index(drop=True), features


def load_reference_feature_cache_manifest(
    path: Path,
    ref_df: pd.DataFrame,
    times: np.ndarray,
    components: list[str],
) -> tuple[pd.DataFrame, np.ndarray] | None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    cache_times = np.asarray(manifest.get("times", []), dtype=np.float32)
    if len(cache_times) != len(times) or not np.allclose(cache_times, times):
        return None
    cache_components = [str(x) for x in manifest.get("components", [])]
    if cache_components != list(components):
        return None
    feature_parts = []
    real_row_parts = []
    for shard_name in manifest.get("shards", []):
        shard_path = path.parent / shard_name
        if not shard_path.exists():
            raise FileNotFoundError(f"missing reference feature cache shard: {shard_path}")
        z = np.load(shard_path, allow_pickle=True)
        feature_parts.append(np.asarray(z["features"], dtype=np.float32))
        real_row_parts.append(np.asarray(z["real_row"], dtype=int))
    if not feature_parts:
        return None
    features = np.concatenate(feature_parts, axis=0)
    real_rows = np.concatenate(real_row_parts, axis=0)
    ref_df = ref_df.set_index("real_row").loc[real_rows].reset_index()
    if len(ref_df) != len(features):
        return None
    return ref_df.reset_index(drop=True), features


def maybe_subsample_reference_rows(ref_df: pd.DataFrame, max_per_priority: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    keep = np.ones(len(ref_df), dtype=bool)
    if max_per_priority > 0:
        rng = np.random.default_rng(seed)
        keep[:] = False
        for _group, idx_df in ref_df.groupby("priority_group", sort=True):
            idx = idx_df.index.to_numpy()
            if len(idx) > max_per_priority:
                idx = np.sort(rng.choice(idx, size=max_per_priority, replace=False))
            keep[idx] = True
    return ref_df[keep].reset_index(drop=True), keep


def maybe_subsample_reference(
    ref_df: pd.DataFrame,
    features: np.ndarray,
    max_per_priority: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    sub_df, keep = maybe_subsample_reference_rows(ref_df, max_per_priority, seed)
    return sub_df, features[keep]


def assign_nearest_reference(
    detail: pd.DataFrame,
    generated: dict[str, Any],
    reference: dict[str, Any],
    chunk_size: int,
) -> pd.DataFrame:
    ref_df = reference["reference_assignments"].reset_index(drop=True)
    ref_x = reference["features"]
    out_rows: list[dict[str, Any]] = []
    for base_target, gen_idx_df in generated["rows"].groupby("base_target", sort=True):
        ref_idx = np.flatnonzero(ref_df["priority_group"].astype(str).to_numpy() == str(base_target))
        if len(ref_idx) == 0:
            print(f"warning: no reference rows for priority_group={base_target}", flush=True)
            continue
        ref_part = ref_x[ref_idx]
        ref_norm = np.einsum("ij,ij->i", ref_part, ref_part)
        gen_indices = gen_idx_df.index.to_numpy()
        for start in range(0, len(gen_indices), chunk_size):
            idx = gen_indices[start : start + chunk_size]
            gx = generated["features"][idx]
            gnorm = np.einsum("ij,ij->i", gx, gx)
            d2 = gnorm[:, None] + ref_norm[None, :] - 2.0 * gx @ ref_part.T
            np.maximum(d2, 0.0, out=d2)
            nearest_local = np.argmin(d2, axis=1)
            nearest_dist = np.sqrt(d2[np.arange(len(idx)), nearest_local])
            for local_i, gen_i in enumerate(idx):
                rpos = int(ref_idx[int(nearest_local[local_i])])
                ref_row = ref_df.iloc[rpos]
                row = dict(generated["rows"].iloc[int(gen_i)])
                row.update(
                    {
                        "nearest_reference_real_row": int(ref_row["real_row"]),
                        "nearest_reference_sample_index": int(ref_row["sample_index"]),
                        "dynamic_family_id": str(ref_row["dynamic_family_id"]),
                        "structural_mode_id": str(ref_row["structural_mode_id"]),
                        "nearest_reference_distance": float(nearest_dist[local_i]),
                        "distance_components": ",".join(generated["components"]),
                        "reference_pool_size_for_base_target": int(len(ref_idx)),
                    }
                )
                out_rows.append(row)
    if not out_rows:
        raise RuntimeError("no rows assigned to scalable reference")
    return pd.DataFrame(out_rows)


def summarize(assigned: pd.DataFrame, ref_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, sub in assigned.groupby(["run_label", "condition_set", "target", "base_target"], dropna=False):
        run_label, condition_set, target, base_target = keys
        ref_part = ref_df[ref_df["priority_group"].astype(str).eq(str(base_target))]
        family_counts = sub["dynamic_family_id"].value_counts()
        structural_counts = sub["structural_mode_id"].value_counts()
        ref_family_count = int(ref_part["dynamic_family_id"].nunique())
        ref_structural_count = int(ref_part["structural_mode_id"].nunique())
        rows.append(
            {
                "run_label": run_label,
                "condition_set": condition_set,
                "target": target,
                "base_target": base_target,
                "n_assigned": int(len(sub)),
                "target_match_rate": mean_bool(sub.get("target_match")),
                "reference_pool_size": int(sub["reference_pool_size_for_base_target"].iloc[0]),
                "dynamic_family_count": int(len(family_counts)),
                "reference_dynamic_family_count": ref_family_count,
                "dynamic_family_coverage_fraction": float(len(family_counts) / max(1, ref_family_count)),
                "top_dynamic_family_id": str(family_counts.index[0]),
                "top_dynamic_family_fraction": float(family_counts.iloc[0] / len(sub)),
                "dynamic_family_entropy_norm": normalized_entropy(family_counts.to_numpy(float)),
                "structural_mode_count": int(len(structural_counts)),
                "reference_structural_mode_count": ref_structural_count,
                "structural_mode_coverage_fraction": float(len(structural_counts) / max(1, ref_structural_count)),
                "top_structural_mode_id": str(structural_counts.index[0]),
                "top_structural_mode_fraction": float(structural_counts.iloc[0] / len(sub)),
                "nearest_reference_distance_median": median_col(sub, "nearest_reference_distance"),
                "nearest_reference_distance_p90": percentile_col(sub, "nearest_reference_distance", 90),
            }
        )
    return pd.DataFrame(rows).sort_values(["run_label", "condition_set", "target"]).reset_index(drop=True)


def summarize_breakdown(assigned: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    group_cols = ["run_label", "condition_set", "target", "base_target", column]
    for keys, sub in assigned.groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, keys))
        denom = len(
            assigned[
                (assigned["run_label"].astype(str) == str(base["run_label"]))
                & (assigned["condition_set"].astype(str) == str(base["condition_set"]))
                & (assigned["target"].astype(str) == str(base["target"]))
            ]
        )
        base.update(
            {
                "n": int(len(sub)),
                "fraction": float(len(sub) / max(1, denom)),
                "target_match_rate": mean_bool(sub.get("target_match")),
                "nearest_reference_distance_median": median_col(sub, "nearest_reference_distance"),
            }
        )
        rows.append(base)
    return pd.DataFrame(rows).sort_values(["run_label", "condition_set", "target", "fraction"], ascending=[True, True, True, False])


def make_feature(eta_t: np.ndarray, path_t: np.ndarray, pop_t: np.ndarray | None, components: list[str]) -> np.ndarray:
    parts = []
    if "eta" in components:
        eta = np.asarray(eta_t, dtype=np.float32).reshape(-1)
        parts.append(eta / math.sqrt(max(1, eta.size)))
    if "path" in components:
        path = np.asarray(path_t, dtype=np.float32).reshape(-1)
        parts.append(path / math.sqrt(max(1, path.size)))
    if "pop" in components and pop_t is not None:
        pop = np.asarray(pop_t, dtype=np.float32).reshape(-1)
        parts.append(pop / math.sqrt(max(1, pop.size)))
    if not parts:
        raise RuntimeError("no feature components selected")
    return np.concatenate(parts).astype(np.float32)


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary: pd.DataFrame,
    family: pd.DataFrame,
    structural: pd.DataFrame,
) -> None:
    lines = [
        "# H27 Scalable Dynamic-Reference Diversity Audit",
        "",
        "## Method",
        "",
        "Generated samples are not reclustered. Instead, each sample is assigned to the nearest row in the 62k real dynamic-reference set.",
        "The distance feature uses `eta_t`, `path_t`, and `pop_t[:, :7]`; it does not use c_l1, PCA, or static-H L2 distance.",
        "",
        "## Inputs",
        "",
        f"- detail: `{args.detail}`",
        f"- trajectories: `{args.trajectories}`",
        f"- raw reference trajectories: `{args.raw}`",
        f"- reference assignments: `{args.reference_assignments}`",
        f"- components: `{args.components}`",
        f"- target_match_only: `{args.target_match_only}`",
        "",
        "## Summary",
        "",
        md_table(summary),
        "",
        "## Dynamic family breakdown",
        "",
        md_table(family.head(80)),
        "",
        "## Structural mode breakdown",
        "",
        md_table(structural.head(80)),
        "",
        "## Interpretation",
        "",
        "- Low `top_dynamic_family_fraction` together with high `dynamic_family_count` suggests that successful samples spread across several reference dynamic families.",
        "- This should be compared with how many families exist for the corresponding target in the reference set itself.",
        "- This nearest-reference assignment is stronger than generated-only k-means, but the generated sample size per target may still be too small to guarantee rare tail-mode detection.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def make_stem(args: argparse.Namespace) -> str:
    bits = []
    if args.run_label:
        bits.append(slug(args.run_label))
    if args.condition_set:
        bits.append(slug(args.condition_set))
    if args.target_match_only:
        bits.append("targetmatch")
    if not bits:
        bits.append("all")
    return "_".join(bits)


def parse_base_target(target: str) -> str:
    for base_name in sorted(BASE_TARGETS, key=len, reverse=True):
        if target == base_name or target.startswith(f"{base_name}_"):
            return base_name
    return target


def interp_vector(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    return np.interp(t_dst, t_src, y).astype(np.float32)


def interp_matrix(t_src: np.ndarray, y: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
    arr = np.asarray(y, dtype=np.float32)
    return np.stack([np.interp(t_dst, t_src, arr[:, j]) for j in range(arr.shape[1])], axis=1).astype(np.float32)


def boolish(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None:
        return False
    if isinstance(value, float) and not math.isfinite(value):
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def mean_bool(values: Any) -> float:
    if values is None:
        return math.nan
    vals = pd.Series(values).map(boolish).astype(float)
    return float(vals.mean()) if len(vals) else math.nan


def normalized_entropy(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=float)
    counts = counts[np.isfinite(counts) & (counts > 0)]
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(counts)))


def median_col(df: pd.DataFrame, col: str) -> float:
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return float(np.nanmedian(vals)) if np.isfinite(vals).any() else math.nan


def percentile_col(df: pd.DataFrame, col: str, q: float) -> float:
    vals = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
    return float(np.nanpercentile(vals, q)) if np.isfinite(vals).any() else math.nan


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_rows ?놁쓬_"
    try:
        return df.to_markdown(index=False, floatfmt=f".{digits}f")
    except ImportError:
        return df.to_string(index=False)


def slug(text: str) -> str:
    keep = []
    for ch in str(text):
        keep.append(ch if ch.isalnum() else "_")
    return "_".join("".join(keep).split("_")).lower()


def clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (np.integer, np.floating)):
        return clean_json(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())

