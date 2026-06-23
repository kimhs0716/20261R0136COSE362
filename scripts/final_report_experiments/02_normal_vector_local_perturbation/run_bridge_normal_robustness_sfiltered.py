#!/usr/bin/env python3
"""Normal-vector robustness sweep for selected D/S same-S bridge targets.

This runner starts from already-computed straight bridge paths.  For each
selected pair/alpha point it builds perturbations in directions orthogonal to
the endpoint-to-endpoint tangent in H_gauge_27, then runs the same dynamics
simulation used by the bridge noise sweep.

It is meant as a Priority D validation runner, not a subtype-discovery script.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
NEW = ROOT / "new"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(NEW) not in sys.path:
    sys.path.insert(0, str(NEW))

from fmo_hamiltonian import sampling, simulator  # noqa: E402
from fmo_hamiltonian.constants import N_SITE  # noqa: E402
from fmo_hamiltonian.trajectory_features import compute_arrival_times, compute_eta_t, compute_path_groups  # noqa: E402

import run_hamiltonian_bridge_noise_sfiltered as bridge  # noqa: E402


DEFAULT_MANIFEST = Path("new/bridge_group_trend_priorityD_validation_plan_20260613/csv/validation_target_manifest.csv")
DEFAULT_RAW = Path("outputs/pilot_sampling/pilot62000_t50_schema_v2_20260603_merged/pilot_raw.npz")
DEFAULT_OUT = Path("outputs/bridge_normal_robustness_priorityD_20260613")
PATH_PANELS = ["site1", "site2", "sink34", "detour567", "trap", "loss", "residual"]


@dataclass(frozen=True)
class RobustnessJob:
    job_order: int
    target_i: int
    alpha_i: int
    lambda_i: int
    radius_i: int
    direction_i: int
    pair_id: str
    target_priority: int
    target_category: str
    group_name: str
    feature_set: str
    alpha: float
    alpha_source_index: int
    lambda_reorg: float
    radius: float
    radius_mode: str
    perturb_norm_h_gauge_27: float
    tangent_norm_h_gauge_27: float
    normal_dot_tangent: float
    h_path: np.ndarray
    h_perturbed: np.ndarray
    times: np.ndarray
    nearest_support_distance: float
    support_level: str


class ProgressLogger:
    def __init__(self, log_path: Path | None):
        self.log_path = log_path
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text("", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        if self.log_path is not None:
            with self.log_path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    p.add_argument("--target-priorities", default="", help="Comma list such as 1,2. Empty means all.")
    p.add_argument("--pairs-file", type=Path, default=None, help="Optional pair-id allowlist.")
    p.add_argument("--max-targets", type=int, default=0, help="0 means all selected targets.")
    p.add_argument("--alpha-values", default="", help="Override manifest alpha_subset for every target.")
    p.add_argument("--lambda-values", default="", help="Override manifest lambda_subset for every target.")
    p.add_argument("--radius-values", default="0,0.25,0.5,1.0,1.5")
    p.add_argument("--radius-mode", choices=["tangent-fraction", "absolute"], default="tangent-fraction")
    p.add_argument("--normal-directions", type=int, default=16)
    p.add_argument("--seed", type=int, default=20260613)
    p.add_argument("--sim-time-step", type=float, default=0.25)
    p.add_argument("--support-space", choices=["H_gauge_27"], default="H_gauge_27")
    p.add_argument("--skip-support-profile", action="store_true")
    p.add_argument("--support-reference-samples", type=int, default=2000)
    p.add_argument("--max-workers", type=int, default=1)
    p.add_argument("--print-every", type=int, default=25)
    p.add_argument("--progress-every-sec", type=float, default=60.0)
    p.add_argument("--log-file", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    targets = load_targets(resolve(args.target_manifest), args)
    radii = parse_float_list(args.radius_values)
    raw = load_raw(resolve(args.raw), mmap=args.dry_run or args.skip_support_profile)
    times = bridge.load_times(raw, args.sim_time_step)
    support_ctx = None if args.skip_support_profile else build_support_context(raw, args.support_reference_samples)
    jobs, curves_meta = build_jobs(targets, args, radii, times, support_ctx)

    out = make_dirs(resolve(args.out_root))
    log_path = resolve(args.log_file) if args.log_file is not None else out["logs"] / "normal_robustness_progress.log"
    write_plan(out["json"] / "normal_robustness_run_plan.json", args, targets, radii, jobs)

    print(f"targets: {len(targets)}")
    print(f"jobs: {len(jobs)}")
    print(f"radius_mode: {args.radius_mode}")
    print(f"support_profile: {support_ctx is not None}")
    print(f"out_root: {out['root']}")
    if args.dry_run:
        return 0

    bridge.check_simulator_dependency()
    detail, curves = execute_jobs(jobs, curves_meta, args.max_workers, args.print_every, args.progress_every_sec, log_path)
    detail_path = out["csv"] / "normal_robustness_detail.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summarize(detail).to_csv(out["csv"] / "normal_robustness_summary.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out["npz"] / "normal_robustness_curves.npz", **curves)
    write_report(out["reports"] / "normal_robustness_report.md", args, targets, detail, started, log_path)

    print(f"detail: {detail_path}")
    print(f"summary: {out['csv'] / 'normal_robustness_summary.csv'}")
    print(f"npz: {out['npz'] / 'normal_robustness_curves.npz'}")
    print(f"report: {out['reports'] / 'normal_robustness_report.md'}")
    print(f"log: {log_path}")
    return 0


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def make_dirs(root: Path) -> dict[str, Path]:
    dirs = {
        "root": root,
        "csv": root / "csv",
        "npz": root / "npz",
        "json": root / "json",
        "reports": root / "reports",
        "logs": root / "logs",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def parse_float_list(text: str) -> list[float]:
    values = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    if not values:
        raise ValueError("empty float list")
    return values


def parse_int_set(text: str) -> set[int]:
    if not str(text).strip():
        return set()
    return {int(x.strip()) for x in str(text).split(",") if x.strip()}


def read_pair_allowlist(path: Path | None) -> set[str]:
    if path is None:
        return set()
    text = resolve(path).read_text(encoding="utf-8").strip()
    return {x.strip() for x in text.replace("\n", ",").split(",") if x.strip()}


def load_targets(path: Path, args: argparse.Namespace) -> pd.DataFrame:
    targets = pd.read_csv(path)
    priorities = parse_int_set(args.target_priorities)
    if priorities:
        targets = targets[targets["target_priority"].astype(int).isin(priorities)].copy()
    pair_allow = read_pair_allowlist(args.pairs_file)
    if pair_allow:
        targets = targets[targets["pair_id"].astype(str).isin(pair_allow)].copy()
    targets = targets.sort_values(["target_priority", "target_category", "pair_id"]).reset_index(drop=True)
    if args.max_targets > 0:
        targets = targets.head(args.max_targets).copy()
    if targets.empty:
        raise ValueError("No validation targets selected.")
    required = {"pair_id", "existing_curves_npz", "existing_curves_pair_index", "alpha_subset", "lambda_subset"}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"target manifest missing columns: {sorted(missing)}")
    return targets


def load_raw(path: Path, *, mmap: bool) -> dict[str, Any]:
    npz = np.load(path, allow_pickle=True, mmap_mode="r" if mmap else None)
    return {key: npz[key] for key in npz.files}


def build_support_context(raw: dict[str, Any], reference_samples: int) -> dict[str, Any]:
    h_cloud = bridge.support_feature_matrix(raw, "H_gauge_27")
    z_cloud, center, scale = bridge.robust_z_fit(h_cloud)
    ref = bridge.support_reference_quantiles(z_cloud, 25, reference_samples)
    return {
        "center": center,
        "scale": scale,
        "nn": bridge.NeighborIndex(z_cloud),
        "query_k": min(max(25 + 4, 8), len(z_cloud)),
        "ref": ref,
    }


def build_jobs(
    targets: pd.DataFrame,
    args: argparse.Namespace,
    radii: list[float],
    times: np.ndarray,
    support_ctx: dict[str, Any] | None,
) -> tuple[list[RobustnessJob], dict[str, Any]]:
    rng = np.random.default_rng(int(args.seed))
    jobs: list[RobustnessJob] = []
    pair_ids: list[str] = []
    h_rows: list[np.ndarray] = []
    job_i = 0
    for target_i, target in targets.iterrows():
        pair_id = str(target["pair_id"])
        npz_path = Path(str(target["existing_curves_npz"]))
        if not npz_path.exists():
            raise FileNotFoundError(f"Missing NPZ for {pair_id}: {npz_path}")
        z = np.load(npz_path, allow_pickle=True, mmap_mode="r")
        pair_index = int(target["existing_curves_pair_index"])
        alphas_available = np.asarray(z["alpha_values"], dtype=float)
        lambdas_available = np.asarray(z["lambda_values"], dtype=float)
        h_bridge = np.asarray(z["H_bridge"][pair_index], dtype=float)
        alphas = choose_values(args.alpha_values, target.get("alpha_subset", ""), alphas_available)
        lambdas = choose_values(args.lambda_values, target.get("lambda_subset", ""), lambdas_available)
        h_vecs = np.stack([matrix_to_h27(h) for h in h_bridge], axis=0)
        tangent = h_vecs[-1] - h_vecs[0]
        tangent_norm = float(np.linalg.norm(tangent))
        normal_dirs = build_normal_directions(rng, tangent, int(args.normal_directions))
        for alpha in alphas:
            alpha_i = nearest_index(alphas_available, alpha)
            h_path = h_bridge[alpha_i]
            h27 = h_vecs[alpha_i]
            for radius_i, radius in enumerate(radii):
                if abs(float(radius)) < 1e-12:
                    directions = [(0, np.zeros_like(h27), 0.0)]
                else:
                    directions = [(i, v, dot_with_tangent(v, tangent)) for i, v in enumerate(normal_dirs)]
                for direction_i, direction, dot in directions:
                    perturb_norm = radius_to_norm(float(radius), args.radius_mode, tangent_norm)
                    h_pert = h27_to_matrix(h27 + perturb_norm * direction)
                    nearest_dist, support_level = support_for_h(h_pert, support_ctx)
                    for lam in lambdas:
                        lambda_i = nearest_index(lambdas_available, lam)
                        job_i += 1
                        h_rows.append(h_pert.astype(np.float32))
                        jobs.append(
                            RobustnessJob(
                                job_order=job_i,
                                target_i=int(target_i),
                                alpha_i=int(alpha_i),
                                lambda_i=int(lambda_i),
                                radius_i=int(radius_i),
                                direction_i=int(direction_i),
                                pair_id=pair_id,
                                target_priority=int(target.get("target_priority", -1)),
                                target_category=str(target.get("target_category", "")),
                                group_name=str(target.get("group_name", "")),
                                feature_set=str(target.get("feature_set", "")),
                                alpha=float(alphas_available[alpha_i]),
                                alpha_source_index=int(alpha_i),
                                lambda_reorg=float(lambdas_available[lambda_i]),
                                radius=float(radius),
                                radius_mode=str(args.radius_mode),
                                perturb_norm_h_gauge_27=float(perturb_norm),
                                tangent_norm_h_gauge_27=float(tangent_norm),
                                normal_dot_tangent=float(dot),
                                h_path=h_path.astype(np.float64),
                                h_perturbed=h_pert.astype(np.float64),
                                times=times.astype(np.float64),
                                nearest_support_distance=float(nearest_dist),
                                support_level=str(support_level),
                            )
                        )
                        pair_ids.append(pair_id)
    curves_meta = {
        "times": times.astype(np.float32),
        "job_pair_id": np.asarray(pair_ids, dtype="U64"),
        "H_perturbed_by_unique_geometry_order": np.asarray(h_rows, dtype=np.float32),
    }
    return jobs, curves_meta


def choose_values(override: str, manifest_text: Any, available: np.ndarray) -> list[float]:
    requested = parse_float_list(override) if str(override).strip() else parse_float_list(str(manifest_text))
    return [float(available[nearest_index(available, x)]) for x in requested]


def nearest_index(values: np.ndarray, target: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=float) - float(target))))


def matrix_to_h27(h: np.ndarray) -> np.ndarray:
    return np.asarray(sampling.gauge_fix_encode(sampling.h_to_params(np.asarray(h, dtype=float))), dtype=float)


def h27_to_matrix(h27: np.ndarray) -> np.ndarray:
    return np.asarray(sampling.h27_to_matrix(np.asarray(h27, dtype=float)), dtype=float)


def build_normal_directions(rng: np.random.Generator, tangent: np.ndarray, n: int) -> list[np.ndarray]:
    n = max(1, int(n))
    tangent = np.asarray(tangent, dtype=float)
    tangent_norm2 = float(np.dot(tangent, tangent))
    out: list[np.ndarray] = []
    tries = 0
    while len(out) < n and tries < n * 100:
        tries += 1
        v = rng.normal(size=tangent.shape)
        if tangent_norm2 > 1e-20:
            v = v - tangent * (float(np.dot(v, tangent)) / tangent_norm2)
        norm = float(np.linalg.norm(v))
        if norm > 1e-12:
            out.append(v / norm)
    if len(out) < n:
        raise RuntimeError("Could not generate enough normal directions.")
    return out


def dot_with_tangent(v: np.ndarray, tangent: np.ndarray) -> float:
    norm = float(np.linalg.norm(tangent))
    if norm <= 1e-12:
        return 0.0
    return float(np.dot(v, tangent / norm))


def radius_to_norm(radius: float, mode: str, tangent_norm: float) -> float:
    if mode == "absolute":
        return float(radius)
    if mode == "tangent-fraction":
        return float(radius) * float(tangent_norm)
    raise ValueError(f"unsupported radius mode: {mode}")


def support_for_h(h: np.ndarray, support_ctx: dict[str, Any] | None) -> tuple[float, str]:
    if support_ctx is None:
        return math.nan, ""
    h27 = matrix_to_h27(h)
    z = (h27 - support_ctx["center"]) / support_ctx["scale"]
    dists, _ = support_ctx["nn"].query(z[None, :], support_ctx["query_k"])
    nearest = float(dists[0, 0])
    q50 = float(support_ctx["ref"].get("q50", math.nan))
    q90 = float(support_ctx["ref"].get("q90", math.nan))
    q95 = float(support_ctx["ref"].get("q95", math.nan))
    if math.isfinite(q50) and nearest <= q50:
        return nearest, "high"
    if math.isfinite(q90) and nearest <= q90:
        return nearest, "medium"
    if math.isfinite(q95) and nearest <= q95:
        return nearest, "low"
    return nearest, "sparse"


def execute_jobs(
    jobs: list[RobustnessJob],
    curves_meta: dict[str, Any],
    max_workers: int,
    print_every: int,
    progress_every_sec: float,
    log_path: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    total = len(jobs)
    times = curves_meta["times"]
    eta = np.full((total, len(times)), np.nan, dtype=np.float32)
    cl1 = np.full_like(eta, np.nan)
    purity = np.full_like(eta, np.nan)
    ipr = np.full_like(eta, np.nan)
    path = np.full((total, len(times), len(PATH_PANELS)), np.nan, dtype=np.float32)
    logger = ProgressLogger(log_path)
    logger.log(f"starting normal robustness sweep: jobs={total}, max_workers={max_workers}")
    rows: list[dict[str, Any]] = []
    ok = 0
    started = time.perf_counter()
    last_log = started

    def handle(result: dict[str, Any], completed: int) -> None:
        nonlocal ok, last_log
        rows.append(result["rec"])
        j = int(result["job_order"]) - 1
        if bool(result["rec"]["solver_success"]):
            ok += 1
            eta[j] = result["eta_t"]
            cl1[j] = result["cl1_t"]
            purity[j] = result["purity_t"]
            ipr[j] = result["ipr_t"]
            path[j] = result["path_t"]
        now = time.perf_counter()
        due_count = print_every > 0 and (completed % print_every == 0 or completed == total)
        due_time = progress_every_sec > 0 and now - last_log >= progress_every_sec
        if due_count or due_time:
            elapsed = now - started
            rate = completed / max(elapsed, 1e-9)
            remain = (total - completed) / rate if rate > 0 else math.nan
            rec = result["rec"]
            logger.log(
                f"progress {completed}/{total}; success={ok}; failed={completed - ok}; "
                f"elapsed={elapsed:.1f}s; eta={format_duration(remain)}; "
                f"latest={rec['pair_id']} alpha={rec['alpha']:.3f} lambda={rec['lambda_reorg']:g} radius={rec['radius']:g}"
            )
            if not bool(rec["solver_success"]):
                logger.log(f"latest failure: {rec['solver_error']}")
            last_log = now

    max_workers = normalize_max_workers(max_workers)
    if max_workers <= 1:
        for completed, job in enumerate(jobs, start=1):
            handle(run_job(job), completed)
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_order = {executor.submit(run_job, job): job.job_order for job in jobs}
            for completed, future in enumerate(as_completed(future_to_order), start=1):
                order = future_to_order[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = failed_result(jobs[order - 1], exc)
                handle(result, completed)
    logger.log(f"finished normal robustness sweep: success={ok}/{total}, failures={total - ok}")
    detail = pd.DataFrame(rows).sort_values("job_order").drop(columns=["job_order"]).reset_index(drop=True)
    curves = dict(curves_meta)
    curves.update(
        {
            "eta_t": eta,
            "cl1_t": cl1,
            "purity_t": purity,
            "ipr_t": ipr,
            "path_t": path,
            "path_group_names": np.asarray(PATH_PANELS, dtype="U16"),
        }
    )
    return detail, curves


def run_job(job: RobustnessJob) -> dict[str, Any]:
    rec = base_record(job)
    result: dict[str, Any] = {"job_order": int(job.job_order), "rec": rec}
    try:
        out = simulator.simulate(job.h_perturbed, float(job.lambda_reorg), tlist=job.times, return_traj=True)
        _, rho_t = out["_traj"]
        pop = np.real(np.diagonal(rho_t, axis1=1, axis2=2)).astype(np.float32)
        eta_t = compute_eta_t(pop)
        path_t, path_names = compute_path_groups(pop)
        cl1_t, purity_t, ipr_t = bridge.quantum_timeseries(rho_t)
        rec.update(bridge.metric_dict(job.times, eta_t, path_t, path_names, cl1_t, purity_t, ipr_t))
        rec["solver_success"] = True
        result["eta_t"] = eta_t.astype(np.float32)
        result["cl1_t"] = cl1_t.astype(np.float32)
        result["purity_t"] = purity_t.astype(np.float32)
        result["ipr_t"] = ipr_t.astype(np.float32)
        result["path_t"] = bridge.reorder_path(path_t, path_names).astype(np.float32)
    except Exception as exc:
        rec["solver_error"] = repr(exc)
    return result


def base_record(job: RobustnessJob) -> dict[str, Any]:
    return {
        "job_order": int(job.job_order),
        "pair_id": job.pair_id,
        "target_priority": int(job.target_priority),
        "target_category": job.target_category,
        "group_name": job.group_name,
        "feature_set": job.feature_set,
        "alpha": float(job.alpha),
        "alpha_source_index": int(job.alpha_source_index),
        "lambda_reorg": float(job.lambda_reorg),
        "radius": float(job.radius),
        "radius_mode": job.radius_mode,
        "normal_direction_i": int(job.direction_i),
        "perturb_norm_h_gauge_27": float(job.perturb_norm_h_gauge_27),
        "tangent_norm_h_gauge_27": float(job.tangent_norm_h_gauge_27),
        "normal_dot_tangent": float(job.normal_dot_tangent),
        "H_fro_perturb_to_path": float(np.linalg.norm(job.h_perturbed - job.h_path)),
        "nearest_support_distance": float(job.nearest_support_distance),
        "support_level": job.support_level,
        "solver_success": False,
        "solver_error": "",
    }


def failed_result(job: RobustnessJob, exc: Exception) -> dict[str, Any]:
    rec = base_record(job)
    rec["solver_error"] = f"worker_failure: {repr(exc)}"
    return {"job_order": int(job.job_order), "rec": rec}


def summarize(detail: pd.DataFrame) -> pd.DataFrame:
    success = detail[detail["solver_success"].astype(bool)].copy()
    if success.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["pair_id", "target_priority", "target_category", "group_name", "alpha", "lambda_reorg", "radius"]
    for keys, group in success.groupby(group_cols, sort=False):
        rec = dict(zip(group_cols, keys))
        eta = group["eta20"].to_numpy(float)
        rec.update(
            {
                "n_directions": int(group["normal_direction_i"].nunique()),
                "eta20_mean": finite(np.nanmean(eta)),
                "eta20_std": finite(np.nanstd(eta)),
                "eta20_min": finite(np.nanmin(eta)),
                "eta20_max": finite(np.nanmax(eta)),
                "eta20_range": finite(np.nanmax(eta) - np.nanmin(eta)),
                "t80_median": nanmedian_or_nan(group["t80"].to_numpy(float)),
                "cl1_mean_0_5ps_mean": finite(np.nanmean(group["cl1_mean_0_5ps"].to_numpy(float))),
                "nearest_support_distance_median": nanmedian_or_nan(group["nearest_support_distance"].to_numpy(float)),
                "support_levels": ",".join(sorted({str(x) for x in group["support_level"].dropna().unique() if str(x)})),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def finite(value: Any) -> float:
    try:
        x = float(value)
    except Exception:
        return math.nan
    return x if math.isfinite(x) else math.nan


def nanmedian_or_nan(values: Any) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return math.nan
    return finite(np.median(arr))


def normalize_max_workers(value: int) -> int:
    if value < 1:
        raise ValueError("--max-workers must be >= 1")
    return min(int(value), int(os.cpu_count() or 1))


def format_duration(seconds: float) -> str:
    if not math.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    if minutes:
        return f"{minutes}m{sec:02d}s"
    return f"{sec}s"


def markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    text = df.copy()
    for col in text.columns:
        text[col] = text[col].map(lambda x: "" if pd.isna(x) else str(x))
    cols = list(text.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in text.iterrows():
        lines.append("| " + " | ".join(str(row[col]).replace("|", "\\|") for col in cols) + " |")
    return "\n".join(lines)


def write_plan(path: Path, args: argparse.Namespace, targets: pd.DataFrame, radii: list[float], jobs: list[RobustnessJob]) -> None:
    payload = {
        "experiment": "bridge_normal_robustness_sfiltered",
        "target_manifest": str(resolve(args.target_manifest)),
        "out_root": str(resolve(args.out_root)),
        "target_count": int(len(targets)),
        "job_count": int(len(jobs)),
        "radius_values": [float(x) for x in radii],
        "radius_mode": str(args.radius_mode),
        "normal_directions": int(args.normal_directions),
        "seed": int(args.seed),
        "support_profile": not bool(args.skip_support_profile),
        "max_workers": int(normalize_max_workers(args.max_workers)),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_report(path: Path, args: argparse.Namespace, targets: pd.DataFrame, detail: pd.DataFrame, started: float, log_path: Path) -> None:
    elapsed = time.perf_counter() - started
    summary = summarize(detail)
    lines = [
        "# Normal-Vector Robustness Report",
        "",
        "This Priority D run tests whether selected bridge observations survive perturbations orthogonal to the straight H(alpha) path.",
        "",
        "## Inputs",
        "",
        f"- target manifest: `{args.target_manifest}`",
        f"- targets: `{len(targets)}`",
        f"- radius mode: `{args.radius_mode}`",
        f"- radius values: `{args.radius_values}`",
        f"- normal directions: `{args.normal_directions}`",
        f"- support profile: `{not args.skip_support_profile}`",
        f"- max workers: `{normalize_max_workers(args.max_workers)}`",
        f"- elapsed seconds: `{elapsed:.2f}`",
        f"- log: `{log_path}`",
        "",
        "## Success",
        "",
        f"- simulations requested: `{len(detail)}`",
        f"- successful simulations: `{int(detail['solver_success'].astype(bool).sum()) if not detail.empty else 0}`",
        "",
        "## Summary Preview",
        "",
        markdown_table(summary.head(40).round(6) if not summary.empty else summary),
        "",
        "## Interpretation Guardrails",
        "",
        "- A robust eta valley under normal perturbation supports a valley-vulnerability reading, not by itself a boundary/bottleneck claim.",
        "- If eta stays high over radii, treat it as functional stability evidence; inspect H/eigenstate features before naming a mechanism.",
        "- Lambda-specific differences are dynamics responses on the same H(alpha); H eigenfeatures do not change with lambda.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
