#!/usr/bin/env python3
"""Validate H27 CFAST generated Hamiltonians with the forward simulator.

This script is for generated samples written by
`notebooks/h27_cfast_condition_flow_colab.ipynb`, whose NPZ keys look like:

    CFAST_CL1_fast_high_H_vec28_trace_zero

It reruns the Bloch-Redfield + trap/loss simulator and recomputes eta10,
eta20, eta50, t80, and related diagnostics. Generated H is not an inverse
design success until this simulator validation passes.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from fmo_hamiltonian import simulator
from fmo_hamiltonian.constants import DEFAULT_LAMBDA_REORG, N_SITE
from fmo_hamiltonian.trajectory_features import compute_arrival_times, compute_eta_t, compute_path_groups


TARGETS = ("fast_high", "very_fast", "late_high", "non_high")
QUERY_TIMES = np.array([5.0, 10.0, 20.0, 50.0], dtype=float)
PATH_QUERY_TIMES = np.array([6.0, 10.0, 20.0, 50.0], dtype=float)
IU = np.triu_indices(N_SITE)
OFF_IDX = np.where(IU[0] != IU[1])[0]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--generated",
        type=Path,
        default=Path("outputs/h27_cfast_condition_flow_colab_20260618/full/CFAST_CL1_generated_samples.npz"),
        help="Generated samples NPZ from the Colab flow notebook.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/h27_cfast_condition_flow_colab_20260618/full/simulator_validation"),
    )
    p.add_argument("--condition-set", default="CFAST_CL1", help="Condition-set prefix to validate, or ALL.")
    p.add_argument("--targets", default=",".join(TARGETS), help="Comma-separated target names, or ALL for every target variant in the NPZ.")
    p.add_argument("--n-per-target", type=int, default=32, help="Use -1 to validate all generated rows.")
    p.add_argument("--selection", choices=["random", "head"], default="random")
    p.add_argument("--seed", type=int, default=20260618)
    p.add_argument("--lambda-reorg", type=float, default=DEFAULT_LAMBDA_REORG)
    p.add_argument("--t-max", type=float, default=50.0)
    p.add_argument("--dt", type=float, default=0.25)
    p.add_argument("--save-trajectories", action="store_true")
    p.add_argument("--print-every", type=int, default=10)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "csv").mkdir(exist_ok=True)
    (args.out_dir / "reports").mkdir(exist_ok=True)
    (args.out_dir / "npz").mkdir(exist_ok=True)

    if not args.generated.exists():
        raise FileNotFoundError(f"missing generated NPZ: {args.generated}")

    start = time.perf_counter()
    generated = np.load(args.generated, allow_pickle=True)
    rows, traj_payload = validate_generated(generated, args)
    detail = pd.DataFrame(rows)
    summary = summarize(detail)

    stem = args.generated.stem
    detail_path = args.out_dir / "csv" / f"{stem}_simulator_validation_detail.csv"
    summary_path = args.out_dir / "csv" / f"{stem}_simulator_validation_summary.csv"
    report_path = args.out_dir / "reports" / f"{stem}_simulator_validation_report_kr.md"
    manifest_path = args.out_dir / "validation_manifest.json"
    trajectory_path = args.out_dir / "npz" / f"{stem}_sampled_trajectories.npz"

    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    write_report(report_path, args, detail, summary, elapsed=time.perf_counter() - start)
    n_trajectory_rows = int(sum(1 for key in traj_payload if key.endswith("_eta_t")))
    trajectory_saved = False
    if args.save_trajectories:
        if traj_payload:
            np.savez_compressed(trajectory_path, **traj_payload)
            trajectory_saved = True
        else:
            np.savez_compressed(
                trajectory_path,
                __empty__=np.array([], dtype=np.float32),
                __reason__=np.array(["no successful simulator trajectories"], dtype=object),
            )
            print(
                "warning: --save-trajectories was requested, but no simulator runs succeeded; "
                "wrote an empty trajectory NPZ so downstream checks can fail explicitly.",
                flush=True,
            )
    manifest = {
        "generated": str(args.generated),
        "out_dir": str(args.out_dir),
        "detail_csv": str(detail_path),
        "summary_csv": str(summary_path),
        "report": str(report_path),
        "trajectory_npz": str(trajectory_path) if args.save_trajectories else None,
        "trajectory_saved": bool(trajectory_saved),
        "n_trajectory_rows": n_trajectory_rows,
        "n_rows": int(len(detail)),
        "n_success": int(detail["simulation_success"].sum()) if "simulation_success" in detail else 0,
        "args": vars(args),
    }
    manifest_path.write_text(json.dumps(clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"detail: {detail_path}")
    print(f"summary: {summary_path}")
    print(f"report: {report_path}")
    if args.save_trajectories:
        print(f"trajectories: {trajectory_path} rows={n_trajectory_rows} saved={trajectory_saved}")
    print(f"rows={len(detail)} success={manifest['n_success']}")
    return 0


def validate_generated(generated: np.lib.npyio.NpzFile, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    rng = np.random.default_rng(args.seed)
    wanted_targets = None if args.targets.strip().upper() == "ALL" else [x.strip() for x in args.targets.split(",") if x.strip()]
    wanted_condition = None if args.condition_set.upper() == "ALL" else args.condition_set
    jobs = list(select_jobs(generated, wanted_condition, wanted_targets, args.n_per_target, args.selection, rng))
    if not jobs:
        keys = list(generated.files)
        preview = ", ".join(keys[:20])
        raise RuntimeError(
            "no generated samples matched validation request; "
            f"condition_set={args.condition_set!r}, targets={args.targets!r}, "
            f"n_keys={len(keys)}, keys_preview=[{preview}]"
        )
    tlist = np.arange(0.0, args.t_max + 0.5 * args.dt, args.dt, dtype=float)
    rows: list[dict[str, Any]] = []
    traj_payload: dict[str, np.ndarray] = {}

    for job_i, job in enumerate(jobs, start=1):
        key, condition_set, target, source_index, vec28 = job
        row: dict[str, Any] = {
            "generated_key": key,
            "condition_set": condition_set,
            "target": target,
            "source_generated_index": int(source_index),
            "validation_order": int(job_i - 1),
            "lambda_reorg": float(args.lambda_reorg),
        }
        t0 = time.perf_counter()
        try:
            mat = vec28_to_hmat(np.asarray(vec28, dtype=np.float64)[None, :])[0]
            row.update(physical_checks(mat, vec28))
            sim = simulator.simulate(mat, args.lambda_reorg, tlist=tlist, return_traj=True)
            _, rho_t = sim["_traj"]
            pop = np.real(np.diagonal(rho_t, axis1=1, axis2=2))
            eta_t = compute_eta_t(pop)
            eta_q = interp_1d(eta_t, tlist, QUERY_TIMES)
            arrivals = compute_arrival_times(eta_t, tlist)
            path_t, path_names = compute_path_groups(pop)
            path_q = interp_path_time_group(path_t, tlist, PATH_QUERY_TIMES)

            row.update(
                {
                    "simulation_success": True,
                    "eta5": finite_float(eta_q[0]),
                    "eta10": finite_float(eta_q[1]),
                    "eta20": finite_float(eta_q[2]),
                    "eta50": finite_float(eta_q[3]),
                    "eta_final": finite_float(sim.get("eta")),
                    "tau_transfer": finite_float(sim.get("tau_transfer")),
                    "t25": finite_float(arrivals.get("t25")),
                    "t50": finite_float(arrivals.get("t50")),
                    "t80": finite_float(arrivals.get("t80")),
                    "t90": finite_float(arrivals.get("t90")),
                    "ipr": finite_float(sim.get("ipr")),
                    "purity": finite_float(sim.get("purity")),
                    "c_l1": finite_float(sim.get("c_l1")),
                }
            )
            name_to_idx = {name: i for i, name in enumerate(path_names)}
            for t_idx, t in enumerate(PATH_QUERY_TIMES):
                suffix = format_time(t)
                for name in ("site1", "site2", "sink34", "detour567", "trap", "loss", "residual"):
                    row[f"{name}_at_{suffix}ps"] = finite_float(path_q[t_idx, name_to_idx[name]])
            row.update(success_flags(row, target))

            if args.save_trajectories:
                label = f"{condition_set}_{target}_{int(source_index):04d}"
                traj_payload[f"{label}_times"] = tlist.astype(np.float32)
                traj_payload[f"{label}_eta_t"] = np.asarray(eta_t, dtype=np.float32)
                traj_payload[f"{label}_path_t"] = np.asarray(path_t, dtype=np.float32)
                traj_payload[f"{label}_pop_t"] = np.asarray(pop, dtype=np.float32)
        except Exception as exc:
            row.update({"simulation_success": False, "error": repr(exc)})
        row["runtime_sec"] = float(time.perf_counter() - t0)
        rows.append(row)
        if args.print_every > 0 and (job_i % args.print_every == 0 or job_i == len(jobs)):
            ok = sum(bool(r.get("simulation_success")) for r in rows)
            print(f"validated {job_i}/{len(jobs)} success={ok} latest={condition_set}/{target}", flush=True)
    return rows, traj_payload


def select_jobs(
    generated: np.lib.npyio.NpzFile,
    condition_set: str | None,
    targets: list[str] | None,
    n_per_target: int,
    selection: str,
    rng: np.random.Generator,
):
    for key in sorted(generated.files):
        parsed = parse_generated_key(key, condition_set)
        if parsed is None:
            continue
        cond, target = parsed
        if condition_set is not None and cond != condition_set:
            continue
        if targets is not None and target not in targets:
            continue
        arr = np.asarray(generated[key], dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != 28:
            raise ValueError(f"{key} must have shape [N, 28], got {arr.shape}")
        n = len(arr) if n_per_target < 0 else min(int(n_per_target), len(arr))
        if n_per_target >= 0 and int(n_per_target) > len(arr):
            print(f"warning: {key} has only {len(arr)} generated rows; requested {n_per_target}, validating {n}", flush=True)
        if selection == "head":
            indices = np.arange(n, dtype=int)
        else:
            indices = np.sort(rng.choice(len(arr), size=n, replace=False))
        for idx in indices:
            yield key, cond, target, int(idx), arr[idx]


def parse_generated_key(key: str, condition_set: str | None = None) -> tuple[str, str] | None:
    suffix = "_H_vec28_trace_zero"
    if not key.endswith(suffix):
        return None
    prefix = key[: -len(suffix)]
    if condition_set is not None:
        head = f"{condition_set}_"
        if prefix.startswith(head):
            return condition_set, prefix[len(head) :]
    for target in sorted(TARGETS, key=len, reverse=True):
        tail = f"_{target}"
        if prefix.endswith(tail):
            return prefix[: -len(tail)], target
    return None


def physical_checks(mat: np.ndarray, vec28: np.ndarray) -> dict[str, float]:
    eigs = np.linalg.eigvalsh(mat)
    diag = np.diag(mat)
    off = np.asarray(vec28, dtype=float)[OFF_IDX]
    return {
        "trace_abs": float(abs(np.trace(mat))),
        "symmetry_abs_max": float(np.max(np.abs(mat - mat.T))),
        "spectral_width": float(eigs[-1] - eigs[0]),
        "min_eigen_gap": float(np.min(np.diff(eigs))) if len(eigs) > 1 else math.nan,
        "offdiag_diag_norm_ratio": float(np.linalg.norm(off) / (np.linalg.norm(diag) + 1e-8)),
    }


def vec28_to_hmat(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec)
    if arr.shape[-1] != 28:
        raise ValueError(f"expected trailing dimension 28, got {arr.shape}")
    h = np.zeros(arr.shape[:-1] + (N_SITE, N_SITE), dtype=arr.dtype)
    h[..., IU[0], IU[1]] = arr
    h = h + np.swapaxes(h, -1, -2)
    diag = np.diagonal(h, axis1=-2, axis2=-1).copy()
    h[..., np.arange(N_SITE), np.arange(N_SITE)] = diag / 2.0
    return h


def interp_1d(values: np.ndarray, times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    return np.interp(query_times, times, np.asarray(values, dtype=float)).astype(float)


def interp_path_time_group(path_t: np.ndarray, times: np.ndarray, query_times: np.ndarray) -> np.ndarray:
    path = np.asarray(path_t, dtype=float)
    if path.ndim != 2 or path.shape[0] != len(times):
        raise ValueError(f"path_t must have shape [time, group], got {path.shape}")
    return np.stack([np.interp(query_times, times, path[:, j]) for j in range(path.shape[1])], axis=-1)


def success_flags(row: dict[str, Any], target: str) -> dict[str, Any]:
    target_base = base_target(target)
    eta10 = as_float(row.get("eta10"))
    eta20 = as_float(row.get("eta20"))
    eta50 = as_float(row.get("eta50"))
    t80 = as_float(row.get("t80"))
    tau = as_float(row.get("tau_transfer"))
    flags = {
        "meets_eta50_ge_0p90": finite_ge(eta50, 0.90),
        "meets_eta20_ge_0p80": finite_ge(eta20, 0.80),
        "meets_eta20_ge_0p90": finite_ge(eta20, 0.90),
        "meets_eta10_ge_0p70": finite_ge(eta10, 0.70),
        "meets_t80_le_10": finite_le(t80, 10.0),
        "meets_t80_le_15": finite_le(t80, 15.0),
        "meets_tau_le_8": finite_le(tau, 8.0),
        "meets_tau_le_10": finite_le(tau, 10.0),
        "fast_like_eta20_t80": finite_ge(eta20, 0.80) and finite_le(t80, 15.0),
        "very_fast_like_eta20_t80": finite_ge(eta20, 0.90) and finite_le(t80, 10.0),
    }
    if target_base == "fast_high":
        match = finite_ge(eta50, 0.90) and finite_le(t80, 15.0)
    elif target_base == "very_fast":
        match = finite_ge(eta50, 0.90) and finite_le(t80, 10.0)
    elif target_base == "late_high":
        match = finite_ge(eta50, 0.90) and (eta20 is not None and eta20 < 0.90)
    elif target_base == "non_high":
        match = eta50 is not None and eta50 < 0.90
    else:
        match = False
    flags["target_match"] = bool(match)
    return flags


def base_target(target: str) -> str:
    text = str(target)
    for base in sorted(TARGETS, key=len, reverse=True):
        if text == base or text.startswith(f"{base}_"):
            return base
    return text


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (condition_set, target), part in df.groupby(["condition_set", "target"], dropna=False):
        ok = part[part["simulation_success"].astype(bool)].copy()
        row: dict[str, Any] = {
            "condition_set": condition_set,
            "target": target,
            "n_total": int(len(part)),
            "n_success": int(len(ok)),
            "simulation_success_rate": float(len(ok) / max(1, len(part))),
        }
        for col in (
            "eta5",
            "eta10",
            "eta20",
            "eta50",
            "eta_final",
            "tau_transfer",
            "t80",
            "t90",
            "c_l1",
            "sink34_at_6ps",
            "detour567_at_6ps",
            "loss_at_50ps",
            "trace_abs",
            "spectral_width",
            "min_eigen_gap",
            "offdiag_diag_norm_ratio",
        ):
            if col in ok:
                vals = pd.to_numeric(ok[col], errors="coerce").to_numpy(float)
                row[f"{col}_median"] = nan_stat(vals, np.nanmedian)
                row[f"{col}_mean"] = nan_stat(vals, np.nanmean)
        for col in (
            "target_match",
            "meets_eta50_ge_0p90",
            "meets_eta20_ge_0p80",
            "meets_eta20_ge_0p90",
            "meets_t80_le_10",
            "meets_t80_le_15",
            "fast_like_eta20_t80",
            "very_fast_like_eta20_t80",
        ):
            if col in ok:
                row[f"{col}_rate"] = float(pd.to_numeric(ok[col], errors="coerce").mean()) if len(ok) else math.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition_set", "target"]).reset_index(drop=True)


def write_report(path: Path, args: argparse.Namespace, detail: pd.DataFrame, summary: pd.DataFrame, elapsed: float) -> None:
    lines = [
        "# H27 CFAST Generated-H Simulator Validation",
        "",
        "## Purpose",
        "",
        "This validation reruns generated trace-zero 28D Hamiltonians through the forward simulator and computes eta10/eta20/eta50/t80.",
        "It checks whether generated candidates satisfy dynamic targets, separately from density-model likelihood.",
        "",
        "## Run settings",
        "",
        f"- generated: `{args.generated}`",
        f"- condition_set: `{args.condition_set}`",
        f"- targets: `{args.targets}`",
        f"- n_per_target: `{args.n_per_target}`",
        f"- selection: `{args.selection}`",
        f"- lambda_reorg: `{args.lambda_reorg}`",
        f"- t_max: `{args.t_max}` ps",
        f"- dt: `{args.dt}` ps",
        f"- elapsed: `{elapsed:.2f}` sec",
        "",
        "## Per-target summary",
        "",
        md_table(summary),
        "",
        "## Failure-error summary",
        "",
        error_summary_table(detail),
        "",
        "## Decision rules",
        "",
        "- `fast_high`: `eta50 >= 0.90` and `t80 <= 15 ps`",
        "- `very_fast`: `eta50 >= 0.90` and `t80 <= 10 ps`",
        "- `late_high`: `eta50 >= 0.90` and `eta20 < 0.90`",
        "- `non_high`: `eta50 < 0.90`",
        "",
        "## Interpretation notes",
        "",
        "- A high `target_match_rate` means generated samples are useful inverse-design candidates.",
        "- If the sample count is small, this should be treated as smoke validation; at least 100 samples per target are recommended for final claims.",
        "- Good density NLL does not guarantee simulator success; density fit and dynamic target control must be evaluated separately.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def error_summary_table(detail: pd.DataFrame, limit: int = 8) -> str:
    if "simulation_success" not in detail or "error" not in detail:
        return "_error column ?놁쓬_"
    failed = detail[~detail["simulation_success"].astype(bool)].copy()
    if failed.empty:
        return "_?ㅽ뙣 ?놁쓬_"
    counts = failed["error"].fillna("<missing error>").value_counts().head(limit).reset_index()
    counts.columns = ["error", "count"]
    return md_table(counts, digits=0)


def md_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "_rows ?놁쓬_"
    preferred = [
        "condition_set",
        "target",
        "n_success",
        "n_total",
        "simulation_success_rate",
        "target_match_rate",
        "eta10_median",
        "eta20_median",
        "eta50_median",
        "t80_median",
        "tau_transfer_median",
        "fast_like_eta20_t80_rate",
        "very_fast_like_eta20_t80_rate",
    ]
    cols = [c for c in preferred if c in df.columns]
    if not cols:
        cols = list(df.columns)
    view = df[cols].copy()
    try:
        return view.to_markdown(index=False, floatfmt=f".{digits}f")
    except ImportError:
        return simple_markdown_table(view, digits)


def simple_markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for col in cols:
            val = row[col]
            if isinstance(val, float):
                vals.append("" if not math.isfinite(val) else f"{val:.{digits}f}")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def nan_stat(vals: np.ndarray, fn) -> float:
    vals = np.asarray(vals, dtype=float)
    return float(fn(vals)) if np.isfinite(vals).any() else math.nan


def as_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    return out if math.isfinite(out) else None


def finite_float(value: Any) -> float:
    out = as_float(value)
    return out if out is not None else math.nan


def finite_ge(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def finite_le(value: float | None, threshold: float) -> bool:
    return value is not None and value <= threshold


def format_time(value: float) -> str:
    return f"{value:g}".replace(".", "p")


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

