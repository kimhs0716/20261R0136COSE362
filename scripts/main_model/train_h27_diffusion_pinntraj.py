#!/usr/bin/env python3
"""Train H27 conditional diffusion generators with optional PINN-trajectory signal.

This is the diffusion counterpart to the recent H27 RealNVP and flow-matching
experiments.  The user-facing condition is intentionally compact, e.g.
``CFAST_ORANGE3``.  Dynamic/path information is not exposed as a generation
condition.  Instead, the optional ``DDPM_PINNTRAJ`` ablation trains a frozen
H -> population-trajectory surrogate and uses the predicted intermediate path
as a differentiable signal during denoising.

Implemented ablations:

- DDPM: conditional diffusion denoising model with compact condition only.
- DDPM_PINNTRAJ: DDPM plus full population-trajectory surrogate guidance.
- DDPM_HTPINNTRAJ: DDPM_PINNTRAJ plus high-transfer target-prototype
  trajectory guidance and stronger sampling-time guidance.
- DDPM_HTBALPINNTRAJ: DDPM_PINNTRAJ plus internal dynamic-mode prototypes,
  target feature-band guidance, batch mode-balance loss, and weaker/later
  sampling guidance.  Dynamic modes are internal training/sampling latents,
  not user-facing conditions.

Generated NPZ keys match ``validate_h27_cfast_generated_simulator.py``.
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
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_h27_early_readout_flow as orange
import train_h27_path_dynamic_flow as base
import train_h27_dynz_pinntraj_flow as pinnmod


DEFAULT_OUT_ROOT = Path("outputs/experiments/20260621_h27_diffusion_pinntraj")
DEFAULT_METHODS = "DDPM,DDPM_PINNTRAJ,DDPM_HTPINNTRAJ,DDPM_HTBALPINNTRAJ"
DEFAULT_BASE_CONDITION = "CFAST_ORANGE3"
TARGET_GROUPS = ("fast_high", "very_fast", "late_high", "non_high")
HTBAL_METHOD = "DDPM_HTBALPINNTRAJ"
DEFAULT_HTBAL_REFERENCE_DIR = Path("FMO_H27_context_ablation/data/clustered_from_clean/dynamic_condition_modes_n1000")
HTBAL_REFERENCE_CONDITION = {
    "fast_high": "c_fast",
    "very_fast": "c_very_fast",
    "late_high": "c_late",
    "non_high": "c_nonhigh",
}
AUX_KEYS = (
    "weighted",
    "traj",
    "feature",
    "phys",
    "support",
    "target_band",
    "mode",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=base.DEFAULT_PREPARED)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-name", choices=["smoke", "full"], default="full")
    p.add_argument("--base-condition", choices=["CFAST_ONLY", "CFAST_ORANGE3", "CFAST_CL1_ORANGE3"], default=DEFAULT_BASE_CONDITION)
    p.add_argument(
        "--methods",
        default=DEFAULT_METHODS,
        help="Comma-separated: DDPM,DDPM_PINNTRAJ,DDPM_HTPINNTRAJ,DDPM_HTBALPINNTRAJ",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--time-features", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--n-generate", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260621)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--pinn-only", action="store_true", help="Train/load the trajectory surrogate and stop before DDPM training.")
    p.add_argument("--generate-only", action="store_true")
    p.add_argument("--checkpoint", type=Path, help="Checkpoint for --generate-only.")
    p.add_argument("--target-groups", default=",".join(TARGET_GROUPS))
    p.add_argument("--residual-mode", choices=["site2_to_site7", "all_sites", "one_minus_trap_loss"], default="site2_to_site7")

    # Required by base.build_context dynamic-submode metadata construction.
    p.add_argument("--dyn-k-fast", type=int, default=3)
    p.add_argument("--dyn-k-very-fast", type=int, default=3)
    p.add_argument("--dyn-k-late", type=int, default=4)
    p.add_argument("--dyn-k-nonhigh", type=int, default=6)
    p.add_argument("--dyn-k-other", type=int, default=1)
    p.add_argument("--kmeans-iter", type=int, default=80)
    p.add_argument("--kmeans-init", type=int, default=8)
    p.add_argument("--min-dyn-target-rows", type=int, default=80)
    p.add_argument("--max-dyn-targets-per-group", type=int, default=3)

    # DDPM/DDIM design.
    p.add_argument("--diffusion-steps", type=int, default=1000)
    p.add_argument("--sample-steps", type=int, default=100)
    p.add_argument("--beta-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--beta-start", type=float, default=1e-4)
    p.add_argument("--beta-end", type=float, default=2e-2)
    p.add_argument("--ddim-eta", type=float, default=0.0)
    p.add_argument("--x0-clip", type=float, default=8.0)
    p.add_argument("--generate-strategies", default="median,mixture", help="Comma-separated: median,mixture")

    # PINN-lite population trajectory surrogate.
    p.add_argument("--pinn-epochs", type=int, default=180)
    p.add_argument("--pinn-hidden", type=int, default=192)
    p.add_argument("--pinn-depth", type=int, default=3)
    p.add_argument("--pinn-dropout", type=float, default=0.10)
    p.add_argument("--pinn-lr", type=float, default=3e-4)
    p.add_argument("--pinn-patience", type=int, default=24)
    p.add_argument("--pinn-min-delta", type=float, default=1e-5)
    p.add_argument("--pinn-batch-size", type=int, default=512)
    p.add_argument("--pinn-eval-batch-size", type=int, default=8192)
    p.add_argument("--pinn-loss-pop", type=float, default=1.0)
    p.add_argument("--pinn-loss-feature", type=float, default=0.35)
    p.add_argument("--pinn-loss-dyn-ce", type=float, default=0.0)
    p.add_argument("--pinn-loss-monotonic", type=float, default=0.05)
    p.add_argument("--pinn-loss-smooth", type=float, default=0.01)
    p.add_argument("--pinn-score-mode", choices=["total", "traj_feature"], default="traj_feature")
    p.add_argument("--pinn-balanced-ce", action="store_true", default=True)
    p.add_argument("--no-pinn-balanced-ce", dest="pinn_balanced_ce", action="store_false")
    p.add_argument("--pinn-class-weight-power", type=float, default=0.5)
    p.add_argument("--pinn-focal-gamma", type=float, default=0.0)
    p.add_argument("--pinn-label-smoothing", type=float, default=0.05)

    # Diffusion-side surrogate signal.  These defaults avoid a label-like
    # dynamic CE and emphasize intermediate population paths.
    p.add_argument("--aux-start-epoch", type=int, default=10)
    p.add_argument("--aux-warmup-epochs", type=int, default=20)
    p.add_argument("--lambda-pinn-traj", type=float, default=0.05)
    p.add_argument("--lambda-pinn-feature", type=float, default=0.02)
    p.add_argument("--lambda-pinn-phys", type=float, default=0.005)
    p.add_argument("--lambda-support", type=float, default=0.001)
    p.add_argument("--lambda-target-traj", type=float, default=0.08)
    p.add_argument("--lambda-target-feature", type=float, default=0.04)
    p.add_argument("--target-guidance-groups", default="fast_high,very_fast,late_high")
    p.add_argument("--target-guidance-weight-high", type=float, default=2.0)
    p.add_argument("--target-guidance-weight-nonhigh", type=float, default=0.35)
    p.add_argument("--lambda-htbal-target-band", type=float, default=0.04)
    p.add_argument("--lambda-htbal-mode-proto", type=float, default=0.03)
    p.add_argument("--lambda-htbal-mode-assign", type=float, default=0.02)
    p.add_argument("--lambda-htbal-mode-balance", type=float, default=0.06)
    p.add_argument("--lambda-htbal-mode-floor", type=float, default=0.02)
    p.add_argument("--htbal-mode-source", choices=["reference", "train_dynamic"], default="reference")
    p.add_argument("--htbal-reference-dir", type=Path, default=DEFAULT_HTBAL_REFERENCE_DIR)
    p.add_argument("--htbal-feature-q-low", type=float, default=0.20)
    p.add_argument("--htbal-feature-q-high", type=float, default=0.80)
    p.add_argument("--htbal-prior-alpha", type=float, default=0.30, help="Blend reference mode prior with uniform prior.")
    p.add_argument("--htbal-prior-min", type=float, default=0.05, help="Minimum desired internal mode mass before renormalization.")
    p.add_argument("--htbal-assign-temp", type=float, default=0.75)
    p.add_argument("--support-clip", type=float, default=6.0)
    p.add_argument("--val-aux-weight", type=float, default=1.0)
    p.add_argument("--val-aux-max", type=int, default=4096)

    # Optional DDIM sampling-time guidance against target-family trajectory
    # prototypes.  Training-time surrogate loss is usually the first thing to
    # compare; this knob tests whether explicit diffusion guidance helps.
    p.add_argument("--sample-guidance-scale", type=float, default=0.03)
    p.add_argument("--sample-guidance-start-frac", type=float, default=0.70)
    p.add_argument("--sample-guidance-every", type=int, default=2)
    p.add_argument("--sample-guidance-ramp-power", type=float, default=1.0)
    p.add_argument("--ht-sample-guidance-scale", type=float, default=0.12)
    p.add_argument("--ht-sample-guidance-start-frac", type=float, default=0.45)
    p.add_argument("--htbal-sample-guidance-scale", type=float, default=0.06)
    p.add_argument("--htbal-sample-guidance-start-frac", type=float, default=0.60)

    # Useful for local smoke tests or fast Colab probes.
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    prepared = base.resolve_prepared(args.prepared)
    out_root = args.out_root
    run_dir = out_root / args.run_name
    for path in [out_root / "metadata", run_dir, run_dir / "checkpoints", run_dir / "figures", run_dir / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.load(prepared, allow_pickle=True)
    base.validate_prepared(raw)
    context = build_orange_context(raw, args)
    condition_data = orange.build_condition_datasets(context, args)
    if args.base_condition not in condition_data:
        raise KeyError(f"unknown base condition: {args.base_condition}")
    ds = condition_data[args.base_condition]
    target_pop = build_target_pop_prototypes(context, args)
    mode_guidance = build_htbal_mode_guidance(context, args)
    write_metadata(out_root, prepared, args, context, ds, target_pop, mode_guidance)

    print(f"prepared: {prepared}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"base_condition: {args.base_condition}", flush=True)
    print(f"methods: {args.methods}", flush=True)
    if args.metadata_only:
        print("metadata-only complete:", out_root / "metadata", flush=True)
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    device = base.choose_device(args.device, torch)
    print(f"device: {device}", flush=True)

    selected = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(selected) - {"DDPM", "DDPM_PINNTRAJ", "DDPM_HTPINNTRAJ", HTBAL_METHOD})
    if unknown:
        raise KeyError(f"unknown methods: {unknown}")

    needs_pinn = args.pinn_only or any(method.endswith("PINNTRAJ") for method in selected) or args.generate_only
    pinn = None
    if needs_pinn:
        if float(args.pinn_loss_dyn_ce) <= 0.0:
            print(
                "note: PINN dynamic-class CE is disabled; surrogate acc/top3 logs are unused classifier diagnostics. "
                "Judge the surrogate by pop/feature losses and split/feature metrics.",
                flush=True,
            )
        pinn = pinnmod.train_or_load_pinn_surrogate(context, args, run_dir, device, torch, TensorDataset, DataLoader)
        print("pinn trajectory surrogate ready", flush=True)
    if args.pinn_only:
        print("pinn-only complete:", run_dir / "pinntraj_surrogate_split_metrics.csv", flush=True)
        return 0

    if args.generate_only:
        if args.checkpoint is None:
            raise ValueError("--generate-only requires --checkpoint")
        state = load_checkpoint(args.checkpoint, device, torch)
        if list(state.get("condition_names", [])) != list(ds.names):
            raise RuntimeError(
                "checkpoint condition names do not match the selected --base-condition; "
                f"checkpoint={state.get('condition_names')} selected={ds.names}"
            )
        method = str(state["method"])
        condition_set = str(state["condition_set"])
        apply_checkpoint_schedule_args(args, state)
        schedule = make_diffusion_schedule(args, device, torch)
        model = build_model_from_state(state, args, device, torch, nn)
        generate_targets(condition_set, method, model, ds, context, target_pop, mode_guidance, pinn, args, run_dir, schedule, device, torch, F)
        return 0

    schedule = make_diffusion_schedule(args, device, torch)
    for method in selected:
        condition_set = f"{args.base_condition}_{method}"
        ready, missing = artifacts_ready(run_dir, condition_set)
        if ready and not args.force:
            print(f"skip existing artifacts: {condition_set}", flush=True)
            continue
        if missing and not args.force:
            print(f"build missing artifacts for {condition_set}: {[str(p) for p in missing]}", flush=True)
        train_one(
            method,
            condition_set,
            ds,
            context,
            target_pop,
            mode_guidance,
            pinn,
            args,
            run_dir,
            schedule,
            device,
            torch,
            nn,
            F,
            TensorDataset,
            DataLoader,
        )

    write_run_summary(run_dir, args, selected)
    print("done:", run_dir, flush=True)
    return 0


def build_orange_context(raw: np.lib.npyio.NpzFile, args: argparse.Namespace) -> dict[str, Any]:
    context = base.build_context(raw, args)
    orange_raw, orange_names, orange_meta = orange.build_orange_readout_features(raw, args.residual_mode)
    context["orange_raw"] = orange_raw
    context["orange_names"] = orange_names
    context["orange_meta"] = orange_meta
    return context


def build_target_pop_prototypes(context: dict[str, Any], args: argparse.Namespace) -> dict[str, np.ndarray]:
    train_idx = np.asarray(context["train_idx"], dtype=np.int64)
    priority = np.asarray(context["priority_group"]).astype(str)
    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for group in [x.strip() for x in args.target_groups.split(",") if x.strip()]:
        idx = train_idx[priority[train_idx] == group]
        if len(idx) == 0:
            continue
        proto = np.nanmedian(pop[idx], axis=0).astype(np.float32)
        proto = proto / np.maximum(proto.sum(axis=1, keepdims=True), 1e-8)
        out[group] = proto
        rows.append(
            {
                "target_group": group,
                "n_train": int(len(idx)),
                "eta10_proto": float(proto[nearest_time_index(context, 10.0), 7]),
                "eta20_proto": float(proto[nearest_time_index(context, 20.0), 7]),
                "eta50_proto": float(proto[nearest_time_index(context, 50.0), 7]),
            }
        )
    context["target_pop_prototype_summary"] = pd.DataFrame(rows)
    return out


def nearest_time_index(context: dict[str, Any], value: float) -> int:
    times = np.asarray(context["raw"]["times"], dtype=np.float32)
    return int(np.argmin(np.abs(times - float(value))))


def build_htbal_mode_guidance(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Build internal dynamic-mode prototypes for HTBAL.

    These prototypes are training/sampling aids only.  They are derived from
    row-aligned training trajectories and dynamic submodes, but are not exposed
    as user-facing generation conditions.
    """

    if str(getattr(args, "htbal_mode_source", "reference")) == "reference":
        try:
            return build_htbal_reference_mode_guidance(context, args)
        except Exception as exc:
            print(
                f"warning: failed to build HTBAL reference-mode guidance from {args.htbal_reference_dir}: {exc}. "
                "Falling back to train_dynamic submodes.",
                flush=True,
            )

    train_idx = np.asarray(context["train_idx"], dtype=np.int64)
    priority = np.asarray(context["priority_group"]).astype(str)
    dyn = np.asarray(context["dynamic_label"]).astype(str)
    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    times = np.asarray(context["raw"]["times"], dtype=np.float32)
    feat = pinnmod.numpy_pop_features(pop, times).astype(np.float32)

    target_groups = [x.strip() for x in str(args.target_guidance_groups).split(",") if x.strip()]
    q_low = float(np.clip(args.htbal_feature_q_low, 0.0, 0.49))
    q_high = float(np.clip(args.htbal_feature_q_high, 0.51, 1.0))
    alpha = float(np.clip(args.htbal_prior_alpha, 0.0, 1.0))
    rho_min = float(max(0.0, args.htbal_prior_min))

    row_group_id = np.full(len(priority), -1, dtype=np.int64)
    row_mode_id = np.full(len(priority), -1, dtype=np.int64)
    groups: list[str] = []
    by_group: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for group in target_groups:
        idx = train_idx[priority[train_idx] == group]
        if len(idx) < 2:
            continue
        mode_names = sorted(pd.unique(dyn[idx].astype(str)).tolist())
        if len(mode_names) < 2:
            continue
        gid = len(groups)
        groups.append(group)
        row_group_id[priority == group] = gid

        group_feat = feat[idx]
        feat_mu = np.nanmean(group_feat, axis=0).astype(np.float32)
        feat_sd = np.nanstd(group_feat, axis=0).astype(np.float32)
        feat_sd = np.where(feat_sd < 1e-6, 1.0, feat_sd).astype(np.float32)
        feat_low = np.nanquantile(group_feat, q_low, axis=0).astype(np.float32)
        feat_high = np.nanquantile(group_feat, q_high, axis=0).astype(np.float32)

        proto_pop: list[np.ndarray] = []
        proto_feat: list[np.ndarray] = []
        counts: list[int] = []
        for mid, mode in enumerate(mode_names):
            mode_rows = idx[dyn[idx] == mode]
            counts.append(int(len(mode_rows)))
            row_mode_id[dyn == mode] = mid
            p = np.nanmedian(pop[mode_rows], axis=0).astype(np.float32)
            p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
            proto_pop.append(p)
            proto_feat.append(pinnmod.numpy_pop_features(p[None, :, :], times)[0].astype(np.float32))

        ref_prior = np.asarray(counts, dtype=np.float32)
        ref_prior = ref_prior / max(float(ref_prior.sum()), 1e-8)
        uniform = np.full_like(ref_prior, 1.0 / len(ref_prior), dtype=np.float32)
        pi_star = (1.0 - alpha) * ref_prior + alpha * uniform
        if rho_min > 0.0:
            pi_star = np.maximum(pi_star, rho_min).astype(np.float32)
            pi_star = pi_star / max(float(pi_star.sum()), 1e-8)

        proto_pop_arr = np.stack(proto_pop, axis=0).astype(np.float32)
        proto_feat_arr = np.stack(proto_feat, axis=0).astype(np.float32)
        proto_z = ((proto_feat_arr - feat_mu[None, :]) / feat_sd[None, :]).astype(np.float32)

        by_group[group] = {
            "group_id": gid,
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
            "counts": counts,
        }
        for mode, count, ref, prior in zip(mode_names, counts, ref_prior, pi_star):
            summary_rows.append(
                {
                    "mode_source": "train_dynamic",
                    "target_group": group,
                    "mode": mode,
                    "n_train": int(count),
                    "reference_fraction": float(ref),
                    "smoothed_internal_prior": float(prior),
                }
            )

    out = {
        "groups": groups,
        "by_group": by_group,
        "row_group_id": row_group_id,
        "row_mode_id": row_mode_id,
        "summary": pd.DataFrame(summary_rows),
    }
    context["htbal_mode_guidance_summary"] = out["summary"]
    return out


def build_htbal_reference_mode_guidance(context: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    reference_dir = Path(args.htbal_reference_dir)
    traces_path = reference_dir / "npz" / "dynamic_condition_mode_traces.npz"
    assignments_path = reference_dir / "csv" / "dynamic_condition_mode_assignments.csv"
    summary_path = reference_dir / "csv" / "dynamic_condition_mode_summary.csv"
    if not traces_path.exists():
        raise FileNotFoundError(traces_path)
    z = np.load(traces_path, allow_pickle=True)
    assignments = pd.read_csv(assignments_path)
    summary = pd.read_csv(summary_path)

    ref_candidate_id = z["candidate_id"].astype(str)
    ref_condition = z["condition"].astype(str)
    ref_pop = np.asarray(z["pop_t"], dtype=np.float32)
    ref_times = np.asarray(z["tlist"], dtype=np.float32)
    ref_feat = pinnmod.numpy_pop_features(ref_pop, ref_times).astype(np.float32)
    ref_pos = {cid: i for i, cid in enumerate(ref_candidate_id.tolist())}

    train_idx = np.asarray(context["train_idx"], dtype=np.int64)
    priority = np.asarray(context["priority_group"]).astype(str)
    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    times = np.asarray(context["raw"]["times"], dtype=np.float32)
    feat = pinnmod.numpy_pop_features(pop, times).astype(np.float32)

    target_groups = [x.strip() for x in str(args.target_guidance_groups).split(",") if x.strip()]
    q_low = float(np.clip(args.htbal_feature_q_low, 0.0, 0.49))
    q_high = float(np.clip(args.htbal_feature_q_high, 0.51, 1.0))
    alpha = float(np.clip(args.htbal_prior_alpha, 0.0, 1.0))
    rho_min = float(max(0.0, args.htbal_prior_min))

    row_group_id = np.full(len(priority), -1, dtype=np.int64)
    row_mode_id = np.full(len(priority), -1, dtype=np.int64)
    groups: list[str] = []
    by_group: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for group in target_groups:
        cond = HTBAL_REFERENCE_CONDITION.get(group)
        if cond is None:
            continue
        ref_summary = summary[summary["condition"].astype(str) == cond].copy()
        ref_assign = assignments[assignments["condition"].astype(str) == cond].copy()
        idx_train = train_idx[priority[train_idx] == group]
        if len(ref_summary) < 2 or len(ref_assign) == 0 or len(idx_train) < 2:
            continue
        if "dynamic_mode" in ref_summary.columns:
            ref_summary = ref_summary.sort_values("dynamic_mode")
        mode_names = ref_summary["dynamic_mode_id"].astype(str).tolist()
        gid = len(groups)
        groups.append(group)
        row_group_id[priority == group] = gid

        group_feat = feat[idx_train]
        feat_mu = np.nanmean(group_feat, axis=0).astype(np.float32)
        feat_sd = np.nanstd(group_feat, axis=0).astype(np.float32)
        feat_sd = np.where(feat_sd < 1e-6, 1.0, feat_sd).astype(np.float32)
        feat_low = np.nanquantile(group_feat, q_low, axis=0).astype(np.float32)
        feat_high = np.nanquantile(group_feat, q_high, axis=0).astype(np.float32)

        proto_pop: list[np.ndarray] = []
        proto_feat: list[np.ndarray] = []
        proto_mode_names: list[str] = []
        counts: list[int] = []
        for mode in mode_names:
            mode_rows = ref_assign[ref_assign["dynamic_mode_id"].astype(str) == str(mode)]
            ref_trace_idx = [ref_pos[str(cid)] for cid in mode_rows["candidate_id"].astype(str).tolist() if str(cid) in ref_pos]
            if not ref_trace_idx:
                continue
            proto_mode_names.append(str(mode))
            counts.append(int(len(ref_trace_idx)))
            p = np.nanmedian(ref_pop[ref_trace_idx], axis=0).astype(np.float32)
            p = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-8)
            proto_pop.append(p)
            proto_feat.append(pinnmod.numpy_pop_features(p[None, :, :], ref_times)[0].astype(np.float32))
        if len(proto_pop) < 2:
            continue

        mode_names = proto_mode_names
        proto_pop_arr = np.stack(proto_pop, axis=0).astype(np.float32)
        proto_feat_arr = np.stack(proto_feat, axis=0).astype(np.float32)
        proto_z = ((proto_feat_arr - feat_mu[None, :]) / feat_sd[None, :]).astype(np.float32)

        group_z = ((feat[priority == group] - feat_mu[None, :]) / feat_sd[None, :]).astype(np.float32)
        dist = np.mean((group_z[:, None, :] - proto_z[None, :, :]) ** 2, axis=-1)
        row_mode_id[priority == group] = np.argmin(dist, axis=1).astype(np.int64)

        ref_prior = np.asarray(counts, dtype=np.float32)
        ref_prior = ref_prior / max(float(ref_prior.sum()), 1e-8)
        uniform = np.full_like(ref_prior, 1.0 / len(ref_prior), dtype=np.float32)
        pi_star = (1.0 - alpha) * ref_prior + alpha * uniform
        if rho_min > 0.0:
            pi_star = np.maximum(pi_star, rho_min).astype(np.float32)
            pi_star = pi_star / max(float(pi_star.sum()), 1e-8)

        by_group[group] = {
            "group_id": gid,
            "reference_condition": cond,
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
            "counts": counts,
        }
        for mode, count, ref, prior in zip(mode_names, counts, ref_prior, pi_star):
            summary_rows.append(
                {
                    "mode_source": "reference",
                    "target_group": group,
                    "reference_condition": cond,
                    "mode": mode,
                    "n_reference": int(count),
                    "reference_fraction": float(ref),
                    "smoothed_internal_prior": float(prior),
                }
            )

    out = {
        "groups": groups,
        "by_group": by_group,
        "row_group_id": row_group_id,
        "row_mode_id": row_mode_id,
        "summary": pd.DataFrame(summary_rows),
    }
    context["htbal_mode_guidance_summary"] = out["summary"]
    return out


def artifacts_ready(run_dir: Path, condition_set: str) -> tuple[bool, list[Path]]:
    required = [
        run_dir / "checkpoints" / f"{condition_set}_best.pt",
        run_dir / f"{condition_set}_loss_history.csv",
        run_dir / f"{condition_set}_test_metrics.csv",
        run_dir / f"{condition_set}_generated_samples.npz",
        run_dir / f"{condition_set}_generated_physical_summary.csv",
    ]
    missing = [p for p in required if (not p.exists()) or p.stat().st_size == 0]
    return len(missing) == 0, missing


def limited_indices(idx: np.ndarray, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or len(idx) <= max_n:
        return idx
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(idx, size=int(max_n), replace=False)).astype(np.int64)


def make_val_aux_keep(n: int, max_n: int, seed: int) -> np.ndarray:
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=int(max_n), replace=False)).astype(np.int64)


def aux_weight(epoch: int, start: int, warmup: int) -> float:
    if epoch < int(start):
        return 0.0
    if warmup <= 0:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - start + 1) / float(warmup))))


def make_diffusion_model(torch, nn, x_dim: int, c_dim: int, hidden: int, depth: int, time_features: int):
    class Denoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_dim = int(x_dim)
            self.c_dim = int(c_dim)
            self.time_features = int(time_features)
            in_dim = self.x_dim + self.c_dim + self.time_features
            layers: list[Any] = []
            for i in range(max(1, int(depth))):
                layers.append(nn.Linear(in_dim if i == 0 else int(hidden), int(hidden)))
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(int(hidden)))
            layers.append(nn.Linear(int(hidden), self.x_dim))
            self.net = nn.Sequential(*layers)

        def time_embed(self, t):
            if t.ndim == 1:
                t = t[:, None]
            n_freq = max(1, self.time_features // 2)
            freqs = (2.0 ** torch.arange(n_freq, dtype=t.dtype, device=t.device)) * math.pi
            phase = t * freqs[None, :]
            emb = torch.cat([torch.sin(phase), torch.cos(phase)], dim=1)
            if emb.shape[1] < self.time_features:
                emb = torch.cat([emb, t], dim=1)
            return emb[:, : self.time_features]

        def forward(self, x_t, t_index, c):
            if t_index.dtype in (torch.int32, torch.int64, torch.long):
                denom = max(1.0, float(int(self.training_steps) - 1))
                t = t_index.to(dtype=x_t.dtype)[:, None] / denom
            else:
                t = t_index.to(dtype=x_t.dtype)
                if t.ndim == 1:
                    t = t[:, None]
            return self.net(torch.cat([x_t, self.time_embed(t), c], dim=1))

    return Denoiser()


def attach_training_steps(model, steps: int):
    model.training_steps = int(steps)
    return model


def make_diffusion_schedule(args, device, torch) -> dict[str, Any]:
    steps = int(args.diffusion_steps)
    if steps < 2:
        raise ValueError("--diffusion-steps must be >= 2")
    if args.beta_schedule == "linear":
        betas_np = np.linspace(float(args.beta_start), float(args.beta_end), steps, dtype=np.float64)
    else:
        betas_np = cosine_betas(steps)
    betas = torch.tensor(betas_np.astype(np.float32), dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(torch.clamp(1.0 - alpha_bars, min=1e-12)),
        "steps": steps,
    }


def cosine_betas(steps: int, s: float = 0.008) -> np.ndarray:
    x = np.linspace(0, steps, steps + 1, dtype=np.float64)
    f = np.cos(((x / steps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alpha_bar = f / f[0]
    betas = 1.0 - (alpha_bar[1:] / np.maximum(alpha_bar[:-1], 1e-12))
    return np.clip(betas, 1e-5, 0.999)


def q_sample(x0, t_idx, schedule, torch):
    noise = torch.randn_like(x0)
    sqrt_ab = schedule["sqrt_alpha_bars"][t_idx][:, None]
    sqrt_om = schedule["sqrt_one_minus_alpha_bars"][t_idx][:, None]
    return sqrt_ab * x0 + sqrt_om * noise, noise


def predict_x0_from_eps(x_t, t_idx, eps, schedule, x0_clip: float):
    sqrt_ab = schedule["sqrt_alpha_bars"][t_idx][:, None]
    sqrt_om = schedule["sqrt_one_minus_alpha_bars"][t_idx][:, None]
    x0 = (x_t - sqrt_om * eps) / torch_clamp_min(sqrt_ab, 1e-6)
    if x0_clip > 0:
        x0 = x0.clamp(-float(x0_clip), float(x0_clip))
    return x0


def torch_clamp_min(x, value: float):
    return x.clamp_min(float(value))


def train_one(
    method: str,
    condition_set: str,
    ds: base.ConditionDataset,
    context: dict[str, Any],
    target_pop: dict[str, np.ndarray],
    mode_guidance: dict[str, Any],
    pinn,
    args: argparse.Namespace,
    run_dir: Path,
    schedule: dict[str, Any],
    device,
    torch,
    nn,
    F,
    TensorDataset,
    DataLoader,
) -> None:
    use_pinn = method in {"DDPM_PINNTRAJ", "DDPM_HTPINNTRAJ", HTBAL_METHOD}
    use_target_guidance = method == "DDPM_HTPINNTRAJ"
    use_htbal = method == HTBAL_METHOD
    if use_pinn and pinn is None:
        raise RuntimeError("DDPM_PINNTRAJ requires a trained PINN trajectory surrogate")

    x = np.asarray(context["h_norm"], dtype=np.float32)
    c = np.asarray(ds.norm, dtype=np.float32)
    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    target_pop_arr, target_weight = make_row_target_guidance(pop, context, target_pop, args)
    row_group_id = np.asarray(mode_guidance.get("row_group_id", np.full(len(x), -1)), dtype=np.int64)
    row_mode_id = np.asarray(mode_guidance.get("row_mode_id", np.full(len(x), -1)), dtype=np.int64)
    train_idx = limited_indices(np.asarray(context["train_idx"], dtype=np.int64), args.max_train_samples, args.seed)
    val_idx = limited_indices(np.asarray(context["val_idx"], dtype=np.int64), args.max_val_samples, args.seed + 1)
    test_idx = np.asarray(context["test_idx"], dtype=np.int64)

    model = attach_training_steps(
        make_diffusion_model(torch, nn, x.shape[1], c.shape[1], args.hidden, args.depth, args.time_features),
        int(args.diffusion_steps),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tensors = TensorDataset(
        torch.tensor(x[train_idx]),
        torch.tensor(c[train_idx]),
        torch.tensor(pop[train_idx]),
        torch.tensor(target_pop_arr[train_idx]),
        torch.tensor(target_weight[train_idx]),
        torch.tensor(row_group_id[train_idx]),
        torch.tensor(row_mode_id[train_idx]),
    )
    loader = DataLoader(tensors, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_x = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    val_c = torch.tensor(c[val_idx], dtype=torch.float32, device=device)
    val_pop = torch.tensor(pop[val_idx], dtype=torch.float32, device=device)
    val_target_pop = torch.tensor(target_pop_arr[val_idx], dtype=torch.float32, device=device)
    val_target_weight = torch.tensor(target_weight[val_idx], dtype=torch.float32, device=device)
    val_group_id = torch.tensor(row_group_id[val_idx], dtype=torch.long, device=device)
    val_mode_id = torch.tensor(row_mode_id[val_idx], dtype=torch.long, device=device)
    val_aux_keep = make_val_aux_keep(len(val_idx), args.val_aux_max, args.seed)

    best = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "checkpoints" / f"{condition_set}_best.pt"
    t0 = time.perf_counter()
    epoch_iter = range(1, args.epochs + 1)
    progress = None
    if not args.no_progress:
        try:
            from tqdm.auto import tqdm

            progress = tqdm(epoch_iter, total=args.epochs, desc=f"{condition_set} {args.run_name}", unit="epoch")
            epoch_iter = progress
        except Exception as exc:
            print(f"tqdm disabled: {exc}", flush=True)

    print(
        f"{condition_set}: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"H_dim={x.shape[1]} C_dim={c.shape[1]} use_pinn={use_pinn}",
        flush=True,
    )
    for epoch in epoch_iter:
        model.train()
        warm = aux_weight(epoch, args.aux_start_epoch, args.aux_warmup_epochs)
        totals = {"loss": 0.0, "eps": 0.0, "x0": 0.0, **{key: 0.0 for key in AUX_KEYS if key != "weighted"}}
        seen = 0
        for xb, cb, pb, tb, wb, gb, mb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            pb = pb.to(device)
            tb = tb.to(device)
            wb = wb.to(device)
            gb = gb.to(device)
            mb = mb.to(device)
            t_idx = torch.randint(0, int(args.diffusion_steps), (len(xb),), dtype=torch.long, device=device)
            x_t, noise = q_sample(xb, t_idx, schedule, torch)
            eps_pred = model(x_t, t_idx, cb)
            eps_loss = F.mse_loss(eps_pred, noise)
            x0_pred = predict_x0_from_eps(x_t, t_idx, eps_pred, schedule, args.x0_clip)
            x0_loss = F.mse_loss(x0_pred, xb)
            aux = zero_aux(xb)
            if use_pinn and warm > 0.0:
                aux = trajectory_aux_loss(x0_pred, pb, pinn, args, torch, F)
                if use_target_guidance:
                    target_aux = target_trajectory_aux_loss(x0_pred, tb, wb, pinn, args, torch, F)
                    aux = add_aux(aux, target_aux)
                if use_htbal:
                    htbal_aux = htbal_mode_balance_aux_loss(x0_pred, gb, mb, mode_guidance, pinn, args, torch, F)
                    aux = add_aux(aux, htbal_aux)
            loss = eps_loss + 0.05 * x0_loss + warm * aux["weighted"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            opt.step()
            n = len(xb)
            seen += n
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["eps"] += float(eps_loss.detach().cpu()) * n
            totals["x0"] += float(x0_loss.detach().cpu()) * n
            for key in [key for key in AUX_KEYS if key != "weighted"]:
                totals[key] += float(aux[key].detach().cpu()) * n

        model.eval()
        with torch.no_grad():
            val = evaluate_ddpm_losses(
                model,
                val_x,
                val_c,
                val_pop,
                pinn if use_pinn else None,
                args,
                schedule,
                torch,
                F,
                val_aux_keep,
                target_pop=val_target_pop,
                target_weight=val_target_weight,
                group_id=val_group_id,
                mode_id=val_mode_id,
                mode_guidance=mode_guidance,
                use_target_guidance=use_target_guidance,
                use_htbal=use_htbal,
            )
            val_score = val["eps"] + 0.05 * val["x0"] + float(args.val_aux_weight) * val["weighted"]
        score = float(val_score.detach().cpu())
        if score < best - float(args.min_delta):
            best = score
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "conditional_ddpm_denoiser",
                    "method": method,
                    "condition_set": condition_set,
                    "base_condition": args.base_condition,
                    "condition_names": ds.names,
                    "condition_mu": ds.mu.tolist(),
                    "condition_sd": ds.sd.tolist(),
                    "condition_flag_mask": ds.flag_mask.astype(bool).tolist(),
                    "x_dim": int(x.shape[1]),
                    "c_dim": int(c.shape[1]),
                    "hidden": int(args.hidden),
                    "depth": int(args.depth),
                    "time_features": int(args.time_features),
                    "diffusion_steps": int(args.diffusion_steps),
                    "beta_schedule": args.beta_schedule,
                    "beta_start": float(args.beta_start),
                    "beta_end": float(args.beta_end),
                    "best_epoch": int(best_epoch),
                    "best_val_score": float(best),
                    "args": base.clean_json(vars(args)),
                },
                best_path,
            )
        else:
            stale += 1
        row = {
            "condition_set": condition_set,
            "method": method,
            "run_name": args.run_name,
            "epoch": int(epoch),
            "aux_weight": float(warm),
            "train_loss": totals["loss"] / max(seen, 1),
            "train_eps_mse": totals["eps"] / max(seen, 1),
            "train_x0_mse": totals["x0"] / max(seen, 1),
            "train_pinn_traj": totals["traj"] / max(seen, 1),
            "train_pinn_feature": totals["feature"] / max(seen, 1),
            "train_pinn_phys": totals["phys"] / max(seen, 1),
            "train_support": totals["support"] / max(seen, 1),
            "train_target_band": totals["target_band"] / max(seen, 1),
            "train_htbal_mode": totals["mode"] / max(seen, 1),
            "val_eps_mse": float(val["eps"].detach().cpu()),
            "val_x0_mse": float(val["x0"].detach().cpu()),
            "val_aux_weighted": float(val["weighted"].detach().cpu()),
            "val_pinn_traj": float(val["traj"].detach().cpu()),
            "val_pinn_feature": float(val["feature"].detach().cpu()),
            "val_pinn_phys": float(val["phys"].detach().cpu()),
            "val_support": float(val["support"].detach().cpu()),
            "val_target_band": float(val["target_band"].detach().cpu()),
            "val_htbal_mode": float(val["mode"].detach().cpu()),
            "val_score": score,
            "best_val_score": float(best),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        history.append(row)
        if progress is not None:
            progress.set_postfix(eps=f"{row['train_eps_mse']:.4f}", val=f"{row['val_eps_mse']:.4f}", best=f"{best:.4f}", stale=stale)
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                f"{condition_set} epoch {epoch}/{args.epochs} "
                f"train_eps={row['train_eps_mse']:.5f} traj={row['train_pinn_traj']:.5f} "
                f"val_eps={row['val_eps_mse']:.5f} score={score:.5f} best={best:.5f}@{best_epoch}",
                flush=True,
            )
        if args.run_name != "smoke" and args.patience > 0 and stale >= args.patience:
            print(f"{condition_set} early stopping at epoch {epoch}", flush=True)
            break
    if progress is not None:
        progress.close()

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / f"{condition_set}_loss_history.csv", index=False)
    plot_ddpm_loss(hist, run_dir / "figures" / f"{condition_set}_loss_curve.png")

    state = load_checkpoint(best_path, device, torch)
    model.load_state_dict(state["state_dict"])
    model.eval()
    metrics = evaluate_split_metrics(
        condition_set,
        method,
        model,
        x,
        c,
        pop,
        context,
        target_pop_arr,
        target_weight,
        row_group_id,
        row_mode_id,
        mode_guidance,
        pinn if use_pinn else None,
        args,
        schedule,
        device,
        torch,
        F,
        use_target_guidance,
        use_htbal,
    )
    pd.DataFrame(metrics).to_csv(run_dir / f"{condition_set}_test_metrics.csv", index=False)
    generated_summary = generate_targets(condition_set, method, model, ds, context, target_pop, mode_guidance, pinn if use_pinn else None, args, run_dir, schedule, device, torch, F)
    write_condition_report(run_dir, condition_set, method, args, ds, metrics, generated_summary, use_pinn)


def zero_aux(x):
    z = x.new_zeros(())
    return {key: z for key in AUX_KEYS}


def add_aux(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    keys = sorted(set(a) | set(b))
    ref = next(iter(a.values())) if a else next(iter(b.values()))
    zero = ref.new_zeros(())
    return {key: a.get(key, zero) + b.get(key, zero) for key in keys}


def trajectory_aux_loss(x0_pred, true_pop, pinn, args, torch, F):
    pred_pop, _logits = pinn.model(x0_pred)
    traj = F.mse_loss(pred_pop, true_pop)
    feature = F.mse_loss(
        pinnmod.torch_pop_features(pred_pop, pinn.model.times_raw, torch),
        pinnmod.torch_pop_features(true_pop, pinn.model.times_raw, torch),
    )
    mono, smooth = pinnmod.population_physics_penalties(pred_pop, torch)
    phys = mono + smooth
    support = torch.relu(torch.abs(x0_pred) - float(args.support_clip)).pow(2).mean()
    weighted = (
        float(args.lambda_pinn_traj) * traj
        + float(args.lambda_pinn_feature) * feature
        + float(args.lambda_pinn_phys) * phys
        + float(args.lambda_support) * support
    )
    aux = zero_aux(x0_pred)
    aux.update({"weighted": weighted, "traj": traj, "feature": feature, "phys": phys, "support": support})
    return aux


def target_trajectory_aux_loss(x0_pred, target_pop, target_weight, pinn, args, torch, F):
    pred_pop, _logits = pinn.model(x0_pred)
    weights = target_weight[:, None, None].to(dtype=pred_pop.dtype)
    denom = torch.clamp(weights.mean(), min=1e-6)
    traj = ((pred_pop - target_pop).pow(2) * weights).mean() / denom
    pred_feat = pinnmod.torch_pop_features(pred_pop, pinn.model.times_raw, torch)
    target_feat = pinnmod.torch_pop_features(target_pop, pinn.model.times_raw, torch)
    feat_weight = target_weight[:, None].to(dtype=pred_feat.dtype)
    feature = ((pred_feat - target_feat).pow(2) * feat_weight).mean() / torch.clamp(feat_weight.mean(), min=1e-6)
    mono, smooth = pinnmod.population_physics_penalties(pred_pop, torch)
    phys = mono + smooth
    support = torch.relu(torch.abs(x0_pred) - float(args.support_clip)).pow(2).mean()
    weighted = (
        float(args.lambda_target_traj) * traj
        + float(args.lambda_target_feature) * feature
        + float(args.lambda_pinn_phys) * phys
        + float(args.lambda_support) * support
    )
    aux = zero_aux(x0_pred)
    aux.update({"weighted": weighted, "traj": traj, "feature": feature, "phys": phys, "support": support})
    return aux


def htbal_mode_balance_aux_loss(x0_pred, group_id, mode_id, mode_guidance: dict[str, Any], pinn, args, torch, F):
    aux = zero_aux(x0_pred)
    if not mode_guidance.get("groups"):
        return aux

    pred_pop, _logits = pinn.model(x0_pred)
    pred_feat = pinnmod.torch_pop_features(pred_pop, pinn.model.times_raw, torch)
    mono, smooth = pinnmod.population_physics_penalties(pred_pop, torch)
    phys = mono + smooth
    support = torch.relu(torch.abs(x0_pred) - float(args.support_clip)).pow(2).mean()

    band_terms = []
    mode_terms = []
    assign_terms = []
    balance_terms = []
    floor_terms = []
    temp = max(1e-4, float(args.htbal_assign_temp))
    eps = 1e-6

    for group in mode_guidance["groups"]:
        info = mode_guidance["by_group"][group]
        gid = int(info["group_id"])
        mask = (group_id == gid) & (mode_id >= 0)
        if not bool(mask.any().detach().cpu()):
            continue
        feat_g = pred_feat[mask]
        mode_g = mode_id[mask].long()

        feat_mu = torch.tensor(info["feat_mu"], dtype=feat_g.dtype, device=feat_g.device)
        feat_sd = torch.tensor(info["feat_sd"], dtype=feat_g.dtype, device=feat_g.device).clamp_min(eps)
        proto_z = torch.tensor(info["proto_z"], dtype=feat_g.dtype, device=feat_g.device)
        feat_low = torch.tensor(info["feat_low"], dtype=feat_g.dtype, device=feat_g.device)
        feat_high = torch.tensor(info["feat_high"], dtype=feat_g.dtype, device=feat_g.device)
        pi_star = torch.tensor(info["pi_star"], dtype=feat_g.dtype, device=feat_g.device)

        z = (feat_g - feat_mu[None, :]) / feat_sd[None, :]
        dist = (z[:, None, :] - proto_z[None, :, :]).pow(2).mean(dim=-1)
        q_pred = torch.softmax(-dist / temp, dim=-1)
        q_bar = q_pred.mean(dim=0).clamp_min(eps)

        valid_mode = mode_g.clamp(min=0, max=dist.shape[1] - 1)
        mode_terms.append(dist[torch.arange(len(dist), device=dist.device), valid_mode].mean())
        assign_terms.append(-torch.log(q_pred[torch.arange(len(dist), device=dist.device), valid_mode].clamp_min(eps)).mean())
        balance_terms.append((pi_star * (torch.log(pi_star.clamp_min(eps)) - torch.log(q_bar))).sum())
        floor_terms.append(torch.relu(float(args.htbal_prior_min) - q_bar).pow(2).sum())
        band_terms.append((torch.relu(feat_low[None, :] - feat_g).pow(2) + torch.relu(feat_g - feat_high[None, :]).pow(2)).mean())

    if not mode_terms:
        return aux

    target_band = torch.stack(band_terms).mean()
    mode_proto = torch.stack(mode_terms).mean()
    mode_assign = torch.stack(assign_terms).mean()
    mode_balance = torch.stack(balance_terms).mean()
    mode_floor = torch.stack(floor_terms).mean()
    mode_total = (
        float(args.lambda_htbal_mode_proto) * mode_proto
        + float(args.lambda_htbal_mode_assign) * mode_assign
        + float(args.lambda_htbal_mode_balance) * mode_balance
        + float(args.lambda_htbal_mode_floor) * mode_floor
    )
    weighted = (
        float(args.lambda_htbal_target_band) * target_band
        + mode_total
        + float(args.lambda_pinn_phys) * phys
        + float(args.lambda_support) * support
    )
    aux.update({"weighted": weighted, "phys": phys, "support": support, "target_band": target_band, "mode": mode_total})
    return aux


def evaluate_ddpm_losses(
    model,
    x,
    c,
    pop,
    pinn,
    args,
    schedule,
    torch,
    F,
    keep_idx: np.ndarray | None = None,
    target_pop=None,
    target_weight=None,
    group_id=None,
    mode_id=None,
    mode_guidance: dict[str, Any] | None = None,
    use_target_guidance: bool = False,
    use_htbal: bool = False,
):
    if keep_idx is not None:
        x = x[keep_idx]
        c = c[keep_idx]
        pop = pop[keep_idx]
        if target_pop is not None:
            target_pop = target_pop[keep_idx]
        if target_weight is not None:
            target_weight = target_weight[keep_idx]
        if group_id is not None:
            group_id = group_id[keep_idx]
        if mode_id is not None:
            mode_id = mode_id[keep_idx]
    if len(x) == 0:
        return zero_aux(c)
    t_idx = torch.randint(0, int(args.diffusion_steps), (len(x),), dtype=torch.long, device=x.device)
    x_t, noise = q_sample(x, t_idx, schedule, torch)
    eps_pred = model(x_t, t_idx, c)
    eps = F.mse_loss(eps_pred, noise)
    x0_pred = predict_x0_from_eps(x_t, t_idx, eps_pred, schedule, args.x0_clip)
    x0 = F.mse_loss(x0_pred, x)
    aux = zero_aux(x)
    if pinn is not None:
        aux = trajectory_aux_loss(x0_pred, pop, pinn, args, torch, F)
        if use_target_guidance and target_pop is not None and target_weight is not None:
            aux = add_aux(aux, target_trajectory_aux_loss(x0_pred, target_pop, target_weight, pinn, args, torch, F))
        if use_htbal and group_id is not None and mode_id is not None and mode_guidance is not None:
            aux = add_aux(aux, htbal_mode_balance_aux_loss(x0_pred, group_id, mode_id, mode_guidance, pinn, args, torch, F))
    return {"eps": eps, "x0": x0, **aux}


def evaluate_split_metrics(
    condition_set: str,
    method: str,
    model,
    x,
    c,
    pop,
    context,
    target_pop_arr,
    target_weight,
    row_group_id,
    row_mode_id,
    mode_guidance,
    pinn,
    args,
    schedule,
    device,
    torch,
    F,
    use_target_guidance: bool,
    use_htbal: bool,
):
    test_idx = np.asarray(context["test_idx"], dtype=np.int64)
    priority = np.asarray(context["priority_group"])[test_idx].astype(str)
    metrics: list[dict[str, Any]] = []
    for group_type, group_values in [("overall", np.array(["overall"] * len(test_idx), dtype=object)), ("priority_group", priority)]:
        for group in sorted(set(group_values.astype(str))):
            rows = test_idx if group_type == "overall" else test_idx[group_values == group]
            if len(rows) == 0:
                continue
            xb = torch.tensor(x[rows], dtype=torch.float32, device=device)
            cb = torch.tensor(c[rows], dtype=torch.float32, device=device)
            pb = torch.tensor(pop[rows], dtype=torch.float32, device=device)
            tb = torch.tensor(target_pop_arr[rows], dtype=torch.float32, device=device)
            wb = torch.tensor(target_weight[rows], dtype=torch.float32, device=device)
            gb = torch.tensor(row_group_id[rows], dtype=torch.long, device=device)
            mb = torch.tensor(row_mode_id[rows], dtype=torch.long, device=device)
            with torch.no_grad():
                losses = evaluate_ddpm_losses(
                    model,
                    xb,
                    cb,
                    pb,
                    pinn,
                    args,
                    schedule,
                    torch,
                    F,
                    target_pop=tb,
                    target_weight=wb,
                    group_id=gb,
                    mode_id=mb,
                    mode_guidance=mode_guidance,
                    use_target_guidance=use_target_guidance,
                    use_htbal=use_htbal,
                )
            metrics.append(
                {
                    "condition_set": condition_set,
                    "method": method,
                    "group_type": group_type,
                    "group": group,
                    "n": int(len(rows)),
                    "eps_mse": float(losses["eps"].detach().cpu()),
                    "x0_mse": float(losses["x0"].detach().cpu()),
                    "pinn_traj": float(losses["traj"].detach().cpu()),
                    "pinn_feature": float(losses["feature"].detach().cpu()),
                    "pinn_phys": float(losses["phys"].detach().cpu()),
                    "support": float(losses["support"].detach().cpu()),
                    "target_band": float(losses["target_band"].detach().cpu()),
                    "htbal_mode": float(losses["mode"].detach().cpu()),
                    "score": float((losses["eps"] + 0.05 * losses["x0"] + losses["weighted"]).detach().cpu()),
                }
            )
    return metrics


def build_model_from_state(state: dict[str, Any], args, device, torch, nn):
    model = attach_training_steps(
        make_diffusion_model(
            torch,
            nn,
            int(state["x_dim"]),
            int(state["c_dim"]),
            int(state.get("hidden", args.hidden)),
            int(state.get("depth", args.depth)),
            int(state.get("time_features", args.time_features)),
        ),
        int(state.get("diffusion_steps", args.diffusion_steps)),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def load_checkpoint(path: Path, device, torch) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def apply_checkpoint_schedule_args(args, state: dict[str, Any]) -> None:
    args.diffusion_steps = int(state.get("diffusion_steps", args.diffusion_steps))
    args.beta_schedule = str(state.get("beta_schedule", args.beta_schedule))
    args.beta_start = float(state.get("beta_start", args.beta_start))
    args.beta_end = float(state.get("beta_end", args.beta_end))


def ddim_timesteps(diffusion_steps: int, sample_steps: int) -> np.ndarray:
    steps = max(1, min(int(sample_steps), int(diffusion_steps)))
    return np.unique(np.linspace(0, int(diffusion_steps) - 1, steps, dtype=np.int64))[::-1].copy()


def sample_ddim(
    model,
    c_batch,
    target_pop_batch,
    pinn,
    args,
    schedule,
    device,
    torch,
    F,
    guidance_scale: float,
    guidance_start_frac: float,
) -> Any:
    n = len(c_batch)
    x = torch.randn((n, 28), dtype=torch.float32, device=device)
    t_steps = ddim_timesteps(int(args.diffusion_steps), int(args.sample_steps))
    model.eval()
    for pos, t_value in enumerate(t_steps):
        t_idx = torch.full((n,), int(t_value), dtype=torch.long, device=device)
        with torch.no_grad():
            eps = model(x, t_idx, c_batch)
            x0 = predict_x0_from_eps(x, t_idx, eps, schedule, args.x0_clip)
        scale_now = step_sample_guidance_scale(pos, len(t_steps), args, guidance_scale, guidance_start_frac)
        if scale_now > 0.0 and pinn is not None and target_pop_batch is not None:
            x0 = guide_x0_with_surrogate(x0, target_pop_batch, pinn, args, torch, F, scale_now)
            abar_t = schedule["alpha_bars"][int(t_value)]
            eps = (x - torch.sqrt(abar_t) * x0) / torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12))
        prev_t = int(t_steps[pos + 1]) if pos + 1 < len(t_steps) else -1
        x = ddim_step(x, x0, eps, int(t_value), prev_t, args, schedule, torch)
    return x


def should_apply_sample_guidance(pos: int, total: int, args, guidance_scale: float, guidance_start_frac: float) -> bool:
    return step_sample_guidance_scale(pos, total, args, guidance_scale, guidance_start_frac) > 0.0


def step_sample_guidance_scale(pos: int, total: int, args, guidance_scale: float, guidance_start_frac: float) -> float:
    if float(guidance_scale) <= 0:
        return 0.0
    start = int(max(0, min(total - 1, math.floor(total * float(guidance_start_frac)))))
    if pos < start:
        return 0.0
    every = max(1, int(args.sample_guidance_every))
    if (pos - start) % every != 0:
        return 0.0
    denom = max(1, total - start)
    frac = max(0.0, min(1.0, (pos - start + 1) / float(denom)))
    power = max(0.0, float(getattr(args, "sample_guidance_ramp_power", 1.0)))
    ramp = frac**power if power > 0 else 1.0
    return float(guidance_scale) * float(ramp)


def guide_x0_with_surrogate(x0, target_pop, pinn, args, torch, F, guidance_scale: float):
    scale = float(guidance_scale)
    if scale <= 0:
        return x0.detach()
    with torch.enable_grad():
        x_req = x0.detach().requires_grad_(True)
        aux = trajectory_aux_loss(x_req, target_pop, pinn, args, torch, F)
        grad = torch.autograd.grad(aux["weighted"], x_req, retain_graph=False, create_graph=False)[0]
        norm = grad.flatten(1).norm(dim=1).clamp_min(1e-6)[:, None]
        guided = x_req - scale * grad / norm
        if args.x0_clip > 0:
            guided = guided.clamp(-float(args.x0_clip), float(args.x0_clip))
    return guided.detach()


def ddim_step(x, x0, eps, t_value: int, prev_t: int, args, schedule, torch):
    abar_t = schedule["alpha_bars"][int(t_value)]
    abar_prev = x.new_tensor(1.0) if prev_t < 0 else schedule["alpha_bars"][int(prev_t)]
    eta = float(args.ddim_eta)
    if eta > 0 and prev_t >= 0:
        sigma = eta * torch.sqrt(
            torch.clamp((1.0 - abar_prev) / torch.clamp(1.0 - abar_t, min=1e-12), min=0.0)
            * torch.clamp(1.0 - abar_t / torch.clamp(abar_prev, min=1e-12), min=0.0)
        )
    else:
        sigma = x.new_tensor(0.0)
    dir_coeff = torch.sqrt(torch.clamp(1.0 - abar_prev - sigma * sigma, min=0.0))
    noise = torch.randn_like(x) if float(sigma.detach().cpu()) > 0 else torch.zeros_like(x)
    return torch.sqrt(abar_prev) * x0 + dir_coeff * eps + sigma * noise


def generate_targets(condition_set: str, method: str, model, ds, context, target_pop, mode_guidance, pinn, args, run_dir: Path, schedule, device, torch, F) -> pd.DataFrame:
    strategies = [x.strip().lower() for x in args.generate_strategies.split(",") if x.strip()]
    unknown = sorted(set(strategies) - {"median", "mixture"})
    if unknown:
        raise KeyError(f"unknown generation strategies: {unknown}")
    h_mu = np.asarray(context["h_mu"], dtype=np.float32)
    h_sd = np.asarray(context["h_sd"], dtype=np.float32)
    train_h_raw = base.gauge_fix_vec28(context["h_norm"][context["train_idx"]] * h_sd + h_mu)
    generated: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + 997)
    guidance_scale, guidance_start_frac = effective_sample_guidance(method, args)

    for target_name, cvec in ds.targets_norm.items():
        group = infer_base_target(target_name, args)
        if group is None:
            continue
        if "median" in strategies:
            c_batch = np.repeat(cvec[None, :].astype(np.float32), int(args.n_generate), axis=0)
            pop_batch, internal_modes = make_generation_guidance_pop_batch(method, group, len(c_batch), target_pop, mode_guidance, rng)
            write_generated_batch(
                condition_set,
                target_name,
                group,
                "median",
                c_batch,
                pop_batch,
                model,
                context,
                pinn,
                args,
                run_dir,
                schedule,
                device,
                torch,
                F,
                guidance_scale,
                guidance_start_frac,
                h_mu,
                h_sd,
                train_h_raw,
                generated,
                rows,
                provenance_rows,
                internal_modes=internal_modes,
            )
        if "mixture" in strategies and target_name == group:
            c_batch, pop_batch, source_rows = make_condition_mixture(group, ds, context, args, rng)
            if method == HTBAL_METHOD:
                pop_batch, internal_modes = make_generation_guidance_pop_batch(method, group, len(c_batch), target_pop, mode_guidance, rng)
            else:
                internal_modes = None
            write_generated_batch(
                condition_set,
                f"{group}_mixture",
                group,
                "mixture",
                c_batch,
                pop_batch,
                model,
                context,
                pinn,
                args,
                run_dir,
                schedule,
                device,
                torch,
                F,
                guidance_scale,
                guidance_start_frac,
                h_mu,
                h_sd,
                train_h_raw,
                generated,
                rows,
                provenance_rows,
                source_rows=source_rows,
                internal_modes=internal_modes,
            )

    np.savez_compressed(run_dir / f"{condition_set}_generated_samples.npz", **generated)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / f"{condition_set}_generated_physical_summary.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(run_dir / f"{condition_set}_generation_provenance.csv", index=False)
    return summary


def write_generated_batch(
    condition_set: str,
    target_key: str,
    base_target: str,
    strategy: str,
    c_batch: np.ndarray,
    pop_batch: np.ndarray | None,
    model,
    context,
    pinn,
    args,
    run_dir: Path,
    schedule,
    device,
    torch,
    F,
    guidance_scale: float,
    guidance_start_frac: float,
    h_mu,
    h_sd,
    train_h_raw,
    generated: dict[str, np.ndarray],
    rows: list[dict[str, Any]],
    provenance_rows: list[dict[str, Any]],
    source_rows: np.ndarray | None = None,
    internal_modes: list[str] | None = None,
) -> None:
    xs: list[np.ndarray] = []
    bs = int(args.batch_size)
    for start in range(0, len(c_batch), bs):
        cb = torch.tensor(c_batch[start : start + bs], dtype=torch.float32, device=device)
        if pop_batch is not None and pinn is not None:
            pb = torch.tensor(pop_batch[start : start + bs], dtype=torch.float32, device=device)
        else:
            pb = None
        xg = sample_ddim(model, cb, pb, pinn, args, schedule, device, torch, F, guidance_scale, guidance_start_frac)
        xs.append(xg.detach().cpu().numpy())
    x_gen_norm = np.concatenate(xs, axis=0).astype(np.float32)
    x_gen_raw = base.gauge_fix_vec28(x_gen_norm * h_sd + h_mu).astype(np.float32)
    key = f"{condition_set}_{target_key}_H_vec28_trace_zero"
    generated[key] = x_gen_raw
    rows.append(
        {
            "condition_set": condition_set,
            "target": target_key,
            "base_target": base_target,
            "strategy": strategy,
            "uses_pinntraj_signal": bool(pinn is not None),
            "sample_guidance_scale": float(guidance_scale) if pinn is not None else 0.0,
            **base.physical_summary(x_gen_raw, train_h_raw),
        }
    )
    if source_rows is None:
        source_rows = np.full(len(c_batch), -1, dtype=np.int64)
    if internal_modes is None:
        internal_modes = [""] * len(c_batch)
    for i, src in enumerate(source_rows):
        provenance_rows.append(
            {
                "condition_set": condition_set,
                "target": target_key,
                "base_target": base_target,
                "strategy": strategy,
                "generated_index": int(i),
                "source_condition_row": int(src),
                "internal_mode": str(internal_modes[i]) if i < len(internal_modes) else "",
            }
        )
    print(f"generated {key}: {x_gen_raw.shape}", flush=True)


def infer_base_target(target_name: str, args) -> str | None:
    groups = [x.strip() for x in args.target_groups.split(",") if x.strip()]
    for group in sorted(groups, key=len, reverse=True):
        if target_name == group or target_name.startswith(f"{group}_"):
            return group
    return None


def make_condition_mixture(group: str, ds, context, args, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_idx = np.asarray(context["train_idx"], dtype=np.int64)
    priority = np.asarray(context["priority_group"]).astype(str)
    idx = train_idx[priority[train_idx] == group]
    if len(idx) == 0:
        raise RuntimeError(f"no train rows for target group {group}")
    source = rng.choice(idx, size=int(args.n_generate), replace=True).astype(np.int64)
    return np.asarray(ds.norm[source], dtype=np.float32), np.asarray(context["raw"]["pop_t"][source], dtype=np.float32), source


def make_generation_guidance_pop_batch(
    method: str,
    group: str,
    n: int,
    target_pop: dict[str, np.ndarray],
    mode_guidance: dict[str, Any],
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, list[str] | None]:
    if method == HTBAL_METHOD and group in mode_guidance.get("by_group", {}):
        info = mode_guidance["by_group"][group]
        probs = np.asarray(info["pi_star"], dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1e-12)
        mode_idx = rng.choice(len(probs), size=int(n), replace=True, p=probs)
        proto = np.asarray(info["proto_pop"], dtype=np.float32)[mode_idx]
        mode_names = [str(info["mode_names"][int(i)]) for i in mode_idx]
        return proto.astype(np.float32), mode_names
    proto = target_pop.get(group)
    if proto is None:
        return None, None
    return np.repeat(proto[None, :, :].astype(np.float32), int(n), axis=0), None


def make_row_target_guidance(pop: np.ndarray, context: dict[str, Any], target_pop: dict[str, np.ndarray], args) -> tuple[np.ndarray, np.ndarray]:
    priority = np.asarray(context["priority_group"]).astype(str)
    out = np.asarray(pop, dtype=np.float32).copy()
    weights = np.full(len(priority), float(args.target_guidance_weight_nonhigh), dtype=np.float32)
    high_groups = {x.strip() for x in str(args.target_guidance_groups).split(",") if x.strip()}
    for group, proto in target_pop.items():
        idx = priority == group
        if not np.any(idx):
            continue
        out[idx] = np.asarray(proto, dtype=np.float32)
        if group in high_groups:
            weights[idx] = float(args.target_guidance_weight_high)
    return out.astype(np.float32), weights.astype(np.float32)


def effective_sample_guidance(method: str, args) -> tuple[float, float]:
    if method == "DDPM_HTPINNTRAJ":
        return float(args.ht_sample_guidance_scale), float(args.ht_sample_guidance_start_frac)
    if method == HTBAL_METHOD:
        return float(args.htbal_sample_guidance_scale), float(args.htbal_sample_guidance_start_frac)
    return float(args.sample_guidance_scale), float(args.sample_guidance_start_frac)


def plot_ddpm_loss(hist: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"plot skipped: {exc}", flush=True)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_eps_mse"], label="train eps")
    ax.plot(hist["epoch"], hist["val_eps_mse"], label="val eps")
    if "val_score" in hist and len(hist):
        ax.plot(hist["epoch"], hist["val_score"], label="val score", alpha=0.7)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE / score")
    ax.set_title(path.stem.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_metadata(out_root: Path, prepared: Path, args, context, ds, target_pop: dict[str, np.ndarray], mode_guidance: dict[str, Any]) -> None:
    meta = out_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    context["dynamic_submode_summary"].to_csv(meta / "dynamic_submode_summary.csv", index=False)
    pd.DataFrame(
        {
            "sample_index": context["raw"]["sample_index"].astype(np.int64),
            "priority_group": context["priority_group"],
            "dynamic_submode": context["dynamic_label"],
        }
    ).to_csv(meta / "dynamic_submode_assignments.csv", index=False)
    pd.DataFrame(
        {
            "feature": context["orange_names"],
            "median_train": np.nanmedian(context["orange_raw"][context["train_idx"]], axis=0),
            "q10_train": np.nanquantile(context["orange_raw"][context["train_idx"]], 0.10, axis=0),
            "q90_train": np.nanquantile(context["orange_raw"][context["train_idx"]], 0.90, axis=0),
        }
    ).to_csv(meta / "orange_readout_feature_summary.csv", index=False)
    context.get("target_pop_prototype_summary", pd.DataFrame()).to_csv(meta / "target_population_prototypes.csv", index=False)
    context.get("htbal_mode_guidance_summary", pd.DataFrame()).to_csv(meta / "htbal_internal_mode_guidance.csv", index=False)
    pd.DataFrame(
        [
            {
                "base_condition": args.base_condition,
                "condition_dim": int(ds.raw.shape[1]),
                "condition_names": json.dumps(ds.names, ensure_ascii=False),
                "target_names": json.dumps(sorted(ds.targets_norm), ensure_ascii=False),
                "methods": args.methods,
            }
        ]
    ).to_csv(meta / "condition_and_method_design.csv", index=False)
    manifest = {
        "purpose": "H27 compact-condition DDPM with optional PINN-lite intermediate population trajectory signal.",
        "prepared": str(prepared),
        "out_root": str(out_root),
        "base_condition": args.base_condition,
        "methods": [x.strip().upper() for x in args.methods.split(",") if x.strip()],
        "condition_boundary": (
            "The generator condition remains compact. Full dynamic/path information is not user-supplied; "
            "DDPM_PINNTRAJ uses a frozen H->pop_t surrogate as a training/sampling signal. "
            "DDPM_HTPINNTRAJ adds target-family trajectory prototypes and stronger high-transfer sampling guidance. "
            "DDPM_HTBALPINNTRAJ uses dynamic-mode prototypes only as internal training/sampling latents."
        ),
        "trajectory_signal": (
            "The surrogate predicts full population paths over the stored time grid. "
            "Diffusion guidance uses path MSE, path-feature MSE, monotonic/smoothness penalties, and support clipping; dynamic CE defaults to zero."
        ),
        "target_pop_prototypes": sorted(target_pop),
        "args": base.clean_json(vars(args)),
        "important_files": {
            "condition_design": str(meta / "condition_and_method_design.csv"),
            "orange_readout_feature_summary": str(meta / "orange_readout_feature_summary.csv"),
            "target_population_prototypes": str(meta / "target_population_prototypes.csv"),
            "dynamic_submode_summary_for_eval": str(meta / "dynamic_submode_summary.csv"),
            "htbal_internal_mode_guidance": str(meta / "htbal_internal_mode_guidance.csv"),
        },
    }
    (meta / "manifest.json").write_text(json.dumps(base.clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    write_readme(out_root, args)


def write_condition_report(run_dir: Path, condition_set: str, method: str, args, ds, metrics, generated_summary, use_pinn: bool) -> None:
    metric_df = pd.DataFrame(metrics)
    lines = [
        f"# {condition_set} H27 diffusion run",
        "",
        f"- method: `{method}`",
        f"- base_condition: `{args.base_condition}`",
        f"- condition_dim: `{ds.raw.shape[1]}`",
        f"- uses_pinntraj_signal: `{use_pinn}`",
        f"- diffusion_steps/sample_steps: `{args.diffusion_steps}` / `{args.sample_steps}`",
        f"- sample_guidance_scale: `{effective_sample_guidance(method, args)[0] if use_pinn else 0.0}`",
        f"- ht_sample_guidance_scale: `{args.ht_sample_guidance_scale if method == 'DDPM_HTPINNTRAJ' else ''}`",
        f"- htbal_sample_guidance_scale: `{args.htbal_sample_guidance_scale if method == HTBAL_METHOD else ''}`",
        "",
        "## Test Denoising Metrics",
        "",
        "```text",
        metric_df.to_string(index=False),
        "```",
        "",
        "## Generated Physical Summary",
        "",
        "```text",
        generated_summary.to_string(index=False),
        "```",
        "",
        "## Interpretation Boundary",
        "",
        "This run should be judged by simulator validation and dynamic-reference diversity assignment. The surrogate signal is a training/guidance signal, not final physical evidence.",
    ]
    (run_dir / "reports" / f"{condition_set}_run_report_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(run_dir: Path, args, selected_methods: list[str]) -> None:
    metric_paths = sorted(run_dir.glob("*_test_metrics.csv"))
    if metric_paths:
        metrics = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)
        metrics.to_csv(run_dir / "all_test_metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "base_condition": args.base_condition,
        "methods": selected_methods,
        "generated_files": [str(p) for p in sorted(run_dir.glob("*_generated_samples.npz"))],
        "checkpoint_files": [str(p) for p in sorted((run_dir / "checkpoints").glob("*_best.pt"))],
        "args": base.clean_json(vars(args)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(base.clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def write_readme(out_root: Path, args) -> None:
    lines = [
        "# H27 Diffusion + PINN-Trajectory Ablation",
        "",
        "Purpose: test whether diffusion denoising plus a trajectory-predicting surrogate improves generated dynamic diversity without exposing full dynamic path labels as user conditions.",
        "",
        "## Methods",
        "",
        "- `DDPM`: compact condition only.",
        "- `DDPM_PINNTRAJ`: same compact condition, but the denoised x0 prediction is passed through a frozen H->pop_t surrogate and penalized against intermediate population paths.",
        "- `DDPM_HTPINNTRAJ`: adds high-transfer target-family trajectory prototype loss and stronger sampling-time guidance.",
        "- `DDPM_HTBALPINNTRAJ`: replaces single target-family prototype attraction with internal dynamic-mode prototypes, target feature-band loss, batch mode-balance loss, and weaker/later mode-specific sampling guidance.",
        "",
        "## Boundary",
        "",
        "The surrogate is not final physical evidence. It supplies a differentiable signal. Final success still requires simulator validation and dynamic-reference diversity assignment.",
        "",
        "`DDPM_HTBALPINNTRAJ` still keeps the user-facing condition compact. Its dynamic modes are internal training/sampling latents only.",
        "",
        "## Main Checks",
        "",
        "1. Inspect `pinntraj_surrogate_split_metrics.csv` before trusting `DDPM_PINNTRAJ`.",
        "2. Compare `*_test_metrics.csv` denoising and surrogate-path losses.",
        "3. Run simulator validation with `--save-trajectories`.",
        "4. Run generated-to-dynamic-reference assignment and compare coverage/largest-fraction/JS against RealNVP and flow-matching baselines.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

