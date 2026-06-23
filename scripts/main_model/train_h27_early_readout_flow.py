#!/usr/bin/env python3
"""Train H27 conditional RealNVP with orange early-readout conditions.

The orange variables come from the dynamic-PC/D-classifier audit:

- residual_10
- source_site1_10
- early_trap_10

This script keeps them as a compact readout condition ablation rather than
exposing full dynamic cluster labels as user-facing conditions.  It reuses the
H27 140k row-aligned prepared artifact and the existing RealNVP training loop
from `train_h27_path_dynamic_flow.py`, so generated NPZ outputs remain
compatible with the existing simulator validation and diversity audit scripts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
for path in (ROOT, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_h27_path_dynamic_flow as base


DEFAULT_PREPARED = base.DEFAULT_PREPARED
DEFAULT_OUT_ROOT = Path("outputs/experiments/20260620_h27_early_readout_flow")
DEFAULT_CONDITION_SETS = "CFAST_ONLY,CFAST_ORANGE3,CFAST_CL1_ORANGE3"
CONDITION_PURPOSE = (
    "H27 conditional RealNVP ablation using compact orange early-readout "
    "conditions that explain dynamic PCs/classes."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=DEFAULT_PREPARED)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-name", choices=["smoke", "full"], default="full")
    p.add_argument("--condition-sets", default=DEFAULT_CONDITION_SETS)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--layers", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--scale-clip", type=float, default=1.8)
    p.add_argument("--patience", type=int, default=12)
    p.add_argument("--min-delta", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--n-generate", type=int, default=512)
    p.add_argument("--seed", type=int, default=20260620)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metadata-only", action="store_true")
    p.add_argument("--target-groups", default=",".join(base.TARGET_GROUPS))
    p.add_argument(
        "--residual-mode",
        choices=["site2_to_site7", "all_sites", "one_minus_trap_loss"],
        default="site2_to_site7",
        help=(
            "Definition for residual_10. `site2_to_site7` is default because "
            "source_site1_10 and early_trap_10 are separate orange features."
        ),
    )

    # Kept for compatibility with base.build_context dynamic-submode metadata.
    p.add_argument("--dyn-k-fast", type=int, default=3)
    p.add_argument("--dyn-k-very-fast", type=int, default=3)
    p.add_argument("--dyn-k-late", type=int, default=4)
    p.add_argument("--dyn-k-nonhigh", type=int, default=6)
    p.add_argument("--dyn-k-other", type=int, default=1)
    p.add_argument("--kmeans-iter", type=int, default=80)
    p.add_argument("--kmeans-init", type=int, default=8)
    p.add_argument("--min-dyn-target-rows", type=int, default=80)
    p.add_argument("--max-dyn-targets-per-group", type=int, default=3)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    prepared = base.resolve_prepared(args.prepared)
    out_root = args.out_root
    run_dir = out_root / args.run_name
    metadata_dir = out_root / "metadata"
    for path in [metadata_dir, run_dir, run_dir / "checkpoints", run_dir / "figures", run_dir / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.load(prepared, allow_pickle=True)
    base.validate_prepared(raw)
    context = base.build_context(raw, args)
    orange_raw, orange_names, orange_meta = build_orange_readout_features(raw, args.residual_mode)
    context["orange_raw"] = orange_raw
    context["orange_names"] = orange_names
    context["orange_meta"] = orange_meta

    condition_data = build_condition_datasets(context, args)
    selected = [x.strip() for x in args.condition_sets.split(",") if x.strip()]
    missing_sets = sorted(set(selected) - set(condition_data))
    if missing_sets:
        raise KeyError(f"unknown condition sets: {missing_sets}; available={sorted(condition_data)}")

    write_metadata(out_root, prepared, args, context, condition_data, selected)
    print(f"prepared: {prepared}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"condition_sets: {selected}", flush=True)
    print(f"orange_names: {orange_names}", flush=True)
    print(f"residual_mode: {args.residual_mode}", flush=True)
    if args.metadata_only:
        print("metadata-only complete:", metadata_dir, flush=True)
        return 0

    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(args.seed)
    device = base.choose_device(args.device, torch)
    print(f"device: {device}", flush=True)
    for condition_set in selected:
        ready, missing = base.artifacts_ready(run_dir, condition_set)
        if ready and not args.force:
            print(f"skip existing artifacts: {condition_set}", flush=True)
            continue
        if missing and not args.force:
            print(f"build missing artifacts for {condition_set}: {[str(p) for p in missing]}", flush=True)
        base.train_one(
            condition_set,
            condition_data[condition_set],
            context,
            args,
            run_dir,
            device,
            torch,
            TensorDataset,
            DataLoader,
            WeightedRandomSampler,
        )
    base.write_run_summary(run_dir, args, selected, condition_data)
    print("done:", run_dir, flush=True)
    return 0


def build_orange_readout_features(raw: np.lib.npyio.NpzFile, residual_mode: str) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    pop = np.asarray(raw["pop_t"], dtype=np.float32)
    times = np.asarray(raw["times"], dtype=np.float32)
    t_idx = int(np.argmin(np.abs(times - 10.0)))
    t_value = float(times[t_idx])
    if abs(t_value - 10.0) > 1e-4:
        raise ValueError(f"10 ps readout unavailable; nearest time is {t_value}")

    source_site1_10 = pop[:, t_idx, 0]
    early_trap_10 = pop[:, t_idx, 7]
    loss_10 = pop[:, t_idx, 8]
    site_sum_10 = pop[:, t_idx, :7].sum(axis=1)

    if residual_mode == "site2_to_site7":
        residual_10 = pop[:, t_idx, 1:7].sum(axis=1)
        residual_definition = "sum(site2..site7 population at 10 ps)"
    elif residual_mode == "all_sites":
        residual_10 = site_sum_10
        residual_definition = "sum(site1..site7 population at 10 ps)"
    elif residual_mode == "one_minus_trap_loss":
        residual_10 = 1.0 - early_trap_10 - loss_10
        residual_definition = "1 - trap_10 - loss_10"
    else:
        raise ValueError(f"unknown residual_mode: {residual_mode}")

    features = np.stack([residual_10, source_site1_10, early_trap_10], axis=1).astype(np.float32)
    names = ["residual_10", "source_site1_10", "early_trap_10"]
    meta: dict[str, Any] = {
        "time_index": t_idx,
        "time_ps": t_value,
        "residual_mode": residual_mode,
        "residual_definition": residual_definition,
        "pop_t_channel_convention": "site1..site7, trap, loss",
        "loss_10_median": float(np.nanmedian(loss_10)),
        "site_sum_10_median": float(np.nanmedian(site_sum_10)),
    }
    return features, names, meta


def build_condition_datasets(context: dict[str, Any], args: argparse.Namespace) -> dict[str, base.ConditionDataset]:
    cfast_raw = np.asarray(context["cfast_raw"], dtype=np.float32)
    cfast_names = list(context["cfast_names"])
    cl1 = np.asarray(context["cl1"], dtype=np.float32)
    orange_raw = np.asarray(context["orange_raw"], dtype=np.float32)
    orange_names = list(context["orange_names"])

    data = {
        "ORANGE3_ONLY": base.make_condition_dataset(
            "ORANGE3_ONLY",
            orange_raw,
            orange_names,
            context,
            flag_names=set(),
            dynamic_label=None,
            balance_label=None,
        ),
        "CFAST_ONLY": base.make_condition_dataset(
            "CFAST_ONLY",
            cfast_raw,
            cfast_names,
            context,
            flag_names={"t80_observed_flag"},
            dynamic_label=None,
            balance_label=None,
        ),
        "CFAST_ORANGE3": base.make_condition_dataset(
            "CFAST_ORANGE3",
            np.concatenate([cfast_raw, orange_raw], axis=1).astype(np.float32),
            cfast_names + orange_names,
            context,
            flag_names={"t80_observed_flag"},
            dynamic_label=None,
            balance_label=None,
        ),
        "CFAST_CL1_ORANGE3": base.make_condition_dataset(
            "CFAST_CL1_ORANGE3",
            np.concatenate([cfast_raw, cl1, orange_raw], axis=1).astype(np.float32),
            cfast_names + ["c_l1"] + orange_names,
            context,
            flag_names={"t80_observed_flag"},
            dynamic_label=None,
            balance_label=None,
        ),
    }
    for ds in data.values():
        ds.targets_raw, ds.targets_norm = base.make_target_conditions(ds, context, args)
    return data


def write_metadata(
    out_root: Path,
    prepared: Path,
    args: argparse.Namespace,
    context: dict[str, Any],
    condition_data: dict[str, base.ConditionDataset],
    selected: list[str],
) -> None:
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "feature": context["orange_names"],
            "median_train": np.nanmedian(context["orange_raw"][context["train_idx"]], axis=0),
            "q10_train": np.nanquantile(context["orange_raw"][context["train_idx"]], 0.10, axis=0),
            "q90_train": np.nanquantile(context["orange_raw"][context["train_idx"]], 0.90, axis=0),
        }
    ).to_csv(metadata_dir / "orange_readout_feature_summary.csv", index=False)

    condition_rows = []
    for name in selected:
        ds = condition_data[name]
        condition_rows.append(
            {
                "condition_set": name,
                "condition_dim": int(ds.raw.shape[1]),
                "n_targets": int(len(ds.targets_norm)),
                "condition_names": json.dumps(ds.names, ensure_ascii=False),
                "target_names": json.dumps(sorted(ds.targets_norm), ensure_ascii=False),
            }
        )
    pd.DataFrame(condition_rows).to_csv(metadata_dir / "condition_design.csv", index=False)

    # Keep dynamic submode metadata available for downstream diversity diagnostics,
    # but do not expose it as a condition in this ablation.
    context["dynamic_submode_summary"].to_csv(metadata_dir / "dynamic_submode_summary.csv", index=False)
    pd.DataFrame(
        {
            "dynamic_submode": context["dynamic_label"],
            "priority_group": context["priority_group"],
            "sample_index": context["raw"]["sample_index"].astype(np.int64),
        }
    ).to_csv(metadata_dir / "dynamic_submode_assignments.csv", index=False)

    manifest = {
        "purpose": CONDITION_PURPOSE,
        "prepared": str(prepared),
        "out_root": str(out_root),
        "selected_condition_sets": selected,
        "orange_features": context["orange_names"],
        "orange_meta": context["orange_meta"],
        "args": base.clean_json(vars(args)),
        "important_files": {
            "condition_design": str(metadata_dir / "condition_design.csv"),
            "orange_readout_feature_summary": str(metadata_dir / "orange_readout_feature_summary.csv"),
            "dynamic_submode_summary_for_eval": str(metadata_dir / "dynamic_submode_summary.csv"),
        },
        "interpretation_boundary": (
            "Orange readouts are compact early trajectory outcome conditions. "
            "They are not full dynamic-cluster labels and do not prove mechanism causality."
        ),
    }
    (metadata_dir / "manifest.json").write_text(
        json.dumps(base.clean_json(manifest), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_readme(out_root, selected)


def write_readme(out_root: Path, selected: list[str]) -> None:
    lines = [
        "# 20260620 H27 Early-Readout Orange Flow",
        "",
        "Purpose: train conditional RealNVP ablations using the three orange early-readout variables from the dynamic-PC/D-classifier audit.",
        "",
        "## Condition Sets",
        "",
        *[f"- `{name}`" for name in selected],
        "",
        "## Orange Variables",
        "",
        "- `residual_10`",
        "- `source_site1_10`",
        "- `early_trap_10`",
        "",
        "Default residual definition is `sum(site2..site7 population at 10 ps)` so it does not duplicate `source_site1_10` or `early_trap_10`.",
        "",
        "## Next Checks",
        "",
        "1. Compare `CFAST_ONLY` vs `CFAST_ORANGE3` / `CFAST_CL1_ORANGE3` test NLL.",
        "2. Run simulator validation for generated NPZ outputs.",
        "3. Run generated dynamic diversity audit and compare target match, dynamic effective dimension, best k, and largest cluster fraction.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

