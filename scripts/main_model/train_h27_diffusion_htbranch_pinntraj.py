#!/usr/bin/env python3
"""Train an H27 branch-conditioned DDPM with internal dynamic branch latents.

This is an additional ablation on top of ``train_h27_diffusion_pinntraj.py``.
The existing ``DDPM_HTBALPINNTRAJ`` model uses dynamic modes through auxiliary
losses and sampling guidance, but the denoiser itself only sees ``x_t``, time,
and the compact user-facing condition.  This script adds the missing structural
piece: an internal branch embedding is passed into the denoiser.

Boundary:
- The user-facing condition is still compact, e.g. ``CFAST_ORANGE3``.
- Dynamic branch labels are not exposed as requested generation conditions.
- During generation, a branch is sampled internally from the reference-mode
  prior ``pi_star`` and used as ``p_theta(H | condition, branch)``.
"""

from __future__ import annotations

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

import train_h27_diffusion_pinntraj as ddpm
import train_h27_early_readout_flow as orange
import train_h27_path_dynamic_flow as base
import train_h27_dynz_pinntraj_flow as pinnmod


METHOD = "DDPM_HTBRANCHPINNTRAJ"
DEFAULT_OUT_ROOT = Path("outputs/experiments/20260622_h27_htbranchpinntraj")
BRANCH_AUX_KEYS = ddpm.AUX_KEYS


def parse_args(argv: list[str] | None = None):
    args = ddpm.parse_args(argv)
    if args.out_root == ddpm.DEFAULT_OUT_ROOT:
        args.out_root = DEFAULT_OUT_ROOT
    default_methods = getattr(ddpm, "DEFAULT_METHODS", "")
    if str(args.methods) == str(default_methods):
        args.methods = METHOD
    # Additional branch-latent knobs.  Keep these as attributes so the old
    # parser remains compatible with the existing notebook/script.
    args.branch_embed_dim = int(getattr(args, "branch_embed_dim", 16))
    args.branch_balance_sampling = bool(getattr(args, "branch_balance_sampling", True))
    args.branch_balance_min_count = int(getattr(args, "branch_balance_min_count", 16))
    args.branch_balance_weight_power = float(getattr(args, "branch_balance_weight_power", 0.75))
    args.branch_kl_weight_scale = float(getattr(args, "branch_kl_weight_scale", 1.0))
    args.methods = ",".join([x.strip().upper() for x in str(args.methods).split(",") if x.strip()])
    return args


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
    context = ddpm.build_orange_context(raw, args)
    condition_data = orange.build_condition_datasets(context, args)
    if args.base_condition not in condition_data:
        raise KeyError(f"unknown base condition: {args.base_condition}")
    ds = condition_data[args.base_condition]
    target_pop = ddpm.build_target_pop_prototypes(context, args)
    mode_guidance = ddpm.build_htbal_mode_guidance(context, args)
    branch_info = build_branch_latent_info(mode_guidance, context, args)

    ddpm.write_metadata(out_root, prepared, args, context, ds, target_pop, mode_guidance)
    write_branch_metadata(out_root, prepared, args, ds, branch_info)

    print(f"prepared: {prepared}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"base_condition: {args.base_condition}", flush=True)
    print(f"methods: {args.methods}", flush=True)
    print(f"n_internal_branches: {branch_info['n_branches']}", flush=True)
    if args.metadata_only:
        print("metadata-only complete:", out_root / "metadata", flush=True)
        return 0

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(args.seed)
    device = base.choose_device(args.device, torch)
    print(f"device: {device}", flush=True)

    selected = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(selected) - {METHOD})
    if unknown:
        raise KeyError(f"{Path(__file__).name} only supports {METHOD}; unknown={unknown}")

    if float(args.pinn_loss_dyn_ce) <= 0.0:
        print(
            "note: PINN dynamic-class CE is disabled; branch signal enters the DDPM denoiser as an internal embedding.",
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
        state = ddpm.load_checkpoint(args.checkpoint, device, torch)
        if list(state.get("condition_names", [])) != list(ds.names):
            raise RuntimeError(
                "checkpoint condition names do not match the selected --base-condition; "
                f"checkpoint={state.get('condition_names')} selected={ds.names}"
            )
        if int(state.get("n_branches", branch_info["n_branches"])) != int(branch_info["n_branches"]):
            raise RuntimeError(
                "checkpoint branch count does not match the current reference branch map; "
                f"checkpoint={state.get('n_branches')} current={branch_info['n_branches']}"
            )
        method = str(state.get("method", METHOD))
        condition_set = str(state.get("condition_set", f"{args.base_condition}_{method}"))
        ddpm.apply_checkpoint_schedule_args(args, state)
        schedule = ddpm.make_diffusion_schedule(args, device, torch)
        model = build_branch_model_from_state(state, args, device, torch, nn)
        generate_targets_branch(
            condition_set,
            method,
            model,
            ds,
            context,
            target_pop,
            branch_info,
            pinn,
            args,
            run_dir,
            schedule,
            device,
            torch,
            F,
        )
        write_branch_run_summary(run_dir, args, selected, branch_info)
        print("generate-only complete:", run_dir, flush=True)
        return 0

    schedule = ddpm.make_diffusion_schedule(args, device, torch)
    for method in selected:
        condition_set = f"{args.base_condition}_{method}"
        ready, missing = ddpm.artifacts_ready(run_dir, condition_set)
        if ready and not args.force:
            print(f"skip existing artifacts: {condition_set}", flush=True)
            continue
        if missing and not args.force:
            print(f"build missing artifacts for {condition_set}: {[str(p) for p in missing]}", flush=True)
        train_one_branch(
            method,
            condition_set,
            ds,
            context,
            target_pop,
            mode_guidance,
            branch_info,
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
            WeightedRandomSampler,
        )

    write_branch_run_summary(run_dir, args, selected, branch_info)
    print("done:", run_dir, flush=True)
    return 0


def build_branch_latent_info(mode_guidance: dict[str, Any], context: dict[str, Any], args=None) -> dict[str, Any]:
    n = len(np.asarray(context["priority_group"]))
    row_group_id = np.asarray(mode_guidance.get("row_group_id", np.full(n, -1)), dtype=np.int64)
    row_mode_id = np.asarray(mode_guidance.get("row_mode_id", np.full(n, -1)), dtype=np.int64)
    target_times = np.asarray(context["raw"]["times"], dtype=np.float32)
    reference_times = load_htbranch_reference_times(args)

    branch_names = ["default"]
    branch_groups = ["default"]
    branch_local_modes = [-1]
    by_group: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = [
        {
            "global_branch_id": 0,
            "branch_name": "default",
            "target_group": "default",
            "local_mode_id": -1,
            "pi_star": 1.0,
            "reference_fraction": np.nan,
        }
    ]
    row_branch_id = np.zeros(n, dtype=np.int64)

    for group in mode_guidance.get("groups", []):
        info = mode_guidance["by_group"][group]
        mode_names = [str(x) for x in info["mode_names"]]
        pi_star = np.asarray(info["pi_star"], dtype=np.float64)
        pi_star = pi_star / max(float(pi_star.sum()), 1e-12)
        ref_prior = np.asarray(info.get("ref_prior", pi_star), dtype=np.float64)
        if len(ref_prior) != len(pi_star):
            ref_prior = pi_star.copy()
        local_to_global: list[int] = []
        for local_id, mode_name in enumerate(mode_names):
            global_id = len(branch_names)
            branch_name = f"{group}:{mode_name}"
            branch_names.append(branch_name)
            branch_groups.append(group)
            branch_local_modes.append(int(local_id))
            local_to_global.append(global_id)
            rows.append(
                {
                    "global_branch_id": int(global_id),
                    "branch_name": branch_name,
                    "target_group": group,
                    "local_mode_id": int(local_id),
                    "pi_star": float(pi_star[local_id]),
                    "reference_fraction": float(ref_prior[local_id]) if local_id < len(ref_prior) else np.nan,
                }
            )
        gid = int(info["group_id"])
        mask = (row_group_id == gid) & (row_mode_id >= 0)
        for local_id, global_id in enumerate(local_to_global):
            row_branch_id[mask & (row_mode_id == local_id)] = int(global_id)
        by_group[group] = {
            "global_ids": np.asarray(local_to_global, dtype=np.int64),
            "mode_names": mode_names,
            "pi_star": pi_star.astype(np.float32),
            "proto_pop": align_population_time_grid(np.asarray(info["proto_pop"], dtype=np.float32), target_times, reference_times),
        }

    if len(branch_names) <= 1:
        raise RuntimeError("No internal HT branch modes were built. Check --htbal-reference-dir and target guidance groups.")

    return {
        "n_branches": int(len(branch_names)),
        "branch_names": branch_names,
        "branch_groups": branch_groups,
        "branch_local_modes": branch_local_modes,
        "row_branch_id": row_branch_id,
        "row_group_id": row_group_id,
        "row_mode_id": row_mode_id,
        "by_group": by_group,
        "summary": pd.DataFrame(rows),
    }


def load_htbranch_reference_times(args) -> np.ndarray | None:
    if args is None:
        return None
    reference_dir = Path(getattr(args, "htbal_reference_dir", ""))
    traces_path = reference_dir / "npz" / "dynamic_condition_mode_traces.npz"
    if not traces_path.exists():
        return None
    try:
        with np.load(traces_path, allow_pickle=True) as z:
            if "tlist" in z:
                return np.asarray(z["tlist"], dtype=np.float32)
            if "times" in z:
                return np.asarray(z["times"], dtype=np.float32)
    except Exception:
        return None
    return None


def align_population_time_grid(pop: np.ndarray | None, target_times: np.ndarray, source_times: np.ndarray | None = None) -> np.ndarray | None:
    """Align reference/generated guidance trajectories to the compact PINN grid."""
    if pop is None:
        return None
    arr = np.asarray(pop, dtype=np.float32)
    if arr.ndim < 3:
        return arr
    target_times = np.asarray(target_times, dtype=np.float32)
    if arr.shape[1] == len(target_times):
        return arr.astype(np.float32)
    if source_times is None or len(source_times) != arr.shape[1]:
        t_min = 0.0 if float(np.nanmin(target_times)) > 0.0 else float(np.nanmin(target_times))
        t_max = float(np.nanmax(target_times))
        source_times = np.linspace(t_min, t_max, arr.shape[1], dtype=np.float32)
    else:
        source_times = np.asarray(source_times, dtype=np.float32)
    idx = np.asarray([int(np.nanargmin(np.abs(source_times - t))) for t in target_times], dtype=np.int64)
    aligned = arr[:, idx, :].astype(np.float32)
    denom = np.maximum(aligned.sum(axis=2, keepdims=True), 1e-8)
    return (aligned / denom).astype(np.float32)


def make_branch_diffusion_model(torch, nn, x_dim: int, c_dim: int, n_branches: int, branch_embed_dim: int, hidden: int, depth: int, time_features: int):
    class BranchDenoiser(nn.Module):
        def __init__(self):
            super().__init__()
            self.x_dim = int(x_dim)
            self.c_dim = int(c_dim)
            self.n_branches = int(n_branches)
            self.branch_embed_dim = int(branch_embed_dim)
            self.time_features = int(time_features)
            self.branch_embedding = nn.Embedding(self.n_branches, self.branch_embed_dim)
            in_dim = self.x_dim + self.c_dim + self.time_features + self.branch_embed_dim
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

        def forward(self, x_t, t_index, c, branch_id):
            if t_index.dtype in (torch.int32, torch.int64, torch.long):
                denom = max(1.0, float(int(self.training_steps) - 1))
                t = t_index.to(dtype=x_t.dtype)[:, None] / denom
            else:
                t = t_index.to(dtype=x_t.dtype)
                if t.ndim == 1:
                    t = t[:, None]
            b = branch_id.long().clamp(0, self.n_branches - 1)
            b_emb = self.branch_embedding(b)
            return self.net(torch.cat([x_t, self.time_embed(t), c, b_emb], dim=1))

    return BranchDenoiser()


def attach_training_steps(model, steps: int):
    model.training_steps = int(steps)
    return model


def build_branch_model_from_state(state: dict[str, Any], args, device, torch, nn):
    model = attach_training_steps(
        make_branch_diffusion_model(
            torch,
            nn,
            int(state["x_dim"]),
            int(state["c_dim"]),
            int(state.get("n_branches", 1)),
            int(state.get("branch_embed_dim", getattr(args, "branch_embed_dim", 16))),
            int(state.get("hidden", args.hidden)),
            int(state.get("depth", args.depth)),
            int(state.get("time_features", args.time_features)),
        ),
        int(state.get("diffusion_steps", args.diffusion_steps)),
    ).to(device)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model


def train_one_branch(
    method: str,
    condition_set: str,
    ds,
    context,
    target_pop,
    mode_guidance,
    branch_info,
    pinn,
    args,
    run_dir: Path,
    schedule,
    device,
    torch,
    nn,
    F,
    TensorDataset,
    DataLoader,
    WeightedRandomSampler,
) -> None:
    x = np.asarray(context["h_norm"], dtype=np.float32)
    c = np.asarray(ds.norm, dtype=np.float32)
    pop = np.asarray(context["raw"]["pop_t"], dtype=np.float32)
    target_pop_arr, target_weight = ddpm.make_row_target_guidance(pop, context, target_pop, args)
    row_group_id = np.asarray(branch_info["row_group_id"], dtype=np.int64)
    row_mode_id = np.asarray(branch_info["row_mode_id"], dtype=np.int64)
    row_branch_id = np.asarray(branch_info["row_branch_id"], dtype=np.int64)
    train_idx = ddpm.limited_indices(np.asarray(context["train_idx"], dtype=np.int64), args.max_train_samples, args.seed)
    val_idx = ddpm.limited_indices(np.asarray(context["val_idx"], dtype=np.int64), args.max_val_samples, args.seed + 1)
    test_idx = np.asarray(context["test_idx"], dtype=np.int64)

    model = attach_training_steps(
        make_branch_diffusion_model(
            torch,
            nn,
            x.shape[1],
            c.shape[1],
            int(branch_info["n_branches"]),
            int(args.branch_embed_dim),
            args.hidden,
            args.depth,
            args.time_features,
        ),
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
        torch.tensor(row_branch_id[train_idx]),
    )
    sampler = None
    shuffle = True
    if args.branch_balance_sampling:
        branch_train = row_branch_id[train_idx]
        counts = np.bincount(branch_train, minlength=int(branch_info["n_branches"])).astype(np.float64)
        counts[counts <= 0] = 1.0
        weights = 1.0 / np.power(counts[branch_train], float(args.branch_balance_weight_power))
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(train_idx), replacement=True)
        shuffle = False
    loader = DataLoader(tensors, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)

    val_x = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    val_c = torch.tensor(c[val_idx], dtype=torch.float32, device=device)
    val_pop = torch.tensor(pop[val_idx], dtype=torch.float32, device=device)
    val_target_pop = torch.tensor(target_pop_arr[val_idx], dtype=torch.float32, device=device)
    val_target_weight = torch.tensor(target_weight[val_idx], dtype=torch.float32, device=device)
    val_group_id = torch.tensor(row_group_id[val_idx], dtype=torch.long, device=device)
    val_mode_id = torch.tensor(row_mode_id[val_idx], dtype=torch.long, device=device)
    val_branch_id = torch.tensor(row_branch_id[val_idx], dtype=torch.long, device=device)
    val_aux_keep = ddpm.make_val_aux_keep(len(val_idx), args.val_aux_max, args.seed)

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
        f"H_dim={x.shape[1]} C_dim={c.shape[1]} branches={branch_info['n_branches']} branch_balanced={args.branch_balance_sampling}",
        flush=True,
    )
    for epoch in epoch_iter:
        model.train()
        warm = ddpm.aux_weight(epoch, args.aux_start_epoch, args.aux_warmup_epochs)
        totals = {"loss": 0.0, "eps": 0.0, "x0": 0.0, **{key: 0.0 for key in BRANCH_AUX_KEYS if key != "weighted"}}
        seen = 0
        for xb, cb, pb, tb, wb, gb, mb, bb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            pb = pb.to(device)
            gb = gb.to(device)
            mb = mb.to(device)
            bb = bb.to(device)
            t_idx = torch.randint(0, int(args.diffusion_steps), (len(xb),), dtype=torch.long, device=device)
            x_t, noise = ddpm.q_sample(xb, t_idx, schedule, torch)
            eps_pred = model(x_t, t_idx, cb, bb)
            eps_loss = F.mse_loss(eps_pred, noise)
            x0_pred = ddpm.predict_x0_from_eps(x_t, t_idx, eps_pred, schedule, args.x0_clip)
            x0_loss = F.mse_loss(x0_pred, xb)
            aux = ddpm.zero_aux(xb)
            if warm > 0.0:
                aux = ddpm.trajectory_aux_loss(x0_pred, pb, pinn, args, torch, F)
                branch_aux = htbranch_mode_aux_loss(x0_pred, gb, mb, mode_guidance, pinn, args, torch, F)
                aux = ddpm.add_aux(aux, branch_aux)
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
            for key in [key for key in BRANCH_AUX_KEYS if key != "weighted"]:
                totals[key] += float(aux[key].detach().cpu()) * n

        model.eval()
        with torch.no_grad():
            val = evaluate_branch_losses(
                model,
                val_x,
                val_c,
                val_branch_id,
                val_pop,
                pinn,
                args,
                schedule,
                torch,
                F,
                val_aux_keep,
                group_id=val_group_id,
                mode_id=val_mode_id,
                mode_guidance=mode_guidance,
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
                    "model_type": "branch_conditional_ddpm_denoiser",
                    "method": method,
                    "condition_set": condition_set,
                    "base_condition": args.base_condition,
                    "condition_names": ds.names,
                    "condition_mu": ds.mu.tolist(),
                    "condition_sd": ds.sd.tolist(),
                    "condition_flag_mask": ds.flag_mask.astype(bool).tolist(),
                    "x_dim": int(x.shape[1]),
                    "c_dim": int(c.shape[1]),
                    "n_branches": int(branch_info["n_branches"]),
                    "branch_embed_dim": int(args.branch_embed_dim),
                    "branch_names": list(branch_info["branch_names"]),
                    "branch_groups": list(branch_info["branch_groups"]),
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
            "train_branch_mode": totals["mode"] / max(seen, 1),
            "val_eps_mse": float(val["eps"].detach().cpu()),
            "val_x0_mse": float(val["x0"].detach().cpu()),
            "val_aux_weighted": float(val["weighted"].detach().cpu()),
            "val_pinn_traj": float(val["traj"].detach().cpu()),
            "val_pinn_feature": float(val["feature"].detach().cpu()),
            "val_pinn_phys": float(val["phys"].detach().cpu()),
            "val_support": float(val["support"].detach().cpu()),
            "val_target_band": float(val["target_band"].detach().cpu()),
            "val_branch_mode": float(val["mode"].detach().cpu()),
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
                f"train_eps={row['train_eps_mse']:.5f} branch={row['train_branch_mode']:.5f} "
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
    ddpm.plot_ddpm_loss(hist, run_dir / "figures" / f"{condition_set}_loss_curve.png")

    state = ddpm.load_checkpoint(best_path, device, torch)
    model.load_state_dict(state["state_dict"])
    model.eval()
    metrics = evaluate_split_metrics_branch(
        condition_set,
        method,
        model,
        x,
        c,
        pop,
        context,
        row_branch_id,
        row_group_id,
        row_mode_id,
        mode_guidance,
        pinn,
        args,
        schedule,
        device,
        torch,
        F,
    )
    pd.DataFrame(metrics).to_csv(run_dir / f"{condition_set}_test_metrics.csv", index=False)
    generated_summary = generate_targets_branch(
        condition_set,
        method,
        model,
        ds,
        context,
        target_pop,
        branch_info,
        pinn,
        args,
        run_dir,
        schedule,
        device,
        torch,
        F,
    )
    write_branch_condition_report(run_dir, condition_set, method, args, ds, metrics, generated_summary, branch_info)


def htbranch_mode_aux_loss(x0_pred, group_id, mode_id, mode_guidance: dict[str, Any], pinn, args, torch, F):
    aux = ddpm.zero_aux(x0_pred)
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
    min_count = max(1, int(args.branch_balance_min_count))

    for group in mode_guidance["groups"]:
        info = mode_guidance["by_group"][group]
        gid = int(info["group_id"])
        mask = (group_id == gid) & (mode_id >= 0)
        n_group = int(mask.detach().sum().cpu())
        if n_group <= 0:
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
        band_terms.append((torch.relu(feat_low[None, :] - feat_g).pow(2) + torch.relu(feat_g - feat_high[None, :]).pow(2)).mean())
        if n_group >= max(min_count, int(2 * len(pi_star))):
            balance_terms.append((pi_star * (torch.log(pi_star.clamp_min(eps)) - torch.log(q_bar))).sum())
            floor_terms.append(torch.relu(float(args.htbal_prior_min) - q_bar).pow(2).sum())

    if not mode_terms:
        return aux

    target_band = torch.stack(band_terms).mean()
    mode_proto = torch.stack(mode_terms).mean()
    mode_assign = torch.stack(assign_terms).mean()
    if balance_terms:
        mode_balance = torch.stack(balance_terms).mean()
        mode_floor = torch.stack(floor_terms).mean()
    else:
        mode_balance = x0_pred.new_zeros(())
        mode_floor = x0_pred.new_zeros(())
    mode_total = (
        float(args.lambda_htbal_mode_proto) * mode_proto
        + float(args.lambda_htbal_mode_assign) * mode_assign
        + float(args.branch_kl_weight_scale) * float(args.lambda_htbal_mode_balance) * mode_balance
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


def evaluate_branch_losses(
    model,
    x,
    c,
    branch_id,
    pop,
    pinn,
    args,
    schedule,
    torch,
    F,
    keep_idx: np.ndarray | None = None,
    group_id=None,
    mode_id=None,
    mode_guidance: dict[str, Any] | None = None,
):
    if keep_idx is not None:
        x = x[keep_idx]
        c = c[keep_idx]
        branch_id = branch_id[keep_idx]
        pop = pop[keep_idx]
        if group_id is not None:
            group_id = group_id[keep_idx]
        if mode_id is not None:
            mode_id = mode_id[keep_idx]
    if len(x) == 0:
        return ddpm.zero_aux(c)
    t_idx = torch.randint(0, int(args.diffusion_steps), (len(x),), dtype=torch.long, device=x.device)
    x_t, noise = ddpm.q_sample(x, t_idx, schedule, torch)
    eps_pred = model(x_t, t_idx, c, branch_id)
    eps = F.mse_loss(eps_pred, noise)
    x0_pred = ddpm.predict_x0_from_eps(x_t, t_idx, eps_pred, schedule, args.x0_clip)
    x0 = F.mse_loss(x0_pred, x)
    aux = ddpm.zero_aux(x)
    if pinn is not None:
        aux = ddpm.trajectory_aux_loss(x0_pred, pop, pinn, args, torch, F)
        if group_id is not None and mode_id is not None and mode_guidance is not None:
            aux = ddpm.add_aux(aux, htbranch_mode_aux_loss(x0_pred, group_id, mode_id, mode_guidance, pinn, args, torch, F))
    return {"eps": eps, "x0": x0, **aux}


def evaluate_split_metrics_branch(
    condition_set,
    method,
    model,
    x,
    c,
    pop,
    context,
    row_branch_id,
    row_group_id,
    row_mode_id,
    mode_guidance,
    pinn,
    args,
    schedule,
    device,
    torch,
    F,
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
            bb = torch.tensor(row_branch_id[rows], dtype=torch.long, device=device)
            pb = torch.tensor(pop[rows], dtype=torch.float32, device=device)
            gb = torch.tensor(row_group_id[rows], dtype=torch.long, device=device)
            mb = torch.tensor(row_mode_id[rows], dtype=torch.long, device=device)
            with torch.no_grad():
                losses = evaluate_branch_losses(
                    model,
                    xb,
                    cb,
                    bb,
                    pb,
                    pinn,
                    args,
                    schedule,
                    torch,
                    F,
                    group_id=gb,
                    mode_id=mb,
                    mode_guidance=mode_guidance,
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
                    "branch_mode": float(losses["mode"].detach().cpu()),
                    "score": float((losses["eps"] + 0.05 * losses["x0"] + losses["weighted"]).detach().cpu()),
                }
            )
    return metrics


def sample_ddim_branch(model, c_batch, branch_batch, target_pop_batch, pinn, args, schedule, device, torch, F, guidance_scale: float, guidance_start_frac: float):
    n = len(c_batch)
    x = torch.randn((n, 28), dtype=torch.float32, device=device)
    t_steps = ddpm.ddim_timesteps(int(args.diffusion_steps), int(args.sample_steps))
    model.eval()
    for pos, t_value in enumerate(t_steps):
        t_idx = torch.full((n,), int(t_value), dtype=torch.long, device=device)
        with torch.no_grad():
            eps = model(x, t_idx, c_batch, branch_batch)
            x0 = ddpm.predict_x0_from_eps(x, t_idx, eps, schedule, args.x0_clip)
        scale_now = ddpm.step_sample_guidance_scale(pos, len(t_steps), args, guidance_scale, guidance_start_frac)
        if scale_now > 0.0 and pinn is not None and target_pop_batch is not None:
            x0 = ddpm.guide_x0_with_surrogate(x0, target_pop_batch, pinn, args, torch, F, scale_now)
            abar_t = schedule["alpha_bars"][int(t_value)]
            eps = (x - torch.sqrt(abar_t) * x0) / torch.sqrt(torch.clamp(1.0 - abar_t, min=1e-12))
        prev_t = int(t_steps[pos + 1]) if pos + 1 < len(t_steps) else -1
        x = ddpm.ddim_step(x, x0, eps, int(t_value), prev_t, args, schedule, torch)
    return x


def generate_targets_branch(condition_set, method, model, ds, context, target_pop, branch_info, pinn, args, run_dir, schedule, device, torch, F):
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
    rng = np.random.default_rng(args.seed + 1701)
    guidance_scale = float(args.htbal_sample_guidance_scale)
    guidance_start_frac = float(args.htbal_sample_guidance_start_frac)

    for target_name, cvec in ds.targets_norm.items():
        group = ddpm.infer_base_target(target_name, args)
        if group is None:
            continue
        if "median" in strategies:
            c_batch = np.repeat(cvec[None, :].astype(np.float32), int(args.n_generate), axis=0)
            branch_batch, pop_batch, branch_names = make_generation_branch_batch(group, len(c_batch), target_pop, branch_info, rng)
            write_generated_batch_branch(
                condition_set,
                target_name,
                group,
                "median",
                c_batch,
                branch_batch,
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
                internal_branches=branch_names,
            )
        if "mixture" in strategies and target_name == group:
            c_batch, _source_pop, source_rows = ddpm.make_condition_mixture(group, ds, context, args, rng)
            branch_batch, pop_batch, branch_names = make_generation_branch_batch(group, len(c_batch), target_pop, branch_info, rng)
            write_generated_batch_branch(
                condition_set,
                f"{group}_mixture",
                group,
                "mixture",
                c_batch,
                branch_batch,
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
                internal_branches=branch_names,
            )

    np.savez_compressed(run_dir / f"{condition_set}_generated_samples.npz", **generated)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / f"{condition_set}_generated_physical_summary.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(run_dir / f"{condition_set}_generation_provenance.csv", index=False)
    return summary


def make_generation_branch_batch(group: str, n: int, target_pop: dict[str, np.ndarray], branch_info: dict[str, Any], rng: np.random.Generator):
    if group in branch_info.get("by_group", {}):
        info = branch_info["by_group"][group]
        probs = np.asarray(info["pi_star"], dtype=np.float64)
        probs = probs / max(float(probs.sum()), 1e-12)
        local = rng.choice(len(probs), size=int(n), replace=True, p=probs)
        global_ids = np.asarray(info["global_ids"], dtype=np.int64)[local]
        proto = np.asarray(info["proto_pop"], dtype=np.float32)[local]
        names = [str(info["mode_names"][int(i)]) for i in local]
        return global_ids.astype(np.int64), proto.astype(np.float32), names
    proto = target_pop.get(group)
    pop = None if proto is None else np.repeat(proto[None, :, :].astype(np.float32), int(n), axis=0)
    return np.zeros(int(n), dtype=np.int64), pop, ["default"] * int(n)


def write_generated_batch_branch(
    condition_set,
    target_key,
    base_target,
    strategy,
    c_batch,
    branch_batch,
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
    source_rows=None,
    internal_branches=None,
):
    xs: list[np.ndarray] = []
    pop_batch = align_population_time_grid(
        pop_batch,
        np.asarray(context["raw"]["times"], dtype=np.float32),
        load_htbranch_reference_times(args),
    )
    bs = int(args.batch_size)
    for start in range(0, len(c_batch), bs):
        cb = torch.tensor(c_batch[start : start + bs], dtype=torch.float32, device=device)
        bb = torch.tensor(branch_batch[start : start + bs], dtype=torch.long, device=device)
        if pop_batch is not None and pinn is not None:
            pb = torch.tensor(pop_batch[start : start + bs], dtype=torch.float32, device=device)
        else:
            pb = None
        xg = sample_ddim_branch(model, cb, bb, pb, pinn, args, schedule, device, torch, F, guidance_scale, guidance_start_frac)
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
            "uses_branch_embedding": True,
            "uses_pinntraj_signal": bool(pinn is not None),
            "sample_guidance_scale": float(guidance_scale) if pinn is not None else 0.0,
            **base.physical_summary(x_gen_raw, train_h_raw),
        }
    )
    if source_rows is None:
        source_rows = np.full(len(c_batch), -1, dtype=np.int64)
    if internal_branches is None:
        internal_branches = ["default"] * len(c_batch)
    for i, src in enumerate(source_rows):
        provenance_rows.append(
            {
                "condition_set": condition_set,
                "target": target_key,
                "base_target": base_target,
                "strategy": strategy,
                "generated_index": int(i),
                "source_condition_row": int(src),
                "internal_branch_id": int(branch_batch[i]),
                "internal_branch": str(internal_branches[i]) if i < len(internal_branches) else "",
            }
        )
    print(f"generated {key}: {x_gen_raw.shape}", flush=True)


def write_branch_metadata(out_root: Path, prepared: Path, args, ds, branch_info: dict[str, Any]) -> None:
    meta = out_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    branch_info["summary"].to_csv(meta / "htbranch_internal_branch_summary.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(branch_info["row_branch_id"]), dtype=np.int64),
            "internal_branch_id": branch_info["row_branch_id"].astype(np.int64),
            "internal_group_id": branch_info["row_group_id"].astype(np.int64),
            "internal_mode_id": branch_info["row_mode_id"].astype(np.int64),
        }
    ).to_csv(meta / "htbranch_row_assignments.csv", index=False)
    manifest = {
        "purpose": "Additional H27 DDPM ablation with real internal branch embeddings in the denoiser.",
        "prepared": str(prepared),
        "out_root": str(out_root),
        "method": METHOD,
        "base_condition": args.base_condition,
        "condition_dim": int(ds.raw.shape[1]),
        "branch_embed_dim": int(args.branch_embed_dim),
        "n_branches": int(branch_info["n_branches"]),
        "branch_boundary": (
            "Branch ids are internal latents. They are used as denoiser embeddings and sampled internally at generation time; "
            "they are not exposed as user-facing generation conditions."
        ),
        "important_files": {
            "branch_summary": str(meta / "htbranch_internal_branch_summary.csv"),
            "row_assignments": str(meta / "htbranch_row_assignments.csv"),
        },
        "args": base.clean_json(vars(args)),
    }
    (meta / "htbranch_manifest.json").write_text(json.dumps(base.clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def write_branch_condition_report(run_dir, condition_set, method, args, ds, metrics, generated_summary, branch_info):
    metric_df = pd.DataFrame(metrics)
    lines = [
        f"# {condition_set} H27 branch-conditioned diffusion run",
        "",
        f"- method: `{method}`",
        f"- base_condition: `{args.base_condition}`",
        f"- condition_dim: `{ds.raw.shape[1]}`",
        f"- n_internal_branches: `{branch_info['n_branches']}`",
        f"- branch_embed_dim: `{args.branch_embed_dim}`",
        f"- user-facing branch condition: `False`",
        "",
        "## Test Metrics",
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
        "This is the true internal-branch ablation: branch embeddings enter the denoiser, but branch ids are still sampled internally. Final success must be judged by simulator validation and dynamic-reference assignment.",
    ]
    (run_dir / "reports" / f"{condition_set}_run_report_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_branch_run_summary(run_dir: Path, args, selected_methods: list[str], branch_info: dict[str, Any]) -> None:
    metric_paths = sorted(run_dir.glob("*_test_metrics.csv"))
    if metric_paths:
        metrics = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)
        metrics.to_csv(run_dir / "all_test_metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "base_condition": args.base_condition,
        "methods": selected_methods,
        "n_internal_branches": int(branch_info["n_branches"]),
        "branch_names": list(branch_info["branch_names"]),
        "generated_files": [str(p) for p in sorted(run_dir.glob("*_generated_samples.npz"))],
        "checkpoint_files": [str(p) for p in sorted((run_dir / "checkpoints").glob("*_best.pt"))],
        "args": base.clean_json(vars(args)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(base.clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

