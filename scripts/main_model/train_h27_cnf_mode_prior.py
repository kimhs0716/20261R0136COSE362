#!/usr/bin/env python3
"""Train H27 exact-likelihood CNF ablations with internal mode priors.

This script is the normalizing-flow counterpart to the HTBRANCH/PINNTRAJ
experiments.  It keeps the user-facing condition compact, but tests whether
rare high-transfer dynamic routes are better controlled through the flow
objective and latent sampling policy:

- CNF: standard compact-condition RealNVP with a standard Gaussian prior.
- CNF_WMODE: same model with target/mode-balanced weighted NLL.
- HTBAL_CNF_MIXPRIOR: branch-conditioned RealNVP trained with an internal
  dynamic-mode mixture prior and stratified branch sampling.
- HTBAL_CNF_GUIDED: the same mixture-prior flow, but generated latent samples
  are lightly refined with the PINN-lite continuous trajectory feature proxy.

Dynamic modes are privileged internal training/sampling signals only.  They are
not exposed as requested generation conditions.
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

import train_h27_diffusion_htbranch_pinntraj as htbranch
import train_h27_diffusion_pinntraj as ddpm
import train_h27_dynz_pinntraj_flow as pinnmod
import train_h27_early_readout_flow as orange
import train_h27_path_dynamic_flow as base


METHODS = ("CNF", "CNF_WMODE", "HTBAL_CNF_MIXPRIOR", "HTBAL_CNF_GUIDED")
MIX_METHODS = {"HTBAL_CNF_MIXPRIOR", "HTBAL_CNF_GUIDED"}
DEFAULT_OUT_ROOT = Path("outputs/experiments/20260622_h27_cnf_mode_prior")


def parse_args(argv: list[str] | None = None):
    flow_parser = argparse.ArgumentParser(add_help=False)
    flow_parser.add_argument("--flow-layers", "--layers", dest="flow_layers", type=int, default=8)
    flow_parser.add_argument("--scale-clip", type=float, default=1.8)
    flow_parser.add_argument("--branch-embed-dim", type=int, default=16)
    flow_parser.add_argument("--mode-weight-beta", type=float, default=0.5)
    flow_parser.add_argument("--mode-weight-min", type=float, default=0.25)
    flow_parser.add_argument("--mode-weight-max", type=float, default=5.0)
    flow_parser.add_argument("--high-target-weight", type=float, default=1.5)
    flow_parser.add_argument("--other-target-weight", type=float, default=1.0)
    flow_parser.add_argument(
        "--train-target-groups",
        default="",
        help="Optional comma-separated priority_group filter for train/val/test rows. Empty keeps all groups.",
    )
    flow_parser.add_argument("--mixprior-latent-separation", type=float, default=2.5)
    flow_parser.add_argument("--mixprior-latent-sigma", type=float, default=0.85)
    flow_parser.add_argument("--mixprior-assign-weight", type=float, default=0.10)
    flow_parser.add_argument("--mixprior-usage-weight", type=float, default=0.05)
    flow_parser.add_argument(
        "--mixprior-branch-nll-weight",
        type=float,
        default=0.0,
        help="Extra supervised branch NLL weight for rows with internal dynamic-mode branch labels.",
    )
    flow_parser.add_argument("--guided-steps", type=int, default=8)
    flow_parser.add_argument("--guided-step-size", type=float, default=0.05)
    flow_parser.add_argument("--guided-target-weight", type=float, default=1.0)
    flow_parser.add_argument("--guided-mode-weight", type=float, default=0.5)
    flow_parser.add_argument("--guided-support-weight", type=float, default=0.1)
    flow_parser.add_argument("--guided-prior-weight", type=float, default=0.1)
    flow_args, remaining = flow_parser.parse_known_args(argv)

    args = ddpm.parse_args(remaining)
    if args.out_root == ddpm.DEFAULT_OUT_ROOT:
        args.out_root = DEFAULT_OUT_ROOT
    default_methods = getattr(ddpm, "DEFAULT_METHODS", "")
    if str(args.methods) == str(default_methods):
        args.methods = ",".join(METHODS[:3])
    args.methods = ",".join([x.strip().upper() for x in str(args.methods).split(",") if x.strip()])
    for key, value in vars(flow_args).items():
        setattr(args, key, value)
    return args


class ConditionalRealNVP:
    def __init__(self, torch, nn, x_dim: int, c_dim: int, n_layers: int, hidden: int, scale_clip: float):
        self.torch = torch
        self.nn = nn
        self.x_dim = int(x_dim)
        self.c_dim = int(c_dim)
        self.scale_clip = float(scale_clip)
        masks = []
        for i in range(int(n_layers)):
            mask = np.zeros(self.x_dim, dtype=np.float32)
            mask[i % 2 :: 2] = 1.0
            masks.append(torch.tensor(mask))
        self.masks = masks
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.x_dim + self.c_dim, int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), 2 * self.x_dim),
                )
                for _ in range(int(n_layers))
            ]
        )
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def module(self):
        return self.nets

    def state_dict(self):
        return self.nets.state_dict()

    def load_state_dict(self, state):
        self.nets.load_state_dict(state)

    def to(self, device):
        self.nets.to(device)
        self.masks = [m.to(device) for m in self.masks]
        return self

    def _st(self, net, x_masked, c):
        out = net(self.torch.cat([x_masked, c], dim=1))
        s, t = out.chunk(2, dim=1)
        s = self.scale_clip * self.torch.tanh(s / self.scale_clip)
        return s, t

    def x_to_z(self, x, c):
        z = x
        logdet = self.torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for mask, net in zip(self.masks, self.nets):
            mask = mask.to(device=x.device, dtype=x.dtype)
            z_masked = z * mask
            s, t = self._st(net, z_masked, c)
            inv = 1.0 - mask
            z = z_masked + inv * ((z - t) * self.torch.exp(-s))
            logdet = logdet - (inv * s).sum(dim=1)
        return z, logdet

    def z_to_x(self, z, c):
        x = z
        for mask, net in reversed(list(zip(self.masks, self.nets))):
            mask = mask.to(device=x.device, dtype=x.dtype)
            x_masked = x * mask
            s, t = self._st(net, x_masked, c)
            inv = 1.0 - mask
            x = x_masked + inv * (x * self.torch.exp(s) + t)
        return x

    def log_prob(self, x, c):
        z, logdet = self.x_to_z(x, c)
        base_logp = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
        return base_logp + logdet

    def sample(self, c, n: int | None = None, z=None):
        if c.ndim == 1:
            if n is None:
                n = 1
            c = c[None, :].expand(int(n), -1)
        if z is None:
            z = self.torch.randn((c.shape[0], self.x_dim), dtype=c.dtype, device=c.device)
        return self.z_to_x(z, c)


class BranchConditionalRealNVP:
    def __init__(
        self,
        torch,
        nn,
        x_dim: int,
        c_dim: int,
        n_branches: int,
        branch_embed_dim: int,
        n_layers: int,
        hidden: int,
        scale_clip: float,
    ):
        self.torch = torch
        self.nn = nn
        self.x_dim = int(x_dim)
        self.c_dim = int(c_dim)
        self.n_branches = int(n_branches)
        self.branch_embed_dim = int(branch_embed_dim)
        self.scale_clip = float(scale_clip)
        self.branch_embedding = nn.Embedding(self.n_branches, self.branch_embed_dim)
        masks = []
        for i in range(int(n_layers)):
            mask = np.zeros(self.x_dim, dtype=np.float32)
            mask[i % 2 :: 2] = 1.0
            masks.append(torch.tensor(mask))
        self.masks = masks
        in_dim = self.x_dim + self.c_dim + self.branch_embed_dim
        self.nets = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(in_dim, int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), int(hidden)),
                    nn.SiLU(),
                    nn.Linear(int(hidden), 2 * self.x_dim),
                )
                for _ in range(int(n_layers))
            ]
        )
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self._module_container = nn.ModuleList([self.branch_embedding, self.nets])

    def module(self):
        return self._module_container

    def state_dict(self):
        return {"branch_embedding": self.branch_embedding.state_dict(), "nets": self.nets.state_dict()}

    def load_state_dict(self, state):
        if "branch_embedding" in state and "nets" in state:
            self.branch_embedding.load_state_dict(state["branch_embedding"])
            self.nets.load_state_dict(state["nets"])
        else:
            self.nets.load_state_dict(state)

    def to(self, device):
        self._module_container.to(device)
        self.masks = [m.to(device) for m in self.masks]
        return self

    def _branch_context(self, branch_id, dtype):
        b = branch_id.long().clamp(0, self.n_branches - 1)
        return self.branch_embedding(b).to(dtype=dtype)

    def _st(self, net, x_masked, c, b_ctx):
        out = net(self.torch.cat([x_masked, c, b_ctx], dim=1))
        s, t = out.chunk(2, dim=1)
        s = self.scale_clip * self.torch.tanh(s / self.scale_clip)
        return s, t

    def x_to_z(self, x, c, branch_id):
        z = x
        b_ctx = self._branch_context(branch_id, x.dtype)
        logdet = self.torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for mask, net in zip(self.masks, self.nets):
            mask = mask.to(device=x.device, dtype=x.dtype)
            z_masked = z * mask
            s, t = self._st(net, z_masked, c, b_ctx)
            inv = 1.0 - mask
            z = z_masked + inv * ((z - t) * self.torch.exp(-s))
            logdet = logdet - (inv * s).sum(dim=1)
        return z, logdet

    def z_to_x(self, z, c, branch_id):
        x = z
        b_ctx = self._branch_context(branch_id, z.dtype)
        for mask, net in reversed(list(zip(self.masks, self.nets))):
            mask = mask.to(device=x.device, dtype=x.dtype)
            x_masked = x * mask
            s, t = self._st(net, x_masked, c, b_ctx)
            inv = 1.0 - mask
            x = x_masked + inv * (x * self.torch.exp(s) + t)
        return x

    def branch_log_prob(self, x, c, branch_id, latent_means, latent_sigma: float):
        z, logdet = self.x_to_z(x, c, branch_id)
        mu = latent_means[branch_id.long().clamp(0, self.n_branches - 1)].to(dtype=z.dtype, device=z.device)
        sigma = float(max(1e-6, latent_sigma))
        dz = (z - mu) / sigma
        base_logp = -0.5 * (dz * dz + math.log(2.0 * math.pi) + 2.0 * math.log(sigma)).sum(dim=1)
        return base_logp + logdet

    def sample(self, c, branch_id, latent_means, latent_sigma: float, z=None):
        if c.ndim == 1:
            c = c[None, :]
        if not self.torch.is_tensor(branch_id):
            branch_id = self.torch.full((c.shape[0],), int(branch_id), dtype=self.torch.long, device=c.device)
        elif branch_id.ndim == 0:
            branch_id = branch_id[None].expand(c.shape[0])
        branch_id = branch_id.to(device=c.device)
        if z is None:
            mu = latent_means[branch_id.long().clamp(0, self.n_branches - 1)].to(dtype=c.dtype, device=c.device)
            z = mu + float(latent_sigma) * self.torch.randn((c.shape[0], self.x_dim), dtype=c.dtype, device=c.device)
        return self.z_to_x(z, c, branch_id)


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
    branch_info = htbranch.build_branch_latent_info(mode_guidance, context, args)

    selected = [x.strip().upper() for x in args.methods.split(",") if x.strip()]
    unknown = sorted(set(selected) - set(METHODS))
    if unknown:
        raise KeyError(f"unknown methods: {unknown}; allowed={METHODS}")

    write_metadata(out_root, prepared, args, context, ds, mode_guidance, branch_info, selected)

    print(f"prepared: {prepared}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"base_condition: {args.base_condition}", flush=True)
    print(f"methods: {selected}", flush=True)
    print(f"n_internal_branches: {branch_info['n_branches']}", flush=True)
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

    pinn = None
    if "HTBAL_CNF_GUIDED" in selected:
        if float(args.pinn_loss_dyn_ce) <= 0.0:
            print("note: guided CNF uses PINN-lite continuous features, not dynamic-class CE.", flush=True)
        pinn = pinnmod.train_or_load_pinn_surrogate(context, args, run_dir, device, torch, TensorDataset, DataLoader)
        print("pinn trajectory surrogate ready", flush=True)

    for method in selected:
        condition_set = f"{args.base_condition}_{method}"
        ready, missing = base.artifacts_ready(run_dir, condition_set)
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
            mode_guidance,
            branch_info,
            pinn,
            args,
            run_dir,
            device,
            torch,
            nn,
            F,
            TensorDataset,
            DataLoader,
        )

    write_run_summary(run_dir, args, selected, branch_info)
    print("done:", run_dir, flush=True)
    return 0


def train_one(
    method: str,
    condition_set: str,
    ds,
    context,
    mode_guidance,
    branch_info,
    pinn,
    args,
    run_dir: Path,
    device,
    torch,
    nn,
    F,
    TensorDataset,
    DataLoader,
) -> None:
    x = np.asarray(context["h_norm"], dtype=np.float32)
    c = np.asarray(ds.norm, dtype=np.float32)
    priority = np.asarray(context["priority_group"]).astype(str)
    row_group_id = np.asarray(branch_info["row_group_id"], dtype=np.int64)
    row_mode_id = np.asarray(branch_info["row_mode_id"], dtype=np.int64)
    row_branch_id = np.asarray(branch_info["row_branch_id"], dtype=np.int64)
    train_idx_raw = filter_indices_by_priority(np.asarray(context["train_idx"], dtype=np.int64), priority, args.train_target_groups)
    val_idx_raw = filter_indices_by_priority(np.asarray(context["val_idx"], dtype=np.int64), priority, args.train_target_groups)
    test_idx = filter_indices_by_priority(np.asarray(context["test_idx"], dtype=np.int64), priority, args.train_target_groups)
    train_idx = ddpm.limited_indices(train_idx_raw, args.max_train_samples, args.seed)
    val_idx = ddpm.limited_indices(val_idx_raw, args.max_val_samples, args.seed + 1)
    if len(train_idx) == 0 or len(val_idx) == 0 or len(test_idx) == 0:
        raise RuntimeError(
            f"empty split after --train-target-groups={args.train_target_groups!r}: "
            f"train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}"
        )

    train_weight = compute_mode_weights(priority, row_group_id, row_mode_id, train_idx, mode_guidance, args).astype(np.float32)
    val_weight = compute_mode_weights(priority, row_group_id, row_mode_id, val_idx, mode_guidance, args).astype(np.float32)

    if method in MIX_METHODS:
        model = BranchConditionalRealNVP(
            torch,
            nn,
            x.shape[1],
            c.shape[1],
            int(branch_info["n_branches"]),
            int(args.branch_embed_dim),
            int(args.flow_layers),
            int(args.hidden),
            float(args.scale_clip),
        ).to(device)
        latent_means_np = make_latent_means(branch_info, x.shape[1], float(args.mixprior_latent_separation)).astype(np.float32)
    else:
        model = ConditionalRealNVP(torch, nn, x.shape[1], c.shape[1], int(args.flow_layers), int(args.hidden), float(args.scale_clip)).to(device)
        latent_means_np = np.zeros((1, x.shape[1]), dtype=np.float32)

    opt = torch.optim.AdamW(model.module().parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tensors = TensorDataset(
        torch.tensor(x[train_idx]),
        torch.tensor(c[train_idx]),
        torch.tensor(row_group_id[train_idx], dtype=torch.long),
        torch.tensor(row_mode_id[train_idx], dtype=torch.long),
        torch.tensor(row_branch_id[train_idx], dtype=torch.long),
        torch.tensor(train_weight, dtype=torch.float32),
    )
    loader = DataLoader(tensors, batch_size=args.batch_size, shuffle=True, drop_last=False)
    latent_means = torch.tensor(latent_means_np, dtype=torch.float32, device=device)
    val_pack = {
        "x": torch.tensor(x[val_idx], dtype=torch.float32, device=device),
        "c": torch.tensor(c[val_idx], dtype=torch.float32, device=device),
        "group": torch.tensor(row_group_id[val_idx], dtype=torch.long, device=device),
        "mode": torch.tensor(row_mode_id[val_idx], dtype=torch.long, device=device),
        "branch": torch.tensor(row_branch_id[val_idx], dtype=torch.long, device=device),
        "weight": torch.tensor(val_weight, dtype=torch.float32, device=device),
    }

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
        f"H_dim={x.shape[1]} C_dim={c.shape[1]} branches={branch_info['n_branches'] if method in MIX_METHODS else 0} "
        f"train_target_groups={target_group_filter_label(args.train_target_groups)}",
        flush=True,
    )

    for epoch in epoch_iter:
        model.module().train()
        totals = {"loss": 0.0, "nll": 0.0, "assign": 0.0, "usage": 0.0, "branch_nll": 0.0}
        seen = 0
        for xb, cb, gb, mb, bb, wb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            gb = gb.to(device)
            mb = mb.to(device)
            bb = bb.to(device)
            wb = wb.to(device)
            if method == "CNF":
                logp = model.log_prob(xb, cb)
                nll = -logp.mean()
                assign = usage = branch_nll = xb.new_zeros(())
                loss = nll
            elif method == "CNF_WMODE":
                logp = model.log_prob(xb, cb)
                nll_vec = -logp
                nll = nll_vec.mean()
                assign = usage = branch_nll = xb.new_zeros(())
                loss = (wb * nll_vec).mean()
            else:
                logp, assign, usage = mixprior_log_prob(model, xb, cb, gb, mb, mode_guidance, branch_info, latent_means, args, torch, F)
                nll_vec = -logp
                nll = nll_vec.mean()
                branch_nll = supervised_branch_nll(model, xb, cb, bb, latent_means, args, sample_weight=wb)
                loss = (
                    (wb * nll_vec).mean()
                    + float(args.mixprior_assign_weight) * assign
                    + float(args.mixprior_usage_weight) * usage
                    + float(args.mixprior_branch_nll_weight) * branch_nll
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module().parameters(), float(args.grad_clip))
            opt.step()
            n = len(xb)
            seen += n
            totals["loss"] += float(loss.detach().cpu()) * n
            totals["nll"] += float(nll.detach().cpu()) * n
            totals["assign"] += float(assign.detach().cpu()) * n
            totals["usage"] += float(usage.detach().cpu()) * n
            totals["branch_nll"] += float(branch_nll.detach().cpu()) * n

        model.module().eval()
        with torch.no_grad():
            val_metrics = eval_objective(method, model, val_pack, mode_guidance, branch_info, latent_means, args, torch, F)
        score = float(val_metrics["score"])
        if score < best - float(args.min_delta):
            best = score
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model_type": "compact_conditional_realnvp" if method not in MIX_METHODS else "branch_mixture_prior_realnvp",
                    "method": method,
                    "condition_set": condition_set,
                    "base_condition": args.base_condition,
                    "condition_names": ds.names,
                    "condition_mu": ds.mu.tolist(),
                    "condition_sd": ds.sd.tolist(),
                    "condition_flag_mask": ds.flag_mask.astype(bool).tolist(),
                    "x_dim": int(x.shape[1]),
                    "c_dim": int(c.shape[1]),
                    "n_branches": int(branch_info["n_branches"] if method in MIX_METHODS else 0),
                    "branch_embed_dim": int(args.branch_embed_dim),
                    "branch_names": list(branch_info["branch_names"]) if method in MIX_METHODS else [],
                    "branch_groups": list(branch_info["branch_groups"]) if method in MIX_METHODS else [],
                    "hidden": int(args.hidden),
                    "flow_layers": int(args.flow_layers),
                    "scale_clip": float(args.scale_clip),
                    "latent_means": latent_means_np.tolist(),
                    "latent_sigma": float(args.mixprior_latent_sigma),
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
            "train_loss": totals["loss"] / max(seen, 1),
            "train_nll": totals["nll"] / max(seen, 1),
            "train_assign": totals["assign"] / max(seen, 1),
            "train_usage": totals["usage"] / max(seen, 1),
            "train_branch_nll": totals["branch_nll"] / max(seen, 1),
            "val_nll": float(val_metrics["nll"]),
            "val_weighted_nll": float(val_metrics["weighted_nll"]),
            "val_assign": float(val_metrics["assign"]),
            "val_usage": float(val_metrics["usage"]),
            "val_branch_nll": float(val_metrics["branch_nll"]),
            "val_score": score,
            "best_val_score": float(best),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        history.append(row)
        if progress is not None:
            progress.set_postfix(nll=f"{row['train_nll']:.3f}", val=f"{row['val_nll']:.3f}", best=f"{best:.3f}", stale=stale)
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                f"{condition_set} epoch {epoch}/{args.epochs} train_nll={row['train_nll']:.4f} "
                f"val_nll={row['val_nll']:.4f} score={score:.4f} assign={row['val_assign']:.4f} "
                f"usage={row['val_usage']:.4f} branch_nll={row['val_branch_nll']:.4f} best={best:.4f}@{best_epoch}",
                flush=True,
            )
        if args.run_name != "smoke" and args.patience > 0 and stale >= args.patience:
            print(f"{condition_set} early stopping at epoch {epoch}", flush=True)
            break
    if progress is not None:
        progress.close()

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / f"{condition_set}_loss_history.csv", index=False)
    base.plot_loss(hist, run_dir / "figures" / f"{condition_set}_loss_curve.png")

    state = ddpm.load_checkpoint(best_path, device, torch)
    model.load_state_dict(state["state_dict"])
    model.module().eval()
    metrics = evaluate_split_metrics(method, condition_set, model, x, c, context, row_group_id, row_mode_id, row_branch_id, mode_guidance, branch_info, latent_means, args, device, torch, F, test_idx)
    pd.DataFrame(metrics).to_csv(run_dir / f"{condition_set}_test_metrics.csv", index=False)
    generated_summary = generate_targets(method, condition_set, model, ds, context, mode_guidance, branch_info, latent_means, pinn, args, run_dir, device, torch, F)
    write_condition_report(run_dir, condition_set, method, args, ds, branch_info, metrics, generated_summary)


def compute_mode_weights(priority, row_group_id, row_mode_id, idx, mode_guidance, args) -> np.ndarray:
    weights = np.full(len(idx), float(args.other_target_weight), dtype=np.float64)
    target_groups = set(str(x) for x in mode_guidance.get("groups", []))
    for pos, row in enumerate(idx):
        if str(priority[row]) in target_groups:
            weights[pos] = float(args.high_target_weight)
    beta = float(args.mode_weight_beta)
    if beta > 0.0:
        for group in mode_guidance.get("groups", []):
            info = mode_guidance["by_group"][group]
            gid = int(info["group_id"])
            pi_design = np.asarray(info["pi_star"], dtype=np.float64)
            pi_design = pi_design / max(float(pi_design.sum()), 1e-12)
            group_mask = row_group_id[idx] == gid
            modes = row_mode_id[idx[group_mask]]
            if len(modes) == 0:
                continue
            counts = np.bincount(modes[modes >= 0], minlength=len(pi_design)).astype(np.float64)
            pi_emp = counts / max(float(counts.sum()), 1e-12)
            ratio = np.ones_like(pi_design)
            valid = pi_emp > 0
            ratio[valid] = pi_design[valid] / pi_emp[valid]
            mode_weights = np.power(ratio, beta)
            mode_weights = np.clip(mode_weights, float(args.mode_weight_min), float(args.mode_weight_max))
            positions = np.where(group_mask)[0]
            for p, m in zip(positions, modes):
                if 0 <= int(m) < len(mode_weights):
                    weights[p] *= float(mode_weights[int(m)])
    weights = weights / max(float(np.mean(weights)), 1e-12)
    return weights.astype(np.float32)


def make_latent_means(branch_info: dict[str, Any], x_dim: int, separation: float) -> np.ndarray:
    means = np.zeros((int(branch_info["n_branches"]), int(x_dim)), dtype=np.float32)
    for group, info in branch_info.get("by_group", {}).items():
        global_ids = np.asarray(info["global_ids"], dtype=np.int64)
        k = len(global_ids)
        if k <= 1:
            continue
        center = np.zeros((k, int(x_dim)), dtype=np.float32)
        for local_id in range(k):
            center[local_id, local_id % int(x_dim)] = float(separation)
        center = center - center.mean(axis=0, keepdims=True)
        for local_id, global_id in enumerate(global_ids):
            means[int(global_id)] = center[local_id]
    return means


def normal_logprob(z, mu, sigma: float):
    dz = (z - mu) / float(sigma)
    return -0.5 * (dz * dz + math.log(2.0 * math.pi) + 2.0 * math.log(float(sigma))).sum(dim=1)


def mixprior_log_prob(model, x, c, group_id, mode_id, mode_guidance, branch_info, latent_means, args, torch, F):
    logp = x.new_empty((len(x),))
    assign_terms = []
    usage_terms = []
    handled = torch.zeros((len(x),), dtype=torch.bool, device=x.device)
    sigma = float(args.mixprior_latent_sigma)
    eps = 1e-8

    for group in mode_guidance.get("groups", []):
        info = mode_guidance["by_group"][group]
        gid = int(info["group_id"])
        mask = group_id == gid
        if int(mask.sum().detach().cpu()) == 0:
            continue
        handled |= mask
        branch_ids_np = np.asarray(branch_info["by_group"][group]["global_ids"], dtype=np.int64)
        branch_ids = torch.tensor(branch_ids_np, dtype=torch.long, device=x.device)
        pi = torch.tensor(np.asarray(info["pi_star"], dtype=np.float32), dtype=x.dtype, device=x.device)
        pi = pi / pi.sum().clamp_min(eps)
        comps = []
        for local_id, bid in enumerate(branch_ids.tolist()):
            bb = torch.full((int(mask.sum()),), int(bid), dtype=torch.long, device=x.device)
            z, logdet = model.x_to_z(x[mask], c[mask], bb)
            lp = torch.log(pi[local_id].clamp_min(eps)) + normal_logprob(z, latent_means[int(bid)].to(dtype=x.dtype, device=x.device), sigma) + logdet
            comps.append(lp)
        comp = torch.stack(comps, dim=1)
        logp[mask] = torch.logsumexp(comp, dim=1)
        posterior = torch.softmax(comp, dim=1)
        local_mode = mode_id[mask].long()
        valid = (local_mode >= 0) & (local_mode < posterior.shape[1])
        if bool(valid.any()):
            assign_terms.append(F.nll_loss(torch.log(posterior[valid].clamp_min(eps)), local_mode[valid], reduction="mean"))
        if posterior.shape[0] >= max(2, posterior.shape[1]):
            rbar = posterior.mean(dim=0).clamp_min(eps)
            usage_terms.append((pi * (torch.log(pi.clamp_min(eps)) - torch.log(rbar))).sum())

    if bool((~handled).any()):
        bb = torch.zeros((int((~handled).sum()),), dtype=torch.long, device=x.device)
        logp[~handled] = model.branch_log_prob(x[~handled], c[~handled], bb, latent_means, sigma)

    zero = x.new_zeros(())
    assign = torch.stack(assign_terms).mean() if assign_terms else zero
    usage = torch.stack(usage_terms).mean() if usage_terms else zero
    return logp, assign, usage


def supervised_branch_nll(model, x, c, branch_id, latent_means, args, sample_weight=None):
    valid = branch_id.long() > 0
    if not bool(valid.any()):
        return x.new_zeros(())
    lp = model.branch_log_prob(x[valid], c[valid], branch_id[valid].long(), latent_means, float(args.mixprior_latent_sigma))
    nll_vec = -lp
    if sample_weight is not None:
        w = sample_weight[valid].to(dtype=nll_vec.dtype, device=nll_vec.device)
        return (w * nll_vec).mean()
    return nll_vec.mean()


def eval_objective(method, model, pack, mode_guidance, branch_info, latent_means, args, torch, F):
    if method == "CNF":
        logp = model.log_prob(pack["x"], pack["c"])
        nll_vec = -logp
        return {"nll": float(nll_vec.mean().detach().cpu()), "weighted_nll": float(nll_vec.mean().detach().cpu()), "assign": 0.0, "usage": 0.0, "branch_nll": 0.0, "score": float(nll_vec.mean().detach().cpu())}
    if method == "CNF_WMODE":
        logp = model.log_prob(pack["x"], pack["c"])
        nll_vec = -logp
        weighted = (pack["weight"] * nll_vec).mean()
        return {"nll": float(nll_vec.mean().detach().cpu()), "weighted_nll": float(weighted.detach().cpu()), "assign": 0.0, "usage": 0.0, "branch_nll": 0.0, "score": float(weighted.detach().cpu())}
    logp, assign, usage = mixprior_log_prob(model, pack["x"], pack["c"], pack["group"], pack["mode"], mode_guidance, branch_info, latent_means, args, torch, F)
    nll_vec = -logp
    weighted = (pack["weight"] * nll_vec).mean()
    branch_nll = supervised_branch_nll(model, pack["x"], pack["c"], pack["branch"], latent_means, args, sample_weight=pack["weight"])
    score = (
        weighted
        + float(args.mixprior_assign_weight) * assign
        + float(args.mixprior_usage_weight) * usage
        + float(args.mixprior_branch_nll_weight) * branch_nll
    )
    return {
        "nll": float(nll_vec.mean().detach().cpu()),
        "weighted_nll": float(weighted.detach().cpu()),
        "assign": float(assign.detach().cpu()),
        "usage": float(usage.detach().cpu()),
        "branch_nll": float(branch_nll.detach().cpu()),
        "score": float(score.detach().cpu()),
    }


def evaluate_split_metrics(method, condition_set, model, x, c, context, row_group_id, row_mode_id, row_branch_id, mode_guidance, branch_info, latent_means, args, device, torch, F, test_idx=None):
    if test_idx is None:
        test_idx = filter_indices_by_priority(np.asarray(context["test_idx"], dtype=np.int64), np.asarray(context["priority_group"]).astype(str), args.train_target_groups)
    else:
        test_idx = np.asarray(test_idx, dtype=np.int64)
    priority = np.asarray(context["priority_group"])[test_idx].astype(str)
    group_id = row_group_id[test_idx]
    mode_id = row_mode_id[test_idx]
    branch_id = row_branch_id[test_idx]
    lps = []
    bs = int(max(256, args.batch_size))
    with torch.no_grad():
        for start in range(0, len(test_idx), bs):
            take = test_idx[start : start + bs]
            xt = torch.tensor(x[take], dtype=torch.float32, device=device)
            ct = torch.tensor(c[take], dtype=torch.float32, device=device)
            if method in MIX_METHODS:
                gt = torch.tensor(row_group_id[take], dtype=torch.long, device=device)
                mt = torch.tensor(row_mode_id[take], dtype=torch.long, device=device)
                lp, _assign, _usage = mixprior_log_prob(model, xt, ct, gt, mt, mode_guidance, branch_info, latent_means, args, torch, F)
            else:
                lp = model.log_prob(xt, ct)
            lps.append(lp.detach().cpu().numpy())
    lp_all = np.concatenate(lps)
    rows: list[dict[str, Any]] = [
        {
            "condition_set": condition_set,
            "method": method,
            "group_type": "overall",
            "group": "overall",
            "n": int(len(lp_all)),
            "mean_logp": float(lp_all.mean()),
            "nll": float(-lp_all.mean()),
        }
    ]
    for group in sorted(set(priority)):
        m = priority == group
        rows.append({"condition_set": condition_set, "method": method, "group_type": "priority_group", "group": str(group), "n": int(m.sum()), "mean_logp": float(lp_all[m].mean()), "nll": float(-lp_all[m].mean())})
    if method in MIX_METHODS:
        for bid, n in pd.Series(branch_id).value_counts().items():
            if int(n) < 20:
                continue
            m = branch_id == int(bid)
            rows.append({"condition_set": condition_set, "method": method, "group_type": "internal_branch", "group": f"B{int(bid):02d}", "n": int(m.sum()), "mean_logp": float(lp_all[m].mean()), "nll": float(-lp_all[m].mean())})
    return rows


def parse_target_group_filter(text: str) -> set[str]:
    return {x.strip() for x in str(text).split(",") if x.strip()}


def target_group_filter_label(text: str) -> str:
    groups = sorted(parse_target_group_filter(text))
    return ",".join(groups) if groups else "ALL"


def filter_indices_by_priority(idx: np.ndarray, priority: np.ndarray, target_groups: str) -> np.ndarray:
    groups = parse_target_group_filter(target_groups)
    idx = np.asarray(idx, dtype=np.int64)
    if not groups:
        return idx
    keep = np.isin(priority[idx].astype(str), sorted(groups))
    return idx[keep]


def generate_targets(method, condition_set, model, ds, context, mode_guidance, branch_info, latent_means, pinn, args, run_dir, device, torch, F) -> pd.DataFrame:
    strategies = [x.strip().lower() for x in str(args.generate_strategies).split(",") if x.strip()]
    unknown = sorted(set(strategies) - {"median", "mixture"})
    if unknown:
        raise KeyError(f"unknown generation strategies: {unknown}")
    h_mu = np.asarray(context["h_mu"], dtype=np.float32)
    h_sd = np.asarray(context["h_sd"], dtype=np.float32)
    train_h_raw = base.gauge_fix_vec28(context["h_norm"][context["train_idx"]] * h_sd + h_mu)
    generated: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed + stable_int(method))

    for target_name, cvec in ds.targets_norm.items():
        group = ddpm.infer_base_target(target_name, args)
        if group is None:
            continue
        if "median" in strategies:
            c_batch = np.repeat(cvec[None, :].astype(np.float32), int(args.n_generate), axis=0)
            branch_batch, branch_names = make_branch_batch_for_method(method, group, len(c_batch), branch_info, rng)
            write_generated_batch(method, condition_set, target_name, group, "median", c_batch, branch_batch, branch_names, model, context, mode_guidance, branch_info, latent_means, pinn, args, run_dir, device, torch, F, h_mu, h_sd, train_h_raw, generated, rows, provenance_rows)
        if "mixture" in strategies and target_name == group:
            c_batch, _source_pop, source_rows = ddpm.make_condition_mixture(group, ds, context, args, rng)
            branch_batch, branch_names = make_branch_batch_for_method(method, group, len(c_batch), branch_info, rng)
            write_generated_batch(method, condition_set, f"{group}_mixture", group, "mixture", c_batch, branch_batch, branch_names, model, context, mode_guidance, branch_info, latent_means, pinn, args, run_dir, device, torch, F, h_mu, h_sd, train_h_raw, generated, rows, provenance_rows, source_rows=source_rows)

    np.savez_compressed(run_dir / f"{condition_set}_generated_samples.npz", **generated)
    summary = pd.DataFrame(rows)
    summary.to_csv(run_dir / f"{condition_set}_generated_physical_summary.csv", index=False)
    pd.DataFrame(provenance_rows).to_csv(run_dir / f"{condition_set}_generation_provenance.csv", index=False)
    return summary


def make_branch_batch_for_method(method: str, group: str, n: int, branch_info: dict[str, Any], rng: np.random.Generator):
    if method not in MIX_METHODS or group not in branch_info.get("by_group", {}):
        return np.zeros(int(n), dtype=np.int64), ["default"] * int(n)
    info = branch_info["by_group"][group]
    global_ids = np.asarray(info["global_ids"], dtype=np.int64)
    probs = np.asarray(info["pi_star"], dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-12)
    local = rng.choice(np.arange(len(global_ids)), size=int(n), replace=True, p=probs)
    branch_batch = global_ids[local].astype(np.int64)
    names = [str(branch_info["branch_names"][int(b)]) for b in branch_batch]
    return branch_batch, names


def write_generated_batch(
    method,
    condition_set,
    target_key,
    base_target,
    strategy,
    c_batch,
    branch_batch,
    branch_names,
    model,
    context,
    mode_guidance,
    branch_info,
    latent_means,
    pinn,
    args,
    run_dir,
    device,
    torch,
    F,
    h_mu,
    h_sd,
    train_h_raw,
    generated,
    rows,
    provenance_rows,
    source_rows=None,
):
    xs: list[np.ndarray] = []
    bs = int(args.batch_size)
    for start in range(0, len(c_batch), bs):
        cb = torch.tensor(c_batch[start : start + bs], dtype=torch.float32, device=device)
        bb_np = branch_batch[start : start + bs]
        if method in MIX_METHODS:
            bb = torch.tensor(bb_np, dtype=torch.long, device=device)
            mu = latent_means[bb.long().clamp(0, latent_means.shape[0] - 1)].to(dtype=cb.dtype, device=device)
            z = mu + float(args.mixprior_latent_sigma) * torch.randn((len(cb), model.x_dim), dtype=cb.dtype, device=device)
            if method == "HTBAL_CNF_GUIDED" and pinn is not None and int(args.guided_steps) > 0:
                z = guided_refine_latent(model, pinn, z, cb, bb, base_target, bb_np, mode_guidance, branch_info, latent_means, args, torch, F)
            with torch.no_grad():
                xg = model.sample(cb, bb, latent_means, float(args.mixprior_latent_sigma), z=z)
        else:
            with torch.no_grad():
                xg = model.sample(cb, len(cb))
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
            "uses_internal_mixture_prior": bool(method in MIX_METHODS),
            "uses_latent_guidance": bool(method == "HTBAL_CNF_GUIDED"),
            **base.physical_summary(x_gen_raw, train_h_raw),
        }
    )
    if source_rows is None:
        source_rows = np.full(len(c_batch), -1, dtype=np.int64)
    for i, src in enumerate(source_rows):
        provenance_rows.append(
            {
                "condition_set": condition_set,
                "target": target_key,
                "base_target": base_target,
                "strategy": strategy,
                "generated_index": int(i),
                "source_condition_row": int(src),
                "internal_branch_id": int(branch_batch[i]) if len(branch_batch) else 0,
                "internal_branch": str(branch_names[i]) if i < len(branch_names) else "default",
            }
        )
    print(f"generated {key}: {x_gen_raw.shape}", flush=True)


def guided_refine_latent(model, pinn, z, c, branch_id, group, branch_np, mode_guidance, branch_info, latent_means, args, torch, F):
    if group not in mode_guidance.get("by_group", {}):
        return z
    info = mode_guidance["by_group"][group]
    global_ids = np.asarray(branch_info["by_group"][group]["global_ids"], dtype=np.int64)
    global_to_local = {int(gid): int(i) for i, gid in enumerate(global_ids)}
    local_modes = torch.tensor([global_to_local.get(int(b), 0) for b in branch_np], dtype=torch.long, device=z.device)
    feat_mu = torch.tensor(info["feat_mu"], dtype=z.dtype, device=z.device)
    feat_sd = torch.tensor(info["feat_sd"], dtype=z.dtype, device=z.device).clamp_min(1e-6)
    feat_low = torch.tensor(info["feat_low"], dtype=z.dtype, device=z.device)
    feat_high = torch.tensor(info["feat_high"], dtype=z.dtype, device=z.device)
    proto_z = torch.tensor(info["proto_z"], dtype=z.dtype, device=z.device)
    z_work = z.detach()
    for _ in range(int(args.guided_steps)):
        z_work = z_work.detach().requires_grad_(True)
        xg = model.z_to_x(z_work, c, branch_id)
        pred_pop, _logits = pinn.model(xg)
        feat = pinnmod.torch_pop_features(pred_pop, pinn.model.times_raw, torch)
        feat_norm = (feat - feat_mu[None, :]) / feat_sd[None, :]
        target_band = (torch.relu(feat_low[None, :] - feat).pow(2) + torch.relu(feat - feat_high[None, :]).pow(2)).mean()
        target_proto = proto_z[local_modes.clamp(0, proto_z.shape[0] - 1)]
        mode_loss = (feat_norm - target_proto).pow(2).mean()
        support = torch.relu(torch.abs(xg) - float(args.support_clip)).pow(2).mean()
        mu = latent_means[branch_id.long().clamp(0, latent_means.shape[0] - 1)].to(dtype=z.dtype, device=z.device)
        prior = -normal_logprob(z_work, mu, float(args.mixprior_latent_sigma)).mean()
        energy = (
            float(args.guided_target_weight) * target_band
            + float(args.guided_mode_weight) * mode_loss
            + float(args.guided_support_weight) * support
            + float(args.guided_prior_weight) * prior
        )
        grad = torch.autograd.grad(energy, z_work, retain_graph=False, create_graph=False)[0]
        grad_norm = grad.flatten(1).norm(dim=1).clamp_min(1e-6)[:, None]
        z_work = z_work - float(args.guided_step_size) * grad / grad_norm
    return z_work.detach()


def write_metadata(out_root: Path, prepared: Path, args, context, ds, mode_guidance, branch_info, selected) -> None:
    meta = out_root / "metadata"
    meta.mkdir(parents=True, exist_ok=True)
    branch_info["summary"].to_csv(meta / "cnf_mode_prior_internal_branch_summary.csv", index=False)
    pd.DataFrame(
        {
            "row_index": np.arange(len(branch_info["row_branch_id"]), dtype=np.int64),
            "priority_group": np.asarray(context["priority_group"]).astype(str),
            "internal_branch_id": branch_info["row_branch_id"].astype(np.int64),
            "internal_group_id": branch_info["row_group_id"].astype(np.int64),
            "internal_mode_id": branch_info["row_mode_id"].astype(np.int64),
        }
    ).to_csv(meta / "cnf_mode_prior_row_assignments.csv", index=False)
    if "htbal_mode_guidance_summary" in context:
        context["htbal_mode_guidance_summary"].to_csv(meta / "cnf_mode_prior_mode_guidance_summary.csv", index=False)
    manifest = {
        "purpose": "Exact-likelihood H27 conditional normalizing-flow ablations for reference-faithful vs diversity-design mode control.",
        "prepared": str(prepared),
        "out_root": str(out_root),
        "methods": selected,
        "base_condition": args.base_condition,
        "train_target_groups": target_group_filter_label(args.train_target_groups),
        "generation_target_groups": args.target_groups,
        "condition_dim": int(ds.raw.shape[1]),
        "n_internal_branches": int(branch_info["n_branches"]),
        "pi_design_source": "mode_guidance pi_star, including htbal prior alpha/min floor",
        "boundary": "Dynamic modes are internal training/sampling signals only; user-facing generation remains compact-condition.",
        "important_files": {
            "branch_summary": str(meta / "cnf_mode_prior_internal_branch_summary.csv"),
            "row_assignments": str(meta / "cnf_mode_prior_row_assignments.csv"),
            "mode_guidance": str(meta / "cnf_mode_prior_mode_guidance_summary.csv"),
        },
        "args": base.clean_json(vars(args)),
    }
    (meta / "cnf_mode_prior_manifest.json").write_text(json.dumps(base.clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")


def write_condition_report(run_dir, condition_set, method, args, ds, branch_info, metrics, generated_summary):
    metric_df = pd.DataFrame(metrics)
    lines = [
        f"# {condition_set} H27 CNF mode-prior run",
        "",
        f"- method: `{method}`",
        f"- base_condition: `{args.base_condition}`",
        f"- train_target_groups: `{target_group_filter_label(args.train_target_groups)}`",
        f"- generation_target_groups: `{args.target_groups}`",
        f"- condition_dim: `{ds.raw.shape[1]}`",
        f"- n_internal_branches: `{branch_info['n_branches'] if method in MIX_METHODS else 0}`",
        f"- flow_layers: `{args.flow_layers}`",
        f"- user-facing dynamic condition: `False`",
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
        "This exact-likelihood ablation tests whether target/mode-balanced likelihood and internal mixture-prior sampling improve high-transfer dynamic-route coverage. Final success must be judged by simulator validation plus reference/design diversity metrics.",
    ]
    (run_dir / "reports" / f"{condition_set}_run_report_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(run_dir: Path, args, selected_methods: list[str], branch_info: dict[str, Any]) -> None:
    metric_paths = sorted(run_dir.glob("*_test_metrics.csv"))
    if metric_paths:
        metrics = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)
        metrics.to_csv(run_dir / "all_test_metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "base_condition": args.base_condition,
        "train_target_groups": target_group_filter_label(args.train_target_groups),
        "generation_target_groups": args.target_groups,
        "methods": selected_methods,
        "n_internal_branches": int(branch_info["n_branches"]),
        "branch_names": list(branch_info["branch_names"]),
        "generated_files": [str(p) for p in sorted(run_dir.glob("*_generated_samples.npz"))],
        "checkpoint_files": [str(p) for p in sorted((run_dir / "checkpoints").glob("*_best.pt"))],
        "args": base.clean_json(vars(args)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(base.clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def stable_int(text: str) -> int:
    out = 0
    for ch in str(text):
        out = (out * 131 + ord(ch)) % 1_000_003
    return int(out)


if __name__ == "__main__":
    raise SystemExit(main())

