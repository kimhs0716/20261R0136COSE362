#!/usr/bin/env python3
"""Train H27 DYNZ flow with a PINN-lite population trajectory surrogate.

This experiment replaces the weak H->dynamic-submode teacher with a richer
H,t->population trajectory surrogate.  It is PINN-lite rather than a full
density-matrix PINN: the prepared 140k artifact contains only population
trajectories, not the full rho(t), so the physics terms are population-level
constraints and trajectory-summary consistency checks.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_h27_path_dynamic_flow as base


DEFAULT_OUT_ROOT = Path("outputs/experiments/20260619_h27_dynz_pinntraj_flow")
BASE_CONDITION_SET = "CFAST_CL1_PATH_DYNZ"
PINN_CONDITION_SET = "CFAST_CL1_PATH_DYNZ_PINNTRAJ"
PINN_FEATURE_NAMES = [
    "eta10",
    "eta20",
    "eta50",
    "sink34_at_5ps",
    "detour567_at_5ps",
    "sink34_at_10ps",
    "detour567_at_10ps",
    "sink34_at_20ps",
    "detour567_at_20ps",
    "residual_at_20ps",
    "loss_at_50ps",
    "residence_sink34_0p5_10ps",
    "residence_detour567_0p5_10ps",
    "q_route_0p5_10ps",
    "residence_sink34_10_20ps",
    "residence_detour567_10_20ps",
    "q_route_10_20ps",
]


@dataclass
class PINNTrajectoryBundle:
    model: Any
    categories: list[str]
    label_ids: np.ndarray
    class_weights: Any
    train_metrics: pd.DataFrame


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, default=base.DEFAULT_PREPARED)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--run-name", choices=["smoke", "full"], default="full")
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
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metadata-only", action="store_true")

    p.add_argument("--dyn-k-fast", type=int, default=3)
    p.add_argument("--dyn-k-very-fast", type=int, default=3)
    p.add_argument("--dyn-k-late", type=int, default=4)
    p.add_argument("--dyn-k-nonhigh", type=int, default=6)
    p.add_argument("--dyn-k-other", type=int, default=1)
    p.add_argument("--kmeans-iter", type=int, default=80)
    p.add_argument("--kmeans-init", type=int, default=8)
    p.add_argument("--min-dyn-target-rows", type=int, default=80)
    p.add_argument("--max-dyn-targets-per-group", type=int, default=3)
    p.add_argument("--target-groups", default=",".join(base.TARGET_GROUPS))

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
    p.add_argument("--pinn-only", action="store_true")

    p.add_argument("--aux-start-epoch", type=int, default=1)
    p.add_argument("--lambda-pinn-traj", type=float, default=0.05)
    p.add_argument("--lambda-pinn-feature", type=float, default=0.02)
    p.add_argument("--lambda-pinn-dyn-ce", type=float, default=0.0)
    p.add_argument("--lambda-pinn-phys", type=float, default=0.005)
    p.add_argument("--lambda-support", type=float, default=0.001)
    p.add_argument("--support-clip", type=float, default=6.0)
    p.add_argument("--val-aux-weight", type=float, default=1.0)
    p.add_argument("--val-aux-max", type=int, default=4096)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    prepared = base.resolve_prepared(args.prepared)
    out_root = args.out_root
    run_dir = out_root / args.run_name
    metadata_dir = out_root / "metadata"
    for path in [run_dir, metadata_dir, run_dir / "checkpoints", run_dir / "figures", run_dir / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.load(prepared, allow_pickle=True)
    base.validate_prepared(raw)
    context = base.build_context(raw, args)
    condition_data = base.build_condition_datasets(context, args)
    ds = clone_condition_dataset(condition_data[BASE_CONDITION_SET])
    ds.targets_raw, ds.targets_norm = condition_data[BASE_CONDITION_SET].targets_raw, condition_data[BASE_CONDITION_SET].targets_norm

    write_metadata(out_root, prepared, args, context, ds)
    print(f"prepared: {prepared}", flush=True)
    print(f"out_root: {out_root}", flush=True)
    print(f"run_dir: {run_dir}", flush=True)
    print(f"condition_set: {PINN_CONDITION_SET}", flush=True)
    if args.metadata_only:
        print("metadata-only complete:", metadata_dir, flush=True)
        return 0

    import torch
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(args.seed)
    device = base.choose_device(args.device, torch)
    print(f"device: {device}", flush=True)

    pinn = train_or_load_pinn_surrogate(context, args, run_dir, device, torch, TensorDataset, DataLoader)
    if args.pinn_only:
        print("pinn-only complete:", run_dir / "pinntraj_surrogate_split_metrics.csv", flush=True)
        return 0
    train_pinn_flow(ds, context, pinn, args, run_dir, device, torch, TensorDataset, DataLoader)
    write_run_summary(run_dir, args)
    print("done:", run_dir, flush=True)
    return 0


def clone_condition_dataset(ds: base.ConditionDataset) -> base.ConditionDataset:
    return base.ConditionDataset(
        name=PINN_CONDITION_SET,
        raw=ds.raw,
        norm=ds.norm,
        names=ds.names,
        mu=ds.mu,
        sd=ds.sd,
        flag_mask=ds.flag_mask,
        dynamic_label=ds.dynamic_label,
        balance_label=None,
        targets_raw=dict(ds.targets_raw),
        targets_norm=dict(ds.targets_norm),
    )


def dynamic_label_ids(labels: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cats = sorted(np.unique(np.asarray(labels).astype(str)).tolist())
    pos = {cat: i for i, cat in enumerate(cats)}
    ids = np.array([pos[str(x)] for x in labels], dtype=np.int64)
    return ids, cats


def make_pinn_surrogate(torch, x_dim: int, n_classes: int, times: np.ndarray, hidden: int, depth: int, dropout: float):
    import torch.nn as nn

    class ResidualBlock(nn.Module):
        def __init__(self, width: int, p_drop: float):
            super().__init__()
            self.net = nn.Sequential(
                nn.LayerNorm(width),
                nn.Linear(width, width),
                nn.SiLU(),
                nn.Dropout(p_drop),
                nn.Linear(width, width),
            )

        def forward(self, x):
            return x + self.net(x)

    class PINNTrajectorySurrogate(nn.Module):
        def __init__(self):
            super().__init__()
            p_drop = max(0.0, float(dropout))
            self.register_buffer("times_raw", torch.tensor(times.astype(np.float32)))
            t_norm = np.log1p(times.astype(np.float32)) / np.log1p(float(np.max(times)))
            self.register_buffer("times_norm", torch.tensor(t_norm.astype(np.float32)))
            h_blocks: list[nn.Module] = [nn.Linear(x_dim, hidden), nn.SiLU(), nn.Dropout(p_drop)]
            for _ in range(max(1, int(depth))):
                h_blocks.append(ResidualBlock(hidden, p_drop))
                h_blocks.append(nn.SiLU())
            h_blocks.append(nn.LayerNorm(hidden))
            self.h_encoder = nn.Sequential(*h_blocks)
            time_dim = 7
            traj_blocks: list[nn.Module] = [nn.Linear(hidden + time_dim, hidden), nn.SiLU(), nn.Dropout(p_drop)]
            for _ in range(max(1, int(depth // 2))):
                traj_blocks.append(ResidualBlock(hidden, p_drop))
                traj_blocks.append(nn.SiLU())
            traj_blocks.append(nn.LayerNorm(hidden))
            self.traj_net = nn.Sequential(*traj_blocks)
            self.pop_head = nn.Linear(hidden, 9)
            self.classifier = nn.Linear(hidden, n_classes)

        def time_features(self, t):
            return torch.stack(
                [
                    t,
                    t * t,
                    torch.sin(np.pi * t),
                    torch.cos(np.pi * t),
                    torch.sin(2.0 * np.pi * t),
                    torch.cos(2.0 * np.pi * t),
                    torch.log1p(50.0 * t) / np.log(51.0),
                ],
                dim=-1,
            )

        def forward(self, h, return_embedding: bool = False):
            emb = self.h_encoder(h)
            b = h.shape[0]
            tfeat = self.time_features(self.times_norm).to(device=h.device, dtype=h.dtype)
            h_rep = emb[:, None, :].expand(b, tfeat.shape[0], emb.shape[1])
            t_rep = tfeat[None, :, :].expand(b, tfeat.shape[0], tfeat.shape[1])
            z = self.traj_net(torch.cat([h_rep, t_rep], dim=-1))
            pop = torch.softmax(self.pop_head(z), dim=-1)
            logits = self.classifier(emb)
            if return_embedding:
                return pop, logits, emb
            return pop, logits

    return PINNTrajectorySurrogate()


def train_or_load_pinn_surrogate(context, args, run_dir, device, torch, TensorDataset, DataLoader) -> PINNTrajectoryBundle:
    x = context["h_norm"].astype(np.float32)
    pop = context["raw"]["pop_t"].astype(np.float32)
    times = context["raw"]["times"].astype(np.float32)
    label_ids, categories = dynamic_label_ids(context["dynamic_label"])
    path = run_dir / "checkpoints" / "pinntraj_surrogate_best.pt"
    metrics_path = run_dir / "pinntraj_surrogate_metrics.csv"
    model = make_pinn_surrogate(
        torch,
        x_dim=x.shape[1],
        n_classes=len(categories),
        times=times,
        hidden=args.pinn_hidden,
        depth=args.pinn_depth,
        dropout=args.pinn_dropout,
    ).to(device)
    class_weights = make_class_weights(label_ids, context["train_idx"], len(categories), args, torch).to(device)

    if path.exists() and not args.force:
        state = base.load_checkpoint(path, device, torch)
        if pinn_checkpoint_compatible(state, args) and metrics_path.exists() and metrics_path.stat().st_size > 0:
            model.load_state_dict(state["state_dict"])
            metrics = pd.read_csv(metrics_path)
            print(f"loaded PINN surrogate: {path}", flush=True)
        else:
            print("existing PINN surrogate checkpoint is incompatible; retraining", flush=True)
            metrics = train_pinn_surrogate_model(model, x, pop, label_ids, class_weights, context, args, run_dir, device, torch, TensorDataset, DataLoader)
    else:
        metrics = train_pinn_surrogate_model(model, x, pop, label_ids, class_weights, context, args, run_dir, device, torch, TensorDataset, DataLoader)

    model.eval()
    write_pinn_eval_artifacts(model, x, pop, label_ids, categories, context, args, run_dir, device, torch)
    for param in model.parameters():
        param.requires_grad_(False)
    return PINNTrajectoryBundle(model=model, categories=categories, label_ids=label_ids, class_weights=class_weights, train_metrics=metrics)


def pinn_checkpoint_compatible(state: dict[str, Any], args) -> bool:
    saved_args = state.get("args", {})
    keys = [
        "pinn_hidden",
        "pinn_depth",
        "pinn_dropout",
        "pinn_loss_pop",
        "pinn_loss_feature",
        "pinn_loss_dyn_ce",
        "pinn_balanced_ce",
        "pinn_class_weight_power",
        "pinn_focal_gamma",
        "pinn_label_smoothing",
        "pinn_score_mode",
    ]
    return all(key in saved_args and str(saved_args[key]) == str(getattr(args, key)) for key in keys)


def train_pinn_surrogate_model(model, x, pop, label_ids, class_weights, context, args, run_dir, device, torch, TensorDataset, DataLoader) -> pd.DataFrame:
    import torch.nn.functional as F

    if float(args.pinn_loss_dyn_ce) <= 0.0:
        print(
            "pinn dynamic-class CE is disabled; printed acc/top3 are unused diagnostics and may stay near random.",
            flush=True,
        )
    train_idx = context["train_idx"]
    val_idx = context["val_idx"]
    train = TensorDataset(
        torch.tensor(x[train_idx], dtype=torch.float32),
        torch.tensor(pop[train_idx], dtype=torch.float32),
        torch.tensor(label_ids[train_idx], dtype=torch.long),
    )
    loader = DataLoader(train, batch_size=args.pinn_batch_size, shuffle=True, drop_last=False)
    val_x = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    val_pop = torch.tensor(pop[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(label_ids[val_idx], dtype=torch.long, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.pinn_lr, weight_decay=args.weight_decay)
    best_score = float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    best_path = run_dir / "checkpoints" / "pinntraj_surrogate_best.pt"
    t0 = time.perf_counter()
    times = model.times_raw
    for epoch in range(1, args.pinn_epochs + 1):
        model.train()
        totals = {"loss": 0.0, "pop": 0.0, "feature": 0.0, "ce": 0.0, "mono": 0.0, "smooth": 0.0, "acc": 0.0}
        seen = 0
        for xb, pb, yb in loader:
            xb = xb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            pred_pop, logits = model(xb)
            losses = pinn_losses(pred_pop, pb, logits, yb, class_weights, args, times, F, torch)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            n = len(xb)
            seen += n
            for key in ["loss", "pop", "feature", "ce", "mono", "smooth"]:
                totals[key] += float(losses[key].detach().cpu()) * n
            totals["acc"] += float((logits.argmax(dim=1) == yb).float().mean().detach().cpu()) * n
        model.eval()
        with torch.no_grad():
            val_pred, val_logits = model(val_x)
            val_losses = pinn_losses(val_pred, val_pop, val_logits, val_y, class_weights, args, times, F, torch)
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean()
            val_top3 = topk_accuracy(val_logits, val_y, k=min(3, val_logits.shape[1]))
            val_macro = macro_recall_torch(val_logits.argmax(dim=1), val_y, val_logits.shape[1], torch)
        val_select = pinn_selection_score(val_losses, args)
        score = float(val_select.detach().cpu())
        if score < best_score - args.pinn_min_delta:
            best_score = score
            stale = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "categories": dynamic_json_categories(context["dynamic_label"]),
                    "times": context["raw"]["times"].astype(np.float32).tolist(),
                    "feature_names": PINN_FEATURE_NAMES,
                    "args": base.clean_json(vars(args)),
                    "best_epoch": int(epoch),
                    "best_val_score": float(best_score),
                },
                best_path,
            )
        else:
            stale += 1
        row = {
            "epoch": int(epoch),
            "train_loss": totals["loss"] / max(seen, 1),
            "train_pop_mse": totals["pop"] / max(seen, 1),
            "train_feature_mse": totals["feature"] / max(seen, 1),
            "train_dyn_ce": totals["ce"] / max(seen, 1),
            "train_monotonic": totals["mono"] / max(seen, 1),
            "train_smooth": totals["smooth"] / max(seen, 1),
            "train_acc": totals["acc"] / max(seen, 1),
            "val_score": score,
            "val_total_loss": float(val_losses["loss"].detach().cpu()),
            "val_pop_mse": float(val_losses["pop"].detach().cpu()),
            "val_feature_mse": float(val_losses["feature"].detach().cpu()),
            "val_dyn_ce": float(val_losses["ce"].detach().cpu()),
            "val_monotonic": float(val_losses["mono"].detach().cpu()),
            "val_smooth": float(val_losses["smooth"].detach().cpu()),
            "val_acc": float(val_acc.detach().cpu()),
            "val_top3_acc": float(val_top3.detach().cpu()),
            "val_macro_recall": float(val_macro.detach().cpu()),
            "best_val_score": float(best_score),
            "stale": int(stale),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        history.append(row)
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            acc_name = "acc_unused" if float(args.pinn_loss_dyn_ce) <= 0.0 else "acc"
            print(
                f"pinn epoch {epoch}/{args.pinn_epochs} "
                f"train={row['train_loss']:.5f} val={score:.5f} "
                f"pop={row['val_pop_mse']:.5f} feat={row['val_feature_mse']:.5f} "
                f"{acc_name}={row['val_acc']:.3f} macro={row['val_macro_recall']:.3f} top3={row['val_top3_acc']:.3f}",
                flush=True,
            )
        if args.pinn_patience > 0 and stale >= args.pinn_patience:
            print(f"pinn surrogate early stopping at epoch {epoch}", flush=True)
            break
    state = base.load_checkpoint(best_path, device, torch)
    model.load_state_dict(state["state_dict"])
    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / "pinntraj_surrogate_metrics.csv", index=False)
    plot_pinn_history(hist, run_dir / "figures" / "pinntraj_surrogate_history.png")
    return hist


def pinn_losses(pred_pop, true_pop, logits, y, class_weights, args, times, F, torch):
    pop_mse = F.mse_loss(pred_pop, true_pop)
    pred_feat = torch_pop_features(pred_pop, times, torch)
    true_feat = torch_pop_features(true_pop, times, torch)
    feature_mse = F.mse_loss(pred_feat, true_feat)
    ce = pinn_classification_loss(logits, y, class_weights, args, F, torch)
    mono, smooth = population_physics_penalties(pred_pop, torch)
    loss = (
        args.pinn_loss_pop * pop_mse
        + args.pinn_loss_feature * feature_mse
        + args.pinn_loss_dyn_ce * ce
        + args.pinn_loss_monotonic * mono
        + args.pinn_loss_smooth * smooth
    )
    return {"loss": loss, "pop": pop_mse, "feature": feature_mse, "ce": ce, "mono": mono, "smooth": smooth}


def pinn_selection_score(losses: dict[str, Any], args):
    if args.pinn_score_mode == "traj_feature":
        return (
            args.pinn_loss_pop * losses["pop"]
            + args.pinn_loss_feature * losses["feature"]
            + args.pinn_loss_monotonic * losses["mono"]
            + args.pinn_loss_smooth * losses["smooth"]
        )
    return losses["loss"]


def population_physics_penalties(pop, torch):
    trap = pop[:, :, 7]
    loss = pop[:, :, 8]
    mono = torch.relu(-(trap[:, 1:] - trap[:, :-1])).pow(2).mean()
    mono = mono + torch.relu(-(loss[:, 1:] - loss[:, :-1])).pow(2).mean()
    if pop.shape[1] >= 3:
        smooth = (pop[:, 2:] - 2.0 * pop[:, 1:-1] + pop[:, :-2]).pow(2).mean()
    else:
        smooth = torch.zeros((), dtype=pop.dtype, device=pop.device)
    return mono, smooth


def torch_pop_features(pop, times, torch):
    t = times.to(device=pop.device, dtype=pop.dtype)
    sink34 = pop[:, :, 2] + pop[:, :, 3]
    detour567 = pop[:, :, 4] + pop[:, :, 5] + pop[:, :, 6]
    trap = pop[:, :, 7]
    loss = pop[:, :, 8]
    residual = pop[:, :, :7].sum(dim=2)

    def at(series, value):
        idx = int(torch.argmin(torch.abs(t - float(value))).detach().cpu().item())
        return series[:, idx]

    sink_early = residence_torch(sink34, t, 0.5, 10.0, torch)
    detour_early = residence_torch(detour567, t, 0.5, 10.0, torch)
    sink_mid = residence_torch(sink34, t, 10.0, 20.0, torch)
    detour_mid = residence_torch(detour567, t, 10.0, 20.0, torch)
    q_early = (sink_early - detour_early) / (sink_early + detour_early + 1e-6)
    q_mid = (sink_mid - detour_mid) / (sink_mid + detour_mid + 1e-6)
    cols = [
        at(trap, 10.0),
        at(trap, 20.0),
        at(trap, 50.0),
        at(sink34, 5.0),
        at(detour567, 5.0),
        at(sink34, 10.0),
        at(detour567, 10.0),
        at(sink34, 20.0),
        at(detour567, 20.0),
        at(residual, 20.0),
        at(loss, 50.0),
        sink_early,
        detour_early,
        q_early,
        sink_mid,
        detour_mid,
        q_mid,
    ]
    return torch.stack(cols, dim=1)


def residence_torch(series, t, start: float, end: float, torch):
    mask = (t >= start) & (t <= end)
    idx = torch.where(mask)[0]
    if len(idx) < 2:
        return torch.zeros(series.shape[0], dtype=series.dtype, device=series.device)
    v = series.index_select(1, idx)
    tw = t.index_select(0, idx)
    dt = tw[1:] - tw[:-1]
    area = (0.5 * (v[:, :-1] + v[:, 1:]) * dt[None, :]).sum(dim=1)
    return area / torch.clamp(tw[-1] - tw[0], min=1e-6)


def make_class_weights(label_ids: np.ndarray, train_idx: np.ndarray, n_classes: int, args, torch):
    counts = np.bincount(label_ids[train_idx], minlength=n_classes).astype(np.float32)
    counts[counts < 1] = 1.0
    if not args.pinn_balanced_ce:
        return torch.ones(n_classes, dtype=torch.float32)
    mean_count = float(np.mean(counts[counts > 0]))
    weights = (mean_count / counts) ** float(args.pinn_class_weight_power)
    weights = weights / max(float(weights.mean()), 1e-8)
    return torch.tensor(weights.astype(np.float32), dtype=torch.float32)


def pinn_classification_loss(logits, y, class_weights, args, F, torch):
    weights = class_weights.to(device=logits.device, dtype=logits.dtype) if class_weights is not None else None
    try:
        ce = F.cross_entropy(logits, y, weight=weights, reduction="none", label_smoothing=float(args.pinn_label_smoothing))
    except TypeError:
        ce = F.cross_entropy(logits, y, weight=weights, reduction="none")
    gamma = float(args.pinn_focal_gamma)
    if gamma > 0:
        with torch.no_grad():
            pt = F.softmax(logits, dim=1).gather(1, y[:, None]).squeeze(1).clamp_min(1e-6)
        ce = ((1.0 - pt) ** gamma) * ce
    return ce.mean()


def topk_accuracy(logits, y, k: int):
    k = max(1, min(int(k), logits.shape[1]))
    pred = logits.topk(k, dim=1).indices
    return (pred == y[:, None]).any(dim=1).float().mean()


def macro_recall_torch(pred, y, n_classes: int, torch):
    recalls = []
    for cls in range(int(n_classes)):
        m = y == cls
        if bool(m.any()):
            recalls.append((pred[m] == cls).float().mean())
    if not recalls:
        return torch.zeros((), dtype=torch.float32, device=y.device)
    return torch.stack(recalls).mean()


def train_pinn_flow(ds, context, pinn: PINNTrajectoryBundle, args, run_dir, device, torch, TensorDataset, DataLoader) -> None:
    import torch.nn.functional as F

    condition_set = PINN_CONDITION_SET
    ready, missing = base.artifacts_ready(run_dir, condition_set)
    if ready and not args.force:
        print(f"skip existing artifacts: {condition_set}", flush=True)
        return
    if missing and not args.force:
        print(f"build missing artifacts for {condition_set}: {[str(p) for p in missing]}", flush=True)

    x = context["h_norm"].astype(np.float32)
    c = ds.norm.astype(np.float32)
    pop = context["raw"]["pop_t"].astype(np.float32)
    y = pinn.label_ids.astype(np.int64)
    train_idx = context["train_idx"]
    val_idx = context["val_idx"]
    test_idx = context["test_idx"]

    model = base.ConditionalRealNVP(x.shape[1], c.shape[1], args.layers, args.hidden, args.scale_clip).to(device)
    opt = torch.optim.AdamW(model.module().parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tensors = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(c[train_idx]), torch.tensor(pop[train_idx]), torch.tensor(y[train_idx], dtype=torch.long))
    loader = DataLoader(tensors, batch_size=args.batch_size, shuffle=True, drop_last=False)

    val_x = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    val_c = torch.tensor(c[val_idx], dtype=torch.float32, device=device)
    val_pop = torch.tensor(pop[val_idx], dtype=torch.float32, device=device)
    val_y = torch.tensor(y[val_idx], dtype=torch.long, device=device)
    if len(val_idx) > args.val_aux_max > 0:
        rng = np.random.default_rng(args.seed + 31)
        keep = np.sort(rng.choice(len(val_idx), size=args.val_aux_max, replace=False))
        val_aux_c = val_c[keep]
        val_aux_pop = val_pop[keep]
        val_aux_y = val_y[keep]
    else:
        val_aux_c = val_c
        val_aux_pop = val_pop
        val_aux_y = val_y

    best_score = float("inf")
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
        f"H_dim={x.shape[1]} C_dim={c.shape[1]} targets={len(ds.targets_norm)}",
        flush=True,
    )

    for epoch in epoch_iter:
        model.module().train()
        totals = {"loss": 0.0, "nll": 0.0, "traj": 0.0, "feature": 0.0, "ce": 0.0, "phys": 0.0, "support": 0.0}
        seen = 0
        use_aux = epoch >= args.aux_start_epoch
        for xb, cb, pb, yb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            pb = pb.to(device)
            yb = yb.to(device)
            nll = -model.log_prob(xb, cb).mean()
            traj = feature = ce = phys = support = torch.zeros((), dtype=xb.dtype, device=device)
            if use_aux and any(w > 0 for w in [args.lambda_pinn_traj, args.lambda_pinn_feature, args.lambda_pinn_dyn_ce, args.lambda_pinn_phys, args.lambda_support]):
                xg = model.sample(cb, len(cb))
                pred_pop, logits = pinn.model(xg)
                traj = F.mse_loss(pred_pop, pb)
                feature = F.mse_loss(
                    torch_pop_features(pred_pop, pinn.model.times_raw, torch),
                    torch_pop_features(pb, pinn.model.times_raw, torch),
                )
                ce = pinn_classification_loss(logits, yb, pinn.class_weights, args, F, torch)
                mono, smooth = population_physics_penalties(pred_pop, torch)
                phys = mono + smooth
                support = torch.relu(torch.abs(xg) - args.support_clip).pow(2).mean()
            loss = (
                nll
                + args.lambda_pinn_traj * traj
                + args.lambda_pinn_feature * feature
                + args.lambda_pinn_dyn_ce * ce
                + args.lambda_pinn_phys * phys
                + args.lambda_support * support
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module().parameters(), 5.0)
            opt.step()
            n = len(xb)
            seen += n
            for key, val in [("loss", loss), ("nll", nll), ("traj", traj), ("feature", feature), ("ce", ce), ("phys", phys), ("support", support)]:
                totals[key] += float(val.detach().cpu()) * n
        model.module().eval()
        with torch.no_grad():
            val_nll = -model.log_prob(val_x, val_c).mean()
            val_aux = flow_aux_loss(model, pinn, val_aux_c, val_aux_pop, val_aux_y, args, torch, F)
            val_score = val_nll + args.val_aux_weight * val_aux["weighted"]
        score = float(val_score.detach().cpu())
        if score < best_score - args.min_delta:
            best_score = score
            best_epoch = int(epoch)
            stale = 0
            torch.save(
                {
                    "state_dict": model.module().state_dict(),
                    "condition_set": condition_set,
                    "condition_names": ds.names,
                    "condition_mu": ds.mu.tolist(),
                    "condition_sd": ds.sd.tolist(),
                    "condition_flag_mask": ds.flag_mask.astype(bool).tolist(),
                    "pinn_categories": pinn.categories,
                    "best_epoch": int(best_epoch),
                    "best_val_score": float(best_score),
                    "args": base.clean_json(vars(args)),
                },
                best_path,
            )
        else:
            stale += 1
        row = {
            "condition_set": condition_set,
            "run_name": args.run_name,
            "epoch": int(epoch),
            "train_loss": totals["loss"] / max(seen, 1),
            "train_nll": totals["nll"] / max(seen, 1),
            "train_pinn_traj": totals["traj"] / max(seen, 1),
            "train_pinn_feature": totals["feature"] / max(seen, 1),
            "train_pinn_dyn_ce": totals["ce"] / max(seen, 1),
            "train_pinn_phys": totals["phys"] / max(seen, 1),
            "train_support": totals["support"] / max(seen, 1),
            "val_nll": float(val_nll.detach().cpu()),
            "val_aux_weighted": float(val_aux["weighted"].detach().cpu()),
            "val_score": score,
            "val_pinn_traj": float(val_aux["traj"].detach().cpu()),
            "val_pinn_feature": float(val_aux["feature"].detach().cpu()),
            "val_pinn_dyn_ce": float(val_aux["ce"].detach().cpu()),
            "val_pinn_phys": float(val_aux["phys"].detach().cpu()),
            "val_support": float(val_aux["support"].detach().cpu()),
            "best_val_score": float(best_score),
            "best_epoch": int(best_epoch),
            "stale": int(stale),
            "elapsed_sec": float(time.perf_counter() - t0),
        }
        history.append(row)
        if progress is not None:
            progress.set_postfix(nll=f"{row['train_nll']:.3f}", val=f"{row['val_nll']:.3f}", score=f"{score:.3f}", stale=stale)
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                f"{condition_set} epoch {epoch}/{args.epochs} "
                f"train_nll={row['train_nll']:.4f} traj={row['train_pinn_traj']:.5f} "
                f"val_nll={row['val_nll']:.4f} val_score={score:.4f} best={best_score:.4f}@{best_epoch}",
                flush=True,
            )
        if args.run_name != "smoke" and args.patience > 0 and stale >= args.patience:
            print(f"{condition_set} early stopping at epoch {epoch}", flush=True)
            break
    if progress is not None:
        progress.close()

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / f"{condition_set}_loss_history.csv", index=False)
    base.plot_loss(hist.rename(columns={"train_nll": "train_nll", "val_nll": "val_nll"}), run_dir / "figures" / f"{condition_set}_loss_curve.png")

    state = base.load_checkpoint(best_path, device, torch)
    model.module().load_state_dict(state["state_dict"])
    model.module().eval()
    metrics = base.eval_model(condition_set, model, x, c, test_idx, context, ds, device, torch)
    pd.DataFrame(metrics).to_csv(run_dir / f"{condition_set}_test_metrics.csv", index=False)
    generated_summary = base.generate_targets(condition_set, model, ds, context, args, run_dir, device, torch)
    write_condition_report(run_dir, condition_set, args, ds, metrics, generated_summary, pinn)


def flow_aux_loss(model, pinn, c, pop, y, args, torch, F):
    xg = model.sample(c, len(c))
    pred_pop, logits = pinn.model(xg)
    traj = F.mse_loss(pred_pop, pop)
    feature = F.mse_loss(
        torch_pop_features(pred_pop, pinn.model.times_raw, torch),
        torch_pop_features(pop, pinn.model.times_raw, torch),
    )
    ce = pinn_classification_loss(logits, y, pinn.class_weights, args, F, torch)
    mono, smooth = population_physics_penalties(pred_pop, torch)
    phys = mono + smooth
    support = torch.relu(torch.abs(xg) - args.support_clip).pow(2).mean()
    weighted = (
        args.lambda_pinn_traj * traj
        + args.lambda_pinn_feature * feature
        + args.lambda_pinn_dyn_ce * ce
        + args.lambda_pinn_phys * phys
        + args.lambda_support * support
    )
    return {"weighted": weighted, "traj": traj, "feature": feature, "ce": ce, "phys": phys, "support": support}


def write_pinn_eval_artifacts(model, x, pop, label_ids, categories, context, args, run_dir, device, torch) -> None:
    splits = {"train": context["train_idx"], "val": context["val_idx"], "test": context["test_idx"]}
    split_rows = []
    per_class_rows = []
    feature_rows = []
    for split_name, idx in splits.items():
        pred_pop, logits = pinn_predict(model, x, idx, args.pinn_eval_batch_size, device, torch)
        y = label_ids[idx]
        true_pop = pop[idx]
        pred = logits.argmax(axis=1)
        split_rows.append(pinn_split_metrics(split_name, pred_pop, true_pop, logits, y, categories))
        per_class_rows.extend(per_class_metrics(split_name, y, pred, categories))
        if split_name in {"val", "test"}:
            feature_rows.extend(pinn_feature_metrics(split_name, pred_pop, true_pop, PINN_FEATURE_NAMES, model.times_raw.detach().cpu().numpy()))
    pd.DataFrame(split_rows).to_csv(run_dir / "pinntraj_surrogate_split_metrics.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(run_dir / "pinntraj_surrogate_per_class_metrics.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(run_dir / "pinntraj_surrogate_feature_metrics.csv", index=False)
    write_pinn_report(run_dir, split_rows, per_class_rows, feature_rows, args)


def pinn_predict(model, x: np.ndarray, idx: np.ndarray, batch_size: int, device, torch) -> tuple[np.ndarray, np.ndarray]:
    pop_out = []
    logits_out = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(idx), max(1, int(batch_size))):
            take = idx[start : start + max(1, int(batch_size))]
            xb = torch.tensor(x[take], dtype=torch.float32, device=device)
            pop, logits = model(xb)
            pop_out.append(pop.detach().cpu().numpy())
            logits_out.append(logits.detach().cpu().numpy())
    return np.concatenate(pop_out, axis=0), np.concatenate(logits_out, axis=0)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-12)


def pinn_split_metrics(split_name: str, pred_pop: np.ndarray, true_pop: np.ndarray, logits: np.ndarray, y: np.ndarray, categories: list[str]) -> dict[str, Any]:
    pred = logits.argmax(axis=1)
    top3 = np.argsort(logits, axis=1)[:, -min(3, logits.shape[1]) :]
    prob = softmax_np(logits)
    ce = -np.log(np.maximum(prob[np.arange(len(y)), y], 1e-12)).mean()
    recalls = []
    f1s = []
    for cls in range(len(categories)):
        true_m = y == cls
        pred_m = pred == cls
        tp = int((true_m & pred_m).sum())
        recall = tp / max(int(true_m.sum()), 1)
        precision = tp / max(int(pred_m.sum()), 1)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        if true_m.any():
            recalls.append(recall)
            f1s.append(f1)
    trap_mae = float(np.mean(np.abs(pred_pop[:, :, 7] - true_pop[:, :, 7])))
    final_eta_mae = float(np.mean(np.abs(pred_pop[:, -1, 7] - true_pop[:, -1, 7])))
    trap_mono_violation = float(((pred_pop[:, 1:, 7] - pred_pop[:, :-1, 7]) < -1e-4).mean())
    loss_mono_violation = float(((pred_pop[:, 1:, 8] - pred_pop[:, :-1, 8]) < -1e-4).mean())
    return {
        "split": split_name,
        "n": int(len(y)),
        "pop_mse": float(np.mean((pred_pop - true_pop) ** 2)),
        "pop_mae": float(np.mean(np.abs(pred_pop - true_pop))),
        "trap_mae": trap_mae,
        "final_eta_mae": final_eta_mae,
        "mass_abs_err_mean": float(np.mean(np.abs(pred_pop.sum(axis=2) - 1.0))),
        "trap_mono_violation_fraction": trap_mono_violation,
        "loss_mono_violation_fraction": loss_mono_violation,
        "dyn_acc": float(np.mean(pred == y)),
        "dyn_top3_acc": float(np.mean((top3 == y[:, None]).any(axis=1))),
        "dyn_macro_recall": float(np.mean(recalls)) if recalls else np.nan,
        "dyn_macro_f1": float(np.mean(f1s)) if f1s else np.nan,
        "dyn_ce_unweighted": float(ce),
    }


def pinn_feature_metrics(split_name: str, pred_pop: np.ndarray, true_pop: np.ndarray, names: list[str], times_np: np.ndarray) -> list[dict[str, Any]]:
    pred_feat = numpy_pop_features(pred_pop, times_np)
    true_feat = numpy_pop_features(true_pop, times_np)
    rows = []
    for j, name in enumerate(names):
        err = pred_feat[:, j] - true_feat[:, j]
        mse = float(np.mean(err**2))
        mae = float(np.mean(np.abs(err)))
        var = float(np.var(true_feat[:, j]))
        rows.append({"split": split_name, "feature": name, "mse": mse, "mae": mae, "r2": float(1.0 - mse / max(var, 1e-8))})
    return rows


def numpy_pop_features(pop: np.ndarray, times: np.ndarray) -> np.ndarray:
    sink34 = pop[:, :, 2] + pop[:, :, 3]
    detour567 = pop[:, :, 4] + pop[:, :, 5] + pop[:, :, 6]
    trap = pop[:, :, 7]
    loss = pop[:, :, 8]
    residual = pop[:, :, :7].sum(axis=2)

    def at(series, value):
        idx = int(np.argmin(np.abs(times - float(value))))
        return series[:, idx]

    def res(series, start, end):
        mask = (times >= start) & (times <= end)
        tw = times[mask]
        if len(tw) < 2:
            return np.zeros(series.shape[0], dtype=np.float32)
        return (np.trapezoid(series[:, mask], tw, axis=1) / max(float(tw[-1] - tw[0]), 1e-6)).astype(np.float32)

    sink_early = res(sink34, 0.5, 10.0)
    detour_early = res(detour567, 0.5, 10.0)
    sink_mid = res(sink34, 10.0, 20.0)
    detour_mid = res(detour567, 10.0, 20.0)
    q_early = (sink_early - detour_early) / (sink_early + detour_early + 1e-6)
    q_mid = (sink_mid - detour_mid) / (sink_mid + detour_mid + 1e-6)
    return np.stack(
        [
            at(trap, 10.0),
            at(trap, 20.0),
            at(trap, 50.0),
            at(sink34, 5.0),
            at(detour567, 5.0),
            at(sink34, 10.0),
            at(detour567, 10.0),
            at(sink34, 20.0),
            at(detour567, 20.0),
            at(residual, 20.0),
            at(loss, 50.0),
            sink_early,
            detour_early,
            q_early,
            sink_mid,
            detour_mid,
            q_mid,
        ],
        axis=1,
    ).astype(np.float32)


def per_class_metrics(split_name: str, y: np.ndarray, pred: np.ndarray, categories: list[str]) -> list[dict[str, Any]]:
    rows = []
    for cls, name in enumerate(categories):
        true_m = y == cls
        pred_m = pred == cls
        tp = int((true_m & pred_m).sum())
        support = int(true_m.sum())
        pred_support = int(pred_m.sum())
        recall = tp / max(support, 1)
        precision = tp / max(pred_support, 1)
        f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
        rows.append({"split": split_name, "class_id": int(cls), "dynamic_submode": name, "support": support, "pred_support": pred_support, "precision": float(precision), "recall": float(recall), "f1": float(f1)})
    return rows


def dynamic_json_categories(labels: np.ndarray) -> list[str]:
    return sorted(np.unique(np.asarray(labels).astype(str)).tolist())


def plot_pinn_history(hist: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[0].plot(hist["epoch"], hist["val_score"], label="val")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[1].plot(hist["epoch"], hist["val_pop_mse"], label="pop")
    axes[1].plot(hist["epoch"], hist["val_feature_mse"], label="feature")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("MSE")
    axes[1].legend()
    axes[2].plot(hist["epoch"], hist["val_acc"], label="acc")
    axes[2].plot(hist["epoch"], hist["val_macro_recall"], label="macro")
    axes[2].plot(hist["epoch"], hist["val_top3_acc"], label="top3")
    axes[2].set_xlabel("epoch")
    axes[2].set_ylabel("dynamic label")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_pinn_report(run_dir: Path, split_rows, per_class_rows, feature_rows, args) -> None:
    split_df = pd.DataFrame(split_rows)
    per_df = pd.DataFrame(per_class_rows)
    feat_df = pd.DataFrame(feature_rows)
    val_low = per_df[(per_df["split"] == "val") & (per_df["support"] > 0)].sort_values("recall").head(8)
    feat_low = feat_df[feat_df["split"] == "val"].sort_values("r2").head(8) if not feat_df.empty else pd.DataFrame()
    lines = [
        "# PINN-lite population trajectory surrogate audit",
        "",
        "이 surrogate는 full density-matrix PINN이 아니라 population trajectory PINN-lite입니다. 현재 artifact에는 rho(t)가 아니라 pop_t만 있으므로, 물리 제약은 population-level constraint로 걸었습니다.",
        "",
        "## Config",
        "",
        f"- hidden/depth/dropout: `{args.pinn_hidden}` / `{args.pinn_depth}` / `{args.pinn_dropout}`",
        f"- loss weights: pop `{args.pinn_loss_pop}`, feature `{args.pinn_loss_feature}`, dyn_ce `{args.pinn_loss_dyn_ce}`, monotonic `{args.pinn_loss_monotonic}`, smooth `{args.pinn_loss_smooth}`",
        f"- score_mode: `{args.pinn_score_mode}`",
        f"- balanced_ce: `{args.pinn_balanced_ce}`",
        f"- label_smoothing: `{args.pinn_label_smoothing}`",
        "",
        "## Split metrics",
        "",
        "```text",
        split_df.to_string(index=False),
        "```",
        "",
        "## Lowest validation dynamic recalls",
        "",
        "```text",
        val_low.to_string(index=False),
        "```",
        "",
        "## Lowest validation feature R2",
        "",
        "```text",
        feat_low.to_string(index=False) if not feat_low.empty else "(no feature metrics)",
        "```",
        "",
        "## Interpretation",
        "",
        "우선 pop_mse/trap_mae/final_eta_mae와 route feature R2를 봐야 합니다. dynamic label accuracy는 보조 지표입니다. 이 teacher가 trajectory를 못 맞추면 flow auxiliary로 쓰면 안 됩니다.",
    ]
    (run_dir / "reports" / "pinntraj_surrogate_audit_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_metadata(out_root: Path, prepared: Path, args, context, ds) -> None:
    metadata_dir = out_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    context["dynamic_submode_summary"].to_csv(metadata_dir / "dynamic_submode_summary.csv", index=False)
    pd.DataFrame(
        {
            "dynamic_submode": context["dynamic_label"],
            "priority_group": context["priority_group"],
            "sample_index": context["raw"]["sample_index"].astype(np.int64),
        }
    ).to_csv(metadata_dir / "dynamic_submode_assignments.csv", index=False)
    pd.DataFrame(
        [
            {
                "condition_set": PINN_CONDITION_SET,
                "base_condition_set": BASE_CONDITION_SET,
                "condition_dim": int(ds.raw.shape[1]),
                "n_targets": int(len(ds.targets_norm)),
                "uses_pinntraj_guidance": True,
                "condition_names": json.dumps(ds.names, ensure_ascii=False),
                "target_names": json.dumps(sorted(ds.targets_norm), ensure_ascii=False),
            }
        ]
    ).to_csv(metadata_dir / "condition_design.csv", index=False)
    manifest = {
        "purpose": "H27 DYNZ conditional flow with PINN-lite population trajectory surrogate guidance",
        "prepared": str(prepared),
        "out_root": str(out_root),
        "condition_set": PINN_CONDITION_SET,
        "base_condition_set": BASE_CONDITION_SET,
        "args": base.clean_json(vars(args)),
        "label_boundary": "The surrogate learns H,t -> pop_t from the row-aligned H27 140k artifact. It is not a full rho(t) PINN.",
        "success_check": "First inspect pinntraj surrogate metrics; then use simulator validation plus diversity audit for generated H.",
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(base.clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    write_experiment_readme(out_root)


def write_experiment_readme(out_root: Path) -> None:
    lines = [
        "# 20260619 H27 DYNZ PINNTRAJ Flow",
        "",
        "Purpose: replace the weak H->dynamic-label surrogate with a richer H,t->population trajectory surrogate.",
        "",
        f"Main condition set: `{PINN_CONDITION_SET}`.",
        "",
        "Important boundary: this is PINN-lite, not a full density-matrix PINN, because the prepared 140k artifact stores population trajectories rather than rho(t).",
        "",
        "First inspect `pinntraj_surrogate_split_metrics.csv`, `pinntraj_surrogate_feature_metrics.csv`, and `reports/pinntraj_surrogate_audit_kr.md`. Only then run the full flow.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def write_condition_report(run_dir, condition_set, args, ds, metrics, generated_summary, pinn) -> None:
    metric_df = pd.DataFrame(metrics)
    lines = [
        f"# {condition_set} H27 PINN-lite trajectory-guided flow run",
        "",
        f"- run_name: `{args.run_name}`",
        f"- condition_dim: `{ds.raw.shape[1]}`",
        f"- targets: `{len(ds.targets_norm)}`",
        f"- pinn_dynamic_classes: `{len(pinn.categories)}`",
        f"- aux weights: TRAJ `{args.lambda_pinn_traj}`, FEATURE `{args.lambda_pinn_feature}`, DYN_CE `{args.lambda_pinn_dyn_ce}`, PHYS `{args.lambda_pinn_phys}`, SUPPORT `{args.lambda_support}`",
        "",
        "## Test NLL",
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
        "This run uses a learned population trajectory surrogate as a differentiable reverse signal. It is not a success until real simulator validation and diversity audit improve over DYNZ/SURRSEP baselines.",
    ]
    (run_dir / "reports" / f"{condition_set}_run_report_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_run_summary(run_dir: Path, args) -> None:
    metric_paths = sorted(run_dir.glob("*_test_metrics.csv"))
    if metric_paths:
        metrics = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)
        metrics.to_csv(run_dir / "all_test_metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "condition_sets": [PINN_CONDITION_SET],
        "generated_files": [str(p) for p in sorted(run_dir.glob("*_generated_samples.npz"))],
        "metric_files": [str(p) for p in sorted(run_dir.glob("*_test_metrics.csv"))],
        "args": base.clean_json(vars(args)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(base.clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
