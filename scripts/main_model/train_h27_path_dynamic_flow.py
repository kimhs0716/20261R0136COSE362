#!/usr/bin/env python3
"""Train H27 CFAST+PATH conditional RealNVP variants with 140k dynamic submodes.

This script deliberately builds dynamic submode labels from the row-aligned
H27 140k prepared artifact. It does not attach the older 62k D/S labels.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PREPARED = Path("outputs/h27_context_ablation_140k_cfast_prepare/prepared_flow_pilot_data.npz")
DEFAULT_OUT_ROOT = Path("outputs/experiments/20260619_h27_path_dyn_flow")
DEFAULT_CONDITION_SETS = "CFAST_CL1_PATH,CFAST_CL1_PATH_DYN,CFAST_CL1_PATH_DYNZ,CFAST_CL1_PATH_DYNBAL"
TARGET_GROUPS = ("fast_high", "very_fast", "late_high", "non_high")
IU = np.triu_indices(7)
DIAG_IDX = np.where(IU[0] == IU[1])[0]
OFF_IDX = np.where(IU[0] != IU[1])[0]


@dataclass
class ConditionDataset:
    name: str
    raw: np.ndarray
    norm: np.ndarray
    names: list[str]
    mu: np.ndarray
    sd: np.ndarray
    flag_mask: np.ndarray
    dynamic_label: np.ndarray | None
    balance_label: np.ndarray | None
    targets_raw: dict[str, np.ndarray]
    targets_norm: dict[str, np.ndarray]


class ConditionalRealNVP:
    def __init__(self, x_dim: int, c_dim: int, n_layers: int, hidden: int, scale_clip: float):
        import torch
        import torch.nn as nn

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
                    nn.Linear(self.x_dim + self.c_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 2 * self.x_dim),
                )
                for _ in range(int(n_layers))
            ]
        )
        for net in self.nets:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)

    def module(self):
        return self.nets

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
        base = -0.5 * (z * z + math.log(2.0 * math.pi)).sum(dim=1)
        return base + logdet

    def sample(self, c, n: int):
        if c.ndim == 1:
            c = c[None, :].expand(int(n), -1)
        z = self.torch.randn(int(n), self.x_dim, device=c.device, dtype=c.dtype)
        return self.z_to_x(z, c)


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
    p.add_argument("--seed", type=int, default=20260619)
    p.add_argument("--device", default="auto")
    p.add_argument("--force", action="store_true")
    p.add_argument("--dyn-k-fast", type=int, default=3)
    p.add_argument("--dyn-k-very-fast", type=int, default=3)
    p.add_argument("--dyn-k-late", type=int, default=4)
    p.add_argument("--dyn-k-nonhigh", type=int, default=6)
    p.add_argument("--dyn-k-other", type=int, default=1)
    p.add_argument("--kmeans-iter", type=int, default=80)
    p.add_argument("--kmeans-init", type=int, default=8)
    p.add_argument("--min-dyn-target-rows", type=int, default=80)
    p.add_argument("--max-dyn-targets-per-group", type=int, default=3)
    p.add_argument("--target-groups", default=",".join(TARGET_GROUPS))
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--metadata-only", action="store_true", help="Build dynamic labels/condition metadata and exit before training.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    np.random.seed(args.seed)
    prepared = resolve_prepared(args.prepared)
    out_root = args.out_root
    run_dir = out_root / args.run_name
    metadata_dir = out_root / "metadata"
    for path in [run_dir, metadata_dir, run_dir / "checkpoints", run_dir / "figures", run_dir / "reports"]:
        path.mkdir(parents=True, exist_ok=True)

    raw = np.load(prepared, allow_pickle=True)
    validate_prepared(raw)
    context = build_context(raw, args)
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
    if args.metadata_only:
        print("metadata-only complete:", out_root / "metadata", flush=True)
        return 0

    import torch
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

    torch.manual_seed(args.seed)
    device = choose_device(args.device, torch)
    print(f"device: {device}", flush=True)

    results = []
    for condition_set in selected:
        ready, missing = artifacts_ready(run_dir, condition_set)
        if ready and not args.force:
            print(f"skip existing artifacts: {condition_set}", flush=True)
            continue
        if missing and not args.force:
            print(f"build missing artifacts for {condition_set}: {[str(p) for p in missing]}", flush=True)
        results.append(
            train_one(
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
        )
    write_run_summary(run_dir, args, selected, condition_data)
    print("done:", run_dir, flush=True)
    return 0


def choose_device(name: str, torch):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_prepared(preferred: Path) -> Path:
    candidates = [
        preferred,
        DEFAULT_PREPARED,
        Path("/content/drive/MyDrive/Colab Notebooks/fmo_research_clean") / DEFAULT_PREPARED,
        Path("/content/drive/MyDrive/fmo_research_clean") / DEFAULT_PREPARED,
        Path("/content/drive/MyDrive") / DEFAULT_PREPARED,
    ]
    for path in candidates:
        if path.exists():
            return path
    roots = [Path.cwd(), Path("/content/drive/MyDrive")]
    for root in roots:
        if not root.exists():
            continue
        try:
            for path in root.rglob("prepared_flow_pilot_data.npz"):
                if path.parent.name == "h27_context_ablation_140k_cfast_prepare":
                    return path
        except Exception:
            pass
    raise FileNotFoundError(f"missing prepared artifact: {preferred}")


def validate_prepared(raw: np.lib.npyio.NpzFile) -> None:
    required = [
        "CFAST_condition_raw",
        "CFAST_condition_names",
        "CFAST_H_norm",
        "CFAST_H_mu",
        "CFAST_H_sd",
        "split_train",
        "split_val",
        "split_test",
        "pop_t",
        "times",
        "c_l1",
    ]
    missing = [k for k in required if k not in raw.files]
    if missing:
        raise KeyError(f"prepared artifact missing keys: {missing}")
    if raw["CFAST_H_norm"].shape[1] != 28:
        raise ValueError("expected CFAST_H_norm to be 28D vec28 coordinates")


def build_context(raw: np.lib.npyio.NpzFile, args: argparse.Namespace) -> dict[str, Any]:
    train_idx = raw["split_train"].astype(np.int64)
    cfast_raw = raw["CFAST_condition_raw"].astype(np.float32)
    cfast_names = [str(x) for x in raw["CFAST_condition_names"].tolist()]
    masks, priority_group = build_groups(cfast_raw, cfast_names)
    path_raw, path_names = build_path_features(raw["pop_t"], raw["times"])

    cl1 = raw["c_l1"].astype(np.float32)[:, None]
    path_selected_names = [
        "sink34_at_5p0ps",
        "detour567_at_5p0ps",
        "sink34_at_10p0ps",
        "detour567_at_10p0ps",
        "sink34_at_20p0ps",
        "detour567_at_20p0ps",
        "residual_at_20p0ps",
        "loss_at_50p0ps",
        "residence_sink34_0p5_10p0ps",
        "residence_detour567_0p5_10p0ps",
        "residence_sink34_10p0_20p0ps",
        "residence_detour567_10p0_20p0ps",
        "q_route_0p5_10p0ps",
        "q_route_10p0_20p0ps",
    ]
    path_selected_idx = [path_names.index(x) for x in path_selected_names]
    path_selected_raw = path_raw[:, path_selected_idx].astype(np.float32)

    dyn_feature_names = cfast_names + ["c_l1"] + path_selected_names
    dyn_feature_raw = np.concatenate([cfast_raw, cl1, path_selected_raw], axis=1).astype(np.float32)
    dyn_flag_mask = np.array([name == "t80_observed_flag" for name in dyn_feature_names], dtype=bool)
    dyn_feature_norm, dyn_mu, dyn_sd = standardize_train(dyn_feature_raw, train_idx, dyn_flag_mask)
    dynamic_label, dyn_summary, dyn_centers = fit_dynamic_submodes(
        dyn_feature_norm,
        dyn_feature_raw,
        dyn_feature_names,
        priority_group,
        train_idx,
        args,
    )
    dyn_oh, dyn_oh_names = one_hot(dynamic_label, "dyn")
    dyn_z, dyn_z_names, pca_meta = pca_features(dyn_feature_norm, train_idx, prefix="dynz", n_components=2)

    return {
        "raw": raw,
        "train_idx": train_idx,
        "val_idx": raw["split_val"].astype(np.int64),
        "test_idx": raw["split_test"].astype(np.int64),
        "h_norm": raw["CFAST_H_norm"].astype(np.float32),
        "h_mu": raw["CFAST_H_mu"].astype(np.float32),
        "h_sd": raw["CFAST_H_sd"].astype(np.float32),
        "cfast_raw": cfast_raw,
        "cfast_names": cfast_names,
        "cl1": cl1,
        "path_selected_raw": path_selected_raw,
        "path_selected_names": path_selected_names,
        "priority_group": priority_group,
        "masks": masks,
        "dynamic_label": dynamic_label,
        "dyn_oh": dyn_oh,
        "dyn_oh_names": dyn_oh_names,
        "dyn_z": dyn_z,
        "dyn_z_names": dyn_z_names,
        "dyn_feature_raw": dyn_feature_raw,
        "dyn_feature_norm": dyn_feature_norm,
        "dyn_feature_names": dyn_feature_names,
        "dynamic_submode_summary": dyn_summary,
        "dynamic_centers": dyn_centers,
        "dynamic_pca_meta": pca_meta,
    }


def standardize_train(x: np.ndarray, train_idx: np.ndarray, flag_mask: np.ndarray | None = None, eps: float = 1e-8):
    x = np.asarray(x, dtype=np.float32)
    mu = np.nanmean(x[train_idx], axis=0).astype(np.float32)
    sd = np.nanstd(x[train_idx], axis=0).astype(np.float32)
    sd[sd < eps] = 1.0
    if flag_mask is not None:
        mu[flag_mask] = 0.0
        sd[flag_mask] = 1.0
    return ((x - mu) / sd).astype(np.float32), mu, sd


def interp_rows(values: np.ndarray, times: np.ndarray, query: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    return np.stack([np.interp([query], times, row)[0] for row in values], axis=0).astype(np.float32)


def residence(values: np.ndarray, times: np.ndarray, start: float, end: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    mask = (times >= start) & (times <= end)
    if mask.sum() < 2:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    width = float(times[mask][-1] - times[mask][0])
    if width <= 0:
        return np.full(values.shape[0], np.nan, dtype=np.float32)
    return (np.trapezoid(values[:, mask], times[mask], axis=1) / width).astype(np.float32)


def build_path_features(pop_t: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, list[str]]:
    pop = np.asarray(pop_t, dtype=np.float32)
    times = np.asarray(times, dtype=np.float32)
    sink34 = pop[:, :, 2] + pop[:, :, 3]
    detour567 = pop[:, :, 4] + pop[:, :, 5] + pop[:, :, 6]
    trap = pop[:, :, 7]
    loss = pop[:, :, 8]
    residual = pop[:, :, :7].sum(axis=2)
    series = {
        "sink34": sink34,
        "detour567": detour567,
        "trap": trap,
        "loss": loss,
        "residual": residual,
    }
    cols: list[np.ndarray] = []
    names: list[str] = []
    for name in ["sink34", "detour567", "residual", "loss"]:
        for t in [5.0, 10.0, 20.0, 50.0]:
            cols.append(interp_rows(series[name], times, t)[:, None])
            names.append(f"{name}_at_{fmt_time(t)}ps")
    for name in ["sink34", "detour567", "residual", "loss"]:
        for start, end in [(0.5, 10.0), (10.0, 20.0), (20.0, 50.0)]:
            cols.append(residence(series[name], times, start, end)[:, None])
            names.append(f"residence_{name}_{fmt_time(start)}_{fmt_time(end)}ps")
    sink_early = residence(sink34, times, 0.5, 10.0)
    detour_early = residence(detour567, times, 0.5, 10.0)
    sink_mid = residence(sink34, times, 10.0, 20.0)
    detour_mid = residence(detour567, times, 10.0, 20.0)
    q_early = (sink_early - detour_early) / (sink_early + detour_early + 1e-6)
    q_mid = (sink_mid - detour_mid) / (sink_mid + detour_mid + 1e-6)
    cols.extend([q_early[:, None].astype(np.float32), q_mid[:, None].astype(np.float32)])
    names.extend(["q_route_0p5_10p0ps", "q_route_10p0_20p0ps"])
    return np.concatenate(cols, axis=1).astype(np.float32), names


def fmt_time(value: float) -> str:
    return str(float(value)).replace(".", "p")


def build_groups(cfast_raw: np.ndarray, cfast_names: list[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    idx = {name: i for i, name in enumerate(cfast_names)}
    eta20 = cfast_raw[:, idx["eta20"]]
    eta50 = cfast_raw[:, idx["eta50"]]
    t80 = cfast_raw[:, idx["t80_cap"]]
    obs = cfast_raw[:, idx["t80_observed_flag"]] > 0.5
    priority = np.full(len(cfast_raw), "other_high", dtype=object)
    non_high = eta50 < 0.90
    late_high = (eta50 >= 0.90) & (eta20 < 0.90)
    fast_high = (eta50 >= 0.90) & obs & (t80 <= 15.0)
    very_fast = (eta50 >= 0.90) & obs & (t80 <= 10.0)
    priority[non_high] = "non_high"
    priority[late_high] = "late_high"
    priority[fast_high] = "fast_high"
    priority[very_fast] = "very_fast"
    names = ["very_fast", "fast_high", "late_high", "non_high", "other_high"]
    return {name: priority == name for name in names}, priority


def fit_dynamic_submodes(
    feature_norm: np.ndarray,
    feature_raw: np.ndarray,
    feature_names: list[str],
    priority_group: np.ndarray,
    train_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, np.ndarray]]:
    k_by_group = {
        "very_fast": args.dyn_k_very_fast,
        "fast_high": args.dyn_k_fast,
        "late_high": args.dyn_k_late,
        "non_high": args.dyn_k_nonhigh,
        "other_high": args.dyn_k_other,
    }
    labels = np.empty(len(priority_group), dtype=object)
    centers_by_group: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    for group, k_requested in k_by_group.items():
        all_mask = priority_group == group
        fit_idx = train_idx[all_mask[train_idx]]
        if len(fit_idx) == 0:
            labels[all_mask] = f"{group}_D00"
            continue
        k = int(max(1, min(k_requested, len(fit_idx))))
        centers = kmeans_fit(
            feature_norm[fit_idx],
            k,
            seed=args.seed + stable_int(group),
            n_iter=args.kmeans_iter,
            n_init=args.kmeans_init,
        )
        group_labels = kmeans_predict(feature_norm[all_mask], centers)
        group_rows = np.where(all_mask)[0]
        for j in range(k):
            labels[group_rows[group_labels == j]] = f"{group}_D{j:02d}"
        centers_by_group[group] = centers.astype(np.float32)

    for label in sorted(set(labels.astype(str))):
        idx = np.where(labels.astype(str) == label)[0]
        train_mask = np.isin(idx, train_idx)
        group = label.rsplit("_D", 1)[0]
        row: dict[str, Any] = {
            "dynamic_submode": label,
            "priority_group": group,
            "n_total": int(len(idx)),
            "n_train": int(train_mask.sum()),
            "fraction_total": float(len(idx) / len(labels)),
        }
        for name in ["eta10", "eta20", "eta50", "t80_cap", "t80_observed_flag"]:
            if name in feature_names:
                col = feature_names.index(name)
                row[f"{name}_median"] = float(np.nanmedian(feature_raw[idx, col]))
        for name in [
            "residence_sink34_0p5_10p0ps",
            "residence_detour567_0p5_10p0ps",
            "q_route_0p5_10p0ps",
            "c_l1",
        ]:
            if name in feature_names:
                col = feature_names.index(name)
                row[f"{name}_median"] = float(np.nanmedian(feature_raw[idx, col]))
        rows.append(row)
    return labels.astype(str), pd.DataFrame(rows), centers_by_group


def stable_int(text: str) -> int:
    return sum((i + 1) * ord(ch) for i, ch in enumerate(text)) % 100000


def kmeans_fit(x: np.ndarray, k: int, seed: int, n_iter: int = 80, n_init: int = 8) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    n = len(x)
    if n == 0:
        raise ValueError("cannot fit kmeans on empty array")
    k = min(max(1, int(k)), n)
    best_centers = None
    best_inertia = np.inf
    for _ in range(max(1, int(n_init))):
        centers = x[rng.choice(n, size=k, replace=False)].copy()
        labels = np.full(n, -1, dtype=np.int64)
        for _step in range(max(1, int(n_iter))):
            d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                m = labels == j
                centers[j] = x[m].mean(axis=0) if m.any() else x[rng.integers(0, n)]
        d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        labels = np.argmin(d2, axis=1)
        inertia = float(((x - centers[labels]) ** 2).sum())
        if inertia < best_inertia:
            best_inertia = inertia
            best_centers = centers.copy()
    assert best_centers is not None
    return best_centers.astype(np.float32)


def kmeans_predict(x: np.ndarray, centers: np.ndarray) -> np.ndarray:
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def one_hot(labels: np.ndarray, prefix: str) -> tuple[np.ndarray, list[str]]:
    labels = np.asarray(labels).astype(str)
    cats = sorted(np.unique(labels).tolist())
    out = np.zeros((len(labels), len(cats)), dtype=np.float32)
    pos = {cat: i for i, cat in enumerate(cats)}
    for i, label in enumerate(labels):
        out[i, pos[label]] = 1.0
    return out, [f"{prefix}_{sanitize_name(cat)}" for cat in cats]


def sanitize_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", str(text))


def pca_features(x: np.ndarray, train_idx: np.ndarray, prefix: str, n_components: int = 2):
    train = np.asarray(x[train_idx], dtype=np.float32)
    mu = train.mean(axis=0)
    centered = train - mu
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:n_components].astype(np.float32)
    z = (np.asarray(x, dtype=np.float32) - mu) @ components.T
    z_norm, z_mu, z_sd = standardize_train(z.astype(np.float32), train_idx)
    total = float((s * s).sum())
    explained = (s[:n_components] * s[:n_components]) / total if total > 0 else np.zeros(n_components)
    names = [f"{prefix}{i + 1}" for i in range(n_components)]
    meta = {
        "feature_mu": mu.astype(np.float32),
        "components": components,
        "z_mu": z_mu,
        "z_sd": z_sd,
        "explained_variance_ratio": explained.astype(np.float32),
    }
    return z_norm.astype(np.float32), names, meta


def build_condition_datasets(context: dict[str, Any], args: argparse.Namespace) -> dict[str, ConditionDataset]:
    cfast_raw = context["cfast_raw"]
    cfast_names = context["cfast_names"]
    cl1 = context["cl1"]
    path = context["path_selected_raw"]
    path_names = context["path_selected_names"]
    dyn_oh = context["dyn_oh"]
    dyn_oh_names = context["dyn_oh_names"]
    dyn_z = context["dyn_z"]
    dyn_z_names = context["dyn_z_names"]
    dyn_label = context["dynamic_label"]

    base_raw = np.concatenate([cfast_raw, cl1, path], axis=1).astype(np.float32)
    base_names = cfast_names + ["c_l1"] + path_names
    data = {
        "CFAST_CL1_PATH": make_condition_dataset(
            "CFAST_CL1_PATH",
            base_raw,
            base_names,
            context,
            flag_names={"t80_observed_flag"},
            dynamic_label=None,
            balance_label=None,
        ),
        "CFAST_CL1_PATH_DYN": make_condition_dataset(
            "CFAST_CL1_PATH_DYN",
            np.concatenate([base_raw, dyn_oh], axis=1).astype(np.float32),
            base_names + dyn_oh_names,
            context,
            flag_names={"t80_observed_flag", *dyn_oh_names},
            dynamic_label=dyn_label,
            balance_label=None,
        ),
        "CFAST_CL1_PATH_DYNZ": make_condition_dataset(
            "CFAST_CL1_PATH_DYNZ",
            np.concatenate([base_raw, dyn_z], axis=1).astype(np.float32),
            base_names + dyn_z_names,
            context,
            flag_names={"t80_observed_flag"},
            dynamic_label=dyn_label,
            balance_label=None,
        ),
        "CFAST_CL1_PATH_DYNBAL": make_condition_dataset(
            "CFAST_CL1_PATH_DYNBAL",
            np.concatenate([base_raw, dyn_oh], axis=1).astype(np.float32),
            base_names + dyn_oh_names,
            context,
            flag_names={"t80_observed_flag", *dyn_oh_names},
            dynamic_label=dyn_label,
            balance_label=dyn_label,
        ),
    }
    for ds in data.values():
        ds.targets_raw, ds.targets_norm = make_target_conditions(ds, context, args)
    return data


def make_condition_dataset(
    name: str,
    raw: np.ndarray,
    names: list[str],
    context: dict[str, Any],
    flag_names: set[str],
    dynamic_label: np.ndarray | None,
    balance_label: np.ndarray | None,
) -> ConditionDataset:
    flag_mask = np.array([n in flag_names for n in names], dtype=bool)
    norm, mu, sd = standardize_train(raw, context["train_idx"], flag_mask=flag_mask)
    return ConditionDataset(
        name=name,
        raw=raw.astype(np.float32),
        norm=norm.astype(np.float32),
        names=list(names),
        mu=mu.astype(np.float32),
        sd=sd.astype(np.float32),
        flag_mask=flag_mask,
        dynamic_label=dynamic_label,
        balance_label=balance_label,
        targets_raw={},
        targets_norm={},
    )


def normalize_condition(raw_vec: np.ndarray, ds: ConditionDataset) -> np.ndarray:
    return ((np.asarray(raw_vec, dtype=np.float32) - ds.mu) / ds.sd).astype(np.float32)


def make_target_conditions(
    ds: ConditionDataset, context: dict[str, Any], args: argparse.Namespace
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    target_groups = [x.strip() for x in args.target_groups.split(",") if x.strip()]
    out_raw: dict[str, np.ndarray] = {}
    out_norm: dict[str, np.ndarray] = {}
    train_idx = context["train_idx"]
    masks = context["masks"]
    raw_c = ds.raw
    for group in target_groups:
        idx = train_idx[masks[group][train_idx]]
        if len(idx) == 0:
            continue
        med = np.nanmedian(raw_c[idx], axis=0).astype(np.float32)
        out_raw[group] = med
        out_norm[group] = normalize_condition(med, ds)

    qcols = [i for i, n in enumerate(ds.names) if n == "q_route_0p5_10p0ps"]
    if qcols:
        qcol = qcols[0]
        for group in target_groups:
            idx = train_idx[masks[group][train_idx]]
            if len(idx) < 50:
                continue
            q = raw_c[idx, qcol]
            lo, hi = np.nanquantile(q, [0.25, 0.75])
            for tag, sel in [("route_low", q <= lo), ("route_high", q >= hi)]:
                sub = idx[sel]
                if len(sub) >= 30:
                    med = np.nanmedian(raw_c[sub], axis=0).astype(np.float32)
                    key = f"{group}_{tag}"
                    out_raw[key] = med
                    out_norm[key] = normalize_condition(med, ds)

    if ds.dynamic_label is not None:
        labels = np.asarray(ds.dynamic_label).astype(str)
        for group in target_groups:
            base_idx = train_idx[masks[group][train_idx]]
            counts = pd.Series(labels[base_idx]).value_counts()
            counts = counts[counts >= args.min_dyn_target_rows].head(args.max_dyn_targets_per_group)
            for dyn_label, n_rows in counts.items():
                sub = base_idx[labels[base_idx] == dyn_label]
                med = np.nanmedian(raw_c[sub], axis=0).astype(np.float32)
                key = f"{group}_dyn_{sanitize_name(dyn_label)}"
                out_raw[key] = med
                out_norm[key] = normalize_condition(med, ds)
    return out_raw, out_norm


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


def train_one(
    condition_set: str,
    ds: ConditionDataset,
    context: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    device,
    torch,
    TensorDataset,
    DataLoader,
    WeightedRandomSampler,
) -> dict[str, Any]:
    x = context["h_norm"].astype(np.float32)
    c = ds.norm.astype(np.float32)
    train_idx = context["train_idx"]
    val_idx = context["val_idx"]
    test_idx = context["test_idx"]

    model = ConditionalRealNVP(x.shape[1], c.shape[1], args.layers, args.hidden, args.scale_clip).to(device)
    opt = torch.optim.AdamW(model.module().parameters(), lr=args.lr, weight_decay=args.weight_decay)
    tensors = TensorDataset(torch.tensor(x[train_idx]), torch.tensor(c[train_idx]))
    sampler = None
    shuffle = True
    if ds.balance_label is not None:
        train_labels = np.asarray(ds.balance_label)[train_idx].astype(str)
        counts = pd.Series(train_labels).value_counts()
        weights = np.array([1.0 / float(counts[label]) for label in train_labels], dtype=np.float64)
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=len(weights), replacement=True)
        shuffle = False
    loader = DataLoader(tensors, batch_size=args.batch_size, shuffle=shuffle, sampler=sampler, drop_last=False)
    val_x = torch.tensor(x[val_idx], dtype=torch.float32, device=device)
    val_c = torch.tensor(c[val_idx], dtype=torch.float32, device=device)

    best_val = float("inf")
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
        f"H_dim={x.shape[1]} C_dim={c.shape[1]} targets={len(ds.targets_norm)} balanced={ds.balance_label is not None}",
        flush=True,
    )

    for epoch in epoch_iter:
        model.module().train()
        total = 0.0
        seen = 0
        for xb, cb in loader:
            xb = xb.to(device)
            cb = cb.to(device)
            loss = -model.log_prob(xb, cb).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.module().parameters(), 5.0)
            opt.step()
            total += float(loss.detach().cpu().item()) * len(xb)
            seen += len(xb)
        train_nll = total / max(seen, 1)
        model.module().eval()
        with torch.no_grad():
            val_nll = float((-model.log_prob(val_x, val_c).mean()).detach().cpu().item())
        if val_nll < best_val - args.min_delta:
            best_val = val_nll
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
                    "best_epoch": int(best_epoch),
                    "best_val_nll": float(best_val),
                    "args": clean_json(vars(args)),
                },
                best_path,
            )
        else:
            stale += 1
        history.append(
            {
                "condition_set": condition_set,
                "run_name": args.run_name,
                "epoch": int(epoch),
                "train_nll": float(train_nll),
                "val_nll": float(val_nll),
                "best_val_nll": float(best_val),
                "best_epoch": int(best_epoch),
                "stale": int(stale),
                "elapsed_sec": float(time.perf_counter() - t0),
            }
        )
        if progress is not None:
            progress.set_postfix(train=f"{train_nll:.3f}", val=f"{val_nll:.3f}", best=f"{best_val:.3f}", stale=stale)
        if epoch == 1 or epoch % max(1, args.log_every) == 0:
            print(
                f"{condition_set} epoch {int(epoch)}/{args.epochs} "
                f"train={train_nll:.4f} val={val_nll:.4f} best={best_val:.4f}@{best_epoch}",
                flush=True,
            )
        if args.run_name != "smoke" and args.patience > 0 and stale >= args.patience:
            print(f"{condition_set} early stopping at epoch {int(epoch)}", flush=True)
            break
    if progress is not None:
        progress.close()

    hist = pd.DataFrame(history)
    hist.to_csv(run_dir / f"{condition_set}_loss_history.csv", index=False)
    plot_loss(hist, run_dir / "figures" / f"{condition_set}_loss_curve.png")

    state = load_checkpoint(best_path, device, torch)
    model.module().load_state_dict(state["state_dict"])
    model.module().eval()
    metrics = eval_model(condition_set, model, x, c, test_idx, context, ds, device, torch)
    pd.DataFrame(metrics).to_csv(run_dir / f"{condition_set}_test_metrics.csv", index=False)
    generated_summary = generate_targets(condition_set, model, ds, context, args, run_dir, device, torch)
    write_condition_report(run_dir, condition_set, args, ds, metrics, generated_summary)
    return {"condition_set": condition_set, "metrics": metrics}


def load_checkpoint(path: Path, device, torch):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)
    except Exception as exc:
        text = str(exc)
        if "Weights only load failed" in text or "Unsupported global" in text:
            print(
                f"warning: retrying trusted local checkpoint with weights_only=False: {path}",
                flush=True,
            )
            return torch.load(path, map_location=device, weights_only=False)
        raise


def eval_model(
    condition_set: str,
    model: ConditionalRealNVP,
    x: np.ndarray,
    c: np.ndarray,
    test_idx: np.ndarray,
    context: dict[str, Any],
    ds: ConditionDataset,
    device,
    torch,
) -> list[dict[str, Any]]:
    xt = torch.tensor(x[test_idx], dtype=torch.float32, device=device)
    ct = torch.tensor(c[test_idx], dtype=torch.float32, device=device)
    with torch.no_grad():
        lp = model.log_prob(xt, ct).detach().cpu().numpy()
    priority = context["priority_group"][test_idx]
    rows = [
        {
            "condition_set": condition_set,
            "group_type": "overall",
            "group": "overall",
            "n": int(len(lp)),
            "mean_logp": float(lp.mean()),
            "nll": float(-lp.mean()),
        }
    ]
    for group in sorted(set(priority)):
        m = priority == group
        rows.append(
            {
                "condition_set": condition_set,
                "group_type": "priority_group",
                "group": str(group),
                "n": int(m.sum()),
                "mean_logp": float(lp[m].mean()),
                "nll": float(-lp[m].mean()),
            }
        )
    if ds.dynamic_label is not None:
        dyn = np.asarray(ds.dynamic_label)[test_idx].astype(str)
        for label, n in pd.Series(dyn).value_counts().items():
            if int(n) < 20:
                continue
            m = dyn == label
            rows.append(
                {
                    "condition_set": condition_set,
                    "group_type": "dynamic_submode",
                    "group": str(label),
                    "n": int(m.sum()),
                    "mean_logp": float(lp[m].mean()),
                    "nll": float(-lp[m].mean()),
                }
            )
    return rows


def generate_targets(
    condition_set: str,
    model: ConditionalRealNVP,
    ds: ConditionDataset,
    context: dict[str, Any],
    args: argparse.Namespace,
    run_dir: Path,
    device,
    torch,
) -> pd.DataFrame:
    generated: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    h_mu = context["h_mu"]
    h_sd = context["h_sd"]
    train_h_raw = gauge_fix_vec28(context["h_norm"][context["train_idx"]] * h_sd + h_mu)
    for target_name, cvec in ds.targets_norm.items():
        c_tensor = torch.tensor(cvec, dtype=torch.float32, device=device)
        with torch.no_grad():
            x_gen_norm = model.sample(c_tensor, args.n_generate).detach().cpu().numpy()
        x_gen_unfixed = x_gen_norm * h_sd + h_mu
        trace_before = trace_abs_max_vec28(x_gen_unfixed)
        x_gen_raw = gauge_fix_vec28(x_gen_unfixed)
        generated[f"{condition_set}_{target_name}_H_vec28_trace_zero"] = x_gen_raw.astype(np.float32)
        rows.append(
            {
                "condition_set": condition_set,
                "target": target_name,
                "trace_abs_max_before_gauge_fix": trace_before,
                **physical_summary(x_gen_raw, train_h_raw),
            }
        )
    np.savez_compressed(run_dir / f"{condition_set}_generated_samples.npz", **generated)
    out = pd.DataFrame(rows)
    out.to_csv(run_dir / f"{condition_set}_generated_physical_summary.csv", index=False)
    return out


def vec28_to_hmat(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec)
    h = np.zeros(arr.shape[:-1] + (7, 7), dtype=arr.dtype)
    h[..., IU[0], IU[1]] = arr
    h = h + np.swapaxes(h, -1, -2)
    diag = np.diagonal(h, axis1=-2, axis2=-1).copy()
    h[..., np.arange(7), np.arange(7)] = diag / 2.0
    return h


def hmat_to_vec28(h: np.ndarray) -> np.ndarray:
    arr = np.asarray(h)
    return arr[..., IU[0], IU[1]].astype(np.float32)


def gauge_fix_vec28(vec: np.ndarray) -> np.ndarray:
    mats = vec28_to_hmat(np.asarray(vec, dtype=np.float64))
    diag = np.diagonal(mats, axis1=-2, axis2=-1).copy()
    offset = diag.mean(axis=-1, keepdims=True)
    mats[..., np.arange(7), np.arange(7)] = diag - offset
    return hmat_to_vec28(mats)


def trace_abs_max_vec28(vec: np.ndarray) -> float:
    mats = vec28_to_hmat(np.asarray(vec, dtype=np.float64))
    return float(np.max(np.abs(np.trace(mats, axis1=1, axis2=2))))


def physical_summary(h_vec28: np.ndarray, train_ref: np.ndarray | None = None) -> dict[str, float]:
    mats = vec28_to_hmat(h_vec28.astype(np.float64))
    eigs = np.linalg.eigvalsh(mats)
    diag = np.diagonal(mats, axis1=1, axis2=2)
    off = h_vec28[:, OFF_IDX]
    rows: dict[str, float] = {
        "n": int(len(h_vec28)),
        "trace_abs_max": float(np.max(np.abs(np.trace(mats, axis1=1, axis2=2)))),
        "symmetry_abs_max": float(np.max(np.abs(mats - np.swapaxes(mats, 1, 2)))),
        "spectral_width_median": float(np.median(eigs[:, -1] - eigs[:, 0])),
        "min_eigen_gap_median": float(np.median(np.diff(eigs, axis=1).min(axis=1))),
        "offdiag_diag_norm_ratio_median": float(np.median(np.linalg.norm(off, axis=1) / (np.linalg.norm(diag, axis=1) + 1e-8))),
    }
    if train_ref is not None:
        lo = np.nanpercentile(train_ref, 1, axis=0)
        hi = np.nanpercentile(train_ref, 99, axis=0)
        rows["outside_train_1_99_coord_fraction_mean"] = float(((h_vec28 < lo) | (h_vec28 > hi)).mean(axis=1).mean())
    return rows


def plot_loss(hist: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hist["epoch"], hist["train_nll"], label="train")
    ax.plot(hist["epoch"], hist["val_nll"], label="val")
    if len(hist):
        best_epoch = int(hist.loc[hist["val_nll"].idxmin(), "epoch"])
        ax.axvline(best_epoch, color="black", ls="--", lw=1)
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.set_title(path.stem.replace("_", " "))
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_metadata(
    out_root: Path,
    prepared: Path,
    args: argparse.Namespace,
    context: dict[str, Any],
    condition_data: dict[str, ConditionDataset],
    selected: list[str],
) -> None:
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
    condition_rows = []
    for name in selected:
        ds = condition_data[name]
        condition_rows.append(
            {
                "condition_set": name,
                "condition_dim": int(ds.raw.shape[1]),
                "n_targets": int(len(ds.targets_norm)),
                "uses_dynamic_label": bool(ds.dynamic_label is not None),
                "balanced_sampler": bool(ds.balance_label is not None),
                "condition_names": json.dumps(ds.names, ensure_ascii=False),
                "target_names": json.dumps(sorted(ds.targets_norm), ensure_ascii=False),
            }
        )
    pd.DataFrame(condition_rows).to_csv(metadata_dir / "condition_design.csv", index=False)
    np.savez_compressed(
        metadata_dir / "dynamic_feature_model.npz",
        dyn_feature_mu=np.asarray(context["dynamic_pca_meta"]["feature_mu"], dtype=np.float32),
        dyn_pca_components=np.asarray(context["dynamic_pca_meta"]["components"], dtype=np.float32),
        dyn_pca_explained=np.asarray(context["dynamic_pca_meta"]["explained_variance_ratio"], dtype=np.float32),
        **{f"kmeans_centers_{sanitize_name(k)}": v for k, v in context["dynamic_centers"].items()},
    )
    manifest = {
        "purpose": "H27 140k CFAST+PATH flow with row-aligned dynamic submode conditions",
        "prepared": str(prepared),
        "out_root": str(out_root),
        "selected_condition_sets": selected,
        "args": clean_json(vars(args)),
        "important_files": {
            "dynamic_submode_summary": str(metadata_dir / "dynamic_submode_summary.csv"),
            "condition_design": str(metadata_dir / "condition_design.csv"),
            "dynamic_feature_model": str(metadata_dir / "dynamic_feature_model.npz"),
        },
        "label_boundary": "dynamic_submode is fitted from H27 140k row-aligned trajectory/path features; older 62k D/S labels are not attached.",
    }
    (metadata_dir / "manifest.json").write_text(json.dumps(clean_json(manifest), indent=2, ensure_ascii=False), encoding="utf-8")
    write_experiment_readme(out_root, selected)


def write_run_summary(
    run_dir: Path,
    args: argparse.Namespace,
    selected: list[str],
    condition_data: dict[str, ConditionDataset],
) -> None:
    metric_paths = sorted(run_dir.glob("*_test_metrics.csv"))
    if metric_paths:
        metrics = pd.concat([pd.read_csv(p) for p in metric_paths], ignore_index=True)
        metrics.to_csv(run_dir / "all_test_metrics.csv", index=False)
    summary = {
        "run_dir": str(run_dir),
        "run_name": args.run_name,
        "condition_sets": selected,
        "generated_files": [str(p) for p in sorted(run_dir.glob("*_generated_samples.npz"))],
        "metric_files": [str(p) for p in sorted(run_dir.glob("*_test_metrics.csv"))],
        "args": clean_json(vars(args)),
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")


def write_condition_report(
    run_dir: Path,
    condition_set: str,
    args: argparse.Namespace,
    ds: ConditionDataset,
    metrics: list[dict[str, Any]],
    generated_summary: pd.DataFrame,
) -> None:
    metric_df = pd.DataFrame(metrics)
    lines = [
        f"# {condition_set} H27 path-dynamic flow run",
        "",
        f"- run_name: `{args.run_name}`",
        f"- condition_dim: `{ds.raw.shape[1]}`",
        f"- targets: `{len(ds.targets_norm)}`",
        f"- uses_dynamic_label: `{ds.dynamic_label is not None}`",
        f"- balanced_sampler: `{ds.balance_label is not None}`",
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
        "Generated Hamiltonians are gauge-fixed and physically sanity-checked here. They are not inverse-design successes until simulator validation and diversity audit pass.",
    ]
    (run_dir / "reports" / f"{condition_set}_run_report_kr.md").write_text("\n".join(lines), encoding="utf-8")


def write_experiment_readme(out_root: Path, selected: list[str]) -> None:
    lines = [
        "# 20260619 H27 PATH + Dynamic Submode Flow",
        "",
        "Purpose: train conditional RealNVP variants that add row-aligned dynamic submode signal to H27 140k `CFAST + c_l1 + PATH` conditions.",
        "",
        "Important boundary: this experiment computes dynamic submodes from the H27 140k prepared artifact. It does not reuse the older 62k `D_global/S_global` labels as direct conditions.",
        "",
        "## Layout",
        "",
        "- `metadata/manifest.json`: experiment manifest and label boundary.",
        "- `metadata/dynamic_submode_summary.csv`: fitted 140k dynamic submode profile.",
        "- `metadata/condition_design.csv`: condition dimensions and target names.",
        "- `smoke/`: quick 2-epoch sanity run.",
        "- `full/`: full training outputs, checkpoints, generated samples, and reports.",
        "",
        "## Default Condition Sets",
        "",
        *[f"- `{name}`" for name in selected],
        "",
        "## Recommended Next Checks",
        "",
        "1. Run simulator validation for every `*_generated_samples.npz` in `full/`.",
        "2. Run dynamic diversity audit on the validation detail CSVs.",
        "3. Compare high/fast largest dynamic cluster fraction against the previous `CFAST_CL1_PATH` branch.",
    ]
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def clean_json(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean_json(v) for v in obj]
    return obj


if __name__ == "__main__":
    raise SystemExit(main())
