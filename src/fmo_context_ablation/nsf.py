from __future__ import annotations

import copy
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from zuko.flows import NSF


def resolve_device(prefer: str = "auto") -> torch.device:
    if prefer == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(prefer)


def build_flow(
    *,
    feature_dim: int,
    context_dim: int,
    device: torch.device,
    transforms: int = 8,
    hidden: int = 128,
    bins: int = 8,
) -> NSF:
    return NSF(
        features=feature_dim,
        context=context_dim,
        transforms=transforms,
        hidden_features=[hidden, hidden],
        bins=bins,
    ).to(device)


def train_nsf(
    flow: NSF,
    x_norm: np.ndarray,
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 2048,
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    lr_patience: int = 20,
    early_stopping_patience: int = 100,
    seed: int = 0,
    num_workers: int = 0,
    log_every: int = 10,
    log_first_n_epochs: int = 5,
) -> dict:
    log_every = max(1, int(log_every))
    log_first_n_epochs = max(0, int(log_first_n_epochs))
    torch.manual_seed(seed)

    x_t = torch.tensor(x_norm, dtype=torch.float32)
    y_t = torch.tensor(y_norm, dtype=torch.float32)
    train_ds = TensorDataset(x_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(x_t[val_idx], y_t[val_idx])
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=lr_patience,
    )

    hist = {"train": [], "val": [], "lr": [], "stale": []}
    best_state = None
    best_epoch = 0
    best_train = float("inf")
    best_val = float("inf")
    stale = 0
    stopped_epoch = epochs
    train_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        flow.train()
        train_loss_sum = 0.0
        train_n = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            loss = -flow(yb).log_prob(xb).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), grad_clip)
            opt.step()
            train_loss_sum += float(loss.item()) * len(xb)
            train_n += len(xb)
        train_loss = train_loss_sum / max(train_n, 1)

        flow.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_loss_sum += float((-flow(yb).log_prob(xb)).sum().item())
                val_n += len(xb)
        val_loss = val_loss_sum / max(val_n, 1)
        scheduler.step(val_loss)

        hist["train"].append(train_loss)
        hist["val"].append(val_loss)
        hist["lr"].append(float(opt.param_groups[0]["lr"]))

        if val_loss < best_val:
            best_val = val_loss
            best_train = train_loss
            best_epoch = epoch
            best_state = copy.deepcopy(flow.state_dict())
            stale = 0
        else:
            stale += 1
        hist["stale"].append(int(stale))

        if epoch <= log_first_n_epochs or epoch % log_every == 0 or epoch == epochs:
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            dt = time.perf_counter() - t0
            elapsed = time.perf_counter() - train_start
            avg_epoch_sec = elapsed / max(epoch, 1)
            eta_sec = avg_epoch_sec * max(epochs - epoch, 0)
            print(
                f"  [{stamp}] ep {epoch:4d}/{epochs} "
                f"train={train_loss:.3f} val={val_loss:.3f} "
                f"best={best_val:.3f} lr={opt.param_groups[0]['lr']:.1e} "
                f"dt={dt:.1f}s eta={_format_duration(eta_sec)}",
                flush=True,
            )

        if early_stopping_patience > 0 and stale >= early_stopping_patience:
            stopped_epoch = epoch
            print(
                f"  [early-stop] no validation improvement for {early_stopping_patience} epochs; "
                f"best epoch {best_epoch} val {best_val:.3f}",
                flush=True,
            )
            break

    if best_state is not None:
        flow.load_state_dict(best_state)

    stop_reason = "early_stopping" if stopped_epoch < epochs else "max_epochs"
    print(
        f"  [done] {stop_reason}; best ep {best_epoch}, val={best_val:.3f}",
        flush=True,
    )

    hist["best_epoch"] = int(best_epoch)
    hist["best_train"] = float(best_train)
    hist["best_val"] = float(best_val)
    hist["stopped_epoch"] = int(stopped_epoch)
    hist["stop_reason"] = stop_reason
    hist["early_stopping_patience"] = int(early_stopping_patience)
    hist["lr_scheduler"] = "ReduceLROnPlateau"
    hist["lr_patience"] = int(lr_patience)
    return hist


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def sample_h27(flow: NSF, y_norm: np.ndarray, stats: dict, *, device: torch.device) -> np.ndarray:
    flow.eval()
    y = torch.tensor(y_norm, dtype=torch.float32, device=device)
    with torch.no_grad():
        x_norm = flow(y).sample().cpu().numpy()
    return (x_norm * stats["x_sd"] + stats["x_mu"]).astype(np.float32)
