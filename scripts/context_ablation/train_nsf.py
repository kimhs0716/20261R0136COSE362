from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import list_contexts
from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump, load_h27_and_context, normalize_train_val, portable_path
from fmo_context_ablation.nsf import build_flow, resolve_device, train_nsf


DATA_PATH = DEFAULT_MERGED_PATH
OUTPUT_DIR = ROOT / "outputs" / "training"
DEFAULT_EPOCHS = 1000
LR_PATIENCE = 20
EARLY_STOPPING_PATIENCE = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NSF p(H27 | context).")
    parser.add_argument("--context", choices=list_contexts(), default="c5")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--batch-size", type=int, default=2048)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Run scripts/merge_datasets.py first: {DATA_PATH}")

    x, y, context_names = load_h27_and_context(DATA_PATH, args.context)
    x_norm, y_norm, train_idx, val_idx, stats = normalize_train_val(x, y, seed=args.seed)
    device = resolve_device(args.device)
    run_name = args.run_name or f"nsf_h27_{args.context}_seed{args.seed}"

    print(f"[data] {DATA_PATH}: {len(x)} samples")
    print(f"[shape] X={x.shape} -> Xn={x_norm.shape}; Y={y.shape} -> Yn={y_norm.shape}")
    print(f"[context] {args.context}: {len(context_names)} features")
    print(f"[run name] {run_name}")
    print(f"device: {device}")

    flow = build_flow(
        feature_dim=x.shape[1],
        context_dim=y.shape[1],
        device=device,
        transforms=8,
        hidden=128,
        bins=8,
    )
    hist = train_nsf(
        flow,
        x_norm,
        y_norm,
        train_idx,
        val_idx,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr_patience=LR_PATIENCE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        seed=args.seed,
        num_workers=0,
    )

    out_dir = OUTPUT_DIR / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "model": "nsf",
        "state_dict": flow.state_dict(),
        "history": hist,
        "stats": stats,
        "context": args.context,
        "context_names": context_names,
        "context_dim": int(y.shape[1]),
        "feature_dim": int(x.shape[1]),
        "data_path": portable_path(DATA_PATH),
        "args": vars(args),
    }
    checkpoint_path = out_dir / "checkpoint.pt"
    torch.save(payload, checkpoint_path)

    summary = {
        "checkpoint": portable_path(checkpoint_path),
        "model": "nsf",
        "samples": int(len(x)),
        "feature_dim": int(x.shape[1]),
        "context": args.context,
        "context_dim": int(y.shape[1]),
        "context_names": context_names,
        "data_path": portable_path(DATA_PATH),
        "final_train": float(hist["train"][-1]),
        "final_val": float(hist["val"][-1]),
        "best_epoch": int(hist["best_epoch"]),
        "best_train": float(hist["best_train"]),
        "best_val": float(hist["best_val"]),
        "stopped_epoch": int(hist["stopped_epoch"]),
        "stop_reason": hist["stop_reason"],
        "checkpoint_policy": "best_val",
        "early_stopping_patience": int(hist["early_stopping_patience"]),
        "lr_scheduler": hist["lr_scheduler"],
        "lr_patience": int(hist["lr_patience"]),
        "seed": int(args.seed),
    }
    json_dump(out_dir / "summary.json", summary)
    print(f"[saved] {checkpoint_path}")
    print(
        f"[summary] context={args.context} samples={len(x)} "
        f"best_ep={summary['best_epoch']} best_val={summary['best_val']:.3f} "
        f"stopped_ep={summary['stopped_epoch']} reason={summary['stop_reason']}"
    )


if __name__ == "__main__":
    main()

