from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import LABELS, build_context
from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump
from fmo_context_ablation.hamiltonian import h27_to_matrix
from fmo_context_ablation.nsf import build_flow, resolve_device, sample_h27
from fmo_context_ablation import simulator as sim


DATA_PATH = DEFAULT_MERGED_PATH
TRAINING_DIR = ROOT / "outputs" / "training"
OUTPUT_DIR = ROOT / "outputs" / "model_performance"
N_TARGETS = 1000
SIM_WORKERS = 6
SEED = 716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate conditional MAE against random Hamiltonian baseline.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--n-targets", type=int, default=N_TARGETS)
    parser.add_argument("--workers", type=int, default=SIM_WORKERS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(run_name: str, device: torch.device):
    run_dir = TRAINING_DIR / run_name
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    payload = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    flow = build_flow(
        feature_dim=int(payload["feature_dim"]),
        context_dim=int(payload["context_dim"]),
        device=device,
        transforms=payload["args"].get("transforms", 8),
        hidden=payload["args"].get("hidden") or 128,
        bins=payload["args"].get("bins", 8),
    )
    flow.load_state_dict(payload["state_dict"])
    flow.eval()
    return summary, payload, flow


TARGET_SPLIT = "val"


def choose_targets(
    d,
    context_name: str,
    n_targets: int,
    rng: np.random.Generator,
    candidate_idx: np.ndarray,
):
    y, context_names = build_context(d, context_name)
    candidate_idx = np.asarray(candidate_idx, dtype=np.int64)
    if len(candidate_idx) == 0:
        raise ValueError("No candidate validation targets were found in checkpoint stats.")
    if n_targets > len(candidate_idx):
        raise ValueError(
            f"Requested n_targets={n_targets}, but only {len(candidate_idx)} validation targets are available."
        )
    idx = rng.choice(candidate_idx, size=n_targets, replace=False)
    target_df = pd.DataFrame({key: d[key][idx] for key in LABELS})
    target_df.insert(0, "target_id", np.arange(n_targets))
    target_df["dataset_index"] = idx
    return idx, y[idx], context_names, target_df


def simulate_task(task):
    method, target_id, h = task
    out = sim.simulate(h, lambda_reorg=35.0, return_traj=False)
    row = {"method": method, "target_id": int(target_id)}
    for key in LABELS:
        row[key] = float(out[key])
    return row


def run_simulations(tasks, workers: int) -> pd.DataFrame:
    rows = []
    t0 = time.perf_counter()
    if workers > 0:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(simulate_task, task) for task in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                rows.append(fut.result())
                if i % 100 == 0 or i == len(futures):
                    elapsed = time.perf_counter() - t0
                    eta = estimate_eta(elapsed, i, len(futures))
                    print(f"simulated {i}/{len(futures)} elapsed={format_duration(elapsed)} eta={eta}", flush=True)
    else:
        for i, task in enumerate(tasks, 1):
            rows.append(simulate_task(task))
            if i % 100 == 0 or i == len(tasks):
                elapsed = time.perf_counter() - t0
                eta = estimate_eta(elapsed, i, len(tasks))
                print(f"simulated {i}/{len(tasks)} elapsed={format_duration(elapsed)} eta={eta}", flush=True)
    return pd.DataFrame(rows)


def estimate_eta(elapsed: float, done: int, total: int) -> str:
    if done <= 0:
        return "unknown"
    remaining = elapsed / done * max(total - done, 0)
    return format_duration(remaining)


def format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes / 60:.1f}h"


def paired_tests(samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in LABELS:
        wide = samples.pivot(index="target_id", columns="method", values=f"abs_err_{key}")
        model = wide["model"].to_numpy(dtype=np.float64)
        random_base = wide["random_baseline"].to_numpy(dtype=np.float64)
        delta = model - random_base
        row = {
            "metric": key,
            "model_mae": float(model.mean()),
            "random_mae": float(random_base.mean()),
            "delta_model_minus_random": float(delta.mean()),
            "mae_reduction_fraction": float(1.0 - model.mean() / max(random_base.mean(), 1e-12)),
            "model_better_fraction": float((model < random_base).mean()),
        }
        if scipy_stats is not None:
            test = scipy_stats.ttest_rel(model, random_base, alternative="less")
            row["paired_t_stat_less"] = float(test.statistic)
            row["paired_t_p_less"] = float(test.pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = resolve_device(args.device)

    summary, payload, flow = load_model(args.run_name, device)
    context = payload["context"]
    stats = payload["stats"]
    out_dir = OUTPUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with np.load(DATA_PATH) as d:
        val_idx = np.asarray(stats["val_idx"], dtype=np.int64)
        target_idx, target_y, context_names, target_df = choose_targets(
            d,
            context,
            args.n_targets,
            rng,
            val_idx,
        )
        target_y_norm = ((target_y - stats["y_mu"]) / stats["y_sd"]).astype(np.float32)
        model_h27 = sample_h27(flow, target_y_norm, stats, device=device)
        model_h = h27_to_matrix(model_h27).astype(np.float32)
        random_h = np.stack([sim.sample_H_geom(rng) for _ in range(args.n_targets)]).astype(np.float32)

    target_df.to_csv(out_dir / "targets.csv", index=False)
    np.savez_compressed(
        out_dir / "generated_hamiltonians.npz",
        model_h27=model_h27,
        model_h=model_h,
        random_h=random_h,
        target_idx=target_idx,
    )

    tasks = []
    for i, h in enumerate(model_h):
        tasks.append(("model", i, h))
    for i, h in enumerate(random_h):
        tasks.append(("random_baseline", i, h))

    print(f"[simulate] run={args.run_name} context={context} workers={args.workers}")
    samples = run_simulations(tasks, args.workers)
    samples = samples.merge(target_df.rename(columns={k: f"target_{k}" for k in LABELS}), on="target_id")
    for key in LABELS:
        samples[f"abs_err_{key}"] = (samples[key] - samples[f"target_{key}"]).abs()
    samples.to_csv(out_dir / "conditional_mae_samples.csv", index=False)

    result = paired_tests(samples)
    result.to_csv(out_dir / "conditional_mae_summary.csv", index=False)
    json_dump(
        out_dir / "conditional_mae_manifest.json",
        {
            "run_name": args.run_name,
            "context": context,
            "context_dim": int(payload["context_dim"]),
            "context_names": context_names,
            "target_split": TARGET_SPLIT,
            "target_pool_size": int(len(stats["val_idx"])),
            "n_targets": int(args.n_targets),
            "workers": int(args.workers),
            "seed": int(args.seed),
            "training_best_epoch": summary.get("best_epoch"),
            "training_best_val": summary.get("best_val"),
        },
    )

    print(result.to_string(index=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

