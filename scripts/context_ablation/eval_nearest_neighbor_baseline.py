from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors

try:
    from scipy import stats as scipy_stats
except Exception:
    scipy_stats = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import LABELS, build_context
from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump


DATA_PATH = DEFAULT_MERGED_PATH
TRAINING_DIR = ROOT / "outputs" / "training"
PERFORMANCE_DIR = ROOT / "outputs" / "model_performance"
OUTPUT_DIR = ROOT / "outputs" / "nearest_neighbor_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nearest-neighbor retrieval baseline for one NSF run.")
    parser.add_argument("--run-name", required=True)
    return parser.parse_args()


def load_payload(run_name: str) -> tuple[dict, dict]:
    run_dir = TRAINING_DIR / run_name
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing trained run files under {run_dir}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return summary, payload


def load_existing_eval(run_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    eval_dir = PERFORMANCE_DIR / run_name
    targets_path = eval_dir / "targets.csv"
    samples_path = eval_dir / "conditional_mae_samples.csv"
    manifest_path = eval_dir / "conditional_mae_manifest.json"
    if not targets_path.exists() or not samples_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            f"Run scripts/eval_conditional_mae.py first for {run_name}: missing {targets_path}, {samples_path}, or {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("target_split") != "val":
        raise ValueError(
            f"Conditional MAE output for {run_name} is not validation-only. "
            f"Found target_split={manifest.get('target_split')!r}."
        )
    return pd.read_csv(targets_path), pd.read_csv(samples_path), manifest


def find_nearest_train_indices(
    y_norm: np.ndarray,
    train_idx: np.ndarray,
    target_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    train_y = y_norm[train_idx]
    target_y = y_norm[target_idx]

    nn = NearestNeighbors(n_neighbors=2, metric="euclidean")
    nn.fit(train_y)
    distances, local_indices = nn.kneighbors(target_y, return_distance=True)
    candidate_global = train_idx[local_indices]

    nearest = candidate_global[:, 0].copy()
    nearest_distance = distances[:, 0].copy()

    same_as_target = nearest == target_idx
    if np.any(same_as_target):
        nearest[same_as_target] = candidate_global[same_as_target, 1]
        nearest_distance[same_as_target] = distances[same_as_target, 1]

    return nearest.astype(np.int64), nearest_distance.astype(np.float64)


def build_nn_samples(
    targets: pd.DataFrame,
    *,
    context: str,
    stats: dict,
) -> pd.DataFrame:
    with np.load(DATA_PATH) as d:
        y, context_names = build_context(d, context)
        y_norm = ((y - stats["y_mu"]) / stats["y_sd"]).astype(np.float32)
        train_idx = np.asarray(stats["train_idx"], dtype=np.int64)
        target_idx = targets["dataset_index"].to_numpy(dtype=np.int64)
        nearest_idx, nearest_distance = find_nearest_train_indices(y_norm, train_idx, target_idx)

        rows = []
        for target_row, neighbor_idx, distance in zip(targets.itertuples(index=False), nearest_idx, nearest_distance):
            row = {
                "method": "nearest_neighbor_baseline",
                "target_id": int(target_row.target_id),
                "dataset_index": int(target_row.dataset_index),
                "neighbor_index": int(neighbor_idx),
                "neighbor_context_distance": float(distance),
            }
            for key in LABELS:
                target_value = float(getattr(target_row, key))
                neighbor_value = float(d[key][neighbor_idx])
                row[key] = neighbor_value
                row[f"target_{key}"] = target_value
                row[f"abs_err_{key}"] = abs(neighbor_value - target_value)
            rows.append(row)

    return pd.DataFrame(rows), context_names


def summarize(samples: pd.DataFrame, nn_samples: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in LABELS:
        model = error_vector(samples, "model", key)
        random_base = error_vector(samples, "random_baseline", key)
        nn_base = nn_samples.sort_values("target_id")[f"abs_err_{key}"].to_numpy(dtype=np.float64)
        if len(model) != len(nn_base) or len(random_base) != len(nn_base):
            raise ValueError(f"Mismatched sample counts for {key}: model={len(model)} random={len(random_base)} nn={len(nn_base)}")

        row = {
            "metric": key,
            "model_mae": float(model.mean()),
            "random_mae": float(random_base.mean()),
            "nearest_neighbor_mae": float(nn_base.mean()),
            "delta_model_minus_nn": float((model - nn_base).mean()),
            "delta_nn_minus_random": float((nn_base - random_base).mean()),
            "model_reduction_vs_nn": float(1.0 - model.mean() / max(nn_base.mean(), 1e-12)),
            "nn_reduction_vs_random": float(1.0 - nn_base.mean() / max(random_base.mean(), 1e-12)),
            "model_better_fraction_vs_nn": float((model < nn_base).mean()),
            "nn_better_fraction_vs_random": float((nn_base < random_base).mean()),
        }
        if scipy_stats is not None:
            model_vs_nn = scipy_stats.ttest_rel(model, nn_base, alternative="less")
            nn_vs_random = scipy_stats.ttest_rel(nn_base, random_base, alternative="less")
            row["model_vs_nn_paired_t_stat_less"] = float(model_vs_nn.statistic)
            row["model_vs_nn_paired_t_p_less"] = float(model_vs_nn.pvalue)
            row["nn_vs_random_paired_t_stat_less"] = float(nn_vs_random.statistic)
            row["nn_vs_random_paired_t_p_less"] = float(nn_vs_random.pvalue)
        rows.append(row)
    return pd.DataFrame(rows)


def error_vector(samples: pd.DataFrame, method: str, key: str) -> np.ndarray:
    part = samples[samples["method"] == method].sort_values("target_id")
    return part[f"abs_err_{key}"].to_numpy(dtype=np.float64)


def main() -> None:
    args = parse_args()
    summary, payload = load_payload(args.run_name)
    targets, samples, eval_manifest = load_existing_eval(args.run_name)

    context = payload["context"]
    out_dir = OUTPUT_DIR / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    nn_samples, context_names = build_nn_samples(targets, context=context, stats=payload["stats"])
    nn_samples.to_csv(out_dir / "nearest_neighbor_samples.csv", index=False)

    result = summarize(samples, nn_samples)
    result.to_csv(out_dir / "nearest_neighbor_summary.csv", index=False)

    json_dump(
        out_dir / "nearest_neighbor_manifest.json",
        {
            "run_name": args.run_name,
            "context": context,
            "context_dim": int(payload["context_dim"]),
            "context_names": context_names,
            "target_split": eval_manifest.get("target_split"),
            "n_targets": int(len(targets)),
            "dataset": DATA_PATH,
            "training_best_epoch": summary.get("best_epoch"),
            "training_best_val": summary.get("best_val"),
            "nearest_neighbor_pool": "training split from checkpoint stats",
        },
    )

    print(result.to_string(index=False))
    print(f"[saved] {out_dir}")


if __name__ == "__main__":
    main()

