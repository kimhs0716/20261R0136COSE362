from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "outputs" / "training"
PERFORMANCE_DIR = ROOT / "outputs" / "model_performance"
OUT_DIR = PERFORMANCE_DIR / "comparison"
TARGET_SPLIT = "val"
LABELS = ("eta", "tau_transfer", "ipr", "purity", "c_l1")


def context_from_run(run_name: str) -> str:
    return run_name.removeprefix("nsf_h27_").removesuffix("_seed0")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    rows = []
    for summary_path in sorted(PERFORMANCE_DIR.glob("nsf_h27_*_seed0/conditional_mae_summary.csv")):
        run_dir = summary_path.parent
        run_name = run_dir.name
        manifest_path = run_dir / "conditional_mae_manifest.json"
        train_summary_path = TRAINING_DIR / run_name / "summary.json"
        if not manifest_path.exists() or not train_summary_path.exists():
            continue

        manifest = load_json(manifest_path)
        if manifest.get("target_split") != TARGET_SPLIT:
            continue

        train_summary = load_json(train_summary_path)
        df = pd.read_csv(summary_path)
        context = manifest.get("context") or context_from_run(run_name)

        for _, row in df.iterrows():
            rows.append(
                {
                    "run_name": run_name,
                    "context": context,
                    "context_dim": int(manifest["context_dim"]),
                    "metric": row["metric"],
                    "model_mae": float(row["model_mae"]),
                    "random_mae": float(row["random_mae"]),
                    "delta_model_minus_random": float(row["delta_model_minus_random"]),
                    "mae_reduction_fraction": float(row["mae_reduction_fraction"]),
                    "model_better_fraction": float(row["model_better_fraction"]),
                    "paired_t_p_less": float(row["paired_t_p_less"]) if "paired_t_p_less" in row else float("nan"),
                    "best_epoch": int(train_summary["best_epoch"]),
                    "stopped_epoch": int(train_summary["stopped_epoch"]),
                    "stop_reason": train_summary.get("stop_reason"),
                    "best_val_nll": float(train_summary["best_val"]),
                    "samples": int(train_summary["samples"]),
                    "seed": int(train_summary["seed"]),
                    "checkpoint_policy": train_summary.get("checkpoint_policy"),
                    "early_stopping_patience": int(train_summary["early_stopping_patience"]),
                    "lr_scheduler": train_summary.get("lr_scheduler"),
                    "lr_patience": int(train_summary["lr_patience"]),
                    "target_split": manifest.get("target_split"),
                    "n_targets": int(manifest["n_targets"]),
                    "eval_seed": int(manifest["seed"]),
                }
            )

    if not rows:
        raise FileNotFoundError(
            f"No validation-only conditional MAE summaries found under {PERFORMANCE_DIR}. "
            "Run scripts/eval_all_runs.py first."
        )

    by_metric = pd.DataFrame(rows).sort_values(["context_dim", "context", "metric"]).reset_index(drop=True)
    summary_rows = []
    for (run_name, context), group in by_metric.groupby(["run_name", "context"], sort=False):
        base = group.iloc[0].to_dict()
        out = {
            "run_name": run_name,
            "context": context,
            "context_dim": int(base["context_dim"]),
            "best_epoch": int(base["best_epoch"]),
            "stopped_epoch": int(base["stopped_epoch"]),
            "stop_reason": base["stop_reason"],
            "best_val_nll": float(base["best_val_nll"]),
            "samples": int(base["samples"]),
            "seed": int(base["seed"]),
            "checkpoint_policy": base["checkpoint_policy"],
            "early_stopping_patience": int(base["early_stopping_patience"]),
            "lr_scheduler": base["lr_scheduler"],
            "lr_patience": int(base["lr_patience"]),
            "target_split": base["target_split"],
            "n_targets": int(base["n_targets"]),
            "eval_seed": int(base["eval_seed"]),
            "mean_model_mae": float(group["model_mae"].mean()),
            "mean_random_mae": float(group["random_mae"].mean()),
            "mean_delta_model_minus_random": float(group["delta_model_minus_random"].mean()),
            "mean_reduction_fraction": float(group["mae_reduction_fraction"].mean()),
            "mean_model_better_fraction": float(group["model_better_fraction"].mean()),
        }
        for label in LABELS:
            part = group[group["metric"] == label]
            if len(part) != 1:
                continue
            r = part.iloc[0]
            out[f"{label}_mae"] = float(r["model_mae"])
            out[f"{label}_reduction"] = float(r["mae_reduction_fraction"])
        summary_rows.append(out)

    summary = pd.DataFrame(summary_rows).sort_values(["context_dim", "context"]).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    by_metric.to_csv(OUT_DIR / "context_performance_by_metric.csv", index=False)
    summary.to_csv(OUT_DIR / "context_performance_summary.csv", index=False)
    (OUT_DIR / "context_performance_manifest.json").write_text(
        json.dumps(
            {
                "runs": summary["run_name"].tolist(),
                "target_split": TARGET_SPLIT,
                "outputs": {
                    "by_metric": "outputs/model_performance/comparison/context_performance_by_metric.csv",
                    "summary": "outputs/model_performance/comparison/context_performance_summary.csv",
                },
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"[saved] {OUT_DIR}")


if __name__ == "__main__":
    main()

