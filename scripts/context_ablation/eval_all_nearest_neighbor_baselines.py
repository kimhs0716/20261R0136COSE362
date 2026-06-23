from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "outputs" / "training"
PERFORMANCE_DIR = ROOT / "outputs" / "model_performance"
OUTPUT_DIR = ROOT / "outputs" / "nearest_neighbor_baseline"
TARGET_SPLIT = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate nearest-neighbor baselines for all completed NSF evaluations.")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def discover_runs() -> list[str]:
    runs = []
    for summary_path in sorted(TRAINING_DIR.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        run_name = summary_path.parent.name
        eval_manifest_path = PERFORMANCE_DIR / run_name / "conditional_mae_manifest.json"
        has_model_eval = False
        if (PERFORMANCE_DIR / run_name / "conditional_mae_samples.csv").exists() and eval_manifest_path.exists():
            try:
                eval_manifest = json.loads(eval_manifest_path.read_text(encoding="utf-8"))
                has_model_eval = eval_manifest.get("target_split") == TARGET_SPLIT
            except Exception:
                has_model_eval = False
        if summary.get("model") == "nsf" and has_model_eval:
            runs.append(summary_path.parent.name)
    return runs


def is_done(run_name: str) -> bool:
    out_dir = OUTPUT_DIR / run_name
    summary_path = out_dir / "nearest_neighbor_summary.csv"
    manifest_path = out_dir / "nearest_neighbor_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("target_split") == TARGET_SPLIT


def main() -> None:
    args = parse_args()
    runs = discover_runs()
    if not runs:
        raise FileNotFoundError(
            f"No NSF runs with completed conditional MAE evaluation found under {TRAINING_DIR}"
        )

    pending = [run for run in runs if args.force or not is_done(run)]
    skipped = [run for run in runs if not args.force and is_done(run)]
    print(f"[runs] found={len(runs)} pending={len(pending)} skipped={len(skipped)}", flush=True)
    if skipped:
        print("[skip] " + ", ".join(skipped), flush=True)

    for i, run_name in enumerate(pending, 1):
        print(f"\n[nn eval {i}/{len(pending)}] {run_name}", flush=True)
        cmd = [sys.executable, "scripts/eval_nearest_neighbor_baseline.py", "--run-name", run_name]
        print("[cmd] " + " ".join(cmd), flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode != 0:
            raise SystemExit(result.returncode)

    print("\n[summary]", flush=True)
    cmd = [sys.executable, "scripts/plot_nearest_neighbor_baseline.py"]
    print("[cmd] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()

