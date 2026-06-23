from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "outputs" / "training"
PERFORMANCE_DIR = ROOT / "outputs" / "model_performance"
TARGET_SPLIT = "val"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate all trained NSF context runs.")
    parser.add_argument("--n-targets", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true", help="Re-run evaluation even if outputs already exist.")
    return parser.parse_args()


def discover_training_runs() -> list[str]:
    if not TRAINING_DIR.exists():
        return []
    runs = []
    for summary_path in sorted(TRAINING_DIR.glob("*/summary.json")):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if summary.get("model") != "nsf":
            continue
        runs.append(summary_path.parent.name)
    return runs


def is_evaluated(run_name: str) -> bool:
    out_dir = PERFORMANCE_DIR / run_name
    summary_path = out_dir / "conditional_mae_summary.csv"
    manifest_path = out_dir / "conditional_mae_manifest.json"
    if not summary_path.exists() or not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return manifest.get("target_split") == TARGET_SPLIT


def run_command(cmd: list[str]) -> None:
    print("[cmd] " + " ".join(cmd), flush=True)
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    runs = discover_training_runs()
    if not runs:
        raise FileNotFoundError(f"No trained NSF runs found under {TRAINING_DIR}")

    pending = [run for run in runs if args.force or not is_evaluated(run)]
    skipped = [run for run in runs if not args.force and is_evaluated(run)]

    print(f"[runs] found={len(runs)} pending={len(pending)} skipped={len(skipped)}", flush=True)
    if skipped:
        print("[skip] " + ", ".join(skipped), flush=True)

    t0 = time.perf_counter()
    completed = 0
    for i, run_name in enumerate(pending, 1):
        print(f"\n[eval {i}/{len(pending)}] {run_name}", flush=True)
        run_t0 = time.perf_counter()
        run_command(
            [
                sys.executable,
                "scripts/eval_conditional_mae.py",
                "--run-name",
                run_name,
                "--n-targets",
                str(args.n_targets),
                "--workers",
                str(args.workers),
            ]
        )
        completed += 1
        elapsed = time.perf_counter() - t0
        run_elapsed = time.perf_counter() - run_t0
        eta = estimate_eta(elapsed, completed, len(pending))
        print(
            f"[eval done] {run_name} dt={format_duration(run_elapsed)} "
            f"overall={completed}/{len(pending)} eta={eta}",
            flush=True,
        )

    print(f"[done] elapsed={format_duration(time.perf_counter() - t0)}", flush=True)

    print("\n[summary]", flush=True)
    run_command([sys.executable, "scripts/summarize_model_performance.py"])
    run_command([sys.executable, "scripts/plot_model_performance.py"])


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


if __name__ == "__main__":
    main()

