from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.context_features import LABELS, build_context
from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump
from fmo_context_ablation.hamiltonian import gauge_fix_encode, matrix_to_h27
from fmo_context_ablation.nsf import build_flow, resolve_device
from fmo_context_ablation import simulator as sim


DATA_PATH = DEFAULT_MERGED_PATH
TRAINING_DIR = ROOT / "outputs" / "training"
SEED = 716
DEFAULT_SEEDS = (716, 717, 718, 719, 720)
OUTPUT_DIR = ROOT / "outputs" / "c3_biology_likelihood"
MAX_DATASET_SAMPLES = 50000
BATCH_SIZE = 8192
FMO_NAME = "FMO_Adolphs_Renger"
FMO_TIMES = np.array([0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate C3: rank the FMO Hamiltonian under learned conditional "
            "distributions p(H | c_FMO)."
        )
    )
    parser.add_argument(
        "--run-name",
        action="append",
        default=None,
        help="Training run to evaluate. Repeat this option to evaluate multiple runs. Defaults to all nsf_h27_* runs.",
    )
    parser.add_argument("--max-samples", type=int, default=MAX_DATASET_SAMPLES)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Single dataset-subset seed. Kept for compatibility; use --seeds for several seeds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="Dataset-subset seeds to evaluate. Defaults to 716 717 718 719 720.",
    )
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_runs(requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    runs = sorted(p.name for p in TRAINING_DIR.glob("nsf_h27_*") if (p / "checkpoint.pt").exists())
    if not runs:
        raise FileNotFoundError(f"No trained runs found under {TRAINING_DIR}")
    return runs


def load_model(run_name: str, device: torch.device):
    run_dir = TRAINING_DIR / run_name
    summary_path = run_dir / "summary.json"
    checkpoint_path = run_dir / "checkpoint.pt"
    if not summary_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint or summary for run: {run_name}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
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


def gauge_fixed_fmo_h() -> np.ndarray:
    """데이터셋과 같은 trace-zero gauge로 맞춘 FMO Hamiltonian을 반환한다."""
    h = np.asarray(sim.H_FMO_CM, dtype=np.float32).copy()
    h[np.diag_indices(sim.N_SITE)] -= float(np.trace(h) / sim.N_SITE)
    return h


def fmo_context_record() -> dict[str, np.ndarray]:
    labels = sim.simulate(sim.H_FMO_CM, lambda_reorg=35.0, return_traj=True)
    tlist, rho_t = labels.pop("_traj")
    idx = np.array([int(np.argmin(np.abs(tlist - t))) for t in FMO_TIMES], dtype=np.int64)
    rho_sel = rho_t[idx]

    pop_t = np.real(np.diagonal(rho_sel, axis1=1, axis2=2))[None, :, :].astype(np.float32)
    sys = rho_sel[:, : sim.N_SITE, : sim.N_SITE]
    tr_sys = np.real(np.trace(sys, axis1=1, axis2=2))

    ipr_t = np.zeros(len(FMO_TIMES), dtype=np.float32)
    purity_t = np.zeros(len(FMO_TIMES), dtype=np.float32)
    cl1_t = np.zeros(len(FMO_TIMES), dtype=np.float32)
    eye = np.eye(sim.N_SITE, dtype=np.float64)
    for i, (rho_sys, tr) in enumerate(zip(sys, tr_sys)):
        if tr <= 1e-12:
            continue
        rho_n = rho_sys / tr
        pops = np.real(np.diag(rho_n))
        ipr_t[i] = float(np.sum(pops**2))
        purity_t[i] = float(np.real(np.trace(rho_n @ rho_n)))
        cl1_t[i] = float(np.sum(np.abs(rho_n) * (1.0 - eye)))

    eigs = np.linalg.eigvalsh(gauge_fixed_fmo_h()).astype(np.float32)[None, :]
    record: dict[str, np.ndarray] = {
        key: np.array([labels[key]], dtype=np.float32) for key in LABELS
    }
    record.update(
        {
            "eigs": eigs,
            "pop_t": pop_t,
            "times": FMO_TIMES,
            "cl1_t": cl1_t[None, :],
            "purity_t": purity_t[None, :],
            "ipr_t": ipr_t[None, :],
        }
    )
    return record


def load_dataset_h27(max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(DATA_PATH) as d:
        n = len(d["H_params"])
        if max_samples > 0 and max_samples < n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        else:
            idx = np.arange(n)
        h27 = gauge_fix_encode(np.asarray(d["H_params"][idx], dtype=np.float32)).astype(np.float32)
    return idx.astype(np.int64), h27


def normalized_h27(h27: np.ndarray, stats: dict) -> np.ndarray:
    return ((h27 - stats["x_mu"]) / stats["x_sd"]).astype(np.float32)


def normalized_context(context: np.ndarray, stats: dict) -> np.ndarray:
    return ((context - stats["y_mu"]) / stats["y_sd"]).astype(np.float32)


def log_prob_h_given_context(
    flow,
    x_norm: np.ndarray,
    y_norm_single: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    out = []
    y_np = np.asarray(y_norm_single, dtype=np.float32).reshape(1, -1)
    flow.eval()
    with torch.no_grad():
        for start in range(0, len(x_norm), batch_size):
            xb_np = x_norm[start : start + batch_size]
            yb_np = np.repeat(y_np, len(xb_np), axis=0)
            xb = torch.tensor(xb_np, dtype=torch.float32, device=device)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=device)
            out.append(flow(yb).log_prob(xb).detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def sample_norm_h_given_context(
    flow,
    y_norm_single: np.ndarray,
    *,
    n_samples: int,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """고정된 c_FMO 조건에서 normalized H sample을 생성한다."""
    out = []
    y_np = np.asarray(y_norm_single, dtype=np.float32).reshape(1, -1)
    flow.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            size = min(batch_size, n_samples - start)
            yb_np = np.repeat(y_np, size, axis=0)
            yb = torch.tensor(yb_np, dtype=torch.float32, device=device)
            out.append(flow(yb).sample().detach().cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def stable_run_seed(run_name: str, subset_seed: int) -> int:
    """run 이름과 subset seed로부터 재현 가능한 torch sampling seed를 만든다."""
    run_hash = sum((i + 1) * ord(ch) for i, ch in enumerate(run_name))
    return int(1_000_003 + subset_seed + run_hash % 1_000_000)


def plot_histogram(
    run_out: Path,
    run_name: str,
    dataset_logp: np.ndarray,
    generated_logp: np.ndarray,
    fmo_logp: float,
    dataset_percentile: float,
    generated_percentile: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.hist(
        dataset_logp,
        bins=80,
        density=True,
        alpha=0.45,
        color="#4c78a8",
        label="dataset H scored under c_FMO",
    )
    ax.hist(
        generated_logp,
        bins=80,
        density=True,
        alpha=0.45,
        color="#f58518",
        label="generated H ~ p(H | c_FMO)",
    )
    ax.axvline(fmo_logp, color="#d62728", linestyle="--", linewidth=2.0, label=f"FMO logp = {fmo_logp:.2f}")
    ax.set_title(f"C3 likelihood rank: {run_name}")
    ax.set_xlabel("log p(H | c_FMO)")
    ax.set_ylabel("density")
    ax.text(
        0.02,
        0.95,
        f"dataset percentile = {dataset_percentile:.1f}\n"
        f"generated percentile = {generated_percentile:.1f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "#cccccc"},
    )
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(run_out / "fmo_likelihood_histogram.png", dpi=180)
    plt.close(fig)


def plot_percentile_summary(out_dir: Path, aggregate_df: pd.DataFrame) -> None:
    df = aggregate_df.sort_values("context_dim")
    percentile_col = (
        "fmo_percentile_generated_mean"
        if "fmo_percentile_generated_mean" in df.columns
        else "fmo_percentile_mean"
    )
    std_col = (
        "fmo_percentile_generated_std"
        if "fmo_percentile_generated_std" in df.columns
        else "fmo_percentile_std"
    )
    title_suffix = "generated baseline" if "fmo_percentile_generated_mean" in df.columns else "dataset baseline"
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#4c78a8" if p < 95.0 else "#2ca02c" for p in df[percentile_col]]
    ax.bar(
        df["context"],
        df[percentile_col],
        yerr=df[std_col],
        capsize=4,
        color=colors,
        alpha=0.88,
    )
    ax.axhline(95.0, color="#d62728", linestyle="--", linewidth=1.6, label="top 5% threshold")
    ax.set_ylim(0, 100)
    ax.set_xlabel("context")
    ax.set_ylabel("FMO percentile in log p(H | c_FMO)")
    ax.set_title(f"C3 quick check by context across seeds ({title_suffix})")
    for i, row in enumerate(df.itertuples(index=False)):
        percentile = float(getattr(row, percentile_col))
        std = float(getattr(row, std_col))
        y = min(98.0, percentile + std + 1.5)
        ax.text(i, y, f"{percentile:.1f}", ha="center", va="bottom", fontsize=9)
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(out_dir / "fmo_percentile_by_context.png", dpi=180)
    plt.close(fig)


def evaluate_run(
    run_name: str,
    dataset_h27: np.ndarray,
    dataset_idx: np.ndarray,
    fmo_record: dict[str, np.ndarray],
    *,
    output_dir: Path,
    subset_seed: int,
    device: torch.device,
    batch_size: int,
) -> dict:
    summary, payload, flow = load_model(run_name, device)
    context_name = payload["context"]
    stats = payload["stats"]

    fmo_context, context_names = build_context(fmo_record, context_name)
    fmo_y_norm = normalized_context(fmo_context, stats)
    fmo_h27 = matrix_to_h27(gauge_fixed_fmo_h())[None, :].astype(np.float32)
    fmo_x_norm = normalized_h27(fmo_h27, stats)
    dataset_x_norm = normalized_h27(dataset_h27, stats)

    dataset_logp = log_prob_h_given_context(
        flow,
        dataset_x_norm,
        fmo_y_norm[0],
        device=device,
        batch_size=batch_size,
    )
    fmo_logp = float(
        log_prob_h_given_context(
            flow,
            fmo_x_norm,
            fmo_y_norm[0],
            device=device,
            batch_size=1,
        )[0]
    )

    generated_seed = stable_run_seed(run_name, subset_seed)
    set_seed(generated_seed)
    generated_x_norm = sample_norm_h_given_context(
        flow,
        fmo_y_norm[0],
        n_samples=len(dataset_h27),
        device=device,
        batch_size=batch_size,
    )
    generated_logp = log_prob_h_given_context(
        flow,
        generated_x_norm,
        fmo_y_norm[0],
        device=device,
        batch_size=batch_size,
    )

    dataset_percentile = float(np.mean(dataset_logp <= fmo_logp) * 100.0)
    dataset_rank_desc = int(1 + np.sum(dataset_logp > fmo_logp))
    dataset_top_fraction = float(dataset_rank_desc / (len(dataset_logp) + 1) * 100.0)

    generated_percentile = float(np.mean(generated_logp <= fmo_logp) * 100.0)
    generated_rank_desc = int(1 + np.sum(generated_logp > fmo_logp))
    generated_top_fraction = float(generated_rank_desc / (len(generated_logp) + 1) * 100.0)

    run_out = output_dir / run_name / f"seed_{subset_seed}"
    run_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        run_out / "c3_likelihood_scores.npz",
        dataset_index=dataset_idx,
        dataset_logp_under_c_fmo=dataset_logp,
        generated_logp_under_c_fmo=generated_logp,
        fmo_logp=np.array([fmo_logp], dtype=np.float32),
        fmo_context=fmo_context.astype(np.float32),
    )
    plot_histogram(
        run_out,
        run_name,
        dataset_logp,
        generated_logp,
        fmo_logp,
        dataset_percentile,
        generated_percentile,
    )

    manifest = {
        "run_name": run_name,
        "context": context_name,
        "context_dim": int(payload["context_dim"]),
        "context_names": context_names,
        "biological_reference": FMO_NAME,
        "comparison": (
            "FMO is scored under c_FMO and ranked against both dataset H_i scored "
            "under c_FMO and generated H ~ p_theta(H | c_FMO). The generated "
            "baseline is the direct check for the original C3 top-likelihood claim."
        ),
        "dataset_subset_seed": int(subset_seed),
        "generated_sampling_seed": int(generated_seed),
        "n_dataset_baseline": int(len(dataset_logp)),
        "n_generated_baseline": int(len(generated_logp)),
        "fmo_logp": fmo_logp,
        "dataset_logp_mean": float(np.mean(dataset_logp)),
        "dataset_logp_median": float(np.median(dataset_logp)),
        "dataset_logp_std": float(np.std(dataset_logp)),
        "generated_logp_mean": float(np.mean(generated_logp)),
        "generated_logp_median": float(np.median(generated_logp)),
        "generated_logp_std": float(np.std(generated_logp)),
        "fmo_percentile_dataset": dataset_percentile,
        "fmo_descending_rank_dataset": dataset_rank_desc,
        "fmo_top_fraction_percent_dataset": dataset_top_fraction,
        "fmo_percentile_generated": generated_percentile,
        "fmo_descending_rank_generated": generated_rank_desc,
        "fmo_top_fraction_percent_generated": generated_top_fraction,
        "supports_dataset_top5_aux": bool(dataset_percentile >= 95.0),
        "supports_original_top5_claim": bool(generated_percentile >= 95.0),
        "training_best_epoch": summary.get("best_epoch"),
        "training_best_val": summary.get("best_val"),
    }
    json_dump(run_out / "c3_likelihood_manifest.json", manifest)
    return manifest


def main() -> None:
    args = parse_args()
    seeds = args.seeds
    if seeds is None:
        seeds = [args.seed] if args.seed is not None else list(DEFAULT_SEEDS)
    seeds = [int(seed) for seed in seeds]

    set_seed(seeds[0])
    device = resolve_device(args.device)
    run_names = discover_runs(args.run_name)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[C3] runs={len(run_names)} device={device} max_samples={args.max_samples} seeds={seeds}")
    print("[FMO] simulating reference trajectory once")
    fmo_record = fmo_context_record()

    rows = []
    for subset_seed in seeds:
        dataset_idx, dataset_h27 = load_dataset_h27(args.max_samples, subset_seed)
        print(f"[baseline] seed={subset_seed} dataset samples={len(dataset_h27)}")
        for run_name in run_names:
            print(f"[run] {run_name} seed={subset_seed}", flush=True)
            rows.append(
                evaluate_run(
                    run_name,
                    dataset_h27,
                    dataset_idx,
                    fmo_record,
                    output_dir=OUTPUT_DIR,
                    subset_seed=subset_seed,
                    device=device,
                    batch_size=args.batch_size,
                )
            )

    summary_df = pd.DataFrame(rows).sort_values(["context_dim", "context", "dataset_subset_seed"])
    summary_df.to_csv(OUTPUT_DIR / "c3_fmo_likelihood_summary_by_seed.csv", index=False)

    aggregate_df = (
        summary_df.groupby(["run_name", "context", "context_dim"], as_index=False)
        .agg(
            fmo_logp=("fmo_logp", "first"),
            fmo_percentile_dataset_mean=("fmo_percentile_dataset", "mean"),
            fmo_percentile_dataset_std=("fmo_percentile_dataset", "std"),
            fmo_percentile_dataset_min=("fmo_percentile_dataset", "min"),
            fmo_percentile_dataset_max=("fmo_percentile_dataset", "max"),
            fmo_top_fraction_percent_dataset_mean=("fmo_top_fraction_percent_dataset", "mean"),
            fmo_top_fraction_percent_dataset_std=("fmo_top_fraction_percent_dataset", "std"),
            fmo_percentile_generated_mean=("fmo_percentile_generated", "mean"),
            fmo_percentile_generated_std=("fmo_percentile_generated", "std"),
            fmo_percentile_generated_min=("fmo_percentile_generated", "min"),
            fmo_percentile_generated_max=("fmo_percentile_generated", "max"),
            fmo_top_fraction_percent_generated_mean=("fmo_top_fraction_percent_generated", "mean"),
            fmo_top_fraction_percent_generated_std=("fmo_top_fraction_percent_generated", "std"),
            supports_dataset_top5_aux_all_seeds=("supports_dataset_top5_aux", "all"),
            supports_dataset_top5_aux_any_seed=("supports_dataset_top5_aux", "any"),
            supports_original_top5_claim_all_seeds=("supports_original_top5_claim", "all"),
            supports_original_top5_claim_any_seed=("supports_original_top5_claim", "any"),
            n_seeds=("dataset_subset_seed", "nunique"),
        )
        .sort_values(["context_dim", "context"])
    )
    # Backward-compatible canonical columns: after this update, fmo_percentile_* means
    # the generated-baseline percentile, which is the direct C3 check.
    aggregate_df["fmo_percentile_mean"] = aggregate_df["fmo_percentile_generated_mean"]
    aggregate_df["fmo_percentile_std"] = aggregate_df["fmo_percentile_generated_std"]
    aggregate_df["fmo_percentile_min"] = aggregate_df["fmo_percentile_generated_min"]
    aggregate_df["fmo_percentile_max"] = aggregate_df["fmo_percentile_generated_max"]
    aggregate_df["fmo_top_fraction_percent_mean"] = aggregate_df["fmo_top_fraction_percent_generated_mean"]
    aggregate_df["fmo_top_fraction_percent_std"] = aggregate_df["fmo_top_fraction_percent_generated_std"]

    fill_cols = [
        "fmo_percentile_dataset_std",
        "fmo_top_fraction_percent_dataset_std",
        "fmo_percentile_generated_std",
        "fmo_top_fraction_percent_generated_std",
        "fmo_percentile_std",
        "fmo_top_fraction_percent_std",
    ]
    aggregate_df[fill_cols] = aggregate_df[fill_cols].fillna(0.0)
    aggregate_df.to_csv(OUTPUT_DIR / "c3_fmo_likelihood_summary.csv", index=False)
    plot_percentile_summary(OUTPUT_DIR, aggregate_df)
    print(
        aggregate_df[
            [
                "run_name",
                "context",
                "context_dim",
                "fmo_logp",
                "fmo_percentile_dataset_mean",
                "fmo_percentile_generated_mean",
                "fmo_percentile_generated_std",
                "fmo_percentile_generated_min",
                "fmo_percentile_generated_max",
                "supports_original_top5_claim_all_seeds",
            ]
        ].to_string(index=False)
    )
    print(f"[saved] {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

