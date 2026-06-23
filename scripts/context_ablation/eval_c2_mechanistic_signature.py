from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fmo_context_ablation.data import DEFAULT_MERGED_PATH, json_dump
from fmo_context_ablation.hamiltonian import h28_to_matrix


OUTPUT_DIR = ROOT / "outputs" / "model_performance" / "comparison"
ETA_HIGH = 0.95
ETA_NONHIGH = 0.85
RANDOM_SEED = 716

N_SITE = 7
IDX_IN = 0
IDX_TRAP_SITE = 2
CM2RADPS = 2 * np.pi * 0.0299792458
KB = 0.6950348
TEMP_K = 300.0
OMEGA_C = 106.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate original C2 mechanistic-signature claim: high-eta H should "
            "simultaneously show bath-spectrum resonance and source-sink "
            "delocalized eigenstate structure."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_MERGED_PATH)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means use all rows.")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--eta-high", type=float, default=ETA_HIGH)
    parser.add_argument("--eta-nonhigh", type=float, default=ETA_NONHIGH)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def bath_spectrum_cm(gap_cm: np.ndarray, lambda_reorg: float = 35.0) -> np.ndarray:
    """Vectorized Drude-Lorentz bath spectrum evaluated at energy gaps in cm^-1."""
    gap = np.asarray(gap_cm, dtype=np.float64)
    w = gap * CM2RADPS
    lam = lambda_reorg * CM2RADPS
    gam = OMEGA_C * CM2RADPS
    kt = KB * TEMP_K * CM2RADPS
    out = np.empty_like(w, dtype=np.float64)
    small = np.abs(w) < 1e-9
    out[small] = 4.0 * lam * kt / gam
    ww = w[~small]
    j = 2.0 * lam * gam * ww / (ww**2 + gam**2)
    out[~small] = j * (1.0 / np.tanh(ww / (2.0 * kt)) + 1.0)
    return out


def load_dataset(path: Path, max_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as d:
        n = len(d["H_params"])
        if max_samples and max_samples < n:
            rng = np.random.default_rng(seed)
            idx = np.sort(rng.choice(n, size=max_samples, replace=False))
        else:
            idx = np.arange(n)
        h28 = np.asarray(d["H_params"][idx], dtype=np.float32)
        eta = np.asarray(d["eta"][idx], dtype=np.float32)
    return idx.astype(np.int64), h28_to_matrix(h28).astype(np.float64), eta


def compute_c2_scores(h_mats: np.ndarray) -> pd.DataFrame:
    evals, evecs = np.linalg.eigh(h_mats)
    pair_i, pair_j = np.triu_indices(N_SITE, k=1)
    gaps = evals[:, pair_j] - evals[:, pair_i]

    # Bath-resonance score: best bath spectral response among all eigenvalue gaps.
    spectrum = bath_spectrum_cm(gaps)
    grid = np.linspace(0.0, 700.0, 5000)
    spectrum_norm = float(np.max(bath_spectrum_cm(grid)))
    bath_score_by_gap = spectrum / max(spectrum_norm, 1e-12)
    best_gap_idx = np.argmax(bath_score_by_gap, axis=1)
    bath_score = bath_score_by_gap[np.arange(len(h_mats)), best_gap_idx]
    best_gap_cm = gaps[np.arange(len(h_mats)), best_gap_idx]

    # Source-sink delocalization: one eigenstate should carry both site1 and site3.
    # score = harmonic mean of source/sink eigenstate weights, so it is large only
    # when both weights are non-negligible in the same eigenstate.
    weights = np.square(evecs)
    source_w = weights[:, IDX_IN, :]
    sink_w = weights[:, IDX_TRAP_SITE, :]
    deloc_by_state = 2.0 * source_w * sink_w / (source_w + sink_w + 1e-12)
    best_state = np.argmax(deloc_by_state, axis=1)
    deloc_score = deloc_by_state[np.arange(len(h_mats)), best_state]
    source_weight_at_best = source_w[np.arange(len(h_mats)), best_state]
    sink_weight_at_best = sink_w[np.arange(len(h_mats)), best_state]
    loose_deloc_state_count = np.sum((source_w >= 0.05) & (sink_w >= 0.05), axis=1)

    return pd.DataFrame(
        {
            "bath_score": bath_score.astype(np.float32),
            "best_bath_gap_cm": best_gap_cm.astype(np.float32),
            "deloc_score": deloc_score.astype(np.float32),
            "best_deloc_state": best_state.astype(np.int16),
            "source_weight_at_best": source_weight_at_best.astype(np.float32),
            "sink_weight_at_best": sink_weight_at_best.astype(np.float32),
            "loose_deloc_state_count": loose_deloc_state_count.astype(np.int16),
        }
    )


def summarize_groups(df: pd.DataFrame, eta_high: float, eta_nonhigh: float) -> pd.DataFrame:
    groups = {
        "all": np.ones(len(df), dtype=bool),
        f"high_eta_ge_{eta_high:.2f}": df["eta"].to_numpy() >= eta_high,
        f"nonhigh_eta_lt_{eta_nonhigh:.2f}": df["eta"].to_numpy() < eta_nonhigh,
        "top10_eta": df["eta"].to_numpy() >= np.quantile(df["eta"].to_numpy(), 0.90),
        "bottom50_eta": df["eta"].to_numpy() <= np.quantile(df["eta"].to_numpy(), 0.50),
    }
    rows = []
    for name, mask in groups.items():
        part = df.loc[mask]
        if len(part) == 0:
            continue
        rows.append(
            {
                "group": name,
                "n": int(len(part)),
                "eta_median": float(part["eta"].median()),
                "bath_score_median": float(part["bath_score"].median()),
                "deloc_score_median": float(part["deloc_score"].median()),
                "bath_pass_rate": float(part["bath_pass"].mean()),
                "deloc_pass_rate": float(part["deloc_pass"].mean()),
                "joint_pass_rate": float(part["joint_pass"].mean()),
                "loose_joint_pass_rate": float(part["loose_joint_pass"].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_pass_rates(summary: pd.DataFrame, out_dir: Path) -> None:
    keep = [g for g in summary["group"] if g != "all"]
    view = summary.set_index("group").loc[keep]
    x = np.arange(len(view))
    width = 0.22
    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    ax.bar(x - width, view["bath_pass_rate"] * 100, width, label="bath resonance")
    ax.bar(x, view["deloc_pass_rate"] * 100, width, label="source-sink delocalization")
    ax.bar(x + width, view["joint_pass_rate"] * 100, width, label="both")
    ax.set_xticks(x)
    ax.set_xticklabels(keep, rotation=15, ha="right")
    ax.set_ylabel("pass rate (%)")
    ax.set_title("C2 mechanistic signature pass rates")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "c2_signature_pass_rates.png", dpi=180)
    plt.close(fig)


def plot_score_scatter(df: pd.DataFrame, out_dir: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = min(8000, len(df))
    idx = rng.choice(len(df), size=n, replace=False)
    part = df.iloc[idx].copy()
    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    sc = ax.scatter(
        part["bath_score"],
        part["deloc_score"],
        c=part["eta"],
        s=10,
        alpha=0.45,
        cmap="viridis",
        linewidths=0,
    )
    ax.axvline(df["bath_threshold"].iloc[0], color="#666666", linestyle="--", linewidth=1.2)
    ax.axhline(df["deloc_threshold"].iloc[0], color="#666666", linestyle="--", linewidth=1.2)
    ax.set_xlabel("bath resonance score")
    ax.set_ylabel("source-sink delocalization score")
    ax.set_title("C2 scores colored by eta")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("eta")
    fig.tight_layout()
    fig.savefig(out_dir / "c2_score_scatter_eta.png", dpi=180)
    plt.close(fig)


def plot_eta_by_signature(df: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    vals = [
        df.loc[~df["joint_pass"], "eta"].to_numpy(),
        df.loc[df["joint_pass"], "eta"].to_numpy(),
    ]
    ax.boxplot(vals, tick_labels=["signature fail", "signature pass"], showfliers=False)
    ax.set_ylabel("eta")
    ax.set_title("Eta distribution by strict C2 joint signature")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "c2_eta_by_joint_signature.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_idx, h_mats, eta = load_dataset(args.data, args.max_samples, args.seed)
    scores = compute_c2_scores(h_mats)
    scores.insert(0, "dataset_index", dataset_idx)
    scores.insert(1, "eta", eta)

    bath_threshold = float(np.quantile(scores["bath_score"], 0.75))
    deloc_threshold = float(np.quantile(scores["deloc_score"], 0.75))
    scores["bath_threshold"] = bath_threshold
    scores["deloc_threshold"] = deloc_threshold
    scores["bath_pass"] = scores["bath_score"] >= bath_threshold
    scores["deloc_pass"] = scores["deloc_score"] >= deloc_threshold
    scores["joint_pass"] = scores["bath_pass"] & scores["deloc_pass"]
    scores["loose_deloc_pass"] = scores["loose_deloc_state_count"] >= 1
    scores["loose_joint_pass"] = scores["bath_pass"] & scores["loose_deloc_pass"]

    summary = summarize_groups(scores, args.eta_high, args.eta_nonhigh)
    summary.to_csv(args.out_dir / "c2_mechanistic_signature_summary.csv", index=False)
    scores.to_csv(args.out_dir / "c2_mechanistic_signature_scores.csv", index=False)
    plot_pass_rates(summary, args.out_dir)
    plot_score_scatter(scores, args.out_dir, args.seed)
    plot_eta_by_signature(scores, args.out_dir)

    all_joint = float(summary.loc[summary["group"] == "all", "joint_pass_rate"].iloc[0])
    high_group = f"high_eta_ge_{args.eta_high:.2f}"
    high_joint = float(summary.loc[summary["group"] == high_group, "joint_pass_rate"].iloc[0])
    manifest = {
        "data": str(args.data),
        "n_samples": int(len(scores)),
        "eta_high": float(args.eta_high),
        "eta_nonhigh": float(args.eta_nonhigh),
        "bath_score_definition": "max normalized Drude-Lorentz bath spectrum over all eigenvalue gaps",
        "deloc_score_definition": "max harmonic mean of site1 and site3 eigenstate weights",
        "strict_thresholds": {
            "bath_pass": "bath_score >= dataset 75th percentile",
            "deloc_pass": "deloc_score >= dataset 75th percentile",
            "bath_threshold": bath_threshold,
            "deloc_threshold": deloc_threshold,
        },
        "loose_deloc_threshold": "at least one eigenstate has both site1 and site3 weights >= 0.05",
        "strong_c2_necessary_condition_supported": bool(high_joint >= 0.90),
        "weak_c2_enrichment_over_all": float(high_joint - all_joint),
        "outputs": {
            "summary_csv": "c2_mechanistic_signature_summary.csv",
            "scores_csv": "c2_mechanistic_signature_scores.csv",
            "pass_rate_figure": "c2_signature_pass_rates.png",
            "scatter_figure": "c2_score_scatter_eta.png",
            "eta_boxplot": "c2_eta_by_joint_signature.png",
        },
    }
    json_dump(args.out_dir / "c2_mechanistic_signature_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()

