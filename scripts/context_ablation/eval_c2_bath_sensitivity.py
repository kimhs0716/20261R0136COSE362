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


OUTPUT_DIR = ROOT / "outputs" / "c2_bath_sensitivity"
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
            "Run C2 bath-score sensitivity checks. The original bath score uses "
            "only the largest Drude-Lorentz response over eigenvalue gaps; this "
            "script tests whether adding eigenstate transition coupling and "
            "source-sink pathway relevance makes the bath signal more enriched "
            "in high-eta Hamiltonians."
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


def normalize_bath(gaps_cm: np.ndarray) -> np.ndarray:
    spectrum = bath_spectrum_cm(gaps_cm)
    grid = np.linspace(0.0, 700.0, 5000)
    spectrum_norm = float(np.max(bath_spectrum_cm(grid)))
    return spectrum / max(spectrum_norm, 1e-12)


def pair_path_relevance(weights: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    source_w = weights[:, IDX_IN, :]
    sink_w = weights[:, IDX_TRAP_SITE, :]
    pair_source = source_w[:, pair_i] + source_w[:, pair_j]
    pair_sink = sink_w[:, pair_i] + sink_w[:, pair_j]
    return 2.0 * pair_source * pair_sink / (pair_source + pair_sink + 1e-12)


def pair_site_bath_coupling(evecs: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    vi = evecs[:, :, pair_i]
    vj = evecs[:, :, pair_j]
    return np.sum((vi * vj) ** 2, axis=1)


def compute_scores(h_mats: np.ndarray) -> pd.DataFrame:
    evals, evecs = np.linalg.eigh(h_mats)
    pair_i, pair_j = np.triu_indices(N_SITE, k=1)
    gaps = evals[:, pair_j] - evals[:, pair_i]

    bath = normalize_bath(gaps)
    weights = np.square(evecs)
    path = pair_path_relevance(weights, pair_i, pair_j)
    coupling = pair_site_bath_coupling(evecs, pair_i, pair_j)

    score_arrays = {
        "spectral_only": bath,
        "bath_x_coupling": bath * coupling,
        "bath_x_path": bath * path,
        "bath_x_coupling_x_path": bath * coupling * path,
    }

    out: dict[str, np.ndarray] = {}
    for name, arr in score_arrays.items():
        best = np.argmax(arr, axis=1)
        out[f"{name}_score"] = arr[np.arange(len(h_mats)), best].astype(np.float32)
        out[f"{name}_gap_cm"] = gaps[np.arange(len(h_mats)), best].astype(np.float32)
        out[f"{name}_path_relevance"] = path[np.arange(len(h_mats)), best].astype(np.float32)
        out[f"{name}_coupling"] = coupling[np.arange(len(h_mats)), best].astype(np.float32)

    return pd.DataFrame(out)


def summarize_metric(df: pd.DataFrame, metric: str, eta_high: float, eta_nonhigh: float) -> list[dict]:
    score_col = f"{metric}_score"
    pass_col = f"{metric}_pass"
    threshold = float(np.quantile(df[score_col], 0.75))
    df[pass_col] = df[score_col] >= threshold

    groups = {
        "all": np.ones(len(df), dtype=bool),
        f"high_eta_ge_{eta_high:.2f}": df["eta"].to_numpy() >= eta_high,
        f"nonhigh_eta_lt_{eta_nonhigh:.2f}": df["eta"].to_numpy() < eta_nonhigh,
        "top10_eta": df["eta"].to_numpy() >= np.quantile(df["eta"].to_numpy(), 0.90),
        "bottom50_eta": df["eta"].to_numpy() <= np.quantile(df["eta"].to_numpy(), 0.50),
    }
    rows = []
    for group, mask in groups.items():
        part = df.loc[mask]
        rows.append(
            {
                "metric": metric,
                "group": group,
                "n": int(len(part)),
                "threshold_q75": threshold,
                "score_median": float(part[score_col].median()),
                "pass_rate": float(part[pass_col].mean()),
                "eta_median": float(part["eta"].median()),
            }
        )
    return rows


def plot_summary(summary: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        "spectral_only",
        "bath_x_coupling",
        "bath_x_path",
        "bath_x_coupling_x_path",
    ]
    labels = [
        "spectrum only",
        "spectrum x\ntransition coupling",
        "spectrum x\npath relevance",
        "spectrum x\ncoupling x route",
    ]
    groups = [f"high_eta_ge_{ETA_HIGH:.2f}", f"nonhigh_eta_lt_{ETA_NONHIGH:.2f}", "bottom50_eta"]
    group_labels = ["high eta", "non-high", "bottom 50%"]
    colors = ["#4C72B0", "#DD8452", "#8DA0CB"]

    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    x = np.arange(len(metrics))
    width = 0.22
    for j, (group, label, color) in enumerate(zip(groups, group_labels, colors)):
        vals = [
            float(summary.loc[(summary["metric"] == metric) & (summary["group"] == group), "pass_rate"].iloc[0]) * 100.0
            for metric in metrics
        ]
        ax.bar(x + (j - 1) * width, vals, width, label=label, color=color)
        for xpos, val in zip(x + (j - 1) * width, vals):
            ax.text(xpos, val + 1.0, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(25.0, color="#6b7280", linestyle="--", linewidth=1.0, label="dataset q75 baseline")
    ax.set_xticks(x, labels)
    ax.set_ylabel("pass rate (%)")
    ax.set_ylim(0, 88)
    ax.set_title("C2 bath-score sensitivity: only the combined transition/path score is enriched")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18), ncol=4)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "c2_bath_sensitivity_pass_rates.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset_idx, h_mats, eta = load_dataset(args.data, args.max_samples, args.seed)
    scores = compute_scores(h_mats)
    scores.insert(0, "dataset_index", dataset_idx)
    scores.insert(1, "eta", eta)

    metrics = ["spectral_only", "bath_x_coupling", "bath_x_path", "bath_x_coupling_x_path"]
    rows = []
    for metric in metrics:
        rows.extend(summarize_metric(scores, metric, args.eta_high, args.eta_nonhigh))
    summary = pd.DataFrame(rows)

    scores.to_csv(args.out_dir / "c2_bath_sensitivity_scores.csv", index=False)
    summary.to_csv(args.out_dir / "c2_bath_sensitivity_summary.csv", index=False)
    plot_summary(summary, args.out_dir)

    manifest = {
        "data": str(args.data),
        "n_samples": int(len(scores)),
        "eta_high": float(args.eta_high),
        "eta_nonhigh": float(args.eta_nonhigh),
        "metrics": {
            "spectral_only": "max normalized Drude-Lorentz bath spectrum over all eigenvalue gaps",
            "bath_x_coupling": "max spectrum score times uncorrelated site-bath transition coupling between eigenstates",
            "bath_x_path": "max spectrum score times source-sink participation of the eigenstate pair",
            "bath_x_coupling_x_path": "max spectrum score times both transition coupling and source-sink path relevance",
        },
        "threshold": "per-metric dataset 75th percentile",
        "outputs": {
            "summary_csv": "c2_bath_sensitivity_summary.csv",
            "scores_csv": "c2_bath_sensitivity_scores.csv",
            "pass_rate_figure": "c2_bath_sensitivity_pass_rates.png",
        },
    }
    json_dump(args.out_dir / "c2_bath_sensitivity_manifest.json", manifest)
    print(summary.to_string(index=False))
    print(f"[saved] {args.out_dir}")


if __name__ == "__main__":
    main()

