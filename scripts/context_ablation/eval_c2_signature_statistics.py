from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "outputs" / "c2_signature_statistics"
MECH_SCRIPT = ROOT / "scripts" / "eval_c2_mechanistic_signature.py"
BATH_SCRIPT = ROOT / "scripts" / "eval_c2_bath_sensitivity.py"
MECH_SCORES = ROOT / "outputs" / "model_performance" / "comparison" / "c2_mechanistic_signature_scores.csv"
BATH_SCORES = ROOT / "outputs" / "c2_bath_sensitivity" / "c2_bath_sensitivity_scores.csv"

ETA_HIGH = 0.95
ETA_NONHIGH = 0.85
RANDOM_SEED = 716


try:
    from scipy.stats import fisher_exact
except Exception:  # pragma: no cover - optional dependency
    fisher_exact = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Statistically evaluate whether C2-style mechanistic signatures are "
            "enriched in high-eta Hamiltonians compared with non-high Hamiltonians."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--eta-high", type=float, default=ETA_HIGH)
    parser.add_argument("--eta-nonhigh", type=float, default=ETA_NONHIGH)
    parser.add_argument("--force-recompute", action="store_true")
    return parser.parse_args()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def run_script_if_needed(script: Path, expected: Path, *, force: bool) -> None:
    if expected.exists() and not force:
        return
    if not script.exists():
        raise FileNotFoundError(script)
    print(f"[prepare] running {script.name}", flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)
    if not expected.exists():
        raise FileNotFoundError(f"Expected output was not created: {expected}")


def ensure_pass_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for score_col in [c for c in out.columns if c.endswith("_score")]:
        pass_col = score_col.replace("_score", "_pass")
        if pass_col not in out.columns:
            threshold = float(out[score_col].quantile(0.75))
            out[pass_col] = out[score_col] >= threshold
    return out


def safe_rate(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def odds_ratio_from_counts(high_pass: int, high_fail: int, non_pass: int, non_fail: int) -> float:
    # Haldane-Anscombe correction keeps the ratio finite when a cell is zero.
    return float(((high_pass + 0.5) * (non_fail + 0.5)) / ((high_fail + 0.5) * (non_pass + 0.5)))


def binomial_ci(
    high_pass: int,
    high_n: int,
    non_pass: int,
    non_n: int,
    *,
    seed: int,
    bootstrap: int,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    if bootstrap <= 0:
        return {
            "enrichment_low": float("nan"),
            "enrichment_high": float("nan"),
            "odds_ratio_low": float("nan"),
            "odds_ratio_high": float("nan"),
        }
    ph = high_pass / max(high_n, 1)
    pn = non_pass / max(non_n, 1)
    high_draw = rng.binomial(high_n, ph, size=bootstrap)
    non_draw = rng.binomial(non_n, pn, size=bootstrap)
    high_rate = high_draw / max(high_n, 1)
    non_rate = non_draw / max(non_n, 1)
    enrich = (high_rate + 1e-12) / (non_rate + 1e-12)
    high_fail = high_n - high_draw
    non_fail = non_n - non_draw
    odds = ((high_draw + 0.5) * (non_fail + 0.5)) / ((high_fail + 0.5) * (non_draw + 0.5))
    return {
        "enrichment_low": float(np.percentile(enrich, 2.5)),
        "enrichment_high": float(np.percentile(enrich, 97.5)),
        "odds_ratio_low": float(np.percentile(odds, 2.5)),
        "odds_ratio_high": float(np.percentile(odds, 97.5)),
    }


def summarize_binary_metric(
    df: pd.DataFrame,
    pass_col: str,
    *,
    metric_name: str,
    source: str,
    eta_high: float,
    eta_nonhigh: float,
    seed: int,
    bootstrap: int,
) -> dict[str, float | str | int]:
    high = df["eta"].to_numpy() >= eta_high
    non = df["eta"].to_numpy() < eta_nonhigh
    passes = df[pass_col].astype(bool).to_numpy()
    high_pass = int(passes[high].sum())
    high_n = int(high.sum())
    non_pass = int(passes[non].sum())
    non_n = int(non.sum())
    high_rate = safe_rate(high_pass, high_n)
    non_rate = safe_rate(non_pass, non_n)
    high_fail = high_n - high_pass
    non_fail = non_n - non_pass
    enrichment = high_rate / max(non_rate, 1e-12)
    odds_ratio = odds_ratio_from_counts(high_pass, high_fail, non_pass, non_fail)

    if fisher_exact is not None:
        _, p_two_sided = fisher_exact([[high_pass, high_fail], [non_pass, non_fail]], alternative="two-sided")
        _, p_greater = fisher_exact([[high_pass, high_fail], [non_pass, non_fail]], alternative="greater")
    else:
        p_two_sided = float("nan")
        p_greater = float("nan")

    ci = binomial_ci(high_pass, high_n, non_pass, non_n, seed=seed, bootstrap=bootstrap)
    return {
        "source": source,
        "metric": metric_name,
        "pass_column": pass_col,
        "high_n": high_n,
        "nonhigh_n": non_n,
        "high_pass": high_pass,
        "nonhigh_pass": non_pass,
        "high_pass_rate": high_rate,
        "nonhigh_pass_rate": non_rate,
        "pass_rate_delta": high_rate - non_rate,
        "enrichment_ratio": enrichment,
        "enrichment_low": ci["enrichment_low"],
        "enrichment_high": ci["enrichment_high"],
        "odds_ratio": odds_ratio,
        "odds_ratio_low": ci["odds_ratio_low"],
        "odds_ratio_high": ci["odds_ratio_high"],
        "fisher_p_two_sided": float(p_two_sided),
        "fisher_p_greater": float(p_greater),
    }


def load_metric_frames(force: bool) -> list[tuple[str, pd.DataFrame, list[tuple[str, str]]]]:
    run_script_if_needed(MECH_SCRIPT, MECH_SCORES, force=force)
    run_script_if_needed(BATH_SCRIPT, BATH_SCORES, force=force)

    mech = ensure_pass_columns(pd.read_csv(MECH_SCORES))
    bath = ensure_pass_columns(pd.read_csv(BATH_SCORES))

    mech_metrics = [
        ("bath_resonance_original", "bath_pass"),
        ("source_sink_delocalization", "deloc_pass"),
        ("strict_joint_original", "joint_pass"),
        ("loose_joint_original", "loose_joint_pass"),
    ]
    bath_metrics = [
        ("spectral_only", "spectral_only_pass"),
        ("bath_x_coupling", "bath_x_coupling_pass"),
        ("bath_x_path", "bath_x_path_pass"),
        ("bath_x_coupling_x_path", "bath_x_coupling_x_path_pass"),
    ]
    return [
        ("original_mechanistic_signature", mech, [(name, col) for name, col in mech_metrics if col in mech.columns]),
        ("bath_sensitivity_variants", bath, [(name, col) for name, col in bath_metrics if col in bath.columns]),
    ]


def plot_forest(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.sort_values("enrichment_ratio", ascending=True).reset_index(drop=True)
    y = np.arange(len(plot_df))
    ratio = plot_df["enrichment_ratio"].to_numpy(float)
    low = plot_df["enrichment_low"].to_numpy(float)
    high = plot_df["enrichment_high"].to_numpy(float)
    labels = plot_df["metric"].str.replace("_", " ", regex=False).to_list()

    colors = []
    for lo, hi in zip(low, high):
        if lo > 1.0:
            colors.append("#2A9D55")
        elif hi < 1.0:
            colors.append("#C73E3A")
        else:
            colors.append("#6B7A90")

    fig, ax = plt.subplots(figsize=(10.5, max(5.8, 0.55 * len(plot_df) + 2.0)))
    xerr = np.vstack([np.maximum(ratio - low, 0.0), np.maximum(high - ratio, 0.0)])
    ax.errorbar(ratio, y, xerr=xerr, fmt="none", ecolor="#7A7A7A", elinewidth=1.6, capsize=4)
    ax.scatter(ratio, y, s=80, c=colors, zorder=3)
    ax.axvline(1.0, color="#222222", linestyle="--", linewidth=1.2)
    ax.set_xscale("log")
    ax.set_yticks(y, labels=labels)
    ax.set_xlabel("high-eta pass rate / non-high pass rate")
    ax.set_title("C2 signature enrichment in high-efficiency Hamiltonians")
    ax.grid(axis="x", alpha=0.25)
    for i, row in plot_df.iterrows():
        ax.text(row["enrichment_high"] * 1.05, i, f"{row['enrichment_ratio']:.2f}x", va="center", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_pass_rates(summary: pd.DataFrame, out_path: Path) -> None:
    plot_df = summary.copy()
    plot_df["metric_label"] = plot_df["metric"].str.replace("_", " ", regex=False)
    x = np.arange(len(plot_df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12.0, 6.2))
    ax.bar(x - width / 2, plot_df["high_pass_rate"] * 100, width, label="high eta >= 0.95", color="#4E79A7")
    ax.bar(x + width / 2, plot_df["nonhigh_pass_rate"] * 100, width, label="non-high eta < 0.85", color="#F28E2B")
    ax.set_xticks(x, labels=plot_df["metric_label"], rotation=28, ha="right")
    ax.set_ylabel("pass rate (%)")
    ax.set_title("C2 signature pass rates: high-eta vs non-high")
    ax.legend(frameon=True)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown(path: Path, summary: pd.DataFrame, figures: dict[str, Path], bootstrap: int) -> None:
    strongest = summary.sort_values("enrichment_ratio", ascending=False).iloc[0]
    weakest = summary.sort_values("enrichment_ratio", ascending=True).iloc[0]
    md = []
    md.append("# C2 signature statistical follow-up\n")
    md.append(
        "This analysis checks whether the physical signatures proposed for C2 occur more often "
        "in high-efficiency Hamiltonians. The comparison groups are eta >= 0.95 and eta < 0.85.\n"
    )
    md.append("## Main Readout\n")
    md.append(
        f"- The strongest enrichment appears for `{strongest['metric']}`: "
        f"high-eta pass rate {strongest['high_pass_rate'] * 100:.1f}%, "
        f"non-high pass rate {strongest['nonhigh_pass_rate'] * 100:.1f}%, "
        f"ratio {strongest['enrichment_ratio']:.2f}x.\n"
    )
    md.append(
        f"- The weakest enrichment appears for `{weakest['metric']}` with ratio {weakest['enrichment_ratio']:.2f}x.\n"
    )
    md.append(
        f"- Confidence intervals are estimated by binomial bootstrap with {bootstrap} resamples; "
        "Fisher exact-test p-values are stored in the output table.\n"
    )
    md.append("## Figures\n")
    for label, fig_path in figures.items():
        rel = fig_path.relative_to(path.parent).as_posix()
        md.append(f"![{label}]({rel})\n")
        if label == "enrichment_forest":
            md.append(
                "The x-axis is the high-eta pass rate divided by the non-high pass rate. "
                "Values above 1 indicate that the signature appears more often in high-eta samples.\n"
            )
        elif label == "pass_rates":
            md.append(
                "This figure directly compares high-eta and non-high pass rates for each signature.\n"
            )
    md.append("## Output Tables\n")
    md.append("- `c2_signature_enrichment.csv`: pass rate, enrichment ratio, odds ratio, confidence interval, p-value\n")
    path.write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for source, df, metrics in load_metric_frames(args.force_recompute):
        for i, (metric_name, pass_col) in enumerate(metrics):
            rows.append(
                summarize_binary_metric(
                    df,
                    pass_col,
                    metric_name=metric_name,
                    source=source,
                    eta_high=args.eta_high,
                    eta_nonhigh=args.eta_nonhigh,
                    seed=args.seed + i,
                    bootstrap=args.bootstrap,
                )
            )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out_dir / "c2_signature_enrichment.csv", index=False)

    figures = {
        "enrichment_forest": figures_dir / "c2_signature_enrichment_forest.png",
        "pass_rates": figures_dir / "c2_signature_pass_rate_comparison.png",
    }
    plot_forest(summary, figures["enrichment_forest"])
    plot_pass_rates(summary, figures["pass_rates"])
    write_markdown(args.out_dir / "c2_signature_statistics_summary.md", summary, figures, args.bootstrap)
    write_json(
        args.out_dir / "manifest.json",
        {
            "eta_high": args.eta_high,
            "eta_nonhigh": args.eta_nonhigh,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "input_scores": {
                "mechanistic": MECH_SCORES,
                "bath_sensitivity": BATH_SCORES,
            },
            "outputs": {
                "summary": "c2_signature_enrichment.csv",
                "markdown": "c2_signature_statistics_summary.md",
                "figures": {k: str(v.relative_to(args.out_dir)) for k, v in figures.items()},
            },
        },
    )

    print("[saved]", args.out_dir)
    print(
        summary[
            [
                "source",
                "metric",
                "high_pass_rate",
                "nonhigh_pass_rate",
                "enrichment_ratio",
                "enrichment_low",
                "enrichment_high",
                "fisher_p_greater",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

