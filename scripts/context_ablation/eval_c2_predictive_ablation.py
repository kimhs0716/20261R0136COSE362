from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs" / "c2_predictive_ablation"
MECH_SCORES = ROOT / "outputs" / "model_performance" / "comparison" / "c2_mechanistic_signature_scores.csv"
BATH_SCORES = ROOT / "outputs" / "c2_bath_sensitivity" / "c2_bath_sensitivity_scores.csv"

ETA_HIGH = 0.95
ETA_NONHIGH = 0.85
RANDOM_SEED = 716


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Held-out classifier ablation for C2 scores. This checks whether each "
            "mechanistic score block can predict high-eta vs non-high labels."
        )
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def load_design_matrix() -> pd.DataFrame:
    if not MECH_SCORES.exists():
        raise FileNotFoundError(MECH_SCORES)
    if not BATH_SCORES.exists():
        raise FileNotFoundError(BATH_SCORES)
    mech = pd.read_csv(MECH_SCORES)
    bath = pd.read_csv(BATH_SCORES)
    df = pd.DataFrame(
        {
            "dataset_index": mech["dataset_index"],
            "eta": mech["eta"],
            "bath_score": mech["bath_score"],
            "deloc_score": mech["deloc_score"],
            "best_bath_gap_cm": mech["best_bath_gap_cm"],
            "source_weight_at_best": mech["source_weight_at_best"],
            "sink_weight_at_best": mech["sink_weight_at_best"],
            "loose_deloc_state_count": mech["loose_deloc_state_count"],
            "spectral_only_score": bath["spectral_only_score"],
            "bath_x_coupling_score": bath["bath_x_coupling_score"],
            "bath_x_path_score": bath["bath_x_path_score"],
            "bath_x_coupling_x_path_score": bath["bath_x_coupling_x_path_score"],
        }
    )
    return df


def evaluate_feature_set(df: pd.DataFrame, feature_cols: list[str], *, seed: int, test_size: float) -> dict:
    mask = (df["eta"] >= ETA_HIGH) | (df["eta"] < ETA_NONHIGH)
    data = df.loc[mask].copy()
    y = (data["eta"] >= ETA_HIGH).astype(int).to_numpy()
    x = data[feature_cols].to_numpy(dtype=np.float64)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs"),
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "features": ", ".join(feature_cols),
        "n_features": len(feature_cols),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "average_precision": float(average_precision_score(y_test, prob)),
        "balanced_accuracy_at_0_5": float(balanced_accuracy_score(y_test, pred)),
    }


def plot_results(results: pd.DataFrame, out_path: Path) -> None:
    df = results.sort_values("roc_auc")
    labels = df["feature_set"].to_list()
    y = np.arange(len(df))
    ap_baseline = float(df["positive_rate_test"].iloc[0])
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.barh(y - 0.18, df["roc_auc"], height=0.34, label="ROC AUC", color="#4E79A7")
    ax.barh(y + 0.18, df["average_precision"], height=0.34, label="average precision", color="#F28E2B")
    ax.axvline(0.5, color="#222222", linestyle="--", linewidth=1.0, label="ROC random baseline")
    ax.axvline(ap_baseline, color="#B45F06", linestyle=":", linewidth=1.4, label="AP random baseline")
    ax.set_yticks(y, labels=labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("held-out score")
    ax.set_title("C2 predictive ablation on held-out high-eta vs non-high split")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=True)
    for i, row in df.iterrows():
        ypos = list(df.index).index(i)
        ax.text(row["roc_auc"] + 0.01, ypos - 0.18, f"{row['roc_auc']:.2f}", va="center", fontsize=8)
        ax.text(row["average_precision"] + 0.01, ypos + 0.18, f"{row['average_precision']:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, results: pd.DataFrame, fig_path: Path) -> None:
    best = results.sort_values("roc_auc", ascending=False).iloc[0]
    deloc = results.loc[results["feature_set"] == "deloc only"].iloc[0]
    bath = results.loc[results["feature_set"] == "bath only"].iloc[0]
    positive_rate = float(results["positive_rate_test"].iloc[0])
    md = []
    md.append("# C2 predictive ablation\n")
    md.append(
        "This analysis checks whether the C2 score sets can separate high-eta and non-high samples "
        "on a held-out split. The middle eta range is excluded; eta >= 0.95 is positive and "
        "eta < 0.85 is negative.\n"
    )
    md.append(f"The test positive rate is {positive_rate * 100:.1f}%, so average precision should be interpreted relative to this baseline.\n")
    md.append(f"![C2 predictive ablation]({fig_path.relative_to(path.parent).as_posix()})\n")
    md.append(
        f"The best ROC AUC is obtained by `{best['feature_set']}` with AUC {best['roc_auc']:.3f} "
        f"and average precision {best['average_precision']:.3f}.\n"
    )
    md.append(
        f"`deloc only` gives AUC {deloc['roc_auc']:.3f} and average precision {deloc['average_precision']:.3f}, "
        f"whereas `bath only` gives AUC {bath['roc_auc']:.3f} and average precision {bath['average_precision']:.3f}.\n"
    )
    path.write_text("".join(md), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load_design_matrix()

    feature_sets = {
        "bath only": ["bath_score"],
        "deloc only": ["deloc_score"],
        "bath + deloc": ["bath_score", "deloc_score"],
        "bath x coupling x path": ["bath_x_coupling_x_path_score"],
        "all C2 scores": [
            "bath_score",
            "deloc_score",
            "source_weight_at_best",
            "sink_weight_at_best",
            "loose_deloc_state_count",
            "bath_x_coupling_score",
            "bath_x_path_score",
            "bath_x_coupling_x_path_score",
        ],
    }

    rows = []
    for name, cols in feature_sets.items():
        row = evaluate_feature_set(df, cols, seed=args.seed, test_size=args.test_size)
        row["feature_set"] = name
        rows.append(row)
    results = pd.DataFrame(rows)[
        [
            "feature_set",
            "n_features",
            "n_train",
            "n_test",
            "positive_rate_test",
            "roc_auc",
            "average_precision",
            "balanced_accuracy_at_0_5",
            "features",
        ]
    ]
    results.to_csv(out_dir / "c2_predictive_ablation.csv", index=False)
    fig_path = out_dir / "figures" / "c2_predictive_ablation.png"
    plot_results(results, fig_path)
    write_summary(out_dir / "c2_predictive_ablation_summary.md", results, fig_path)
    write_json(
        out_dir / "manifest.json",
        {
            "eta_high": ETA_HIGH,
            "eta_nonhigh": ETA_NONHIGH,
            "seed": args.seed,
            "test_size": args.test_size,
            "inputs": {
                "mechanistic_scores": MECH_SCORES,
                "bath_sensitivity_scores": BATH_SCORES,
            },
            "outputs": {
                "results": "c2_predictive_ablation.csv",
                "summary": "c2_predictive_ablation_summary.md",
                "figure": "figures/c2_predictive_ablation.png",
            },
        },
    )
    print("[saved]", out_dir)
    print(results.sort_values("roc_auc", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

