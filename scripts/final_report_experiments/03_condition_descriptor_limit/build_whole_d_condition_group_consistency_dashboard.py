from __future__ import annotations

import importlib.util
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import entropy, kurtosis, skew
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "new" / "build_whole_d_dynamic_condition_explainability_eta50_ge080.py"
OUT = ROOT / "new" / "whole_d_condition_group_consistency_20260620"
CSV = OUT / "csv"
FIG = OUT / "figures"
REPORT = OUT / "reports"
HTML_ASSETS = ROOT / "htmls" / "63_whole_d_condition_group_consistency_assets"
HTML_OUT = ROOT / "htmls" / "63_whole_d_condition_group_consistency_dashboard.html"

RANDOM_SEED = 20260620
MAX_PAIR_SAMPLES = 60000
MAX_SCATTER_POINTS = 9000
KNN_K = 15


def import_base_module():
    spec = importlib.util.spec_from_file_location("base_eta50_ge080", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = import_base_module()
DESCRIPTORS: dict[str, str] = BASE.DESCRIPTORS
ABSTRACT_DESCRIPTORS = ["cl1_10_20", "IPR_10_20", "purity_10_20"]
READOUT_DESCRIPTORS = ["early_trap_10", "source_site1_10", "residual_10"]
ALL_DESCRIPTORS = list(DESCRIPTORS.keys())


def ensure_dirs() -> None:
    for path in [OUT, CSV, FIG, REPORT, HTML_ASSETS]:
        path.mkdir(parents=True, exist_ok=True)


def savefig(name: str) -> str:
    path = FIG / name
    fig = plt.gcf()
    if not getattr(fig, "_skip_tight_layout", False):
        plt.tight_layout()
    plt.savefig(path, dpi=175, bbox_inches="tight")
    plt.close()
    shutil.copy2(path, HTML_ASSETS / name)
    return name


def style(ax, title: str | None = None) -> None:
    ax.grid(True, color="#e7ebf0", linewidth=0.75)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=11, pad=7)


def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    df = BASE.load_dataset().copy()
    z_dynamic, profile_cols, _ = BASE.robust_profile(df)
    cond_cols = [DESCRIPTORS[d] for d in ALL_DESCRIPTORS]
    cond_values = df[cond_cols].replace([np.inf, -np.inf], np.nan)
    cond_values = cond_values.fillna(cond_values.median(numeric_only=True))
    cond_scaled = RobustScaler().fit_transform(cond_values)
    cond_scaled_df = pd.DataFrame(cond_scaled, index=df.index, columns=ALL_DESCRIPTORS)
    abstract_scaled = cond_scaled_df[ABSTRACT_DESCRIPTORS].to_numpy()
    d_order = sorted(df["dynamic_family_id"].dropna().unique().tolist())
    return df, z_dynamic, profile_cols, cond_scaled_df, cond_scaled, abstract_scaled, d_order


def profile_spread(z_slice: np.ndarray) -> float:
    if len(z_slice) < 3:
        return float("nan")
    return float(np.nanmean(np.nanstd(z_slice, axis=0)))


def group_medians(df: pd.DataFrame, values: pd.DataFrame, d_order: list[str]) -> pd.DataFrame:
    rows = []
    for d in d_order:
        idx = df["dynamic_family_id"].eq(d).to_numpy()
        row = {"D_family": d, "n": int(idx.sum())}
        for col in values.columns:
            row[col] = float(np.nanmedian(values.loc[idx, col]))
        rows.append(row)
    return pd.DataFrame(rows)


def group_iqrs(df: pd.DataFrame, values: pd.DataFrame, d_order: list[str]) -> pd.DataFrame:
    rows = []
    for d in d_order:
        idx = df["dynamic_family_id"].eq(d).to_numpy()
        row = {"D_family": d, "n": int(idx.sum())}
        for col in values.columns:
            row[col] = float(np.nanquantile(values.loc[idx, col], 0.75) - np.nanquantile(values.loc[idx, col], 0.25))
        rows.append(row)
    return pd.DataFrame(rows)


def variance_decomposition(df: pd.DataFrame, values: pd.DataFrame, d_order: list[str]) -> pd.DataFrame:
    rows = []
    labels = df["dynamic_family_id"].to_numpy()
    n_total = len(df)
    for desc in values.columns:
        x = values[desc].to_numpy(float)
        valid = np.isfinite(x)
        total_var = float(np.nanvar(x[valid]))
        weighted_within = 0.0
        weighted_between = 0.0
        global_mean = float(np.nanmean(x[valid]))
        median_iqrs = []
        medians = []
        for d in d_order:
            mask = (labels == d) & valid
            if mask.sum() < 2:
                continue
            xd = x[mask]
            w = mask.sum() / n_total
            weighted_within += w * float(np.nanvar(xd))
            weighted_between += w * (float(np.nanmean(xd)) - global_mean) ** 2
            median_iqrs.append(float(np.nanquantile(xd, 0.75) - np.nanquantile(xd, 0.25)))
            medians.append(float(np.nanmedian(xd)))
        rows.append(
            {
                "descriptor": desc,
                "total_variance": total_var,
                "within_D_variance": weighted_within,
                "between_D_variance": weighted_between,
                "between_variance_fraction": weighted_between / total_var if total_var > 0 else np.nan,
                "median_within_D_iqr": float(np.nanmedian(median_iqrs)),
                "D_median_range": float(np.nanmax(medians) - np.nanmin(medians)),
            }
        )
    return pd.DataFrame(rows)


def histogram_overlap_matrix(df: pd.DataFrame, values: pd.DataFrame, d_order: list[str]) -> pd.DataFrame:
    mat = np.zeros((len(d_order), len(d_order)), dtype=float)
    bins = np.linspace(-4, 4, 49)
    for i, da in enumerate(d_order):
        for j, db in enumerate(d_order):
            overlaps = []
            for desc in values.columns:
                xa = values.loc[df["dynamic_family_id"].eq(da), desc].dropna().to_numpy()
                xb = values.loc[df["dynamic_family_id"].eq(db), desc].dropna().to_numpy()
                if len(xa) < 10 or len(xb) < 10:
                    continue
                ha, _ = np.histogram(np.clip(xa, -4, 4), bins=bins)
                hb, _ = np.histogram(np.clip(xb, -4, 4), bins=bins)
                pa = ha / ha.sum() if ha.sum() else ha
                pb = hb / hb.sum() if hb.sum() else hb
                overlaps.append(float(np.minimum(pa, pb).sum()))
            mat[i, j] = float(np.nanmean(overlaps)) if overlaps else np.nan
    return pd.DataFrame(mat, index=d_order, columns=d_order)


def train_condition_classifiers(df: pd.DataFrame, X_abs: np.ndarray, X_all: np.ndarray):
    y = df["dynamic_family_id"].to_numpy()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    models = {
        "abstract": RandomForestClassifier(
            n_estimators=180,
            max_depth=8,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED,
        ),
        "all_condition": RandomForestClassifier(
            n_estimators=220,
            max_depth=10,
            min_samples_leaf=6,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_SEED + 1,
        ),
    }
    outputs = {}
    for name, model in models.items():
        X = X_abs if name == "abstract" else X_all
        pred = cross_val_predict(model, X, y, cv=cv, n_jobs=None, method="predict")
        proba = cross_val_predict(model, X, y, cv=cv, n_jobs=None, method="predict_proba")
        classes = np.array(sorted(np.unique(y)))
        outputs[name] = {
            "pred": pred,
            "proba": proba,
            "classes": classes,
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "confusion": confusion_matrix(y, pred, labels=classes, normalize="true"),
        }
    return outputs


def local_purity(df: pd.DataFrame, X: np.ndarray, label: str) -> pd.DataFrame:
    y = df["dynamic_family_id"].to_numpy()
    nbrs = NearestNeighbors(n_neighbors=KNN_K + 1, metric="euclidean").fit(X)
    _, inds = nbrs.kneighbors(X)
    rows = []
    for i in range(len(df)):
        neigh = inds[i, 1:]
        rows.append(
            {
                "sample_index": df.iloc[i]["sample_index"],
                "D_family": y[i],
                "feature_set": label,
                "local_D_purity": float(np.mean(y[neigh] == y[i])),
            }
        )
    return pd.DataFrame(rows)


def entropy_over_condition_bins(df: pd.DataFrame, values: pd.DataFrame, d_order: list[str]) -> pd.DataFrame:
    rows = []
    for desc in values.columns:
        bins = pd.qcut(values[desc], q=5, labels=False, duplicates="drop")
        for b in sorted(pd.Series(bins).dropna().unique()):
            mask = bins == b
            counts = df.loc[mask, "dynamic_family_id"].value_counts().reindex(d_order, fill_value=0)
            probs = counts.to_numpy(float)
            probs = probs / probs.sum() if probs.sum() else probs
            ent = float(entropy(probs, base=math.e))
            norm = ent / math.log(len(d_order)) if len(d_order) > 1 else np.nan
            rows.append(
                {
                    "descriptor": desc,
                    "condition_bin": int(b),
                    "n": int(mask.sum()),
                    "D_entropy": ent,
                    "normalized_D_entropy": norm,
                    "n_D_present": int((counts > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def dynamic_bin_multimodality(df: pd.DataFrame, values: pd.DataFrame, z_dynamic: np.ndarray) -> pd.DataFrame:
    pc1 = PCA(n_components=1, random_state=RANDOM_SEED).fit_transform(z_dynamic).ravel()
    rows = []
    for desc in values.columns:
        bins = pd.qcut(values[desc], q=5, labels=False, duplicates="drop")
        for b in sorted(pd.Series(bins).dropna().unique()):
            mask = (bins == b).to_numpy()
            y = df.loc[mask, "eta50p0"].to_numpy(float)
            p = pc1[mask]
            if mask.sum() < 40:
                bc = np.nan
                eta_iqr = np.nan
                dyn_spread = np.nan
            else:
                p_kurt = kurtosis(p, fisher=False, nan_policy="omit")
                bc = float((skew(p, nan_policy="omit") ** 2 + 1) / p_kurt) if p_kurt and np.isfinite(p_kurt) else np.nan
                eta_iqr = float(np.nanquantile(y, 0.75) - np.nanquantile(y, 0.25))
                dyn_spread = profile_spread(z_dynamic[mask])
            rows.append(
                {
                    "descriptor": desc,
                    "condition_bin": int(b),
                    "n": int(mask.sum()),
                    "eta50_iqr": eta_iqr,
                    "dynamic_pc1_bimodality_coefficient": bc,
                    "dynamic_profile_spread": dyn_spread,
                }
            )
    return pd.DataFrame(rows)


def centroid_distance_table(df: pd.DataFrame, X_cond: np.ndarray, z_dynamic: np.ndarray, d_order: list[str]) -> pd.DataFrame:
    rows = []
    labels = df["dynamic_family_id"].to_numpy()
    for i, da in enumerate(d_order):
        for db in d_order[i + 1 :]:
            ma = labels == da
            mb = labels == db
            cdist = float(np.linalg.norm(np.nanmedian(X_cond[ma], axis=0) - np.nanmedian(X_cond[mb], axis=0)))
            ddist = float(np.linalg.norm(np.nanmedian(z_dynamic[ma], axis=0) - np.nanmedian(z_dynamic[mb], axis=0)))
            rows.append({"D_a": da, "D_b": db, "condition_centroid_distance": cdist, "dynamic_centroid_distance": ddist})
    return pd.DataFrame(rows)


def residualized_dynamic_prediction(df: pd.DataFrame, X_abs: np.ndarray, X_all: np.ndarray, z_dynamic: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=8, random_state=RANDOM_SEED)
    Y = pca.fit_transform(z_dynamic)
    y_labels = df["dynamic_family_id"].astype(str).to_numpy()
    one_hot = pd.get_dummies(y_labels, prefix="D", dtype=float).to_numpy()
    X_sets = {
        "abstract_conditions": X_abs,
        "all_conditions": X_all,
        "all_conditions_plus_D": np.hstack([X_all, one_hot]),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    residual_rows = []
    for name, X in X_sets.items():
        model = Ridge(alpha=1.0)
        pred = cross_val_predict(model, X, Y, cv=cv)
        for k in range(Y.shape[1]):
            score = float(r2_score(Y[:, k], pred[:, k]))
            resid_ratio = float(np.nanstd(Y[:, k] - pred[:, k]) / np.nanstd(Y[:, k]))
            rows.append(
                {
                    "feature_set": name,
                    "dynamic_pc": f"PC{k+1}",
                    "explained_variance_ratio": float(pca.explained_variance_ratio_[k]),
                    "cv_r2": score,
                    "residual_std_ratio": resid_ratio,
                }
            )
        residual_rows.append(
            {
                "feature_set": name,
                "weighted_mean_cv_r2": float(np.average([r["cv_r2"] for r in rows if r["feature_set"] == name], weights=pca.explained_variance_ratio_)),
                "mean_residual_std_ratio": float(np.nanmean([r["residual_std_ratio"] for r in rows if r["feature_set"] == name])),
            }
        )
    # A shallow nonlinear check for all conditions only.
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    pred_rf = cross_val_predict(rf, X_all, Y[:, :5], cv=cv, n_jobs=None)
    for k in range(5):
        rows.append(
            {
                "feature_set": "all_conditions_random_forest",
                "dynamic_pc": f"PC{k+1}",
                "explained_variance_ratio": float(pca.explained_variance_ratio_[k]),
                "cv_r2": float(r2_score(Y[:, k], pred_rf[:, k])),
                "residual_std_ratio": float(np.nanstd(Y[:, k] - pred_rf[:, k]) / np.nanstd(Y[:, k])),
            }
        )
    residual_rows.append(
        {
            "feature_set": "all_conditions_random_forest",
            "weighted_mean_cv_r2": float(
                np.average(
                    [r["cv_r2"] for r in rows if r["feature_set"] == "all_conditions_random_forest"],
                    weights=pca.explained_variance_ratio_[:5],
                )
            ),
            "mean_residual_std_ratio": float(np.nanmean([r["residual_std_ratio"] for r in rows if r["feature_set"] == "all_conditions_random_forest"])),
        }
    )
    return pd.DataFrame(rows), pd.DataFrame(residual_rows)


def sample_pair_distances(df: pd.DataFrame, X_cond: np.ndarray, z_dynamic: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(df)
    a = rng.integers(0, n, size=MAX_PAIR_SAMPLES)
    b = rng.integers(0, n, size=MAX_PAIR_SAMPLES)
    mask = a != b
    a, b = a[mask], b[mask]
    cond_dist = np.linalg.norm(X_cond[a] - X_cond[b], axis=1) / math.sqrt(X_cond.shape[1])
    dyn_dist = np.linalg.norm(z_dynamic[a] - z_dynamic[b], axis=1) / math.sqrt(z_dynamic.shape[1])
    labels = df["dynamic_family_id"].to_numpy()
    return pd.DataFrame(
        {
            "sample_a": df.iloc[a]["sample_index"].to_numpy(),
            "sample_b": df.iloc[b]["sample_index"].to_numpy(),
            "D_a": labels[a],
            "D_b": labels[b],
            "same_D": labels[a] == labels[b],
            "condition_distance": cond_dist,
            "dynamic_distance": dyn_dist,
            "eta50_distance": np.abs(df.iloc[a]["eta50p0"].to_numpy(float) - df.iloc[b]["eta50p0"].to_numpy(float)),
        }
    )


def hard_negative_cases(df: pd.DataFrame, X_all: np.ndarray, z_dynamic: np.ndarray, clf_out: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df["dynamic_family_id"].to_numpy()
    pred = clf_out["all_condition"]["pred"]
    proba = clf_out["all_condition"]["proba"]
    classes = clf_out["all_condition"]["classes"]
    max_prob = proba.max(axis=1)
    wrong = np.where(pred != y)[0]
    hard_idx = wrong[np.argsort(max_prob[wrong])[::-1][:80]]
    hard = pd.DataFrame(
        {
            "sample_index": df.iloc[hard_idx]["sample_index"].to_numpy(),
            "true_D": y[hard_idx],
            "pred_D": pred[hard_idx],
            "confidence": max_prob[hard_idx],
            "eta50": df.iloc[hard_idx]["eta50p0"].to_numpy(float),
        }
    )
    # Pair counterexamples.
    nbrs = NearestNeighbors(n_neighbors=25).fit(X_all)
    _, inds = nbrs.kneighbors(X_all)
    cond_pairs = []
    seen_pairs = set()
    for i in range(len(df)):
        best = None
        for j in inds[i, 1:]:
            if y[i] == y[j]:
                continue
            cd = float(np.linalg.norm(X_all[i] - X_all[j]) / math.sqrt(X_all.shape[1]))
            dd = float(np.linalg.norm(z_dynamic[i] - z_dynamic[j]) / math.sqrt(z_dynamic.shape[1]))
            score = dd / (cd + 1e-6)
            if best is None or score > best["score"]:
                best = {"i": i, "j": int(j), "condition_distance": cd, "dynamic_distance": dd, "score": score}
        if best is not None:
            key = tuple(sorted((best["i"], best["j"])))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            cond_pairs.append(best)
    cond_pairs = sorted(cond_pairs, key=lambda r: r["score"], reverse=True)[:20]
    rows = []
    for r in cond_pairs:
        i, j = r["i"], r["j"]
        rows.append(
            {
                "case_type": "same_condition_different_dynamic",
                "sample_a": df.iloc[i]["sample_index"],
                "sample_b": df.iloc[j]["sample_index"],
                "D_a": y[i],
                "D_b": y[j],
                "eta50_a": df.iloc[i]["eta50p0"],
                "eta50_b": df.iloc[j]["eta50p0"],
                "condition_distance": r["condition_distance"],
                "dynamic_distance": r["dynamic_distance"],
                "score": r["score"],
            }
        )
    return hard, pd.DataFrame(rows)


def draw_heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, name: str, cmap="viridis", vmin=None, vmax=None, annotate=False) -> str:
    fig, ax = plt.subplots(figsize=(11.5, 6.5))
    arr = matrix.to_numpy(float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(matrix.shape[1]), matrix.columns, rotation=42, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]), matrix.index, fontsize=9)
    ax.set_title(title, fontsize=14, pad=12)
    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = arr[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color="#111827")
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cb.set_label(cbar_label)
    return savefig(name)


def figure_01_boxplots(df, d_order) -> str:
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), sharex=True)
    for ax, desc in zip(axes.ravel(), ALL_DESCRIPTORS):
        col = DESCRIPTORS[desc]
        data = [df.loc[df["dynamic_family_id"].eq(d), col].dropna().to_numpy() for d in d_order]
        ax.boxplot(data, tick_labels=d_order, showfliers=False)
        ax.set_title(desc)
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        style(ax)
    fig.suptitle("Condition distributions inside each whole-D dynamic group", fontsize=15, y=1.0)
    return savefig("fig01_condition_distribution_by_D.png")


def figure_04_variance(var_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    order = var_df.sort_values("between_variance_fraction", ascending=False)
    axes[0].bar(order["descriptor"], order["between_variance_fraction"], color="#4c78a8")
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("between-D variance / total variance")
    axes[0].set_title("How much condition variance is between D groups?")
    axes[0].tick_params(axis="x", rotation=35)
    axes[1].scatter(order["D_median_range"], order["median_within_D_iqr"], s=80, color="#f58518", edgecolor="white")
    for _, row in order.iterrows():
        axes[1].text(row["D_median_range"], row["median_within_D_iqr"], row["descriptor"], fontsize=8)
    axes[1].set_xlabel("range of D medians")
    axes[1].set_ylabel("median within-D IQR")
    axes[1].set_title("D separation vs within-D spread")
    for ax in axes:
        style(ax)
    return savefig("fig04_variance_decomposition_and_separability.png")


def figure_06_classifier_confusion(d_order, clf_outputs) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 6.5))
    fig.subplots_adjust(right=0.88, wspace=0.34)
    for ax, key, title in [
        (axes[0], "abstract", "Abstract conditions only"),
        (axes[1], "all_condition", "All six condition/readout descriptors"),
    ]:
        mat = clf_outputs[key]["confusion"]
        im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"{title}\nbalanced acc={clf_outputs[key]['balanced_accuracy']:.3f}")
        ax.set_xticks(range(len(d_order)), d_order, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(d_order)), d_order, fontsize=8)
        ax.set_xlabel("predicted D")
        ax.set_ylabel("true D")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j] >= 0.12:
                    ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=6.5, color="#111827")
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cax, label="row-normalized fraction")
    fig._skip_tight_layout = True
    return savefig("fig06_condition_only_classifier_confusion.png")


def figure_07_error_network(d_order, clf_outputs) -> str:
    mat = clf_outputs["all_condition"]["confusion"].copy()
    np.fill_diagonal(mat, 0)
    weights = []
    for i in range(len(d_order)):
        for j in range(i + 1, len(d_order)):
            w = mat[i, j] + mat[j, i]
            if w > 0.04:
                weights.append((i, j, float(w)))
    theta = np.linspace(0, 2 * np.pi, len(d_order), endpoint=False)
    pos = np.column_stack([np.cos(theta), np.sin(theta)])
    fig, ax = plt.subplots(figsize=(7.6, 7.6))
    maxw = max([w for _, _, w in weights], default=1)
    for i, j, w in weights:
        ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="#d62728", alpha=0.18 + 0.65 * w / maxw, linewidth=0.8 + 5 * w / maxw)
    for i, d in enumerate(d_order):
        ax.scatter(pos[i, 0], pos[i, 1], s=520, color="#4c78a8", edgecolor="white", zorder=3)
        ax.text(pos[i, 0], pos[i, 1], d, color="white", ha="center", va="center", fontsize=9, weight="bold", zorder=4)
    ax.set_title("Frequently confused D pairs by condition-only classifier")
    ax.text(-1.25, -1.25, "edge width = symmetric row-normalized confusion", fontsize=9, color="#475467")
    ax.axis("off")
    return savefig("fig07_condition_classifier_error_network.png")


def figure_08_distance_scatter(pair_df: pd.DataFrame) -> str:
    plot_df = pair_df.sample(min(MAX_SCATTER_POINTS, len(pair_df)), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    colors = np.where(plot_df["same_D"], "#4c78a8", "#f58518")
    ax.scatter(plot_df["condition_distance"], plot_df["dynamic_distance"], s=8, alpha=0.22, c=colors, linewidths=0)
    bins = np.quantile(pair_df["condition_distance"], np.linspace(0, 1, 16))
    bins = np.unique(bins)
    xs, ys = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = pair_df["condition_distance"].between(lo, hi, inclusive="left")
        if m.sum() > 20:
            xs.append(float(pair_df.loc[m, "condition_distance"].median()))
            ys.append(float(pair_df.loc[m, "dynamic_distance"].median()))
    ax.plot(xs, ys, color="#111827", linewidth=2.2, marker="o", label="median dynamic distance by condition-distance bin")
    ax.set_xlabel("condition-space distance")
    ax.set_ylabel("dynamic-profile distance")
    ax.set_title("Dynamic distance is not fully determined by condition distance")
    ax.scatter([], [], c="#4c78a8", label="same D")
    ax.scatter([], [], c="#f58518", label="different D")
    ax.legend(frameon=False)
    style(ax)
    return savefig("fig08_dynamic_vs_condition_distance.png")


def figure_09_dendrogram(df, X_cond, z_dynamic, d_order) -> str:
    labels = df["dynamic_family_id"].to_numpy()
    cond_cent = np.vstack([np.nanmedian(X_cond[labels == d], axis=0) for d in d_order])
    dyn_cent = np.vstack([np.nanmedian(z_dynamic[labels == d], axis=0) for d in d_order])
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    dendrogram(linkage(cond_cent, method="ward"), labels=d_order, ax=axes[0], color_threshold=None)
    axes[0].set_title("D clustering by condition centroids")
    dendrogram(linkage(dyn_cent, method="ward"), labels=d_order, ax=axes[1], color_threshold=None)
    axes[1].set_title("D clustering by dynamic-profile centroids")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    return savefig("fig09_condition_vs_dynamic_centroid_dendrogram.png")


def figure_10_local_purity(purity_df: pd.DataFrame, d_order: list[str]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)
    for ax, fset in zip(axes, ["abstract_conditions", "all_condition_descriptors"]):
        sub = purity_df[purity_df["feature_set"].eq(fset)]
        data = [sub.loc[sub["D_family"].eq(d), "local_D_purity"].to_numpy() for d in d_order]
        ax.boxplot(data, tick_labels=d_order, showfliers=False)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"{fset}: kNN local D purity")
        ax.set_ylabel("fraction of nearest neighbors with same D")
        ax.tick_params(axis="x", rotation=45)
        style(ax)
    return savefig("fig10_local_purity_in_condition_space.png")


def figure_11_entropy(ent_df: pd.DataFrame) -> str:
    pvt = ent_df.pivot(index="descriptor", columns="condition_bin", values="normalized_D_entropy").reindex(ALL_DESCRIPTORS)
    return draw_heatmap(pvt, "D-label entropy inside condition quantile bins", "normalized entropy (0 separated, 1 mixed)", "fig11_D_entropy_over_condition_bins.png", cmap="magma", vmin=0, vmax=1, annotate=True)


def figure_12_multimodality(mm_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8))
    p1 = mm_df.pivot(index="descriptor", columns="condition_bin", values="dynamic_pc1_bimodality_coefficient").reindex(ALL_DESCRIPTORS)
    p2 = mm_df.pivot(index="descriptor", columns="condition_bin", values="eta50_iqr").reindex(ALL_DESCRIPTORS)
    im1 = axes[0].imshow(p1, aspect="auto", cmap="viridis", vmin=0.25, vmax=max(0.75, np.nanmax(p1.to_numpy())))
    axes[0].set_title("Dynamic PC1 bimodality proxy")
    im2 = axes[1].imshow(p2, aspect="auto", cmap="magma", vmin=0, vmax=max(0.15, np.nanmax(p2.to_numpy())))
    axes[1].set_title("eta50 IQR inside condition bins")
    for ax, p in zip(axes, [p1, p2]):
        ax.set_xticks(range(p.shape[1]), [f"bin {int(x)}" for x in p.columns])
        ax.set_yticks(range(p.shape[0]), p.index)
    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.02)
    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.02)
    return savefig("fig12_condition_bin_multimodality_proxy.png")


def figure_13_centroid_distance(cd: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.scatter(cd["condition_centroid_distance"], cd["dynamic_centroid_distance"], s=56, color="#4c78a8", alpha=0.78, edgecolor="white")
    # label a few discordant cases
    cd2 = cd.assign(discordance=lambda x: x["dynamic_centroid_distance"] / (x["condition_centroid_distance"] + 1e-6))
    for _, row in pd.concat([cd2.nlargest(5, "discordance"), cd2.nsmallest(3, "discordance")]).drop_duplicates(["D_a", "D_b"]).iterrows():
        ax.text(row["condition_centroid_distance"], row["dynamic_centroid_distance"], f"{row['D_a']}-{row['D_b']}", fontsize=8)
    ax.set_xlabel("D-pair condition centroid distance")
    ax.set_ylabel("D-pair dynamic centroid distance")
    ax.set_title("Condition centroid distance vs dynamic centroid distance")
    style(ax)
    return savefig("fig13_condition_vs_dynamic_centroid_distance.png")


def figure_14_residual_prediction(pc_scores: pd.DataFrame, residual_summary: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    pc_pvt = pc_scores.pivot(index="dynamic_pc", columns="feature_set", values="cv_r2")
    pc_pvt = pc_pvt[[c for c in ["abstract_conditions", "all_conditions", "all_conditions_plus_D", "all_conditions_random_forest"] if c in pc_pvt.columns]]
    pc_pvt.plot(kind="bar", ax=axes[0])
    axes[0].set_ylabel("CV R2")
    axes[0].set_title("Predict dynamic PCs from condition features")
    axes[0].legend(fontsize=8)
    residual_summary.plot(kind="bar", x="feature_set", y="mean_residual_std_ratio", ax=axes[1], color="#f58518", legend=False)
    axes[1].set_ylabel("mean residual std / original std")
    axes[1].set_title("Residual dynamic variation after prediction")
    axes[1].tick_params(axis="x", rotation=30)
    for ax in axes:
        style(ax)
    return savefig("fig14_residualized_dynamic_diversity.png")


def figure_15_condition_plus_D(pc_scores: pd.DataFrame) -> str:
    base = pc_scores[pc_scores["feature_set"].eq("all_conditions")].set_index("dynamic_pc")
    plus = pc_scores[pc_scores["feature_set"].eq("all_conditions_plus_D")].set_index("dynamic_pc")
    joined = base[["cv_r2", "explained_variance_ratio"]].rename(columns={"cv_r2": "condition_only_r2"}).join(
        plus[["cv_r2"]].rename(columns={"cv_r2": "condition_plus_D_r2"})
    )
    joined["delta_r2_from_D"] = joined["condition_plus_D_r2"] - joined["condition_only_r2"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.bar(joined.index, joined["delta_r2_from_D"], color="#54a24b")
    ax.set_ylabel("CV R2 gain by adding D label")
    ax.set_title("Does D label add dynamic information after conditions?")
    style(ax)
    return savefig("fig15_conditional_independence_D_gain.png")


def figure_16_calibration(df, clf_outputs) -> str:
    y = df["dynamic_family_id"].to_numpy()
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    rows = []
    for key, label, color in [("abstract", "abstract conditions", "#4c78a8"), ("all_condition", "all condition descriptors", "#f58518")]:
        proba = clf_outputs[key]["proba"]
        pred = clf_outputs[key]["pred"]
        conf = proba.max(axis=1)
        correct = pred == y
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
            if m.sum() < 20:
                continue
            xs.append(float(conf[m].mean()))
            ys.append(float(correct[m].mean()))
            rows.append({"feature_set": key, "confidence_mean": xs[-1], "accuracy": ys[-1], "n": int(m.sum())})
        ax.plot(xs, ys, marker="o", linewidth=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", label="perfect calibration")
    ax.set_xlabel("mean predicted confidence")
    ax.set_ylabel("empirical accuracy")
    ax.set_title("Condition-only D classifier calibration")
    ax.legend(frameon=False)
    style(ax)
    pd.DataFrame(rows).to_csv(CSV / "classifier_calibration_bins.csv", index=False, encoding="utf-8-sig")
    return savefig("fig16_condition_classifier_calibration.png")


def figure_17_hard_negatives(df, X_all, hard_df) -> str:
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    xy = pca.fit_transform(X_all)
    sample = df.sample(min(MAX_SCATTER_POINTS, len(df)), random_state=RANDOM_SEED)
    idx_map = {int(s): i for i, s in enumerate(df["sample_index"].astype(int).to_numpy())}
    hard_indices = [idx_map[int(s)] for s in hard_df["sample_index"].head(30) if int(s) in idx_map]
    fig, ax = plt.subplots(figsize=(8.4, 6.8))
    ax.scatter(xy[sample.index, 0], xy[sample.index, 1], s=6, alpha=0.18, color="#94a3b8", label="sample")
    if hard_indices:
        ax.scatter(xy[hard_indices, 0], xy[hard_indices, 1], s=52, color="#d62728", edgecolor="white", label="confident wrong D prediction")
    ax.set_xlabel("condition PC1")
    ax.set_ylabel("condition PC2")
    ax.set_title("Hard negatives in condition space")
    ax.legend(frameon=False)
    style(ax)
    return savefig("fig17_hard_negative_condition_space.png")


def figure_18_counterexample_panels(df, pairs_df, z_dynamic, profile_cols) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.2))
    axes = axes.ravel()
    top = pairs_df.head(4)
    idx_by_sample = {int(s): i for i, s in enumerate(df["sample_index"].astype(int).to_numpy())}
    for ax, (_, row) in zip(axes, top.iterrows()):
        ia = idx_by_sample.get(int(row["sample_a"]))
        ib = idx_by_sample.get(int(row["sample_b"]))
        if ia is None or ib is None:
            continue
        xs = np.arange(len(profile_cols))
        ax.plot(xs, z_dynamic[ia], marker="o", linewidth=1.6, label=f"{row['D_a']} sample {int(row['sample_a'])}")
        ax.plot(xs, z_dynamic[ib], marker="o", linewidth=1.6, label=f"{row['D_b']} sample {int(row['sample_b'])}")
        ax.set_title(f"condition close, dynamic far\ncond={row['condition_distance']:.2f}, dyn={row['dynamic_distance']:.2f}")
        ax.set_xticks(xs)
        ax.set_xticklabels(profile_cols, rotation=75, ha="right", fontsize=6)
        ax.legend(fontsize=7)
        style(ax)
    return savefig("fig18_representative_counterexample_profiles.png")


def figure_19_redundancy_network(values: pd.DataFrame) -> str:
    X = values.to_numpy(float)
    corr = pd.DataFrame(values).corr(method="spearman").to_numpy()
    cov = LedoitWolf().fit(StandardScaler().fit_transform(X)).covariance_
    prec = np.linalg.pinv(cov)
    d = np.sqrt(np.diag(prec))
    partial = -prec / np.outer(d, d)
    np.fill_diagonal(partial, 1)
    fig, axes = plt.subplots(1, 2, figsize=(14.8, 5.8))
    fig.subplots_adjust(right=0.88, wspace=0.34)
    for ax, mat, title in [(axes[0], corr, "Spearman feature redundancy"), (axes[1], partial, "Partial correlation network proxy")]:
        im = ax.imshow(mat, cmap="coolwarm", norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
        ax.set_title(title)
        ax.set_xticks(range(len(values.columns)), values.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(values.columns)), values.columns)
    cax = fig.add_axes([0.91, 0.18, 0.018, 0.64])
    fig.colorbar(im, cax=cax, label="correlation")
    fig._skip_tight_layout = True
    return savefig("fig19_feature_redundancy_partial_correlation.png")


def figure_20_condition_coverage_map(df, X_all, d_order) -> str:
    rng = np.random.default_rng(RANDOM_SEED)
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    xy = pca.fit_transform(X_all)
    n = len(df)
    keep = rng.choice(n, size=min(MAX_SCATTER_POINTS, n), replace=False)
    labels = df["dynamic_family_id"].to_numpy()
    cmap = plt.cm.get_cmap("tab20", len(d_order))
    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    for i, d in enumerate(d_order):
        m = keep[labels[keep] == d]
        ax.scatter(xy[m, 0], xy[m, 1], s=5, alpha=0.23, color=cmap(i), label=d)
        allm = labels == d
        cent = np.nanmedian(xy[allm], axis=0)
        ax.scatter(cent[0], cent[1], s=105, color=cmap(i), edgecolor="white", linewidth=1.2)
        ax.text(cent[0], cent[1], d, fontsize=8, ha="center", va="center", color="white", weight="bold")
    ax.set_xlabel("condition PCA 1")
    ax.set_ylabel("condition PCA 2")
    ax.set_title("Condition-space coverage map by D")
    ax.legend(fontsize=7, ncol=2, frameon=False, loc="best")
    style(ax)
    return savefig("fig20_condition_space_coverage_map.png")


def write_html(manifest: list[dict], meta: dict) -> None:
    sections = []
    for item in manifest:
        sections.append(
            f"""
    <section>
      <h2>{item['id']}. {item['title']}</h2>
      <p>{item['shows']}</p>
      <p class=\"metric\"><b>척도:</b> {item['metric']}</p>
      <p class=\"claim\"><b>claim 관련성:</b> {item['claim']}</p>
      <img src=\"63_whole_d_condition_group_consistency_assets/{item['file']}\" alt=\"{item['title']}\">
    </section>
"""
        )
    html = f"""<!doctype html>
<html lang=\"ko\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Whole-D Condition Group Consistency Dashboard</title>
  <style>
    body {{ margin:0; font-family:system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; background:#f5f7fb; color:#172033; }}
    header {{ background:#fff; border-bottom:1px solid #dce3ee; padding:28px 34px 18px; }}
    main {{ max-width:1320px; margin:0 auto; padding:24px 26px 46px; }}
    h1 {{ margin:0 0 8px; font-size:28px; letter-spacing:0; }}
    h2 {{ margin:0 0 10px; font-size:20px; letter-spacing:0; }}
    p {{ margin:0 0 8px; line-height:1.55; }}
    .meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(220px,1fr)); gap:10px; margin-top:16px; font-size:14px; }}
    .meta div {{ background:#eef2f8; border:1px solid #dbe3ee; padding:10px 12px; border-radius:6px; }}
    section {{ background:#fff; border:1px solid #dfe5ef; border-radius:8px; padding:18px; margin-bottom:20px; }}
    img {{ width:100%; height:auto; display:block; border:1px solid #e3e7ef; border-radius:6px; background:#fff; }}
    .note {{ margin-top:12px; background:#fff8e6; border:1px solid #f1d18b; border-radius:6px; padding:12px 14px; }}
    .metric {{ color:#334155; }}
    .claim {{ color:#31415f; }}
    code {{ font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }}
  </style>
</head>
<body>
<header>
  <h1>Whole-D condition group consistency dashboard</h1>
  <p>60번 eta50 >= 0.8 whole-D subset을 기준으로, condition 값만으로 dynamic group을 대표하거나 구분할 수 있는지 점검하는 추가 시각화입니다.</p>
  <div class=\"meta\">
    <div><b>population</b><br>eta50 >= 0.8 whole-D subset</div>
    <div><b>rows</b><br>{meta['n_rows']}</div>
    <div><b>D families</b><br>{meta['n_D']}</div>
    <div><b>abstract descriptors</b><br><code>cl1/IPR/purity</code></div>
    <div><b>condition/readout descriptors</b><br>{len(ALL_DESCRIPTORS)}</div>
    <div><b>source script</b><br><code>new/build_whole_d_dynamic_condition_explainability_eta50_ge080.py</code></div>
  </div>
  <div class=\"note\">주의: 이 자료는 mechanism proof가 아니라 condition-only sufficiency와 group separability의 한계를 점검하는 진단 자료입니다. Outcome-proximal readout이 잘 맞는 경우와 추상 condition이 약한 경우를 구분해서 읽어야 합니다.</div>
</header>
<main>
{''.join(sections)}
</main>
</body>
</html>
"""
    HTML_OUT.write_text(html, encoding="utf-8")
    (OUT / HTML_OUT.name).write_text(html, encoding="utf-8")


def write_report(manifest: list[dict], metrics: dict) -> None:
    lines = [
        "# Whole-D condition group consistency figure guide",
        "",
        "## 목적",
        "",
        "이 dashboard는 두 질문을 직접 확인하기 위해 만들었다.",
        "",
        "1. 같은 dynamic group 안에서 condition들이 일정한가, 아니면 spread가 큰가?",
        "2. 다른 dynamic group들이 condition만으로 잘 구분되는가, 아니면 condition space에서 서로 겹치는가?",
        "",
        "기준 claim은 `condition 값 하나 또는 단순 condition set만으로 dynamic diversity를 직접적이고 안정적으로 만들기는 어렵다`이다.",
        "",
        "## 핵심 수치",
        "",
        f"- Abstract condition-only D classifier balanced accuracy: {metrics['abstract_balanced_accuracy']:.3f}",
        f"- All-condition D classifier balanced accuracy: {metrics['all_balanced_accuracy']:.3f}",
        f"- Median local D purity, abstract conditions: {metrics['abstract_local_purity_median']:.3f}",
        f"- Median local D purity, all condition descriptors: {metrics['all_local_purity_median']:.3f}",
        f"- Median D entropy inside condition bins: {metrics['condition_bin_entropy_median']:.3f}",
        f"- Weighted mean CV R2 for dynamic PCs, abstract conditions: {metrics['abstract_dynamic_pc_r2']:.3f}",
        f"- Weighted mean CV R2 for dynamic PCs, all conditions: {metrics['all_dynamic_pc_r2']:.3f}",
        f"- Weighted mean CV R2 for dynamic PCs, all conditions + D label: {metrics['all_plus_D_dynamic_pc_r2']:.3f}",
        "",
        "## Figure별 설명",
        "",
    ]
    for item in manifest:
        lines.extend(
            [
                f"### {item['id']}. {item['title']}",
                "",
                f"- 무엇을 보이나: {item['shows']}",
                f"- 척도: {item['metric']}",
                f"- claim 관련성: {item['claim']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 해석 caveat",
            "",
            "- 이 분석은 D label이 자연 mechanism임을 증명하지 않는다. D는 whole-D dynamic family scaffold다.",
            "- Condition-only classifier가 어느 정도 맞더라도, 그것은 condition이 dynamic diversity를 충분히 제어한다는 뜻이 아니다. 어떤 D가 섞이는지와 residual dynamic variation을 함께 봐야 한다.",
            "- Outcome-proximal readout(`early_trap_10`, `residual_10`)은 eta/dynamics를 잘 읽을 수 있지만, 독립적인 controllable condition knob이라고 단정하면 안 된다.",
            "- PCA, entropy binning, bimodality coefficient는 탐색적 진단이다. 강한 통계 claim은 핵심 figure인 classifier, local purity, residualized diversity, condition-bin spread와 함께 봐야 한다.",
        ]
    )
    (REPORT / "figure_description_and_claim_relevance.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df, z_dynamic, profile_cols, cond_df, X_all, X_abs, d_order = load_data()
    df = df.reset_index(drop=True)
    cond_df = cond_df.reset_index(drop=True)
    # z_dynamic is in the same order as df returned by BASE.load_dataset.
    labels = df["dynamic_family_id"].to_numpy()

    med = group_medians(df, cond_df, d_order)
    iqr = group_iqrs(df, cond_df, d_order)
    var_df = variance_decomposition(df, cond_df, d_order)
    overlap = histogram_overlap_matrix(df, cond_df, d_order)
    clf = train_condition_classifiers(df, X_abs, X_all)
    purity = pd.concat(
        [
            local_purity(df, X_abs, "abstract_conditions"),
            local_purity(df, X_all, "all_condition_descriptors"),
        ],
        ignore_index=True,
    )
    entropy_bins = entropy_over_condition_bins(df, cond_df, d_order)
    multimodal = dynamic_bin_multimodality(df, cond_df, z_dynamic)
    centroid_dist = centroid_distance_table(df, X_all, z_dynamic, d_order)
    pc_scores, residual_summary = residualized_dynamic_prediction(df, X_abs, X_all, z_dynamic)
    pair_df = sample_pair_distances(df, X_all, z_dynamic)
    hard, counter_pairs = hard_negative_cases(df, X_all, z_dynamic, clf)

    # Save CSV outputs.
    med.to_csv(CSV / "condition_median_by_D_scaled.csv", index=False, encoding="utf-8-sig")
    iqr.to_csv(CSV / "condition_iqr_by_D_scaled.csv", index=False, encoding="utf-8-sig")
    var_df.to_csv(CSV / "condition_variance_decomposition.csv", index=False, encoding="utf-8-sig")
    overlap.to_csv(CSV / "condition_distribution_overlap_matrix.csv", encoding="utf-8-sig")
    purity.to_csv(CSV / "local_purity_condition_space.csv", index=False, encoding="utf-8-sig")
    entropy_bins.to_csv(CSV / "D_entropy_over_condition_bins.csv", index=False, encoding="utf-8-sig")
    multimodal.to_csv(CSV / "condition_bin_multimodality_proxy.csv", index=False, encoding="utf-8-sig")
    centroid_dist.to_csv(CSV / "condition_vs_dynamic_centroid_distance.csv", index=False, encoding="utf-8-sig")
    pc_scores.to_csv(CSV / "residualized_dynamic_pc_prediction.csv", index=False, encoding="utf-8-sig")
    residual_summary.to_csv(CSV / "residualized_dynamic_prediction_summary.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(CSV / "sampled_condition_dynamic_pair_distances.csv", index=False, encoding="utf-8-sig")
    hard.to_csv(CSV / "hard_negative_condition_classifier_cases.csv", index=False, encoding="utf-8-sig")
    counter_pairs.to_csv(CSV / "representative_condition_dynamic_counterexample_pairs.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "feature_set": k,
                "balanced_accuracy": v["balanced_accuracy"],
            }
            for k, v in clf.items()
        ]
    ).to_csv(CSV / "condition_only_classifier_scores.csv", index=False, encoding="utf-8-sig")

    manifest = []
    def add(id_, title, file, shows, metric, claim):
        manifest.append({"id": id_, "title": title, "file": file, "shows": shows, "metric": metric, "claim": claim})

    add("Figure 1", "Condition distribution by D", figure_01_boxplots(df, d_order), "각 whole-D dynamic group 내부에서 condition 값 분포가 좁은지 넓은지 보여 준다.", "descriptor raw value의 boxplot, outlier 숨김", "같은 D 안에서 spread가 크면 condition 값 하나로 D group을 대표하기 어렵다는 근거가 된다.")
    add("Figure 2", "D x condition median heatmap", draw_heatmap(med.set_index("D_family")[ALL_DESCRIPTORS], "D-level condition median, robust-scaled", "median z", "fig02_D_condition_median_heatmap.png", cmap="coolwarm", vmin=-2.5, vmax=2.5, annotate=True), "D별 condition 중심값이 서로 다른지 본다.", "전체 high-eta subset에서 robust-scaled condition median", "group별 condition signature가 있는지 확인한다. 단 median만으로는 내부 spread를 알 수 없다.")
    add("Figure 3", "D x condition spread heatmap", draw_heatmap(iqr.set_index("D_family")[ALL_DESCRIPTORS], "Within-D condition spread, robust-scaled IQR", "IQR in robust-scaled units", "fig03_D_condition_spread_heatmap.png", cmap="magma", vmin=0, vmax=max(2.0, iqr[ALL_DESCRIPTORS].to_numpy().max()), annotate=True), "같은 D 내부의 condition spread를 보여 준다.", "robust-scaled condition IQR", "median signature가 있어도 spread가 크면 condition-only label이 불안정하다는 근거가 된다.")
    add("Figure 4", "Variance decomposition and separability", figure_04_variance(var_df), "condition variance가 D 사이 차이에서 오는지 D 내부 변동에서 오는지 분해한다.", "between-D variance fraction, D median range, median within-D IQR", "between fraction이 낮거나 within-D IQR이 크면 condition만으로 D를 대표하기 어렵다.")
    add("Figure 5", "Condition distribution overlap matrix", draw_heatmap(overlap, "Average 1D condition distribution overlap by D pair", "average histogram overlap (0 separated, 1 overlapping)", "fig05_condition_distribution_overlap_matrix.png", cmap="viridis", vmin=0, vmax=1, annotate=True), "서로 다른 D pair가 condition 분포상 얼마나 겹치는지 본다.", "descriptor별 histogram overlap의 평균", "서로 다른 dynamic group이 condition space에서 많이 겹치면 condition-only 학습은 group을 혼동하기 쉽다.")
    add("Figure 6", "Condition-only D classifier confusion", figure_06_classifier_confusion(d_order, clf), "condition만 입력했을 때 D group을 얼마나 맞히고 어디서 섞이는지 보여 준다.", "row-normalized confusion matrix, balanced accuracy", "condition-only로 D를 완전히 구분하지 못하거나 특정 D가 섞이면 dynamic diversity 표현에 한계가 있음을 보여 준다.")
    add("Figure 7", "Condition classifier error network", figure_07_error_network(d_order, clf), "condition-only classifier가 자주 혼동하는 D pair를 네트워크로 보여 준다.", "symmetric row-normalized confusion edge weight", "어떤 dynamic groups가 condition 기준으로 특히 애매한지 직관적으로 보여 준다.")
    add("Figure 8", "Dynamic distance vs condition distance", figure_08_distance_scatter(pair_df), "sample pair 수준에서 condition 거리와 dynamic 거리의 관계를 본다.", "robust-scaled condition Euclidean distance, dynamic-profile Euclidean distance", "condition이 비슷해도 dynamics가 다르거나, condition-distance가 dynamic-distance를 충분히 설명하지 못하면 condition-only claim이 약해진다.")
    add("Figure 9", "Condition vs dynamic centroid dendrogram", figure_09_dendrogram(df, X_all, z_dynamic, d_order), "D centroid를 condition space와 dynamic space에서 각각 clustering해 비교한다.", "Ward linkage on D centroids", "condition 기준 group 구조가 dynamic 기준 group 구조와 다르면 condition이 dynamic group structure를 대체하기 어렵다.")
    add("Figure 10", "Local D purity in condition space", figure_10_local_purity(purity, d_order), "condition space에서 각 sample의 nearest neighbors가 같은 D인지 본다.", f"k={KNN_K} nearest-neighbor same-D fraction", "local purity가 낮으면 condition space에서 서로 다른 dynamic groups가 섞인다는 직접 근거다.")
    add("Figure 11", "D entropy over condition bins", figure_11_entropy(entropy_bins), "condition quantile bin 안에 D labels가 얼마나 섞이는지 본다.", "normalized entropy, 0=single D, 1=all D mixed", "같은 condition 구간에 여러 D가 섞이면 condition-only 학습으로 dynamic group을 구분하기 어렵다.")
    add("Figure 12", "Condition-bin multimodality proxy", figure_12_multimodality(multimodal), "같은 condition bin 안에서 dynamic PC1 또는 eta가 단봉적/다봉적 신호를 보이는지 탐색한다.", "dynamic PC1 bimodality coefficient, eta50 IQR", "condition이 같아도 내부 dynamic outcome이 여러 형태일 수 있는지 보는 탐색적 보조 figure다.")
    add("Figure 13", "Condition centroid distance vs dynamic centroid distance", figure_13_centroid_distance(centroid_dist), "D pair별 condition centroid distance와 dynamic centroid distance를 비교한다.", "D-pair centroid distance in condition space and dynamic space", "dynamic으로 먼 group이 condition으로는 가까우면 condition-only로 중요한 dynamic contrast를 놓칠 수 있다.")
    add("Figure 14", "Residualized dynamic diversity", figure_14_residual_prediction(pc_scores, residual_summary), "condition feature로 dynamic PC를 예측한 뒤에도 residual dynamic variation이 얼마나 남는지 본다.", "cross-validated R2, residual std/original std", "여러 condition을 함께 써도 dynamic profile을 충분히 설명하지 못하면 condition-only sufficiency가 약해진다.")
    add("Figure 15", "Additional dynamic information from D label", figure_15_condition_plus_D(pc_scores), "condition에 D label을 추가했을 때 dynamic PC 예측력이 얼마나 증가하는지 본다.", "CV R2 gain by adding D one-hot to all conditions", "D가 조건 이후에도 추가 설명력을 갖는다면 condition이 D-level dynamic context를 완전히 대체하지 못한다는 뜻이다.")
    add("Figure 16", "Condition-only classifier calibration", figure_16_calibration(df, clf), "condition-only D classifier의 confidence가 실제 정확도와 맞는지 본다.", "predicted confidence vs empirical accuracy", "높은 confidence에서도 틀리는 구간이 있으면 condition-only group assignment를 신뢰하기 어렵다.")
    add("Figure 17", "Hard negatives in condition space", figure_17_hard_negatives(df, X_all, hard), "condition-only classifier가 높은 확신으로 틀린 sample들이 condition PCA 공간 어디에 있는지 보여 준다.", "condition PCA, confident wrong predictions highlighted", "condition space에서 그럴듯해 보여도 dynamic group이 다른 hard negative가 있음을 직관적으로 보여 준다.")
    add("Figure 18", "Representative counterexample profiles", figure_18_counterexample_panels(df, counter_pairs, z_dynamic, profile_cols), "condition은 가까운데 dynamic profile은 먼 sample pair를 직접 비교한다.", "condition distance, dynamic profile distance, robust-scaled dynamic feature profile", "claim을 발표용으로 직관화한다. 같은 condition 근처에도 다른 dynamic profile이 남을 수 있음을 보여 준다.")
    add("Figure 19", "Feature redundancy and partial correlation", figure_19_redundancy_network(cond_df), "condition features끼리의 redundancy와 partial relation을 본다.", "Spearman correlation, LedoitWolf partial correlation proxy", "condition features가 독립 knob처럼 움직이지 않을 수 있음을 보여 준다.")
    add("Figure 20", "Condition-space coverage map", figure_20_condition_coverage_map(df, X_all, d_order), "condition PCA 공간에서 D groups가 차지하는 영역과 겹침을 보여 준다.", "PCA of condition descriptors, D-colored sample cloud and centroids", "condition space에서 group coverage가 겹치는지 직관적으로 확인하는 overview다.")

    pd.DataFrame(manifest).to_csv(CSV / "figure_manifest.csv", index=False, encoding="utf-8-sig")

    residual_lookup = residual_summary.set_index("feature_set")["weighted_mean_cv_r2"].to_dict()
    metrics = {
        "n_rows": int(len(df)),
        "n_D": int(len(d_order)),
        "abstract_balanced_accuracy": clf["abstract"]["balanced_accuracy"],
        "all_balanced_accuracy": clf["all_condition"]["balanced_accuracy"],
        "abstract_local_purity_median": float(purity.loc[purity["feature_set"].eq("abstract_conditions"), "local_D_purity"].median()),
        "all_local_purity_median": float(purity.loc[purity["feature_set"].eq("all_condition_descriptors"), "local_D_purity"].median()),
        "condition_bin_entropy_median": float(entropy_bins["normalized_D_entropy"].median()),
        "abstract_dynamic_pc_r2": float(residual_lookup.get("abstract_conditions", np.nan)),
        "all_dynamic_pc_r2": float(residual_lookup.get("all_conditions", np.nan)),
        "all_plus_D_dynamic_pc_r2": float(residual_lookup.get("all_conditions_plus_D", np.nan)),
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(manifest, metrics)
    write_report(manifest, metrics)
    (OUT / "run_command.md").write_text(
        "`C:\\\\Users\\\\User\\\\anaconda3\\\\envs\\\\py311-cu132\\\\python.exe new\\\\build_whole_d_condition_group_consistency_dashboard.py`\n",
        encoding="utf-8",
    )
    print(json.dumps({"html": str(HTML_OUT), "out": str(OUT), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
