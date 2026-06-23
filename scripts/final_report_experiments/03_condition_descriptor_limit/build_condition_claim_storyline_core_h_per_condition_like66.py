from __future__ import annotations

import json
import math
import os
import shutil
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "8")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "new" / "whole_d_condition_claim_storyline_h_per_condition_like66_20260620"
CSV = OUT / "csv"
FIG = OUT / "figures"
REPORT = OUT / "reports"
HTMLS = ROOT / "htmls"
HTML_ASSETS = HTMLS / "68_condition_claim_storyline_core_h_per_condition_like66_assets"
HTML_OUT = HTMLS / "68_condition_claim_storyline_core_h_per_condition_like66_dashboard.html"

PILOT_SUMMARY = ROOT / "outputs" / "pilot_sampling" / "pilot62000_t50_schema_v2_20260603_merged" / "pilot_summary.csv"
D_ASSIGN = ROOT / "outputs" / "scalable_mode_clustering_20260604" / "csv" / "dynamic_family_assignments.csv"

RANDOM_SEED = 20260620
ETA_FILTER = 0.8
MAX_PAIRS_PER_D_FEATURE = 500

ABSTRACT = {
    "cl1_10_20": "cl1_mean_10p0_20p0",
    "IPR_10_20": "ipr_mean_10p0_20p0",
    "purity_10_20": "purity_mean_10p0_20p0",
}

H_STRUCTURAL = {
    "diag_std": "diag_std",
    "diag_range": "diag_range",
    "offdiag_mean_abs": "offdiag_mean_abs",
    "offdiag_max_abs": "offdiag_max_abs",
    "offdiag_fro_norm": "offdiag_fro_norm",
    "spectral_width": "spectral_width",
    "min_eigen_gap": "min_eigen_gap",
    "max_eigen_gap": "max_eigen_gap",
    "site1_to_others_abs_sum": "site1_to_others_abs_sum",
    "site2_to_sink34_abs_sum": "site2_to_sink34_abs_sum",
    "site1_to_sink34_abs_sum": "site1_to_sink34_abs_sum",
    "detour_coupling_abs_sum": "detour_coupling_abs_sum",
    "direct_shortcut_to_sink_abs_sum": "direct_shortcut_to_sink_abs_sum",
}

EARLY_READOUT = {
    "early_trap_10": "trap10p0",
    "source_site1_10": "site110p0",
    "residual_10": "residual10p0",
}

DESCRIPTORS = {**ABSTRACT, **H_STRUCTURAL, **EARLY_READOUT}
FEATURE_KIND = {
    **{k: "abstract" for k in ABSTRACT},
    **{k: "H_structural" for k in H_STRUCTURAL},
    **{k: "early_readout" for k in EARLY_READOUT},
}

PROFILE_FEATURES = [
    "eta10p0",
    "eta50p0",
    "site110p0",
    "site120p0",
    "site150p0",
    "site210p0",
    "site220p0",
    "sink3410p0",
    "sink3420p0",
    "detour56710p0",
    "detour56720p0",
    "trap10p0",
    "trap20p0",
    "trap50p0",
    "loss20p0",
    "residual20p0",
    "cl1_mean_10p0_20p0",
    "purity_mean_10p0_20p0",
    "ipr_mean_10p0_20p0",
]


def ensure_dirs() -> None:
    for path in [OUT, CSV, FIG, REPORT, HTMLS, HTML_ASSETS]:
        path.mkdir(parents=True, exist_ok=True)


def descriptor_order() -> list[str]:
    return list(ABSTRACT) + list(H_STRUCTURAL) + list(EARLY_READOUT)


def load_data() -> pd.DataFrame:
    usecols = sorted(
        set(
            [
                "sample_index",
                "solver_success",
                "eta50p0",
                "trap20p0",
                "residual20p0",
                *DESCRIPTORS.values(),
                *PROFILE_FEATURES,
            ]
        )
    )
    pilot = pd.read_csv(PILOT_SUMMARY, usecols=lambda c: c in usecols)
    assign = pd.read_csv(D_ASSIGN, usecols=["sample_index", "dynamic_family_id", "structural_mode_id"])
    df = pilot.merge(assign, on="sample_index", how="left", validate="one_to_one")
    for col in df.columns:
        if col not in {"solver_success", "dynamic_family_id", "structural_mode_id"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["solver_success"].astype(bool)].copy()
    df = df[df["eta50p0"] >= ETA_FILTER].copy()
    df = df.dropna(subset=["dynamic_family_id"]).copy()
    return df.reset_index(drop=True)


def dynamic_profile(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in PROFILE_FEATURES if c in df.columns]
    values = df[cols].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True))
    return RobustScaler().fit_transform(values), cols


def scaled_descriptor_frame(df: pd.DataFrame) -> pd.DataFrame:
    cols = [DESCRIPTORS[label] for label in descriptor_order()]
    values = df[cols].replace([np.inf, -np.inf], np.nan)
    values = values.fillna(values.median(numeric_only=True))
    scaled = RobustScaler().fit_transform(values)
    return pd.DataFrame(scaled, columns=descriptor_order(), index=df.index)


def savefig(name: str) -> str:
    path = FIG / name
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
    asset = HTML_ASSETS / name
    shutil.copy2(path, asset)
    return f"{HTML_ASSETS.name}/{name}"


def style(ax) -> None:
    ax.grid(True, color="#e6e6e6", linewidth=0.7)
    ax.set_axisbelow(True)


def profile_spread(z_slice: np.ndarray) -> float:
    if z_slice.size == 0:
        return np.nan
    return float(np.nanmean(np.nanstd(z_slice, axis=0)))


def single_condition_predictive_r2(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [("ALL", df)] + list(df.groupby("dynamic_family_id", sort=True))
    targets = {"eta50": "eta50p0", "trap20": "trap20p0", "residual20": "residual20p0"}
    for d, g in families:
        if len(g) < 200:
            continue
        for label, col in DESCRIPTORS.items():
            x = g[[col]].to_numpy(float)
            for target, tcol in targets.items():
                y = g[tcol].to_numpy(float)
                mask = np.isfinite(x[:, 0]) & np.isfinite(y)
                if mask.sum() < 200:
                    continue
                x1, y1 = x[mask], y[mask]
                kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
                r2s = []
                for tr, te in kf.split(x1):
                    model = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=50, random_state=RANDOM_SEED)
                    model.fit(x1[tr], y1[tr])
                    r2s.append(r2_score(y1[te], model.predict(x1[te])))
                rows.append(
                    {
                        "D_family": str(d),
                        "descriptor": label,
                        "feature_kind": FEATURE_KIND[label],
                        "target": target,
                        "n": int(mask.sum()),
                        "cv_r2_mean": float(np.mean(r2s)),
                        "cv_r2_std": float(np.std(r2s)),
                    }
                )
    return pd.DataFrame(rows)


def condition_spread_by_d(df: pd.DataFrame) -> pd.DataFrame:
    values = scaled_descriptor_frame(df)
    rows = []
    for d, g in df.groupby("dynamic_family_id", sort=True):
        idx = g.index
        row = {"D_family": d, "n": int(len(g))}
        for desc in descriptor_order():
            x = values.loc[idx, desc].to_numpy(float)
            row[desc] = float(np.nanquantile(x, 0.75) - np.nanquantile(x, 0.25))
        rows.append(row)
    return pd.DataFrame(rows)


def condition_median_by_d(df: pd.DataFrame) -> pd.DataFrame:
    values = scaled_descriptor_frame(df)
    rows = []
    for d, g in df.groupby("dynamic_family_id", sort=True):
        idx = g.index
        row = {"D_family": d, "n": int(len(g))}
        for desc in descriptor_order():
            row[desc] = float(np.nanmedian(values.loc[idx, desc].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows)


def condition_between_d_variance(df: pd.DataFrame) -> pd.DataFrame:
    values = scaled_descriptor_frame(df)
    labels = df["dynamic_family_id"].astype(str)
    n_total = len(df)
    rows = []
    for desc in descriptor_order():
        x = values[desc].to_numpy(float)
        total_var = float(np.nanvar(x))
        global_mean = float(np.nanmean(x))
        weighted_within = 0.0
        weighted_between = 0.0
        median_iqrs = []
        for d, g in df.groupby("dynamic_family_id", sort=True):
            idx = labels.eq(str(d)).to_numpy()
            xd = values.loc[idx, desc].to_numpy(float)
            w = len(xd) / n_total
            weighted_within += w * float(np.nanvar(xd))
            weighted_between += w * (float(np.nanmean(xd)) - global_mean) ** 2
            median_iqrs.append(float(np.nanquantile(xd, 0.75) - np.nanquantile(xd, 0.25)))
        rows.append(
            {
                "descriptor": desc,
                "feature_kind": FEATURE_KIND[desc],
                "total_variance": total_var,
                "weighted_within_D_variance": weighted_within,
                "weighted_between_D_variance": weighted_between,
                "between_D_variance_fraction": weighted_between / total_var if total_var > 0 else np.nan,
                "median_within_D_iqr": float(np.nanmedian(median_iqrs)),
            }
        )
    return pd.DataFrame(rows)


def condition_bin_diversity(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    rows = []
    z_df = pd.DataFrame(z, index=df.index)
    for d, g in df.groupby("dynamic_family_id", sort=True):
        if len(g) < 100:
            continue
        for label, col in DESCRIPTORS.items():
            try:
                bins = pd.qcut(g[col], q=5, labels=False, duplicates="drop")
            except ValueError:
                continue
            for b in sorted(pd.Series(bins).dropna().unique()):
                idx = g.index[bins == b]
                gg = g.loc[idx]
                rows.append(
                    {
                        "D_family": d,
                        "descriptor": label,
                        "feature_kind": FEATURE_KIND[label],
                        "condition_bin": int(b),
                        "n": int(len(gg)),
                        "eta50_iqr": float(gg["eta50p0"].quantile(0.75) - gg["eta50p0"].quantile(0.25)),
                        "dynamic_profile_spread": profile_spread(z_df.loc[idx].to_numpy()),
                    }
                )
    return pd.DataFrame(rows)


def matched_condition_pairs(df: pd.DataFrame, z: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    z_df = pd.DataFrame(z, index=df.index)
    rows = []
    for d, g in df.groupby("dynamic_family_id", sort=True):
        if len(g) < 50:
            continue
        for label, col in DESCRIPTORS.items():
            gg = g[[col, "eta50p0"]].dropna().sort_values(col)
            if len(gg) < 50:
                continue
            vals = gg[col].to_numpy(float)
            diffs = np.diff(vals)
            thresh = np.nanquantile(diffs, 0.25)
            positions = np.where(diffs <= thresh)[0]
            if len(positions) > MAX_PAIRS_PER_D_FEATURE:
                positions = rng.choice(positions, size=MAX_PAIRS_PER_D_FEATURE, replace=False)
            for pos in positions:
                i, j = gg.index[pos], gg.index[pos + 1]
                zi = z_df.loc[i].to_numpy(float)
                zj = z_df.loc[j].to_numpy(float)
                rows.append(
                    {
                        "D_family": d,
                        "descriptor": label,
                        "feature_kind": FEATURE_KIND[label],
                        "condition_distance": abs(float(df.loc[i, col]) - float(df.loc[j, col])),
                        "eta50_distance": abs(float(df.loc[i, "eta50p0"]) - float(df.loc[j, "eta50p0"])),
                        "dynamic_distance": float(np.linalg.norm(zi - zj) / math.sqrt(len(zi))),
                    }
                )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    return (
        pairs.groupby(["D_family", "descriptor", "feature_kind"], as_index=False)
        .agg(
            n_matched_pairs=("dynamic_distance", "size"),
            eta50_distance_median=("eta50_distance", "median"),
            eta50_distance_q90=("eta50_distance", lambda x: x.quantile(0.9)),
            dynamic_distance_median=("dynamic_distance", "median"),
            dynamic_distance_q90=("dynamic_distance", lambda x: x.quantile(0.9)),
        )
    )


def condition_bin_d_entropy(df: pd.DataFrame) -> pd.DataFrame:
    d_order = sorted(df["dynamic_family_id"].astype(str).unique())
    rows = []
    for label, col in DESCRIPTORS.items():
        try:
            bins = pd.qcut(df[col], q=5, labels=False, duplicates="drop")
        except ValueError:
            continue
        for b in sorted(pd.Series(bins).dropna().unique()):
            mask = bins == b
            counts = df.loc[mask, "dynamic_family_id"].astype(str).value_counts().reindex(d_order, fill_value=0)
            probs = counts.to_numpy(float)
            probs = probs / probs.sum() if probs.sum() else probs
            ent = -float(np.sum([p * math.log(p) for p in probs if p > 0]))
            rows.append(
                {
                    "descriptor": label,
                    "feature_kind": FEATURE_KIND[label],
                    "condition_bin": int(b),
                    "n": int(mask.sum()),
                    "normalized_D_entropy": ent / math.log(len(d_order)) if len(d_order) > 1 else np.nan,
                    "n_D_present": int((counts > 0).sum()),
                }
            )
    return pd.DataFrame(rows)


def single_condition_d_classifier(df: pd.DataFrame) -> pd.DataFrame:
    y = df["dynamic_family_id"].astype(str).to_numpy()
    rows = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    for label, col in DESCRIPTORS.items():
        x = df[[col]].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).to_numpy(float)
        preds = np.empty(len(df), dtype=object)
        for tr, te in cv.split(x, y):
            model = DecisionTreeClassifier(max_leaf_nodes=12, min_samples_leaf=50, class_weight="balanced", random_state=RANDOM_SEED)
            model.fit(x[tr], y[tr])
            preds[te] = model.predict(x[te])
        rows.append(
            {
                "descriptor": label,
                "feature_kind": FEATURE_KIND[label],
                "balanced_accuracy": float(balanced_accuracy_score(y, preds)),
            }
        )
    return pd.DataFrame(rows)


def single_condition_dynamic_pc_prediction(df: pd.DataFrame, z: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    pca = PCA(n_components=8, random_state=RANDOM_SEED)
    y_pc = pca.fit_transform(z)
    weights = pca.explained_variance_ratio_
    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    summary = []
    for label, col in DESCRIPTORS.items():
        x = df[[col]].replace([np.inf, -np.inf], np.nan)
        x = x.fillna(x.median(numeric_only=True)).to_numpy(float)
        local = []
        for k in range(y_pc.shape[1]):
            pred = np.full(len(df), np.nan)
            for tr, te in cv.split(x):
                model = DecisionTreeRegressor(max_leaf_nodes=8, min_samples_leaf=50, random_state=RANDOM_SEED)
                model.fit(x[tr], y_pc[tr, k])
                pred[te] = model.predict(x[te])
            r2 = float(r2_score(y_pc[:, k], pred))
            resid = float(np.nanstd(y_pc[:, k] - pred) / np.nanstd(y_pc[:, k]))
            row = {
                "descriptor": label,
                "feature_kind": FEATURE_KIND[label],
                "dynamic_pc": f"PC{k + 1}",
                "explained_variance_ratio": float(weights[k]),
                "cv_r2": r2,
                "residual_std_ratio": resid,
            }
            rows.append(row)
            local.append(row)
        summary.append(
            {
                "descriptor": label,
                "feature_kind": FEATURE_KIND[label],
                "weighted_mean_cv_r2": float(np.average([r["cv_r2"] for r in local], weights=weights)),
                "mean_residual_std_ratio": float(np.nanmean([r["residual_std_ratio"] for r in local])),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)


def one_condition_eta_residual_spread(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("dynamic_family_id", sort=True):
        eta = g["eta50p0"].to_numpy(float)
        overall_std = float(np.nanstd(eta))
        for label, col in DESCRIPTORS.items():
            x = g[col].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(eta)
            if valid.sum() < 30 or overall_std <= 1e-12:
                rows.append(
                    {
                        "D_family": d,
                        "descriptor": label,
                        "feature_kind": FEATURE_KIND[label],
                        "n": int(valid.sum()),
                        "residual_eta_std_ratio": np.nan,
                    }
                )
                continue
            xv = x[valid]
            yv = eta[valid]
            qs = np.unique(np.nanquantile(xv, np.linspace(0, 1, 11)))
            if len(qs) <= 2:
                ratio = np.nan
            else:
                bins = np.searchsorted(qs[1:-1], xv, side="right")
                stds, counts = [], []
                for b in np.unique(bins):
                    vals = yv[bins == b]
                    if len(vals) >= 15:
                        stds.append(float(np.nanstd(vals)))
                        counts.append(len(vals))
                ratio = float(np.average(stds, weights=counts) / overall_std) if counts else np.nan
            rows.append(
                {
                    "D_family": d,
                    "descriptor": label,
                    "feature_kind": FEATURE_KIND[label],
                    "n": int(valid.sum()),
                    "residual_eta_std_ratio": ratio,
                }
            )
    return pd.DataFrame(rows)


def descriptor_eta_relation(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for d, g in df.groupby("dynamic_family_id", sort=True):
        eta = g["eta50p0"].to_numpy(float)
        for label, col in DESCRIPTORS.items():
            x = g[col].to_numpy(float)
            valid = np.isfinite(x) & np.isfinite(eta)
            if valid.sum() < 20 or np.nanstd(x[valid]) <= 1e-12:
                rho, r2 = np.nan, np.nan
            else:
                rho = float(pd.Series(x[valid]).corr(pd.Series(eta[valid]), method="spearman"))
                coef = np.polyfit(x[valid], eta[valid], 1)
                pred = np.polyval(coef, x[valid])
                r2 = float(r2_score(eta[valid], pred))
            rows.append(
                {
                    "D_family": d,
                    "descriptor": label,
                    "feature_kind": FEATURE_KIND[label],
                    "spearman_eta50": rho,
                    "linear_r2": r2,
                }
            )
    return pd.DataFrame(rows)


def draw_heatmap(df: pd.DataFrame, title: str, cbar_label: str, name: str, cmap: str, vmin=None, vmax=None) -> str:
    fig, ax = plt.subplots(figsize=(max(11.5, df.shape[1] * 0.68), max(6.2, df.shape[0] * 0.36)))
    im = ax.imshow(df.to_numpy(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(df.shape[0]), df.index, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=cbar_label)
    return savefig(name)


def fig00_condition_spread(spread_df: pd.DataFrame) -> str:
    mat = spread_df.set_index("D_family").reindex(columns=descriptor_order())
    return draw_heatmap(
        mat,
        "Within-D condition spread across expanded conditions",
        "robust-scaled IQR inside D",
        "fig00_within_D_condition_spread_expanded_conditions.png",
        cmap="viridis",
        vmin=0,
    )


def fig00b_condition_median_and_between_d(median_df: pd.DataFrame, var_df: pd.DataFrame) -> str:
    mat = median_df.set_index("D_family").reindex(columns=descriptor_order())
    ordered_var = var_df.set_index("descriptor").reindex(descriptor_order()).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.4), gridspec_kw={"width_ratios": [1.65, 0.95]})
    vmax = float(np.nanquantile(np.abs(mat.to_numpy(float)), 0.98))
    vmax = max(vmax, 0.1)
    im = axes[0].imshow(mat.to_numpy(float), aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    axes[0].set_title("D-wise condition median trend")
    axes[0].set_xticks(range(mat.shape[1]), mat.columns, rotation=55, ha="right", fontsize=8)
    axes[0].set_yticks(range(mat.shape[0]), mat.index)
    fig.colorbar(im, ax=axes[0], fraction=0.035, pad=0.02, label="robust-scaled median inside D")

    colors = ordered_var["feature_kind"].map({"abstract": "#6b7280", "H_structural": "#4c78a8", "early_readout": "#f58518"})
    axes[1].barh(ordered_var["descriptor"], ordered_var["between_D_variance_fraction"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("between-D variance fraction")
    axes[1].set_title("How much of each condition varies between D?")
    axes[1].scatter([], [], color="#6b7280", label="abstract")
    axes[1].scatter([], [], color="#4c78a8", label="H structural")
    axes[1].scatter([], [], color="#f58518", label="early readout")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    style(axes[1])
    return savefig("fig00b_D_condition_median_and_betweenD_variance_expanded_conditions.png")


def fig01_predictive(pred: pd.DataFrame) -> str:
    order = descriptor_order()
    all_eta = pred[(pred["D_family"].eq("ALL")) & (pred["target"].eq("eta50"))]
    left = all_eta.set_index("descriptor").reindex(order)[["cv_r2_mean"]]
    per_d = pred[(~pred["D_family"].eq("ALL")) & (pred["target"].eq("eta50"))]
    right = per_d.pivot(index="D_family", columns="descriptor", values="cv_r2_mean").reindex(columns=order)
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.6), gridspec_kw={"width_ratios": [0.75, 1.85]})
    im0 = axes[0].imshow(left.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=max(0.55, np.nanmax(left.to_numpy(float))))
    axes[0].set_title("ALL: single descriptor CV R2")
    axes[0].set_yticks(range(len(left.index)), left.index)
    axes[0].set_xticks([0], ["eta50"])
    fig.colorbar(im0, ax=axes[0], fraction=0.045, pad=0.02)
    im1 = axes[1].imshow(right.to_numpy(float), aspect="auto", cmap="viridis", vmin=0, vmax=max(0.5, np.nanquantile(right.to_numpy(float), 0.98)))
    axes[1].set_title("Per-D: CV R2 predicting eta50")
    axes[1].set_xticks(range(len(right.columns)), right.columns, rotation=55, ha="right", fontsize=8)
    axes[1].set_yticks(range(len(right.index)), right.index)
    fig.colorbar(im1, ax=axes[1], fraction=0.025, pad=0.02)
    return savefig("fig01_single_condition_predictive_sufficiency_expanded_conditions.png")


def fig02_bin_residual(bin_df: pd.DataFrame) -> str:
    order = descriptor_order()
    agg = bin_df.groupby(["descriptor", "condition_bin"], as_index=False).agg(
        dynamic_profile_spread=("dynamic_profile_spread", "median"),
        eta50_iqr=("eta50_iqr", "median"),
    )
    p1 = agg.pivot(index="descriptor", columns="condition_bin", values="dynamic_profile_spread").reindex(order)
    p2 = agg.pivot(index="descriptor", columns="condition_bin", values="eta50_iqr").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8))
    im1 = axes[0].imshow(p1.to_numpy(float), aspect="auto", cmap="viridis")
    axes[0].set_title("Median dynamic spread inside condition bins")
    im2 = axes[1].imshow(p2.to_numpy(float), aspect="auto", cmap="magma")
    axes[1].set_title("Median eta50 IQR inside condition bins")
    for ax, p in zip(axes, [p1, p2]):
        ax.set_xticks(range(p.shape[1]), [f"bin {int(c)}" for c in p.columns])
        ax.set_yticks(range(p.shape[0]), p.index)
    fig.colorbar(im1, ax=axes[0], fraction=0.03, pad=0.02)
    fig.colorbar(im2, ax=axes[1], fraction=0.03, pad=0.02)
    return savefig("fig02_condition_bin_residual_diversity_expanded_conditions.png")


def fig03_matched_pairs(pair_df: pd.DataFrame) -> str:
    order = descriptor_order()
    data_dyn = [pair_df.loc[pair_df["descriptor"].eq(desc), "dynamic_distance_q90"].dropna().to_numpy() for desc in order]
    data_eta = [pair_df.loc[pair_df["descriptor"].eq(desc), "eta50_distance_q90"].dropna().to_numpy() for desc in order]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.8), sharey=True)
    pos = np.arange(len(order)) + 1
    axes[0].boxplot(data_dyn, positions=pos, vert=False, showfliers=False)
    axes[0].set_title("Q90 dynamic distance among matched-condition pairs")
    axes[0].set_xlabel("dynamic distance q90")
    axes[1].boxplot(data_eta, positions=pos, vert=False, showfliers=False)
    axes[1].set_title("Q90 eta50 distance among matched-condition pairs")
    axes[1].set_xlabel("eta50 distance q90")
    for ax in axes:
        ax.set_yticks(pos, order)
        ax.invert_yaxis()
        style(ax)
    return savefig("fig03_matched_condition_pair_distance_expanded_conditions.png")


def fig04_entropy(ent_df: pd.DataFrame) -> str:
    order = descriptor_order()
    p = ent_df.pivot(index="descriptor", columns="condition_bin", values="normalized_D_entropy").reindex(order)
    return draw_heatmap(
        p,
        "D-label entropy inside condition quantile bins",
        "normalized entropy (0 separated, 1 mixed)",
        "fig04_D_entropy_over_condition_bins_expanded_conditions.png",
        cmap="magma",
        vmin=0,
        vmax=1,
    )


def fig05_single_d_classifier(acc_df: pd.DataFrame) -> str:
    order = descriptor_order()
    p = acc_df.set_index("descriptor").reindex(order)
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    colors = p["feature_kind"].map({"abstract": "#6b7280", "H_structural": "#4c78a8", "early_readout": "#f58518"})
    ax.barh(p.index, p["balanced_accuracy"], color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV balanced accuracy")
    ax.set_title("Single-condition D classifier accuracy")
    ax.scatter([], [], color="#6b7280", label="abstract")
    ax.scatter([], [], color="#4c78a8", label="H structural")
    ax.scatter([], [], color="#f58518", label="early readout")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    style(ax)
    return savefig("fig05_single_condition_d_classifier_accuracy_expanded_conditions.png")


def fig06_dynamic_pc(summary: pd.DataFrame) -> str:
    order = descriptor_order()
    p = summary.set_index("descriptor").reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(17.8, 7.8), sharey=True)
    colors = p["feature_kind"].map({"abstract": "#6b7280", "H_structural": "#4c78a8", "early_readout": "#f58518"})
    axes[0].barh(p.index, p["weighted_mean_cv_r2"], color=colors)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("weighted mean CV R2")
    axes[0].set_title("Single-condition dynamic PC prediction")
    axes[1].barh(p.index, p["mean_residual_std_ratio"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean residual std / original std")
    axes[1].set_title("Residual dynamic variation")
    for ax in axes:
        style(ax)
    axes[1].scatter([], [], color="#6b7280", label="abstract")
    axes[1].scatter([], [], color="#4c78a8", label="H structural")
    axes[1].scatter([], [], color="#f58518", label="early readout")
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    return savefig("fig06_single_condition_dynamic_pc_prediction_expanded_conditions.png")


def fig07_one_condition_residual(resid: pd.DataFrame) -> str:
    agg = resid.groupby(["descriptor", "feature_kind"], as_index=False)["residual_eta_std_ratio"].median()
    order = descriptor_order()
    p = agg.set_index("descriptor").reindex(order)
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    colors = p["feature_kind"].map({"abstract": "#6b7280", "H_structural": "#4c78a8", "early_readout": "#f58518"})
    ax.barh(p.index, p["residual_eta_std_ratio"], color=colors)
    ax.axvline(1.0, color="#111827", linestyle="--", linewidth=0.9, alpha=0.65)
    ax.invert_yaxis()
    ax.set_xlabel("median residual eta50 std ratio after one-condition binning")
    ax.set_title("Residual eta spread after one-condition binning")
    ax.scatter([], [], color="#6b7280", label="abstract")
    ax.scatter([], [], color="#4c78a8", label="H structural")
    ax.scatter([], [], color="#f58518", label="early readout")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.01, 0.5))
    style(ax)
    return savefig("fig07_residual_eta_spread_after_one_condition_binning_expanded_conditions.png")


def fig08_eta_relation(rel: pd.DataFrame) -> str:
    order = descriptor_order()
    p = rel.pivot(index="descriptor", columns="D_family", values="spearman_eta50").reindex(order)
    return draw_heatmap(
        p,
        "Descriptor eta50 relation by D",
        "Spearman rho",
        "fig08_descriptor_eta_relation_by_D_expanded_conditions.png",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )


def write_html(figures: list[dict], metrics: dict) -> None:
    cards = []
    for i, fig in enumerate(figures, 1):
        cards.append(
            f"""
    <section class="figure-block">
      <div class="figure-meta">
        <div class="step">Step {i}</div>
        <div class="flow">{fig['flow']}</div>
        <h2>{fig['title']}</h2>
        <p><b>무엇을 보나.</b> {fig['shows']}</p>
        <p><b>H-condition 추가.</b> {fig['h_added']}</p>
        <p><b>해석.</b> {fig['interpretation']}</p>
      </div>
      <a href="{fig['src']}" target="_blank" rel="noreferrer"><img src="{fig['src']}" alt="{fig['title']}"></a>
    </section>
"""
        )
    html = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>68. 66-like Per-Condition Storyline with H Conditions</title>
  <style>
    body {{ margin:0; font-family: Arial, "Malgun Gothic", sans-serif; color:#172033; background:#f4f6fa; }}
    header {{ background:#fff; border-bottom:1px solid #dfe5ef; padding:28px 38px 20px; }}
    main {{ max-width:1380px; margin:0 auto; padding:24px 26px 46px; }}
    h1 {{ margin:0 0 8px; font-size:28px; }}
    h2 {{ margin:0 0 10px; font-size:20px; }}
    p {{ line-height:1.55; color:#526070; }}
    .summary {{ max-width:1160px; }}
    .metric-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:10px; max-width:1160px; margin-top:16px; }}
    .metric {{ border:1px solid #dfe5ef; border-radius:8px; padding:12px 14px; background:#fbfcff; }}
    .metric strong {{ display:block; font-size:18px; color:#172033; margin-bottom:3px; }}
    .note {{ max-width:1210px; border-left:4px solid #4c78a8; background:#fff; padding:14px 16px; margin-bottom:20px; }}
    .figure-block {{ background:#fff; border:1px solid #dfe5ef; border-radius:8px; padding:18px; margin-bottom:24px; }}
    .step {{ color:#4c78a8; font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.06em; }}
    .flow {{ color:#697586; font-size:14px; margin:4px 0 8px; }}
    img {{ display:block; width:100%; max-width:1240px; margin:14px auto 0; border:1px solid #dfe5ef; border-radius:6px; background:#fff; }}
    footer {{ padding:16px 38px 34px; color:#697586; font-size:13px; }}
  </style>
</head>
<body>
  <header>
    <h1>68. 66-like Per-Condition Storyline with H Conditions</h1>
    <p class="summary">
      66번의 논리 흐름을 유지하되, feature-set 그룹 비교를 하지 않고 condition descriptor 목록만 확장한 버전입니다.
      기존 abstract/readout condition 뒤에 H-derived structural/eigenvalue condition 후보를 추가해 같은 plot 방식으로 비교합니다.
    </p>
    <div class="metric-grid">
      <div class="metric"><strong>{metrics['n_samples']:,}</strong>eta50 &ge; {ETA_FILTER:.2f} samples</div>
      <div class="metric"><strong>{metrics['n_d_groups']}</strong>whole-D groups</div>
      <div class="metric"><strong>{metrics['n_conditions']}</strong>condition descriptors</div>
      <div class="metric"><strong>{metrics['best_single_eta_r2']:.3f}</strong>best single-condition eta50 CV R2</div>
      <div class="metric"><strong>{metrics['median_h_eta_r2']:.3f}</strong>median H-condition eta50 CV R2</div>
      <div class="metric"><strong>{metrics['median_h_between_d_fraction']:.3f}</strong>median H-condition between-D fraction</div>
      <div class="metric"><strong>{metrics['median_h_entropy']:.3f}</strong>median H-condition D entropy</div>
      <div class="metric"><strong>{metrics['median_h_within_d_iqr']:.3f}</strong>median H-condition within-D IQR</div>
    </div>
  </header>
  <main>
    <div class="note">
      <b>읽는 법.</b> 이 dashboard는 각 condition을 개별 축으로 본다. H structural condition도 readout보다 안전한 후보지만,
      단일 condition 실험이므로 multi-feature structural condition 가능성을 부정하는 자료로 읽으면 안 된다.
    </div>
    {''.join(cards)}
  </main>
  <footer>Generated by new/build_condition_claim_storyline_core_h_per_condition_like66.py.</footer>
</body>
</html>"""
    HTML_OUT.write_text(html, encoding="utf-8")


def write_report(figures: list[dict], metrics: dict) -> None:
    lines = [
        "# 68. 66 흐름 유지 + per-condition H 후보 추가 설명",
        "",
        "66번 core dashboard의 논리 흐름을 유지하되, feature-set 그룹 비교 없이 condition descriptor 목록만 확장했다.",
        "",
        "## 핵심 수치",
        "",
        f"- 샘플 수: {metrics['n_samples']:,}",
        f"- D group 수: {metrics['n_d_groups']}",
        f"- condition descriptor 수: {metrics['n_conditions']}",
        f"- best single-condition eta50 CV R2: {metrics['best_single_eta_r2']:.3f}",
        f"- median H-condition eta50 CV R2: {metrics['median_h_eta_r2']:.3f}",
        f"- median H-condition between-D variance fraction: {metrics['median_h_between_d_fraction']:.3f}",
        f"- median H-condition D entropy: {metrics['median_h_entropy']:.3f}",
        f"- median H-condition within-D robust-scaled IQR: {metrics['median_h_within_d_iqr']:.3f}",
        "",
        "## Figure별 설명",
        "",
    ]
    for fig in figures:
        lines.extend(
            [
                f"### {fig['title']}",
                "",
                f"- 무엇을 보나: {fig['shows']}",
                f"- H-condition 추가: {fig['h_added']}",
                f"- 해석: {fig['interpretation']}",
                f"- image: `{fig['src']}`",
                "",
            ]
        )
    (REPORT / "figure_description_and_claim_relevance.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ensure_dirs()
    df = load_data()
    z, profile_cols = dynamic_profile(df)
    d_order = sorted(df["dynamic_family_id"].astype(str).unique())

    median_df = condition_median_by_d(df)
    spread_df = condition_spread_by_d(df)
    between_var_df = condition_between_d_variance(df)
    pred = single_condition_predictive_r2(df)
    bin_df = condition_bin_diversity(df, z)
    pair_df = matched_condition_pairs(df, z)
    ent_df = condition_bin_d_entropy(df)
    clf_df = single_condition_d_classifier(df)
    pc_scores, pc_summary = single_condition_dynamic_pc_prediction(df, z)
    resid_df = one_condition_eta_residual_spread(df)
    rel_df = descriptor_eta_relation(df)

    median_df.to_csv(CSV / "D_condition_median_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    spread_df.to_csv(CSV / "within_D_condition_spread_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    between_var_df.to_csv(CSV / "condition_between_D_variance_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(CSV / "single_condition_predictive_sufficiency_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    bin_df.to_csv(CSV / "condition_bin_residual_diversity_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    pair_df.to_csv(CSV / "matched_condition_pair_distance_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    ent_df.to_csv(CSV / "condition_bin_D_entropy_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    clf_df.to_csv(CSV / "single_condition_D_classifier_accuracy_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    pc_scores.to_csv(CSV / "single_condition_dynamic_pc_prediction_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    pc_summary.to_csv(CSV / "single_condition_dynamic_pc_prediction_summary_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    resid_df.to_csv(CSV / "one_condition_eta_residual_spread_expanded_conditions.csv", index=False, encoding="utf-8-sig")
    rel_df.to_csv(CSV / "descriptor_eta_relation_by_D_expanded_conditions.csv", index=False, encoding="utf-8-sig")

    figures = [
        {
            "flow": "2. Across-D condition trend",
            "title": "D-wise condition trend",
            "src": fig00b_condition_median_and_between_d(median_df, between_var_df),
            "shows": "D마다 각 condition의 중심값이 어떻게 다른지, 그리고 전체 condition variance 중 D 사이 차이가 차지하는 비율을 본다.",
            "h_added": "H structural condition 각각에 대해 D별 robust-scaled median과 between-D variance fraction을 추가했다.",
            "interpretation": "다른 D가 condition space에서 서로 다른 경향을 가지는지 확인한다.",
        },
        {
            "flow": "2. Within-group condition spread",
            "title": "Within-D condition spread",
            "src": fig00_condition_spread(spread_df),
            "shows": "같은 D 내부에서 각 condition 값이 얼마나 넓게 퍼져 있는지 robust-scaled IQR로 본다.",
            "h_added": "기존 abstract/readout condition뿐 아니라 H structural condition 각각의 within-D spread를 같은 heatmap에 추가했다.",
            "interpretation": "D를 condition 값 하나로 대표할 수 있는지, 또는 같은 D 안에서도 condition variation이 큰지 확인한다.",
        },
        {
            "flow": "3. Single-descriptor sufficiency",
            "title": "Single-condition predictive sufficiency",
            "src": fig01_predictive(pred),
            "shows": "각 condition 하나만으로 eta50을 얼마나 예측하는지 본다.",
            "h_added": "기존 condition 목록에 H structural/eigenvalue/coupling summary를 condition row로 추가했다.",
            "interpretation": "단일 H-condition이 eta50 설명축으로 충분한지 기존 방식 그대로 비교한다.",
        },
        {
            "flow": "4. Residual diversity after condition binning",
            "title": "Condition-bin residual diversity",
            "src": fig02_bin_residual(bin_df),
            "shows": "각 condition quantile bin 안에 dynamic spread와 eta IQR이 얼마나 남는지 본다.",
            "h_added": "H structural condition 각각에 대해 같은 q=5 binning을 적용했다.",
            "interpretation": "H-condition 값 범위를 맞춰도 dynamic/eta spread가 남는지 확인한다.",
        },
        {
            "flow": "4. Matched-condition countercheck",
            "title": "Matched-condition pair distance",
            "src": fig03_matched_pairs(pair_df),
            "shows": "각 condition 값이 거의 같은 pair 사이의 dynamic/eta distance를 본다.",
            "h_added": "H structural condition 각각에 대해서도 adjacent matched pair를 구성했다.",
            "interpretation": "단일 H-condition이 가까운 것만으로 dynamic similarity가 보장되는지 확인한다.",
        },
        {
            "flow": "5. Condition-bin mixing",
            "title": "D entropy over condition bins",
            "src": fig04_entropy(ent_df),
            "shows": "각 condition bin 안에 D labels가 얼마나 섞이는지 본다.",
            "h_added": "H structural condition 각각을 동일하게 q=5 binning해 entropy를 계산했다.",
            "interpretation": "비슷한 H-condition 값 구간이 여러 D group을 포함하는지 확인한다.",
        },
        {
            "flow": "6. Condition-only classification",
            "title": "Single-condition D classifier accuracy",
            "src": fig05_single_d_classifier(clf_df),
            "shows": "각 condition 하나만으로 D group을 맞히는 balanced accuracy를 본다.",
            "h_added": "H structural condition 각각을 single-condition classifier 입력으로 평가했다.",
            "interpretation": "단일 H-condition이 dynamic group 구분에 얼마나 유용한지 비교한다.",
        },
        {
            "flow": "6. Residual dynamic information",
            "title": "Single-condition dynamic PC prediction",
            "src": fig06_dynamic_pc(pc_summary),
            "shows": "각 condition 하나로 dynamic PC를 예측했을 때 R2와 residual variation을 본다.",
            "h_added": "H structural condition 각각을 같은 single-condition prediction 방식으로 평가했다.",
            "interpretation": "단일 H-condition만으로 dynamic profile variation을 얼마나 설명하는지 확인한다.",
        },
        {
            "flow": "4. Shared one-condition support check",
            "title": "Residual eta spread after one-condition binning",
            "src": fig07_one_condition_residual(resid_df),
            "shows": "각 condition 하나로 binning한 뒤 D 내부 eta spread가 얼마나 남는지 본다.",
            "h_added": "H structural condition 각각에 대해 decile-like one-condition binning을 적용했다.",
            "interpretation": "단일 H-condition이 D 내부 eta variation을 충분히 줄이는지 확인한다.",
        },
        {
            "flow": "2. Within-group condition trend",
            "title": "Descriptor eta50 relation by D",
            "src": fig08_eta_relation(rel_df),
            "shows": "각 D 내부에서 condition과 eta50의 Spearman 관계 방향과 크기를 본다.",
            "h_added": "H structural condition 각각을 기존 descriptor-eta relation heatmap에 추가했다.",
            "interpretation": "H-condition의 eta 관계가 D마다 일관적인지 또는 context-dependent인지 확인한다.",
        },
    ]

    all_eta = pred[(pred["D_family"].eq("ALL")) & (pred["target"].eq("eta50"))]
    h_eta = all_eta[all_eta["feature_kind"].eq("H_structural")]
    h_entropy = ent_df[ent_df["feature_kind"].eq("H_structural")]
    h_between = between_var_df[between_var_df["feature_kind"].eq("H_structural")]
    h_spread_values = spread_df[[d for d in descriptor_order() if d in H_STRUCTURAL]].to_numpy(float)
    metrics = {
        "n_samples": int(len(df)),
        "n_d_groups": int(len(d_order)),
        "n_conditions": int(len(DESCRIPTORS)),
        "best_single_eta_r2": float(all_eta["cv_r2_mean"].max()),
        "median_h_eta_r2": float(h_eta["cv_r2_mean"].median()),
        "median_h_between_d_fraction": float(h_between["between_D_variance_fraction"].median()),
        "median_h_entropy": float(h_entropy["normalized_D_entropy"].median()),
        "median_h_within_d_iqr": float(np.nanmedian(h_spread_values)),
    }
    (OUT / "metrics_summary.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(figures, metrics)
    write_report(figures, metrics)
    (OUT / "run_command.md").write_text(
        "`$env:PYTHONIOENCODING='utf-8'; C:\\\\Users\\\\User\\\\anaconda3\\\\envs\\\\py311-cu132\\\\python.exe new\\\\build_condition_claim_storyline_core_h_per_condition_like66.py`\n",
        encoding="utf-8",
    )
    print(json.dumps({"html": str(HTML_OUT), "out": str(OUT), **metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
