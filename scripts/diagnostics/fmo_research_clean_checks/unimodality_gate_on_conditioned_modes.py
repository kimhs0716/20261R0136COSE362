#!/usr/bin/env python3
"""Unimodality gate on the EXACT trajectories the 40/41 dashboards cluster.

Cross-review tool. It reuses `dynamic_condition_modes_n1000/.../dynamic_condition_mode_traces.npz`
(the already-simulated generated-H trajectories, 1000 per condition) and the SAME dynamic
distance as `cluster_conditioned_h_dynamic_modes.py`
(`sqrt(eta_rms^2 + path_rms^2 + site_rms^2)`).

For each condition it contrasts:
  A. "Dashboard view" ??Agglomerative(average linkage) at k=2,3,4 + silhouette + cluster sizes.
     (This is the number the 40/41/44 dashboards report ??but clustering ALWAYS returns k pieces.)
  B. "Unimodality gate" ??does the data ACTUALLY have separated modes?
       - HDBSCAN (precomputed) : density-separated clusters vs one blob + noise (k not forced)
       - Hartigan dip test     : formal unimodality null on the dynamic 1-D axis (MDS1 + centroid axis)
       - KDE valley depth      : is there a real low-density gap between peaks?

If A shows "k modes, silhouette ~0.2" but B says unimodal (dip p>0.05, HDBSCAN ~0 clusters,
valley<0.3) -> the "modes" are a continuous gradient sliced into k, not discrete mechanism families.

Usage:  python scripts/unimodality_gate_on_conditioned_modes.py [--n 800]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
from scipy.stats import gaussian_kde
import diptest

TRACES = ROOT / "outputs/conditioned_H_distribution_20260604/dynamic_condition_modes_n1000/npz/dynamic_condition_mode_traces.npz"
CONDITIONS = ["c_nonhigh", "c_late", "c_fast", "c_very_fast", "c_fmo"]


def their_dynamic_distance(eta, path, site, ds=2):
    """EXACT distance of cluster_conditioned_h_dynamic_modes.py: sqrt(eta_rms^2+path_rms^2+site_rms^2),
    vectorized. rms over time/groups == euclidean / sqrt(#elements)."""
    eta = eta[:, ::ds]
    path = path[:, ::ds, :].reshape(len(path), -1)
    site = site[:, ::ds, :].reshape(len(site), -1)
    T = eta.shape[1]
    d_eta = pdist(eta) / np.sqrt(T)
    d_path = pdist(path) / np.sqrt(path.shape[1])
    d_site = pdist(site) / np.sqrt(site.shape[1])
    D = np.sqrt(d_eta**2 + d_path**2 + d_site**2)
    return squareform(D).astype(float)


def classical_mds(D, k=6):
    n = len(D)
    D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    w = np.clip(w[idx], 0, None)
    return V[:, idx] * np.sqrt(w + 1e-12)


def dip_p(x):
    x = np.asarray(x, float)
    if x.std() < 1e-12 or len(np.unique(x)) < 4:
        return 1.0
    return float(diptest.diptest(x)[1])


def kde_valley(x, grid=400):
    x = np.asarray(x, float)
    if x.std() < 1e-12:
        return 1, 0.0
    xs = np.linspace(x.min(), x.max(), grid)
    dens = gaussian_kde(x)(xs)
    peaks = [i for i in range(1, grid - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    if len(peaks) < 2:
        return len(peaks) or 1, 0.0
    p2 = sorted(peaks, key=lambda i: dens[i])[-2:]
    lo, hi = min(p2), max(p2)
    valley = dens[lo:hi + 1].min()
    return len(peaks), float(1 - valley / (min(dens[p2[0]], dens[p2[1]]) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=800, help="samples per condition (<=1000)")
    ap.add_argument("--ds", type=int, default=2, help="time downsample")
    args = ap.parse_args()

    t = np.load(TRACES, allow_pickle=True)
    cond = t["condition"].astype(str)
    eta_t, path_t, pop_t = t["eta_t"], t["path_t"], t["pop_t"][:, :, :7]
    rng = np.random.default_rng(0)

    print("=" * 80, flush=True)
    print("UNIMODALITY GATE on the exact trajectories of dashboards 40/41 (generated H, n/cond)", flush=True)
    print("  A = dashboard view (forced-k Agglomerative)   B = formal gate (dip/KDE/HDBSCAN)", flush=True)
    print("=" * 80, flush=True)

    results = []
    for c in CONDITIONS:
        idx = np.flatnonzero(cond == c)
        if len(idx) > args.n:
            idx = np.sort(rng.choice(idx, args.n, replace=False))
        D = their_dynamic_distance(eta_t[idx], path_t[idx], pop_t[idx], ds=args.ds)
        n = len(D)

        # A. dashboard view: average-linkage at k=2,3,4 (their method)
        dash = {}
        for k in (2, 3, 4):
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                          linkage="average").fit_predict(D)
            sizes = np.bincount(lab).tolist()
            sil = silhouette_score(D, lab, metric="precomputed")
            dash[k] = (float(sil), sizes)
        best_k = max(dash, key=lambda k: dash[k][0])

        # B. unimodality gate
        coords = classical_mds(D, k=6)
        mds1 = coords[:, 0]
        lab2 = AgglomerativeClustering(n_clusters=2, metric="precomputed",
                                       linkage="average").fit_predict(D)
        c0, c1 = coords[lab2 == 0].mean(0), coords[lab2 == 1].mean(0)
        u = (c1 - c0); u /= (np.linalg.norm(u) + 1e-12)
        cax = (coords - coords.mean(0)) @ u
        dp_mds, dp_cax = dip_p(mds1), dip_p(cax)
        nmodes, valley = kde_valley(mds1)
        mcs = max(20, n // 20)
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=10, metric="precomputed",
                      copy=True).fit_predict(D.astype(np.float64))
        n_hdb = len(np.unique(hdb[hdb >= 0]))
        noise = float((hdb < 0).mean())

        verdict = ("MULTIMODAL" if (n_hdb >= 2 and min(dp_mds, dp_cax) < 0.05 and valley > 0.3)
                   else "UNIMODAL / continuous" if (n_hdb < 2 and min(dp_mds, dp_cax) > 0.05)
                   else "weak / ambiguous")

        print(f"\n?? {c}  (n={n})", flush=True)
        print(f"   A dashboard: best k={best_k} sil={dash[best_k][0]:.2f} sizes={dash[best_k][1]}  "
              f"(k2 sil={dash[2][0]:.2f}, k3={dash[3][0]:.2f}, k4={dash[4][0]:.2f})", flush=True)
        print(f"   B gate:  HDBSCAN clusters={n_hdb} (noise {100*noise:.0f}%)  "
              f"dip MDS1 p={dp_mds:.3f} / centroid p={dp_cax:.3f}  KDE valley={valley:.2f}", flush=True)
        print(f"   ??{verdict}", flush=True)

        results.append({
            "condition": c, "n": n,
            "dashboard_best_k": best_k, "dashboard_silhouette": dash[best_k][0],
            "dashboard_sizes": dash[best_k][1],
            "silhouette_by_k": {k: dash[k][0] for k in dash},
            "hdbscan_clusters": n_hdb, "hdbscan_noise": noise,
            "dip_p_mds1": dp_mds, "dip_p_centroid": dp_cax,
            "kde_modes": nmodes, "kde_valley_depth": valley, "verdict": verdict,
        })

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY  (dashboard 'modes' vs formal gate)", flush=True)
    for r in results:
        print(f"   {r['condition']:13s}: dashboard k={r['dashboard_best_k']} sil={r['dashboard_silhouette']:.2f}"
              f"  ?? gate: {r['verdict']}  (HDBSCAN {r['hdbscan_clusters']}, dip {min(r['dip_p_mds1'],r['dip_p_centroid']):.2f}, valley {r['kde_valley_depth']:.2f})",
              flush=True)
    mm = sum(r["verdict"] == "MULTIMODAL" for r in results)
    print(f"\n   MULTIMODAL conditions: {mm}/{len(results)}", flush=True)
    print("   Interpretation: dashboard-style k partitions can return moderate silhouette scores,", flush=True)
    print("         but dip/KDE/HDBSCAN checks are needed before calling them discrete modes.", flush=True)

    out = ROOT / "outputs/unimodality_gate_conditioned_modes.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nsaved: {out}\nDONE.", flush=True)


if __name__ == "__main__":
    main()

