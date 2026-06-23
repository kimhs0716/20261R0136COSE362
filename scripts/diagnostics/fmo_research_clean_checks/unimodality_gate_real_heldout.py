#!/usr/bin/env python3
"""Unimodality gate on REAL held-out H (test split), complementing the generated-H gate.

Mirrors `unimodality_gate_on_conditioned_modes.py` but on real Hamiltonians from
`prepared_flow_pilot_data.npz` (62k), restricted to the held-out TEST split, simulated
fresh with THIS repo's simulator. For each priority group it contrasts:
  A. dashboard view  : Agglomerative(average) k=2,3,4 + silhouette + sizes  (the "modes")
  B. unimodality gate: HDBSCAN(precomputed) + Hartigan dip + KDE valley       (real?)

Decode is validated against stored CFAST labels (eta10/eta20/eta50) before clustering.

Usage:
  python scripts/unimodality_gate_real_heldout.py --validate-only
  python scripts/unimodality_gate_real_heldout.py --n 200
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
import diptest

from fmo_hamiltonian import simulator
from fmo_hamiltonian.sampling import h27_to_matrix
from fmo_hamiltonian.trajectory_features import compute_eta_t, compute_path_groups

PREP = ROOT / "outputs/flow_pilot_pH_given_c_20260603_62k_prepare/prepared_flow_pilot_data.npz"
GROUPS = ["fast_high", "very_fast", "late_high"]
TLIST = np.arange(0.0, 50.0 + 0.25, 0.5, dtype=float)   # 101 pts, dt=0.5 (matches dashboards)


def simulate_one(h27):
    H = h27_to_matrix(np.asarray(h27, float))
    out = simulator.simulate(H, tlist=TLIST, return_traj=True)
    _, rho = out["_traj"]
    pop = np.real(np.diagonal(rho, axis1=1, axis2=2)).astype(np.float32)   # (T,9)
    eta = compute_eta_t(pop).astype(np.float32)
    path, _names = compute_path_groups(pop)
    return pop[:, :7], eta, np.asarray(path, np.float32)


def their_dynamic_distance(eta, path, site, ds=2):
    eta = eta[:, ::ds]
    path = path[:, ::ds, :].reshape(len(path), -1)
    site = site[:, ::ds, :].reshape(len(site), -1)
    d_eta = pdist(eta) / np.sqrt(eta.shape[1])
    d_path = pdist(path) / np.sqrt(path.shape[1])
    d_site = pdist(site) / np.sqrt(site.shape[1])
    return squareform(np.sqrt(d_eta**2 + d_path**2 + d_site**2)).astype(float)


def classical_mds(D, k=6):
    n = len(D); D2 = D**2
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None) + 1e-12)


def dip_p(x):
    x = np.asarray(x, float)
    if x.std() < 1e-12 or len(np.unique(x)) < 4:
        return 1.0
    return float(diptest.diptest(x)[1])


def kde_valley(x, grid=400):
    x = np.asarray(x, float)
    if x.std() < 1e-12:
        return 0.0
    xs = np.linspace(x.min(), x.max(), grid)
    dens = gaussian_kde(x)(xs)
    peaks = [i for i in range(1, grid - 1) if dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]]
    if len(peaks) < 2:
        return 0.0
    p2 = sorted(peaks, key=lambda i: dens[i])[-2:]
    valley = dens[min(p2):max(p2) + 1].min()
    return float(1 - valley / (min(dens[p2[0]], dens[p2[1]]) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    d = np.load(PREP, allow_pickle=True)
    H27 = d["H_gauge_27"]
    test = d["split_test"]
    pg = d["priority_group"].astype(str)
    cond_raw = d["CFAST_condition_raw"]            # [eta10, eta20, eta50, t80_cap, flag]
    rng = np.random.default_rng(0)

    # --- decode/sim validation against stored labels ---
    print("=== decode+simulate validation (real H) ===", flush=True)
    vidx = test[rng.choice(len(test), 5, replace=False)]
    ok = True
    for i in vidx:
        site, eta, _ = simulate_one(H27[i])
        e10 = float(np.interp(10.0, TLIST, eta)); e20 = float(np.interp(20.0, TLIST, eta))
        e50 = float(eta[-1])
        s10, s20, s50 = cond_raw[i, 0], cond_raw[i, 1], cond_raw[i, 2]
        good = abs(e10 - s10) < 0.05 and abs(e20 - s20) < 0.05 and abs(e50 - s50) < 0.05
        ok = ok and good
        print(f"  idx {i}: sim eta10/20/50 = {e10:.3f}/{e20:.3f}/{e50:.3f}  "
              f"stored {s10:.3f}/{s20:.3f}/{s50:.3f}  {'OK' if good else 'MISMATCH'}", flush=True)
    print(f"  decode/sim reproduces stored labels: {ok}", flush=True)
    if args.validate_only:
        return
    if not ok:
        print("  WARNING: label mismatch ??gauge/sim convention differs; results suspect.", flush=True)

    test_set = set(int(x) for x in test)
    results = []
    print("\n" + "=" * 80, flush=True)
    print("UNIMODALITY GATE on REAL held-out H (test split), simulated fresh", flush=True)
    print("=" * 80, flush=True)
    for g in GROUPS:
        idx = np.array([i for i in test if pg[i] == g], dtype=int)
        if len(idx) > args.n:
            idx = np.sort(rng.choice(idx, args.n, replace=False))
        n = len(idx)
        t0 = time.perf_counter()
        site = np.zeros((n, len(TLIST), 7), np.float32)
        eta = np.zeros((n, len(TLIST)), np.float32)
        path = np.zeros((n, len(TLIST), 7), np.float32)
        for j, i in enumerate(idx):
            site[j], eta[j], path[j] = simulate_one(H27[i])
            if (j + 1) % 50 == 0:
                print(f"  [{g}] simulated {j+1}/{n} ({time.perf_counter()-t0:.0f}s)", flush=True)
        D = their_dynamic_distance(eta, path, site)

        dash = {}
        for k in (2, 3, 4):
            lab = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average").fit_predict(D)
            dash[k] = (float(silhouette_score(D, lab, metric="precomputed")), np.bincount(lab).tolist())
        best_k = max(dash, key=lambda k: dash[k][0])

        coords = classical_mds(D, 6); mds1 = coords[:, 0]
        lab2 = AgglomerativeClustering(n_clusters=2, metric="precomputed", linkage="average").fit_predict(D)
        c0, c1 = coords[lab2 == 0].mean(0), coords[lab2 == 1].mean(0)
        u = (c1 - c0); u /= (np.linalg.norm(u) + 1e-12)
        cax = (coords - coords.mean(0)) @ u
        dp_mds, dp_cax = dip_p(mds1), dip_p(cax)
        valley = kde_valley(mds1)
        mcs = max(20, n // 20)
        hdb = HDBSCAN(min_cluster_size=mcs, min_samples=10, metric="precomputed", copy=True).fit_predict(D.astype(np.float64))
        n_hdb = len(np.unique(hdb[hdb >= 0])); noise = float((hdb < 0).mean())
        verdict = ("MULTIMODAL" if (n_hdb >= 2 and min(dp_mds, dp_cax) < 0.05 and valley > 0.3)
                   else "UNIMODAL / continuous" if (n_hdb < 2 and min(dp_mds, dp_cax) > 0.05)
                   else "weak / ambiguous")

        print(f"\n?? {g}  (real held-out, n={n})", flush=True)
        print(f"   A dashboard: best k={best_k} sil={dash[best_k][0]:.2f} sizes={dash[best_k][1]}  "
              f"(k2={dash[2][0]:.2f}, k3={dash[3][0]:.2f}, k4={dash[4][0]:.2f})", flush=True)
        print(f"   B gate: HDBSCAN clusters={n_hdb} (noise {100*noise:.0f}%)  "
              f"dip MDS1 p={dp_mds:.3f} / centroid p={dp_cax:.3f}  KDE valley={valley:.2f}", flush=True)
        print(f"   ??{verdict}", flush=True)
        results.append({"group": g, "n": n, "dashboard_best_k": best_k,
                        "dashboard_silhouette": dash[best_k][0], "dashboard_sizes": dash[best_k][1],
                        "silhouette_by_k": {k: dash[k][0] for k in dash},
                        "hdbscan_clusters": n_hdb, "hdbscan_noise": noise,
                        "dip_p_mds1": dp_mds, "dip_p_centroid": dp_cax,
                        "kde_valley_depth": valley, "verdict": verdict})

    print("\n" + "=" * 80, flush=True)
    print("SUMMARY (real held-out H)", flush=True)
    for r in results:
        print(f"   {r['group']:13s}: dashboard k={r['dashboard_best_k']} sil={r['dashboard_silhouette']:.2f}"
              f"  ?? gate: {r['verdict']}  (HDBSCAN {r['hdbscan_clusters']}, dip {min(r['dip_p_mds1'],r['dip_p_centroid']):.2f}, valley {r['kde_valley_depth']:.2f})", flush=True)
    mm = sum(r["verdict"] == "MULTIMODAL" for r in results)
    print(f"\n   MULTIMODAL groups: {mm}/{len(results)}", flush=True)
    out = ROOT / "outputs/unimodality_gate_real_heldout.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"saved: {out}\nDONE.", flush=True)


if __name__ == "__main__":
    main()

