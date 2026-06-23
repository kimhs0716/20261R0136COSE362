"""Does the 472-d dynamic embedding (D000-D012 feature) separate into ~13 families, or is it a
continuous blob sliced into k pieces?

Rebuilds the teammate's feature on the SAME data lineage (re-simulated dense trajectories):
  eta(t) [51] + d eta/dt [51] + path-group(t) [51x7=357] + summary metrics [13] = 472.
Path groups = [site1, site2, sink34=s3+s4, detour567=s5+s6+s7, trap, loss, residual=sum sites].

Separation tests (the question the doc never asks):
  - matched-Gaussian: intrinsic dim + kNN path  (corridor-style: separated/filament vs generic blob)
  - silhouette at k=2..15 (ward) vs the SAME on random/Gaussian — does k=13 (or any k) clear 0.5?
  - HDBSCAN: how many density-separated clusters actually exist (vs 13 claimed)?
  - GMM BIC k=1..15: is there a clear best-k, or does it keep improving (continuum)?
  - dip test on PC1 and the ward-centroid axis.
"""
import os, sys, json, time, threading
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))
from sklearn.cluster import AgglomerativeClustering, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import diptest
from src.cnf import gauge_fix_encode, H27_to_matrix
from src import simulator as sim

TS = np.arange(0.0, 50.0 + 1e-9, 1.0)   # 51 snapshots, 1 ps


def build_472(h27):
    out = sim.simulate(H27_to_matrix(h27), 35.0, return_traj=True)
    tl, rho = out["_traj"]
    pop = np.real(np.diagonal(rho, axis1=1, axis2=2))            # (T,9)
    P = np.stack([np.interp(TS, tl, pop[:, s]) for s in range(9)], axis=1)  # (51,9) vectorized
    eta = P[:, 7]                                                # trap = eta(t)  (51)
    deta = np.gradient(eta, TS)                                  # (51)
    pg = np.stack([P[:, 0], P[:, 1], P[:, 2] + P[:, 3], P[:, 4] + P[:, 5] + P[:, 6],
                   P[:, 7], P[:, 8], P[:, :7].sum(1)], axis=1)   # (51,7)
    # 13 summary metrics
    def at(t): return float(np.interp(t, TS, eta))
    def arr(thr):
        idx = np.where(eta >= thr)[0]
        return float(TS[idx[0]]) if len(idx) else 50.0
    summ = [at(5), at(10), at(20), eta[-1], arr(0.5), arr(0.8), float(deta.max()),
            float(TS[np.argmax(deta)]), float(P[-1, 7]), float(P[-1, 8]),
            float(P[-1, :7].sum()), float(pg[:, 2].mean()), float(pg[:, 3].mean())]
    return np.concatenate([eta, deta, pg.reshape(-1), summ]).astype(np.float32)  # 472


def build_472_timeout(h27, timeout=8.0):
    """Run build_472 with a per-sample timeout to skip stiff/pathological H (brmesolve hang)."""
    box = {}
    def run():
        try: box["r"] = build_472(h27)
        except Exception: pass
    th = threading.Thread(target=run, daemon=True); th.start(); th.join(timeout)
    return box.get("r")          # None if timed out or errored


def twonn_id(X, discard=0.1):
    from sklearn.neighbors import NearestNeighbors
    d, _ = NearestNeighbors(n_neighbors=3).fit(X).kneighbors(X)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12); mu = mu[mu > 1 + 1e-9]
    mu = np.sort(mu)[: int(len(mu) * (1 - discard))]
    return float(len(mu) / np.sum(np.log(mu)))


def char_path(X, k=15, n_src=120):
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import shortest_path, connected_components
    g = kneighbors_graph(X, k, mode="distance"); g = g.maximum(g.T)
    nc, lab = connected_components(g, directed=False)
    idx = np.flatnonzero(lab == np.argmax(np.bincount(lab)))
    src = np.random.default_rng(0).choice(len(idx), min(n_src, len(idx)), replace=False)
    dd = shortest_path(g[idx][:, idx], method="D", indices=src)
    fin = dd[np.isfinite(dd) & (dd > 0)]
    return float(fin.mean()), nc


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=800); args = ap.parse_args()
    Hs = []
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        Hs.append(gauge_fix_encode(np.load(p)["H_params"]))
    H = np.concatenate(Hs)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(H), args.n, replace=False)
    print(f"building 472-d on {args.n} real H (re-sim dense, timeout-protected)...", flush=True)
    t0 = time.perf_counter()
    F = np.full((args.n, 472), np.nan, np.float32)
    nskip = 0
    for i, j in enumerate(idx):
        r = build_472_timeout(H[j])
        if r is not None: F[i] = r
        else: nskip += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{args.n} ({time.perf_counter()-t0:.0f}s, skipped {nskip})", flush=True)
    ok = np.isfinite(F).all(1); F = F[ok]
    print(f"  usable {ok.sum()}/{args.n} (skipped {nskip} stiff/hung H)", flush=True)
    Z = StandardScaler().fit_transform(F)
    print(f"\nN={len(Z)}, dim=472", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("SEPARATION TEST — 472-d dynamic embedding: 13 families or continuous blob?", flush=True)
    print("=" * 78, flush=True)

    # 1. matched-Gaussian
    G = rng.multivariate_normal(Z.mean(0), np.cov(Z.T), size=len(Z))
    idd, gidd = twonn_id(Z), twonn_id(G)
    cp, nc = char_path(Z); gcp, _ = char_path(G)
    print(f"\n[matched-Gaussian] ID data {idd:.2f} / G {gidd:.2f}  |  path data {cp:.2f} / G {gcp:.2f}"
          f"  |  {nc} components", flush=True)

    # 2. silhouette across k (ward) vs Gaussian
    D = squareform(pdist(Z)); GD = squareform(pdist(G))
    print("\n[silhouette] k: data(ward) vs Gaussian(ward)   (>0.5 = strong separation)", flush=True)
    for k in (2, 5, 8, 10, 13, 15):
        ld = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
        lg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(G)
        sd = silhouette_score(D, ld, metric="precomputed")
        sg = silhouette_score(GD, lg, metric="precomputed")
        flag = "  ← claimed k" if k == 13 else ""
        print(f"   k={k:2d}: data {sd:+.3f}   gaussian {sg:+.3f}   Δ {sd-sg:+.3f}{flag}", flush=True)

    # 3. HDBSCAN on PCA
    Zp = PCA(n_components=20).fit_transform(Z)
    hdb = HDBSCAN(min_cluster_size=max(20, len(Z)//30), min_samples=10).fit_predict(Zp)
    nclu = len(np.unique(hdb[hdb >= 0])); noise = float((hdb < 0).mean())
    print(f"\n[HDBSCAN] density-separated clusters: {nclu}  (noise {100*noise:.0f}%)   vs 13 claimed", flush=True)

    # 4. GMM BIC across k
    print("\n[GMM BIC] (lower=better; clear minimum = real #modes, monotone = continuum):", flush=True)
    bics = []
    for k in (1, 2, 5, 8, 10, 13, 15):
        gm = GaussianMixture(k, covariance_type="diag", random_state=0, n_init=1).fit(Zp)
        bics.append((k, gm.bic(Zp)))
    best = min(bics, key=lambda x: x[1])[0]
    print("   " + "  ".join(f"k{k}:{b:.0f}" for k, b in bics), flush=True)
    print(f"   → best k = {best}", flush=True)

    # 5. dip on PC1 + ward-centroid axis (k=2)
    pc1 = Zp[:, 0]
    l2 = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Zp)
    u = Zp[l2 == 1].mean(0) - Zp[l2 == 0].mean(0); u /= np.linalg.norm(u) + 1e-12
    cax = (Zp - Zp.mean(0)) @ u
    print(f"\n[dip test] PC1 p={diptest.diptest(pc1)[1]:.3f}   centroid-axis p={diptest.diptest(cax)[1]:.3f}"
          f"   (p>0.05 = unimodal)", flush=True)

    print("\n" + "=" * 78, flush=True)
    print("판정: HDBSCAN가 13이 아니라 ~0-2이고, silhouette k=13이 낮고(<0.5) Gaussian과 비슷하며,", flush=True)
    print("      GMM BIC가 단조이고, dip가 unimodal이면 → '13 families'는 연속체의 임의 분할.", flush=True)
    print("=" * 78, flush=True)
    Path("outputs/metrics/dfamily_separation.json").write_text(json.dumps(
        {"n": int(len(Z)), "id_data": idd, "id_gauss": gidd, "path_data": cp, "path_gauss": gcp,
         "hdbscan_clusters": nclu, "gmm_best_k": best}, indent=2), encoding="utf-8")
    print("saved: outputs/metrics/dfamily_separation.json", flush=True)


if __name__ == "__main__":
    main()
