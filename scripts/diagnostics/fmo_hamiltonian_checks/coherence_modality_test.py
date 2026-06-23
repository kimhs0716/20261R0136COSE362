"""B1+B3 — coherence 표현 + HDBSCAN 밀도 클러스터로 숨은 multimodality 탐색.

현재 결론("unimodal")의 사각지대 2개를 친다:
  B1 표현: site population 궤적 대신 *off-diagonal 결맞음* |ρ_ij(t)| (21 쌍) 으로 클러스터.
           ENAQT 의 본질이 coherence 라, 같은 population 을 다른 결맞음 경로로 만드는 두
           메커니즘이 있다면 population L2 는 못 잡지만 여기선 잡힌다.
  B3 metric: agglomerative(연결 가정) 대신 HDBSCAN(밀도 기반) — 저밀도로 분리된
           disconnected component(진짜 위상적 multimodality)를 탐지 + noise 분리.

fixed-condition 안에서, population 표현과 coherence 표현을 *나란히* 비교.
각 표현: 표준화 → PCA(10) → HDBSCAN + dip test(PC1 / ward-centroid 축).

사용:  python coherence_modality_test.py [--k 100]
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import diptest

from src.cnf import load_data, normalize, H27_to_matrix
from src import simulator as sim

DS = 10                                        # 시간 다운샘플 (1500 → 150)
_IU = np.triu_indices(7, k=1)                  # 21 off-diagonal 쌍


def pop_and_coh(H7):
    """7×7 H → (population 궤적 flat, coherence 궤적 flat, 쌍별 적분 결맞음 21d)."""
    out = sim.simulate(H7, 35.0, return_traj=True)
    tlist, rho = out.pop("_traj")
    sysb = rho[:, :7, :7]                       # (1500, 7, 7) complex
    pops = np.real(np.diagonal(sysb, axis1=1, axis2=2))        # (1500, 7)
    coh = np.abs(sysb[:, _IU[0], _IU[1]])                      # (1500, 21) |ρ_ij|
    pop_flat = pops[::DS].reshape(-1)                          # (~150*7,)
    coh_flat = coh[::DS].reshape(-1)                           # (~150*21,)
    coh_integ = np.trapz(coh, tlist, axis=0)                  # (21,) 쌍별 적분 결맞음
    return pop_flat.astype(np.float32), coh_flat.astype(np.float32), coh_integ.astype(np.float32)


def dip_p(x):
    x = np.asarray(x, float)
    if x.std() < 1e-12 or len(np.unique(x)) < 4:
        return 1.0
    return float(diptest.diptest(x)[1])


def analyze_rep(flat, tag):
    """표준화 → PCA(10) → HDBSCAN + dip. 반환 dict."""
    Z = StandardScaler().fit_transform(flat)
    Z = PCA(n_components=min(10, flat.shape[0] - 1, flat.shape[1])).fit_transform(Z)
    hdb = HDBSCAN(min_cluster_size=12, min_samples=5, copy=True).fit_predict(Z)
    labels = hdb[hdb >= 0]
    n_clusters = len(np.unique(labels))
    noise = float((hdb < 0).mean())
    sizes = np.bincount(hdb[hdb >= 0]).tolist() if n_clusters else []
    pc1 = Z[:, 0]
    lbl2 = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Z)
    c0, c1 = Z[lbl2 == 0].mean(0), Z[lbl2 == 1].mean(0)
    u = (c1 - c0); u /= (np.linalg.norm(u) + 1e-12)
    cax = (Z - Z.mean(0)) @ u
    dp_pc1, dp_cax = dip_p(pc1), dip_p(cax)
    verdict = "MULTIMODAL" if (n_clusters >= 2 and min(dp_pc1, dp_cax) < 0.05) \
        else "weak" if n_clusters >= 2 or min(dp_pc1, dp_cax) < 0.05 else "UNIMODAL"
    return dict(rep=tag, hdbscan_n_clusters=n_clusters, hdbscan_noise=noise,
                hdbscan_sizes=sizes, dip_pc1=dp_pc1, dip_centroid=dp_cax,
                verdict=verdict, _Z2=Z[:, :2], _hdb=hdb)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    args = ap.parse_args()
    K = args.k

    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, _ = normalize(X, Y)
    L = Yn[:, :5]; eta, tau = Y[:, 0], Y[:, 1]

    specs = [(0.95, 10, "η0.95·빠른 (대조군)"), (0.95, 90, "η0.95·느린"),
             (0.50, 50, "η0.50·중간"), (0.50, 90, "η0.50·느린 (MM 후보)")]
    anchors = []
    for teta, q, d in specs:
        sub = np.where(np.abs(eta - teta) < 0.03)[0]
        anchors.append((int(sub[np.argmin(np.abs(tau[sub] - np.percentile(tau[sub], q)))]),
                        teta, d))

    os.makedirs("outputs/figures/coherence_modality", exist_ok=True)
    print("=" * 76, flush=True)
    print("B1+B3 — coherence 표현 + HDBSCAN. population vs coherence 나란히 (fixed-condition)",
          flush=True)
    print("=" * 76, flush=True)

    all_res = []
    for n, (a, teta, desc) in enumerate(anchors):
        knn = np.argsort(np.linalg.norm(L - L[a], axis=1))[:K]
        triples = [pop_and_coh(H27_to_matrix(X[i])) for i in knn]
        pop = np.stack([t[0] for t in triples])
        coh = np.stack([t[1] for t in triples])
        coh_int = np.stack([t[2] for t in triples])

        r_pop = analyze_rep(pop, "population")
        r_coh = analyze_rep(coh, "coherence")

        # ③ coherence pathway fingerprint: 어느 site-쌍이 결맞음 주도하나 (categorical)
        dom_pair = [f"{_IU[0][j]+1}-{_IU[1][j]+1}" for j in coh_int.argmax(1)]
        uniq, cnt = np.unique(dom_pair, return_counts=True)
        order = np.argsort(-cnt)
        topcoh = {uniq[i]: int(cnt[i]) for i in order[:4]}

        print(f"\n── anchor {n}: {desc}  (η={eta[a]:.3f} τ={tau[a]:.2f}ps)", flush=True)
        for r in (r_pop, r_coh):
            print(f"   [{r['rep']:10s}] HDBSCAN 군집 {r['hdbscan_n_clusters']}개 "
                  f"(noise {100*r['hdbscan_noise']:.0f}%, sizes {r['hdbscan_sizes']})  | "
                  f"dip PC1 {r['dip_pc1']:.3f} / centroid {r['dip_centroid']:.3f}  ▶ {r['verdict']}",
                  flush=True)
        print(f"   결맞음 주도 site-쌍 분포(top4): {topcoh}", flush=True)

        # 그림: coherence PCA-2D, HDBSCAN 색
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
        for r, axi, ttl in ((r_pop, ax[0], "population"), (r_coh, ax[1], "coherence")):
            Z2, hdb = r["_Z2"], r["_hdb"]
            for cl in np.unique(hdb):
                m = hdb == cl
                lab = "noise" if cl < 0 else f"C{cl} (n={int(m.sum())})"
                axi.scatter(Z2[m, 0], Z2[m, 1], s=14, alpha=0.6,
                            color="lightgray" if cl < 0 else None, label=lab)
            axi.set_title(f"{ttl}  HDBSCAN={r['hdbscan_n_clusters']}  "
                          f"dip={min(r['dip_pc1'], r['dip_centroid']):.2f}")
            axi.set_xlabel("PC1"); axi.set_ylabel("PC2"); axi.legend(fontsize=7)
        fig.suptitle(f"anchor {n} (eta={eta[a]:.2f}, tau={tau[a]:.1f}ps): pop vs coherence")
        fig.tight_layout(); fig.savefig(f"outputs/figures/coherence_modality/anchor{n}.png",
                                        dpi=110); plt.close(fig)

        for r in (r_pop, r_coh):
            r.pop("_Z2"); r.pop("_hdb")
        all_res.append({"anchor": n, "desc": desc, "eta": float(eta[a]), "tau": float(tau[a]),
                        "population": r_pop, "coherence": r_coh, "top_coh_pairs": topcoh})

    print("\n" + "=" * 76, flush=True)
    print("종합 (population → coherence):", flush=True)
    for r in all_res:
        print(f"   {r['desc']:26s}  pop:{r['population']['verdict']:11s} → "
              f"coh:{r['coherence']['verdict']:11s}", flush=True)
    coh_mm = sum(r["coherence"]["verdict"] == "MULTIMODAL" for r in all_res)
    print(f"\n해석: coherence 표현에서 MULTIMODAL {coh_mm}/{len(all_res)} anchor.", flush=True)
    print("  · 0개면 → population 으로 못 본 결맞음 multimodality 도 없음 = unimodal 결론 강화.", flush=True)
    print("  · ≥1개면 → population L2 가 놓친 *결맞음 메커니즘 분기* 발견 (중요!).", flush=True)

    with open("outputs/metrics/coherence_modality_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("\n저장: outputs/metrics/coherence_modality_results.json + "
          "outputs/figures/coherence_modality/*.png\nDONE.", flush=True)


if __name__ == "__main__":
    main()
