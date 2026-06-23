"""D — prior 를 바꿔도 conditional unimodal 인가? (결론의 robustness 최종 점검)

우리 결론은 엄밀히 "4nm-uniform geometric prior 가 unimodal" 이다. prior 를 바꿔 검정:
  D1 box 크기:  2 / 4(control) / 8 nm — compact(강결합) vs sparse(약결합)
  D2 FMO 주변:  실제 H_FMO_CM 을 섭동 — *실제 FMO 를 prior 안에 넣음* (원 질문에 가장 관련)

각 prior: N개 생성 → 그 prior 의 고효율 영역에서 anchor 선택 → 5-d 라벨 KNN-K 이웃 →
동역학 시뮬 → dip(PC1·centroid) + HDBSCAN + ward silhouette → conditional 다봉 판정.

사용:  python prior_ablation.py [--n 2000] [--k 80]
"""
import json
import os
import sys
import time
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

import warnings
warnings.filterwarnings("ignore")
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import HDBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import diptest

from src import simulator as sim
from src.cnf import H_params_to_matrix

DS = 10
N_SITE = 7
_IU_ALL = np.triu_indices(N_SITE)


def place(rng, box, rmin, max_restart=500):
    """excluded-volume 배치 (restart 로 robust)."""
    for _ in range(max_restart):
        pos = np.zeros((N_SITE, 3)); ok = True
        for i in range(N_SITE):
            placed = False
            for _t in range(3000):
                p = rng.uniform(0, box, 3)
                if i == 0 or np.all(np.linalg.norm(pos[:i] - p, axis=1) > rmin):
                    pos[i] = p; placed = True; break
            if not placed:
                ok = False; break
        if ok:
            return pos
    raise RuntimeError(f"placement 실패 box={box} rmin={rmin}")


def sample_box(rng, box):
    """sample_H_geom 의 box-가변 버전. rmin 은 작은 box 에서 축소(배치 가능하도록)."""
    rmin = min(sim.GEOM_RMIN, 0.45 * box)
    pos = place(rng, box, rmin)
    mu = rng.normal(size=(N_SITE, 3)); mu /= np.linalg.norm(mu, axis=1, keepdims=True)
    H = np.zeros((N_SITE, N_SITE))
    for i in range(N_SITE):
        for j in range(i + 1, N_SITE):
            d = pos[i] - pos[j]; r = float(np.linalg.norm(d)); rhat = d / r
            kappa = mu[i] @ mu[j] - 3.0 * (mu[i] @ rhat) * (mu[j] @ rhat)
            H[i, j] = H[j, i] = sim.GEOM_DIP * kappa / r**3
    diag = rng.uniform(*sim.SITE_ENERGY_RANGE, size=N_SITE)
    H[np.diag_indices(N_SITE)] = diag - diag.mean()
    return H


def sample_fmo(rng, sig_off=25.0, sig_diag=50.0):
    """실제 FMO Hamiltonian 을 섭동 (실제 FMO 를 prior 중심에). 대각 평균 0 재고정."""
    H = sim.H_FMO_CM.copy()
    noise = rng.normal(scale=sig_off, size=(N_SITE, N_SITE))
    noise = np.triu(noise, 1); noise = noise + noise.T
    H = H + noise
    d = np.diag(H) + rng.normal(scale=sig_diag, size=N_SITE)
    np.fill_diagonal(H, d - d.mean())
    return H


def gen(prior, n, seed=0):
    rng = np.random.default_rng(seed)
    Hs = np.zeros((n, 28), dtype=np.float32)
    labs = np.zeros((n, 5), dtype=np.float32)
    keys = ("eta", "tau_transfer", "ipr", "purity", "c_l1")
    t0 = time.perf_counter()
    for i in range(n):
        H = sample_box(rng, prior["box"]) if prior["kind"] == "box" else \
            sample_fmo(rng, **prior.get("p", {}))
        out = sim.simulate(H, 35.0)
        Hs[i] = H[_IU_ALL]
        labs[i] = [out[k] if np.isfinite(out[k]) else 0.0 for k in keys]
        if (i + 1) % 400 == 0:
            el = time.perf_counter() - t0
            print(f"    {prior['name']}: {i+1}/{n}  {el:.0f}s", flush=True)
    return Hs, labs


def site_traj(h28):
    out = sim.simulate(H_params_to_matrix(h28), 35.0, return_traj=True)
    _, rho = out.pop("_traj")
    pops = np.real(np.diagonal(rho[:, :N_SITE, :N_SITE], axis1=1, axis2=2))
    return pops[::DS].reshape(-1)


def dip_p(x):
    x = np.asarray(x, float)
    if x.std() < 1e-12 or len(np.unique(x)) < 4:
        return 1.0
    return float(diptest.diptest(x)[1])


def test_condition(Hs, traj):
    """한 anchor 이웃의 동역학 → dip + HDBSCAN + ward silhouette → verdict."""
    Z = StandardScaler().fit_transform(traj)
    Z = PCA(n_components=min(10, len(traj) - 1)).fit_transform(Z)
    hdb = HDBSCAN(min_cluster_size=12, min_samples=5, copy=True).fit_predict(Z)
    n_cl = len(np.unique(hdb[hdb >= 0]))
    D = squareform(pdist(Z))
    l2 = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(Z)
    sil = silhouette_score(D, l2)
    c0, c1 = Z[l2 == 0].mean(0), Z[l2 == 1].mean(0)
    u = (c1 - c0); u /= (np.linalg.norm(u) + 1e-12)
    cax = (Z - Z.mean(0)) @ u
    dpc1, dcax = dip_p(Z[:, 0]), dip_p(cax)
    verdict = "MULTIMODAL" if (n_cl >= 2 and min(dpc1, dcax) < 0.05 and sil > 0.5) \
        else "weak" if (n_cl >= 2 or min(dpc1, dcax) < 0.05) else "UNIMODAL"
    return dict(hdbscan=n_cl, silhouette=float(sil), dip_pc1=dpc1,
                dip_centroid=dcax, verdict=verdict)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--k", type=int, default=80)
    ap.add_argument("--anchors", type=int, default=3)
    args = ap.parse_args()
    K = args.k

    priors = [
        {"name": "box2nm", "kind": "box", "box": 2.0},
        {"name": "box4nm(control)", "kind": "box", "box": 4.0},
        {"name": "box8nm", "kind": "box", "box": 8.0},
        {"name": "FMO-perturbed", "kind": "fmo", "p": {"sig_off": 25.0, "sig_diag": 50.0}},
    ]

    print("=" * 76, flush=True)
    print("D — prior 변경 robustness: 각 prior 에서 conditional 다봉인가?", flush=True)
    print("=" * 76, flush=True)

    all_res = []
    for pr in priors:
        print(f"\n■ prior = {pr['name']}  (N={args.n} 생성...)", flush=True)
        Hs, labs = gen(pr, args.n, seed=0)
        eta, tau = labs[:, 0], labs[:, 1]
        # 정규화 라벨 (KNN 용)
        L = (labs - labs.mean(0)) / (labs.std(0) + 1e-8)
        print(f"   η 분포: median {np.median(eta):.3f}  [{eta.min():.2f},{eta.max():.2f}]  "
              f"η>0.85 {(eta>0.85).mean()*100:.0f}%", flush=True)
        # 이 prior 의 '고효율' 영역(상위 30%)에서 τ 백분위로 anchor
        hi = np.where(eta >= np.percentile(eta, 70))[0]
        anchors = [int(hi[np.argmin(np.abs(tau[hi] - np.percentile(tau[hi], q)))])
                   for q in np.linspace(25, 75, args.anchors)]
        anchors = list(dict.fromkeys(anchors))
        verdicts = []
        for a in anchors:
            knn = np.argsort(np.linalg.norm(L - L[a], axis=1))[:K]
            traj = np.stack([site_traj(Hs[i]) for i in knn])
            r = test_condition(Hs[knn], traj)
            verdicts.append(r["verdict"])
            print(f"   anchor η={eta[a]:.3f} τ={tau[a]:.1f}ps: HDBSCAN {r['hdbscan']}, "
                  f"sil {r['silhouette']:.2f}, dip PC1 {r['dip_pc1']:.2f}/cen {r['dip_centroid']:.2f}"
                  f"  ▶ {r['verdict']}", flush=True)
        mm = verdicts.count("MULTIMODAL")
        summary = "MULTIMODAL 우세" if mm >= max(2, len(verdicts)//2 + 1) else \
                  "UNIMODAL 우세" if verdicts.count("UNIMODAL") >= len(verdicts)//2 + 1 else "혼재"
        print(f"   ▶▶ {pr['name']}: {verdicts} → {summary}", flush=True)
        all_res.append({"prior": pr["name"], "eta_median": float(np.median(eta)),
                        "anchors_verdicts": verdicts, "summary": summary})

    print("\n" + "=" * 76, flush=True)
    print("종합 — prior 별 conditional modality:", flush=True)
    for r in all_res:
        print(f"   {r['prior']:18s} (η med {r['eta_median']:.2f}): {r['summary']}  "
              f"{r['anchors_verdicts']}", flush=True)
    print("\n해석: 모든 prior 가 UNIMODAL 우세 → 결론이 prior 에 robust (4nm 한정 아님).", flush=True)
    print("      어떤 prior 가 MULTIMODAL → '특정 prior 에서만 메커니즘 분기' = 흥미로운 발견.",
          flush=True)

    with open("outputs/metrics/prior_ablation_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("\n저장: outputs/metrics/prior_ablation_results.json\nDONE.", flush=True)


if __name__ == "__main__":
    main()
