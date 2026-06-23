"""Q1 결정적 측정 — fixed-condition multimodality (docs/09 Diagnostic 2, 과학적 버전).

질문: "같은 물리 조건(η,τ,ipr,purity,c_l1)을 내는 H 들이 *한 가지* 동역학 메커니즘인가,
       *여러* 메커니즘(multimodal)인가?" = ENAQT 분포가 여러 개인가의 정확한 정의.

marginal(η 밴드 평균) 측정의 오염 — '조건이 달라서 H가 다른 것' — 을 제거하기 위해
*한 조건점의 최근접 이웃만* 모아 그 안에서 multimodality 를 본다.

흐름:
  1. η≈0.95 에서 τ_transfer 가 다른 anchor M개 선택 (효율 같고 속도 다른 점들)
  2. 각 anchor 의 5-d 라벨 공간 최근접 이웃 K개 (전체 데이터에서)
  3. 이웃 H 들의 7-site 동역학 궤적 시뮬 → pairwise dist → silhouette + GMM BIC
  4. anchor 전반의 일관된 판정: conditional 이 unimodal 인가 multimodal 인가

사용:  python fixed_condition_test.py [--k 100] [--anchors 5]
"""
import argparse
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

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from src.cnf import load_data, normalize, H27_to_matrix
from src import simulator as sim


def site_traj(H7):
    """7×7 H → flat 7-site population trajectory (1500*7,)."""
    out = sim.simulate(H7, 35.0, return_traj=True)
    _, rho = out.pop("_traj")
    pops = np.real(np.diagonal(rho, axis1=1, axis2=2))[:, :7]   # (1500, 7)
    return pops.reshape(-1)


def best_gmm_k(Xflat, n_pc=10):
    Z = PCA(n_components=min(n_pc, Xflat.shape[0] - 1)).fit_transform(Xflat)
    bics = []
    for k in (1, 2, 3, 4):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=0, n_init=3).fit(Z)
        bics.append(gm.bic(Z))
    return int(np.argmin(bics)) + 1, bics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100, help="anchor 당 최근접 이웃 수")
    ap.add_argument("--anchors", type=int, default=5)
    ap.add_argument("--eta", type=float, default=0.95, help="고정할 목표 효율 η")
    ap.add_argument("--eta_tol", type=float, default=0.03)
    args = ap.parse_args()
    K = args.k

    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    L = Yn[:, :5]                                   # 5-d 정규화 라벨 (η,τ,ipr,purity,c_l1)
    eta, tau = Y[:, 0], Y[:, 1]

    # η≈target subset 에서 τ_transfer 가 서로 다른 anchor 선택
    subset = np.where(np.abs(eta - args.eta) < args.eta_tol)[0]
    tau_sub = tau[subset]
    qs = np.linspace(10, 90, args.anchors)
    anchor_idx = [subset[np.argmin(np.abs(tau_sub - np.percentile(tau_sub, q)))] for q in qs]
    anchor_idx = list(dict.fromkeys(anchor_idx))    # dedupe

    print(f"전체 N={len(X)}, η≈{args.eta}(±{args.eta_tol}) subset={len(subset)}, "
          f"anchor {len(anchor_idx)}개\n", flush=True)
    print("=" * 70, flush=True)
    print(f"Q1 결정적: η≈{args.eta} 고정 조건(5-d label) 안에서 동역학이 갈라지는가? "
          f"(K={K} 이웃)", flush=True)
    print("=" * 70, flush=True)

    # 이웃 전부 모아 dedupe 후 한 번에 시뮬 (anchor 간 중복 절약)
    neigh = {}
    for a in anchor_idx:
        d = np.linalg.norm(L - L[a], axis=1)
        knn = np.argsort(d)[:K]
        neigh[a] = knn
    all_idx = sorted(set(int(i) for knn in neigh.values() for i in knn))
    print(f"시뮬 대상 고유 H = {len(all_idx)}개 ...", flush=True)
    traj_cache = {}
    for n, i in enumerate(all_idx):
        traj_cache[i] = site_traj(H27_to_matrix(X[i]))
        if (n + 1) % 100 == 0:
            print(f"  {n+1}/{len(all_idx)}", flush=True)

    results = []
    for a in anchor_idx:
        knn = neigh[a]
        Xf = np.stack([traj_cache[int(i)] for i in knn])
        D = squareform(pdist(Xf, metric="euclidean"))
        # 이 조건점의 '고정도': 이웃의 라벨 spread
        eta_sd = eta[knn].std(); tau_sd = tau[knn].std()
        gk, bics = best_gmm_k(Xf)
        sils = {}
        for k in (2, 3):
            for link in ("average", "ward"):
                if link == "ward":
                    lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xf)
                else:
                    lbl = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                                  linkage="average").fit_predict(D)
                sils[f"{link[:3]}_k{k}"] = (silhouette_score(D, lbl, metric="precomputed"),
                                            np.bincount(lbl, minlength=k).tolist())
        max_sil = max(v[0] for v in sils.values())
        verdict = ("MULTIMODAL" if (gk >= 2 and max_sil > 0.5)
                   else "UNIMODAL" if (gk == 1 or max_sil < 0.35)
                   else "AMBIGUOUS")
        print(f"\n── anchor (idx {a}):  η={eta[a]:.3f} τ={tau[a]:.2f}ps  "
              f"[이웃 η±{eta_sd:.3f}, τ±{tau_sd:.2f}]", flush=True)
        print(f"     GMM best-k = {gk}   max silhouette = {max_sil:.2f}", flush=True)
        for key, (s, sz) in sils.items():
            print(f"     {key}: sil {s:+.2f}  sizes {sz}", flush=True)
        print(f"     ▶ {verdict}", flush=True)
        results.append({"idx": int(a), "eta": float(eta[a]), "tau": float(tau[a]),
                        "neigh_eta_sd": float(eta_sd), "neigh_tau_sd": float(tau_sd),
                        "gmm_best_k": gk, "max_silhouette": float(max_sil),
                        "verdict": verdict,
                        "sils": {k: [float(v[0]), v[1]] for k, v in sils.items()}})

    # 종합
    verdicts = [r["verdict"] for r in results]
    print("\n" + "=" * 70, flush=True)
    print("종합 판정 (anchor 별):", verdicts, flush=True)
    n_multi = verdicts.count("MULTIMODAL")
    n_uni = verdicts.count("UNIMODAL")
    print(f"  MULTIMODAL {n_multi} / UNIMODAL {n_uni} / AMBIGUOUS "
          f"{len(verdicts)-n_multi-n_uni}  (anchor {len(verdicts)}개)", flush=True)
    if n_multi >= max(2, len(verdicts) // 2):
        concl = ("→ 같은 조건 안에서도 동역학이 갈라짐 = conditional MULTIMODAL. "
                 "ENAQT '여러 분포' 주장 근거 있음 → 모델이 둘 다 내야 함 (collapse 서사 유효).")
    elif n_uni >= max(2, len(verdicts) // 2):
        concl = ("→ 같은 조건 안에서는 단일 메커니즘 = conditional ~UNIMODAL. "
                 "ENAQT '여러 분포'는 주로 '조건이 달라서'였음. 'collapse 서사'는 약화 "
                 "(한 family 만 내는 게 정답).")
    else:
        concl = "→ 혼재/모호 — K↑, anchor↑, multi-seed 필요."
    print(concl, flush=True)
    print("\n  ⚠ marginal 측정과의 핵심 차이: 여기선 '조건 고정'(이웃 η/τ spread 작음) — "
          "순수 conditional multimodality 만 본다.", flush=True)

    os.makedirs("outputs/metrics", exist_ok=True)
    out = f"outputs/metrics/fixed_condition_results_eta{int(round(args.eta*100)):03d}.json"
    with open(out, "w") as f:
        json.dump({"eta": args.eta, "k": K, "anchors": results, "verdicts": verdicts,
                   "conclusion": concl}, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out}\nDONE.", flush=True)


if __name__ == "__main__":
    main()
