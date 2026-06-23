"""Q1 직접 검정 — η≈0.95 학습 데이터 p(H|η≈0.95) 가 정말 multimodal 인가?

family_ratio_cache.npz 의 traj_tr (300 학습 궤적, 9-state) 재사용 — 새 시뮬 없음.
여러 각도로 modality 검정:
  1. 9-state vs 7-site-only 거리 (trap/loss 단조 채널이 metric 을 흐리는지)
  2. Agglomerative(average/ward) silhouette across k — discrete cluster 분리도
  3. GMM BIC across k (PCA 10d) — k=1 이 이기면 unimodal
  4. PCA 분산 구조 — 'blob 위 연속체' vs '분리된 덩어리'
"""
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

from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

cache = np.load("outputs/cache/family_ratio_cache.npz")
traj = cache["traj_tr"]                       # (300, 1500*9) normalized flat
N = len(traj)
T9 = traj.reshape(N, 1500, 9)
sites = T9[:, :, :7].reshape(N, -1)           # 7-site only
print(f"N={N}, 9-state dim={traj.shape[1]}, 7-site dim={sites.shape[1]}\n", flush=True)


def agglo_silhouette(Xflat, tag):
    D = squareform(pdist(Xflat, metric="euclidean"))
    print(f"[{tag}]  Agglomerative silhouette (높을수록 분리 뚜렷; 보통 >0.5 강함):", flush=True)
    for linkage in ("average", "ward"):
        line = f"   {linkage:8s}"
        for k in (2, 3, 4, 5, 6):
            if linkage == "ward":
                lbl = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xflat)
            else:
                lbl = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                              linkage="average").fit_predict(D)
            sil = silhouette_score(D, lbl, metric="precomputed")
            sizes = np.bincount(lbl, minlength=k)
            line += f"  k{k}:{sil:+.2f}{sizes.tolist()}"
        print(line, flush=True)
    print(flush=True)


def gmm_bic(Xflat, tag, n_pc=10):
    Z = PCA(n_components=min(n_pc, Xflat.shape[0] - 1)).fit_transform(Xflat)
    print(f"[{tag}]  GMM BIC across k (PCA {Z.shape[1]}d; 최소 BIC = 최적 k):", flush=True)
    bics = []
    for k in (1, 2, 3, 4, 5):
        gm = GaussianMixture(n_components=k, covariance_type="full",
                             random_state=0, n_init=3).fit(Z)
        bics.append(gm.bic(Z))
    best = int(np.argmin(bics)) + 1
    print("   " + "  ".join(f"k{i+1}:{b:9.0f}" for i, b in enumerate(bics)), flush=True)
    print(f"   → 최적 k = {best}  ({'UNIMODAL' if best == 1 else f'{best}-modal 선호'})\n",
          flush=True)
    return best


def pca_structure(Xflat, tag):
    p = PCA(n_components=10).fit(Xflat)
    ev = p.explained_variance_ratio_
    print(f"[{tag}]  PCA 분산비 (상위 5): "
          f"{['%.2f' % x for x in ev[:5]]}  누적5={ev[:5].sum():.2f}", flush=True)
    print(flush=True)


print("=" * 64, flush=True)
print("Q1: η≈0.95 학습 분포는 multimodal 인가? (300 sample, 단일 seed)", flush=True)
print("=" * 64 + "\n", flush=True)

for Xf, tag in ((traj, "9-state"), (sites, "7-site")):
    agglo_silhouette(Xf, tag)
    gmm_bic(Xf, tag)
    pca_structure(Xf, tag)

print("판정 가이드:", flush=True)
print("  · GMM k=1 이 이기고 silhouette<0.5 → 사실상 unimodal blob (multiple 주장 약함)", flush=True)
print("  · GMM k≥2 가 이기고 silhouette>0.5 인 k 존재 → 진짜 분리된 family 근거", flush=True)
print("  · average 만 큰 sil 인데 ward 는 아님 → chaining artifact (분리 아님)", flush=True)
