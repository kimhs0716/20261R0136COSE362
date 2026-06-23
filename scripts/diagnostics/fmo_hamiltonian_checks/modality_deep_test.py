"""①②③ — fixed-condition 안에서 데이터 multimodality 를 *형식 검정 + 물리 descriptor* 로.

현재 약점 보완:
  ① Dip test (Hartigan)  → GMM BIC 의 '비대칭=다봉' 착각 제거. 단봉성 null 의 형식 검정.
  ② KDE density valley    → multimodal 의 정의(봉우리 사이 저밀도 골) 직접 확인.
  ③ 물리 pathway descriptor → 27-d 거리 함정 우회, 해석가능한 1-D 로 메커니즘 분리 시각화.

모두 *한 조건점의 최근접 이웃 100개* 안에서 (조건섞임 제거). 대조군(η0.95 빠른=unimodal 기대)과
MM 후보(η0.5 느린)를 나란히.

사용:  python modality_deep_test.py
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
from scipy.stats import gaussian_kde
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import diptest

from src.cnf import load_data, normalize, H27_to_matrix
from src import simulator as sim

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz   # numpy 1.x/2.x 호환

TLIST = np.linspace(0.0, sim.T_MAX, sim.N_TIME)
INTER_IDX = [1, 3, 4, 5, 6]                 # 중간 site 2,4,5,6,7 (입력1·sink인접3 제외)


def full_traj(H7):
    out = sim.simulate(H7, 35.0, return_traj=True)
    _, rho = out.pop("_traj")
    return np.real(np.diagonal(rho, axis1=1, axis2=2))      # (1500, 9)


def descriptors(pops):
    """물리 pathway descriptor — 해석가능한 스칼라들."""
    sites = pops[:, :7]
    integ = trapz(sites, TLIST, axis=0)                    # (7,) 시간적분 점유
    I_inter = integ[INTER_IDX]
    relay_site = INTER_IDX[int(np.argmax(I_inter))] + 1     # 지배 relay site (1-based)
    direct_frac = float(integ[1] / (I_inter.sum() + 1e-12)) # site2 (직접경로) 점유 분율
    trap = pops[:, 7]; fin = float(trap[-1])
    if fin > 1e-6:
        above = np.where(trap >= 0.5 * fin)[0]
        sink_thalf = float(TLIST[above[0]]) if len(above) else 50.0
    else:
        sink_thalf = 50.0
    return dict(direct_frac=direct_frac, site2_int=float(integ[1]),
                sink_thalf=sink_thalf, relay_site=relay_site)


def dip_of(x):
    x = np.asarray(x, float)
    if x.std() < 1e-9 or len(np.unique(x)) < 4:
        return 0.0, 1.0
    d, p = diptest.diptest(x)
    return float(d), float(p)


def kde_modes(x, grid=400):
    """KDE 봉우리 수 + valley depth (두 봉 사이 최저 / 작은 봉)."""
    x = np.asarray(x, float)
    if x.std() < 1e-9:
        return 1, 0.0
    xs = np.linspace(x.min(), x.max(), grid)
    dens = gaussian_kde(x)(xs)
    # 국소최대
    peaks = [i for i in range(1, grid - 1) if dens[i] > dens[i-1] and dens[i] >= dens[i+1]]
    if len(peaks) < 2:
        return len(peaks) or 1, 0.0
    # 가장 큰 두 봉 사이 valley depth
    p2 = sorted(peaks, key=lambda i: dens[i])[-2:]
    lo, hi = min(p2), max(p2)
    valley = dens[lo:hi+1].min()
    small_peak = min(dens[p2[0]], dens[p2[1]])
    depth = float(1 - valley / (small_peak + 1e-12))        # 0=골없음, 1=완전분리
    return len(peaks), depth


def main():
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    L = Yn[:, :5]
    eta, tau = Y[:, 0], Y[:, 1]
    K = 100

    # anchor: (목표 η, τ 백분위, 설명)
    specs = [(0.95, 10, "η0.95·빠른 (unimodal 대조군)"),
             (0.95, 90, "η0.95·느린"),
             (0.50, 50, "η0.50·중간"),
             (0.50, 90, "η0.50·느린 (MM 후보)")]
    anchors = []
    for teta, q, desc in specs:
        sub = np.where(np.abs(eta - teta) < 0.03)[0]
        a = sub[np.argmin(np.abs(tau[sub] - np.percentile(tau[sub], q)))]
        anchors.append((int(a), teta, q, desc))

    os.makedirs("outputs/figures/modality_deep", exist_ok=True)
    print("=" * 74, flush=True)
    print("①②③ fixed-condition 데이터 modality — dip test · KDE valley · 물리 descriptor",
          flush=True)
    print("=" * 74, flush=True)

    all_res = []
    for n, (a, teta, q, desc) in enumerate(anchors):
        knn = np.argsort(np.linalg.norm(L - L[a], axis=1))[:K]
        pops = [full_traj(H27_to_matrix(X[i])) for i in knn]
        D = [descriptors(p) for p in pops]
        direct = np.array([d["direct_frac"] for d in D])
        thalf = np.array([d["sink_thalf"] for d in D])
        relays = np.array([d["relay_site"] for d in D])
        s2 = np.array([d["site2_int"] for d in D])

        # 데이터-구동 판별축: 7-site 궤적 PC1 + ward-centroid 연결선 투영
        flat = np.array([p[:, :7].reshape(-1) for p in pops])
        pc1 = PCA(n_components=1).fit_transform(flat).ravel()
        lbl = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(flat)
        c0, c1 = flat[lbl == 0].mean(0), flat[lbl == 1].mean(0)
        u = (c1 - c0); u = u / (np.linalg.norm(u) + 1e-12)
        cproj = (flat - flat.mean(0)) @ u

        # ① dip test (여러 축)
        axes = {"direct_frac": direct, "sink_thalf": thalf, "site2_int": s2,
                "traj_PC1": pc1, "centroid_axis": cproj}
        dips = {k: dip_of(v) for k, v in axes.items()}

        # ② 가장 다봉스러운 축에서 KDE valley
        best_axis = min(dips, key=lambda k: dips[k][1])
        nmodes, depth = kde_modes(axes[best_axis])

        # ③ relay site 분포 (categorical)
        uniq, cnt = np.unique(relays, return_counts=True)
        relay_dist = {int(s): int(c) for s, c in zip(uniq, cnt)}
        top_relay = int(uniq[np.argmax(cnt)]); top_frac = cnt.max() / K

        print(f"\n── anchor {n}: {desc}  (η={eta[a]:.3f} τ={tau[a]:.2f}ps)", flush=True)
        print(f"   ① dip test p-value (p<0.05 = 다봉):", flush=True)
        for k, (dd, pp) in dips.items():
            flag = "  ◀ 다봉!" if pp < 0.05 else ""
            print(f"        {k:14s} dip={dd:.3f}  p={pp:.3f}{flag}", flush=True)
        print(f"   ② KDE on '{best_axis}': 봉우리 {nmodes}개, valley depth {depth:.2f} "
              f"(>0.3 = 뚜렷한 골)", flush=True)
        print(f"   ③ relay site 분포: {relay_dist}  → 지배 site {top_relay} ({100*top_frac:.0f}%)",
              flush=True)
        sig = [k for k, (_, pp) in dips.items() if pp < 0.05]
        verdict = ("MULTIMODAL" if (len(sig) >= 2 or (nmodes >= 2 and depth > 0.3))
                   else "UNIMODAL" if not sig else "WEAK")
        print(f"   ▶ {verdict}  (dip 유의 축 {len(sig)}개: {sig})", flush=True)

        # 그림: direct_frac / PC1 / relay 막대
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        ax[0].hist(direct, bins=25, color="steelblue", alpha=0.8)
        ax[0].set_title(f"direct_frac (site2 share)\ndip p={dips['direct_frac'][1]:.3f}")
        ax[0].set_xlabel("fraction of intermediate occupation on site 2")
        ax[1].hist(cproj, bins=25, color="indianred", alpha=0.8)
        ax[1].set_title(f"centroid-axis projection\ndip p={dips['centroid_axis'][1]:.3f}")
        ax[1].set_xlabel("projection onto ward-centroid line")
        ax[2].bar([str(s) for s in uniq], cnt, color="seagreen", alpha=0.8)
        ax[2].set_title("dominant relay site")
        ax[2].set_xlabel("site")
        fig.suptitle(f"anchor {n}: {desc}  (eta={eta[a]:.3f}, tau={tau[a]:.1f}ps)  -> {verdict}")
        fig.tight_layout()
        fp = f"outputs/figures/modality_deep/anchor{n}.png"
        fig.savefig(fp, dpi=110); plt.close(fig)

        all_res.append({"anchor": n, "desc": desc, "eta": float(eta[a]), "tau": float(tau[a]),
                        "dip": {k: [v[0], v[1]] for k, v in dips.items()},
                        "kde_axis": best_axis, "kde_nmodes": nmodes, "kde_valley_depth": depth,
                        "relay_dist": relay_dist, "top_relay": top_relay,
                        "verdict": verdict, "dip_sig_axes": sig})

    print("\n" + "=" * 74, flush=True)
    print("종합:", flush=True)
    for r in all_res:
        print(f"   {r['desc']:28s} → {r['verdict']:11s} "
              f"(dip유의 {len(r['dip_sig_axes'])}, KDE봉 {r['kde_nmodes']}/valley "
              f"{r['kde_valley_depth']:.2f}, relay {r['top_relay']})", flush=True)
    print("\n해석: 대조군(η0.95빠른)이 UNIMODAL 이고 MM후보(η0.5느린)도 UNIMODAL 이면 → "
          "데이터 unimodal 강력. MM후보만 다봉이면 → '저효율·느린 구간 메커니즘 분기' 단서.",
          flush=True)

    with open("outputs/metrics/modality_deep_results.json", "w") as f:
        json.dump(all_res, f, indent=2, ensure_ascii=False)
    print("\n저장: outputs/metrics/modality_deep_results.json + "
          "outputs/figures/modality_deep/*.png\nDONE.", flush=True)


if __name__ == "__main__":
    main()
