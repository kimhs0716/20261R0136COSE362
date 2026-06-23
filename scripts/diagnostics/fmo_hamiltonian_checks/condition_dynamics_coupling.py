"""Can scalar conditions control dynamic trajectory diversity? (team verification 1+2, grouping-free)

Measures, WITHOUT any clustering (avoids the circular "dynamic groups differ in dynamics"):
  1. Condition entanglement — correlation + effective dimensionality of the 5 scalar conditions.
     If they lie on a low-dim manifold, you cannot dial them independently.
  2. R²(trajectory ~ conditions) — how much of the trajectory variation the scalar conditions
     explain (linear + nonlinear), as the condition set grows. Low => conditions underdetermine
     the trajectory => cannot control dynamic diversity by conditions alone.
  3. Residual trajectory variance at fixed condition — KNN in condition space; trajectory spread
     among condition-neighbors / global trajectory spread. High => conditions don't pin the path.

Uses dataset_full.npz (same simulator/sampler lineage as the team repo) which already stores the
5 scalar labels (incl. c_l1, IPR) AND the trajectory snapshots pop_t — no new simulation needed.
"""
import os, sys
from pathlib import Path
import numpy as np

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors

COND_NAMES = ["eta", "tau_transfer", "ipr", "purity", "c_l1"]


def load():
    Cs, Ps = [], []
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        d = np.load(p)
        Cs.append(np.stack([d[k] for k in COND_NAMES], axis=1).astype(float))
        Ps.append(d["pop_t"][:, :, :7].reshape(len(d["pop_t"]), -1).astype(float))  # (N, 7t*7site=49)
    C = np.concatenate(Cs); P = np.concatenate(Ps)
    ok = np.isfinite(C).all(1) & np.isfinite(P).all(1)
    return C[ok], P[ok]


def r2_for(Cset, Pstd, Pca10, rng, n_rf=15000):
    """linear R² (full traj) + nonlinear RF R² (on 10 traj-PCs)."""
    Cstd = StandardScaler().fit_transform(Cset)
    lin = LinearRegression().fit(Cstd, Pstd).score(Cstd, Pstd)   # multioutput uniform avg R²
    idx = rng.choice(len(Cstd), min(n_rf, len(Cstd)), replace=False)
    rf = RandomForestRegressor(n_estimators=120, max_depth=12, n_jobs=-1, random_state=0)
    rf.fit(Cstd[idx], Pca10[idx])
    # variance-weighted R² across the 10 PCs
    from sklearn.metrics import r2_score
    pred = rf.predict(Cstd[idx])
    var = Pca10[idx].var(0)
    r2_per = np.array([r2_score(Pca10[idx, j], pred[:, j]) for j in range(Pca10.shape[1])])
    rf_r2 = float(np.average(np.clip(r2_per, -1, 1), weights=var))
    return float(lin), rf_r2


def main():
    C, P = load()
    rng = np.random.default_rng(0)
    print(f"N={len(C)}  conditions={COND_NAMES}  trajectory dim={P.shape[1]} (7 times x 7 sites)\n")

    # ---- 1. condition entanglement ----
    print("=" * 70)
    print("1. CONDITION ENTANGLEMENT (can you dial them independently?)")
    print("=" * 70)
    Cz = StandardScaler().fit_transform(C)
    corr = np.corrcoef(Cz.T)
    print("   correlation matrix (5 scalar conditions):")
    print("        " + "  ".join(f"{n[:6]:>7}" for n in COND_NAMES))
    for i, n in enumerate(COND_NAMES):
        print(f"   {n[:6]:>6} " + "  ".join(f"{corr[i,j]:+7.2f}" for j in range(5)))
    # effective dimensionality
    ev = PCA().fit(Cz).explained_variance_ratio_
    cum = np.cumsum(ev)
    eff90 = int(np.searchsorted(cum, 0.90) + 1)
    print(f"\n   PCA explained var ratio: {[f'{x:.2f}' for x in ev]}")
    print(f"   effective dimensionality: {eff90}/5 PCs reach 90% variance")
    iu = np.triu_indices(5, 1)
    big = [(COND_NAMES[i], COND_NAMES[j], corr[i, j]) for i, j in zip(*iu) if abs(corr[i, j]) > 0.4]
    print(f"   strong pairwise correlations (|r|>0.4): {[(a,b,round(r,2)) for a,b,r in big]}")

    # ---- 2. R²(trajectory ~ conditions) ----
    print("\n" + "=" * 70)
    print("2. R²(trajectory ~ conditions) — how much of dynamics do conditions explain?")
    print("=" * 70)
    Pstd = StandardScaler().fit_transform(P)
    Pca10 = PCA(n_components=10).fit_transform(Pstd)
    print(f"   (trajectory: 10 PCs capture {PCA(n_components=10).fit(Pstd).explained_variance_ratio_.sum():.2f} of variance)")
    subsets = [("eta only", [0]), ("eta+tau", [0, 1]), ("all 5 (eta,tau,ipr,purity,c_l1)", [0, 1, 2, 3, 4])]
    for name, cols in subsets:
        lin, rf = r2_for(C[:, cols], Pstd, Pca10, rng)
        print(f"   {name:32s}: linear R²={lin:.3f}   nonlinear(RF) R²={rf:.3f}")
    print("   → 낮을수록 조건이 trajectory를 덜 설명 = 조건만으로 dynamic diversity 통제 어려움")

    # ---- 3. residual trajectory variance at fixed condition ----
    print("\n" + "=" * 70)
    print("3. RESIDUAL trajectory variance at FIXED condition (KNN in 5-d condition space)")
    print("=" * 70)
    global_var = P.var(0).sum()
    for K in (30, 80, 200):
        nn = NearestNeighbors(n_neighbors=K).fit(Cz)
        anchors = rng.choice(len(Cz), 400, replace=False)
        _, nbr = nn.kneighbors(Cz[anchors])
        within = np.array([P[nbr[m]].var(0).sum() for m in range(len(anchors))])
        frac = float(np.median(within) / global_var)
        # how tight is the condition neighborhood?
        cond_spread = np.median([np.linalg.norm(Cz[nbr[m]] - Cz[nbr[m]].mean(0), axis=1).mean()
                                 for m in range(len(anchors))])
        print(f"   K={K:3d}: residual traj variance / global = {frac:.3f}  "
              f"(condition neighborhood spread {cond_spread:.2f} std-units)")
    print("   → 1에 가까우면 조건을 고정해도 trajectory가 거의 그대로 다양 = 조건이 path를 안 묶음")
    print("     0에 가까우면 조건이 trajectory를 거의 결정")

    print("\n" + "=" * 70)
    print("판정 가이드 (두 갈래 결정):")
    print("  · R²(5d) 높고(>0.8) residual 낮으면 → 조건이 dynamics 통제 가능 (branch 1: 조건 조절 + 얽힘 보정)")
    print("  · R²(5d) 중간/낮고 residual 큼 + 조건 얽힘 → scalar 조건 부족 (branch 2: PINN/dynamics 신호)")
    print("=" * 70)


if __name__ == "__main__":
    main()
