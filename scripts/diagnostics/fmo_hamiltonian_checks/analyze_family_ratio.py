"""Phase-2 — 진짜 mode-collapse 측정 (docs/09 의 'Diagnostic 1' 핵심).

흐름 (final_analysis.ipynb 의 main result 로직):
  1. η≈0.95 학습 데이터 N개 → 동역학 궤적 → pairwise dynamics distance
     → Agglomerative k=2 → baseline family ratio (docs: ~58.5 : 41.5)
     → 두 family centroid (flat trajectory 공간) 계산
  2. 표준 FMO H → 가장 가까운 centroid 로 분류 → 어느 family 가 'FMO-like' 인가
  3. 학습된 Diffusion 에서 η≈0.95 샘플 N개 → 동역학 궤적 → 같은 centroid 로 분류
     → Diffusion 의 family ratio
  4. 비교: |Diffusion ratio − 학습 ratio| 작으면 multimodal coverage, 크면 mode collapse

비용: (N_train + N_diff + 1) 회 시뮬레이션 (~0.2-0.4 s 각). 기본 300+300.

사용:  python analyze_family_ratio.py [--n 300]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

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

from src.cnf import load_data, normalize, denormalize_H, H27_to_matrix
from src.diffusion import build_diffusion, sample_diffusion
from src import simulator as sim

T_FLAT = sim.N_TIME * sim.DIM        # 1500 * 9


def traj_of(H7):
    """7×7 H → flat site-population trajectory (1500*9,), notebook 과 동일 정규화."""
    out = sim.simulate(H7, 35.0, return_traj=True)
    _, rho = out.pop("_traj")
    pops = np.real(np.diagonal(rho, axis1=1, axis2=2))    # (1500, 9)
    return pops.reshape(-1) / np.sqrt(T_FLAT)


def cluster_sizes(D, ks=(2, 3, 4, 5)):
    res = {}
    for k in ks:
        lbl = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                      linkage="average").fit_predict(D)
        res[k] = np.bincount(lbl, minlength=k).tolist()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300, help="train/diffusion 각 시뮬 샘플 수")
    ap.add_argument("--ckpt", default="outputs/checkpoints/diffusion_v1.pt")
    args = ap.parse_args()
    N = args.n

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}", flush=True)

    # ---- 데이터 + 모델 로드 ----
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model, sched = build_diffusion(device, cfg["feature_dim"], cfg["context_dim"],
                                   hidden=cfg["hidden"], layers=cfg["layers"],
                                   T=cfg["T"], schedule=cfg["schedule"])
    model.load_state_dict(ck["state_dict"])
    print(f"loaded {args.ckpt} (epochs={len(ck['hist']['train'])}, "
          f"final val {ck['hist']['val'][-1]:.4f})", flush=True)

    subset = np.where(np.abs(Y[:, 0] - 0.95) < 0.05)[0]
    rng = np.random.default_rng(42)

    # ---- 1. 학습 데이터 baseline (N개 η≈0.95) ----
    idx_tr = rng.choice(subset, size=min(N, len(subset)), replace=False)
    print(f"\n[1] 학습 데이터 {len(idx_tr)}개 시뮬레이션...", flush=True)
    traj_tr = np.stack([traj_of(H27_to_matrix(X[i])) for i in idx_tr])
    D_tr = squareform(pdist(traj_tr, metric="euclidean"))
    lbl_tr = AgglomerativeClustering(n_clusters=2, metric="precomputed",
                                     linkage="average").fit_predict(D_tr)
    cents = np.stack([traj_tr[lbl_tr == c].mean(0) for c in range(2)])    # (2, T_FLAT)
    sizes_tr = np.bincount(lbl_tr, minlength=2)
    print(f"    학습 k=2 family sizes: {sizes_tr.tolist()}  "
          f"({100*sizes_tr[0]/N:.1f} : {100*sizes_tr[1]/N:.1f})", flush=True)
    print(f"    학습 k-sweep: {cluster_sizes(D_tr)}", flush=True)

    def classify(traj):
        """가장 가까운 학습 family centroid 로 분류 (binary L2)."""
        d = np.linalg.norm(traj[None, :] - cents, axis=1)
        return int(np.argmin(d))

    # ---- 2. 표준 FMO 위치 ----
    traj_fmo = traj_of(sim.H_FMO_CM)
    fmo_fam = classify(traj_fmo)
    d_fmo = np.linalg.norm(traj_fmo[None, :] - cents, axis=1)
    print(f"\n[2] 표준 FMO → family {fmo_fam} "
          f"(centroid 거리 {d_fmo[0]:.4f} / {d_fmo[1]:.4f})", flush=True)
    maj_fam = int(np.argmax(sizes_tr))     # 학습에서 더 큰 family
    print(f"    학습 majority family = {maj_fam} "
          f"({'FMO 와 같음' if maj_fam == fmo_fam else 'FMO 와 다름'})", flush=True)

    # ---- 3. Diffusion 샘플 N개 (η≈0.95 조건) ----
    print(f"\n[3] Diffusion 샘플 {N}개 생성 + 시뮬레이션...", flush=True)
    idx_cond = rng.choice(subset, size=N, replace=False)
    S = np.stack([sample_diffusion(model, sched, Yn[i], n=1, device=device,
                                   n_steps=50, eta=0.0)[0] for i in idx_cond])
    S_real = denormalize_H(S, stats)
    traj_df = np.stack([traj_of(H27_to_matrix(h)) for h in S_real])
    fam_df = np.array([classify(t) for t in traj_df])
    sizes_df = np.bincount(fam_df, minlength=2)
    D_df = squareform(pdist(traj_df, metric="euclidean"))

    # ---- 4. 결과 ----
    print("\n" + "=" * 64, flush=True)
    print("FAMILY RATIO — mode collapse 측정 (η≈0.95)", flush=True)
    print("=" * 64, flush=True)
    r_tr = sizes_tr / sizes_tr.sum()
    r_df = sizes_df / sizes_df.sum()
    print(f"  학습 데이터 (baseline):  fam0 {100*r_tr[0]:5.1f}%   fam1 {100*r_tr[1]:5.1f}%",
          flush=True)
    print(f"  Diffusion (centroid분류): fam0 {100*r_df[0]:5.1f}%   fam1 {100*r_df[1]:5.1f}%",
          flush=True)
    delta = abs(r_df[maj_fam] - r_tr[maj_fam]) * 100
    print(f"  |Δ majority ratio| = {delta:.1f}%p", flush=True)
    print(f"  Diffusion 자체 k-sweep: {cluster_sizes(D_df)}", flush=True)
    print("\n  해석:", flush=True)
    if delta < 12:
        print("    → Diffusion 이 학습 분포의 두 family 를 모두 커버 (mode collapse 해결 방향)",
              flush=True)
    elif r_df[maj_fam] > 0.85:
        print("    → Diffusion 이 majority family 로 쏠림 = mode collapse 잔존", flush=True)
    else:
        print("    → 부분적 coverage", flush=True)
    print("\n  ⚠ 한계 (docs/09): 단일 seed · N=%d · binary centroid 분류 — "
          "robustness 위해 multi-seed/soft-assignment 권장." % N, flush=True)

    # ---- 저장 ----
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/cache", exist_ok=True)
    results = {
        "n_per_group": N,
        "train_family_sizes": sizes_tr.tolist(),
        "train_family_ratio": r_tr.tolist(),
        "train_ksweep": cluster_sizes(D_tr),
        "diffusion_family_sizes": sizes_df.tolist(),
        "diffusion_family_ratio": r_df.tolist(),
        "diffusion_ksweep": cluster_sizes(D_df),
        "fmo_family": fmo_fam,
        "majority_family": maj_fam,
        "fmo_is_majority": bool(maj_fam == fmo_fam),
        "delta_majority_pct": float(delta),
    }
    with open("outputs/metrics/family_ratio_results.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    np.savez_compressed("outputs/cache/family_ratio_cache.npz",
                        traj_tr=traj_tr.astype(np.float32), lbl_tr=lbl_tr.astype(np.int8),
                        traj_df=traj_df.astype(np.float32), fam_df=fam_df.astype(np.int8),
                        traj_fmo=traj_fmo.astype(np.float32), cents=cents.astype(np.float32))
    print("\n저장: outputs/metrics/family_ratio_results.json", flush=True)
    print("저장: outputs/cache/family_ratio_cache.npz", flush=True)
    print("\nDONE.", flush=True)


if __name__ == "__main__":
    main()
