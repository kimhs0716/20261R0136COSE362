"""A. Diversity-collapse diagnostic — does the generative model cover the conditional spread?

At a FIXED condition c*, compares the model's sample distribution to the DATA's conditional
distribution p(H|c*):
  - spread ratio (model / data) in parameter (27-d H) and trajectory (site-pop) space.
      ~1 = covers the spread;  <<1 = variance collapse (model concentrates toward the mean).
  - coverage (recall) / precision in trajectory space.
  - support: model-H nearest-train-distance vs data's (are generated H in-distribution).

win-win: covers => "generative model captures the broad conditional (NN gives one point)".
         collapses => variance-collapse, motivates dynamics-aware PINN.

Uses the trained diffusion (outputs/checkpoints/diffusion_v1.pt). Data trajectories from stored
pop_t (7 snapshots); model-H trajectories via fresh simulation at the same 7 times.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import torch

for _s in (sys.stdout, sys.stderr):
    try: _s.reconfigure(encoding="utf-8")
    except Exception: pass

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

from sklearn.neighbors import NearestNeighbors
from src.cnf import load_data, normalize, denormalize_H, H27_to_matrix
from src.diffusion import build_diffusion, sample_diffusion
from src import simulator as sim


def spread(X):
    """sqrt(total variance) = sqrt(trace(cov))."""
    return float(np.sqrt(np.sum(X.var(0))))


def coverage(data, model, radius):
    """recall = frac data covered by some model pt; precision = frac model near some data pt."""
    nn_d = NearestNeighbors(n_neighbors=1).fit(model)
    d_to_model, _ = nn_d.kneighbors(data)
    recall = float((d_to_model[:, 0] <= radius).mean())
    nn_m = NearestNeighbors(n_neighbors=1).fit(data)
    d_to_data, _ = nn_m.kneighbors(model)
    precision = float((d_to_data[:, 0] <= radius).mean())
    return recall, precision


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/diffusion_v1.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    # stored data trajectories (site pop at 7 times), same order as load_data
    pops, times = [], None
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        d = np.load(p); pops.append(d["pop_t"][:, :, :7].reshape(len(d["pop_t"]), -1)); times = d["times"]
    POP = np.concatenate(pops).astype(float)            # (N, 49) site pop at 7 times
    TIMES = np.asarray(times, float)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model, sched = build_diffusion(device, cfg["feature_dim"], cfg["context_dim"],
                                   hidden=cfg["hidden"], layers=cfg["layers"], T=cfg["T"])
    model.load_state_dict(ck["state_dict"])
    _v = ck["hist"].get("val", ck["hist"].get("mse", [float("nan")]))[-1]
    print(f"loaded {args.ckpt} (last {_v:.3f}); N={len(X)}", flush=True)

    eta = Y[:, 0]
    subset = np.where(np.abs(eta - 0.95) < 0.03)[0]
    rng = np.random.default_rng(0)
    nn33 = NearestNeighbors(n_neighbors=100).fit(Yn)     # data conditional via 33-d KNN
    nn_supp = NearestNeighbors(n_neighbors=2).fit(X)      # support: nearest-train-distance (param)

    K, NM = 100, 80
    anchors = subset[rng.choice(len(subset), 4, replace=False)]
    results = []
    print("\n" + "=" * 78, flush=True)
    print("DIVERSITY-COLLAPSE: model spread/coverage vs data conditional p(H|c*)", flush=True)
    print("=" * 78, flush=True)
    for a in anchors:
        c = Yn[a]
        # data conditional (33-d neighbors)
        _, nbr = nn33.kneighbors(c[None]); nbr = nbr[0]
        dX, dPOP = X[nbr], POP[nbr]
        # model conditional (sample at c*, simulate, extract site pop at the 7 data times)
        S = sample_diffusion(model, sched, c, n=NM, device=device, n_steps=50, eta=0.0)
        mX = denormalize_H(S, stats)
        mPOP = np.zeros((NM, 49))
        for i, h in enumerate(mX):
            out = sim.simulate(H27_to_matrix(h), 35.0, return_traj=True)
            tl, rho = out["_traj"]
            mPOP[i] = _interp_pop(rho, tl, TIMES)

        # --- parameter space (27-d H) ---
        mu, sd = dX.mean(0), dX.std(0) + 1e-9
        r_param = spread((mX - mu) / sd) / spread((dX - mu) / sd)
        # --- trajectory space (49-d site pop) ---
        muP, sdP = dPOP.mean(0), dPOP.std(0) + 1e-9
        dPz, mPz = (dPOP - muP) / sdP, (mPOP - muP) / sdP
        r_traj = spread(mPz) / spread(dPz)
        # coverage in trajectory space
        rad = np.median(NearestNeighbors(n_neighbors=2).fit(dPz).kneighbors(dPz)[0][:, 1])
        recall, precision = coverage(dPz, mPz, rad)
        # support: model nearest-train-distance vs data's
        msupp = np.median(nn_supp.kneighbors(mX)[0][:, 0])
        dsupp = np.median(nn_supp.kneighbors(dX)[0][:, 1])

        print(f"\n── anchor η={eta[a]:.3f} τ={Y[a,1]:.1f}ps", flush=True)
        print(f"   spread ratio (model/data):  param {r_param:.2f}   trajectory {r_traj:.2f}", flush=True)
        print(f"   trajectory coverage: recall {recall:.2f}  precision {precision:.2f}", flush=True)
        print(f"   support: model nearest-train {msupp:.1f} vs data {dsupp:.1f} cm⁻¹", flush=True)
        results.append(dict(eta=float(eta[a]), r_param=r_param, r_traj=r_traj,
                            recall=recall, precision=precision, supp_model=float(msupp),
                            supp_data=float(dsupp)))

    rp = np.mean([r["r_param"] for r in results]); rt = np.mean([r["r_traj"] for r in results])
    rc = np.mean([r["recall"] for r in results]); pr = np.mean([r["precision"] for r in results])
    print("\n" + "=" * 78, flush=True)
    print(f"평균: spread ratio param {rp:.2f} / traj {rt:.2f}  | recall {rc:.2f} precision {pr:.2f}", flush=True)
    print("판정:", flush=True)
    if rt > 0.8 and rc > 0.7:
        print("  → 모델이 조건부 spread를 잘 덮음 = NO collapse. '생성 모델이 분포를 준다'(NN은 점) 긍정.", flush=True)
    elif rt < 0.5 or rc < 0.5:
        print("  → variance COLLAPSE (모델이 평균으로 쏠림). PINN/dynamics 신호 정당화.", flush=True)
    else:
        print("  → 부분 coverage. 일부 collapse — PINN으로 보강 여지.", flush=True)
    print("=" * 78, flush=True)
    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/diversity_collapse_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print("saved: outputs/metrics/diversity_collapse_results.json", flush=True)


def _interp_pop(rho, tl, TIMES):
    diag = np.real(np.diagonal(rho, axis1=1, axis2=2))[:, :7]   # (T,7)
    out = np.array([[np.interp(t, tl, diag[:, s]) for s in range(7)] for t in TIMES])
    return out.reshape(-1)


if __name__ == "__main__":
    main()
