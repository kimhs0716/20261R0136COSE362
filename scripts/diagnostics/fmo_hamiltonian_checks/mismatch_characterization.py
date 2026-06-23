"""Where does the model's conditional dynamics diverge from the data conditional?

Follows diversity_collapse_test.py (model conditional is broad but mis-aligned, recall ~0.27).
Here we localize the mismatch in (time x state) space at fixed conditions:
  - mean-shift[t,s] = |mean_model - mean_data| / std_data   (systematic bias)
  - var-ratio[t,s]  = std_model / std_data                  (over/under scatter)
averaged over anchors. The 33-d condition explicitly pins site populations at t=1,5,10 ps
(POP_T_IDX), so we flag conditioned vs free times: if mismatch is low at conditioned times and
high at free/late times, the model satisfies the explicit condition but not the rest of the
dynamics -> that free/late dynamics is exactly what a dynamics-aware PINN signal should enforce.

States: 7 sites + trap(=eta) + loss. Times: [0.5,1,2,5,10,20,50] ps.
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

STATE_NAMES = [f"site{i+1}" for i in range(7)] + ["trap(eta)", "loss"]
COND_TIMES_IDX = [1, 3, 4]   # t=1,5,10 ps are in the 33-d condition (POP_T_IDX)


def interp_states(rho, tl, TIMES):
    diag = np.real(np.diagonal(rho, axis1=1, axis2=2))      # (T, 9)
    return np.array([[np.interp(t, tl, diag[:, s]) for s in range(9)] for t in TIMES])  # (7,9)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/checkpoints/diffusion_v1.pt")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, Y = load_data(["data/dataset_full.npz", "data/dataset_full_v2.npz"])
    Xn, Yn, stats = normalize(X, Y)
    pops, times = [], None
    for p in ("data/dataset_full.npz", "data/dataset_full_v2.npz"):
        d = np.load(p); pops.append(d["pop_t"].reshape(len(d["pop_t"]), 7, 9)); times = d["times"]
    POP = np.concatenate(pops).astype(float)                # (N, 7times, 9states)
    TIMES = np.asarray(times, float)

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ck["config"]
    model, sched = build_diffusion(device, cfg["feature_dim"], cfg["context_dim"],
                                   hidden=cfg["hidden"], layers=cfg["layers"], T=cfg["T"])
    model.load_state_dict(ck["state_dict"])

    eta = Y[:, 0]
    subset = np.where(np.abs(eta - 0.95) < 0.03)[0]
    rng = np.random.default_rng(0)
    nn33 = NearestNeighbors(n_neighbors=100).fit(Yn)
    anchors = subset[rng.choice(len(subset), 5, replace=False)]

    K, NM = 100, 80
    shift = np.zeros((len(anchors), 7, 9)); vrat = np.zeros((len(anchors), 7, 9))
    for ai, a in enumerate(anchors):
        c = Yn[a]
        _, nbr = nn33.kneighbors(c[None]); nbr = nbr[0]
        dS = POP[nbr]                                        # (K,7,9)
        S = sample_diffusion(model, sched, c, n=NM, device=device, n_steps=50, eta=0.0)
        mX = denormalize_H(S, stats)
        mS = np.zeros((NM, 7, 9))
        for i, h in enumerate(mX):
            out = sim.simulate(H27_to_matrix(h), 35.0, return_traj=True)
            tl, rho = out["_traj"]; mS[i] = interp_states(rho, tl, TIMES)
        sd = dS.std(0) + 1e-6
        shift[ai] = np.abs(mS.mean(0) - dS.mean(0)) / sd
        vrat[ai] = (mS.std(0) + 1e-6) / sd
        print(f"  anchor {ai+1}/{len(anchors)} done (η={eta[a]:.3f})", flush=True)

    shift_m = shift.mean(0); vrat_m = vrat.mean(0)           # (7,9)

    print("\n" + "=" * 76, flush=True)
    print("MISMATCH LOCALIZATION (model vs data conditional, mean over 5 anchors)", flush=True)
    print("=" * 76, flush=True)
    print("\n[per TIME] mean-shift & var-ratio (avg over 9 states); * = conditioned time", flush=True)
    print("   t(ps)   cond?   mean-shift   var-ratio", flush=True)
    for ti, t in enumerate(TIMES):
        cflag = "*" if ti in COND_TIMES_IDX else " "
        print(f"   {t:5.1f}    {cflag}      {shift_m[ti].mean():6.2f}      {vrat_m[ti].mean():6.2f}", flush=True)
    cond_sh = shift_m[COND_TIMES_IDX].mean(); free_sh = shift_m[[0,2,5,6]].mean()
    print(f"\n   conditioned times (1,5,10): mean-shift {cond_sh:.2f}", flush=True)
    print(f"   free times (0.5,2,20,50):   mean-shift {free_sh:.2f}", flush=True)

    print("\n[per STATE] mean-shift & var-ratio (avg over 7 times)", flush=True)
    print("   state        mean-shift   var-ratio", flush=True)
    order = np.argsort(-shift_m.mean(0))
    for s in order:
        print(f"   {STATE_NAMES[s]:11s}   {shift_m[:,s].mean():6.2f}      {vrat_m[:,s].mean():6.2f}", flush=True)

    # top (time,state) cells
    flat = [(shift_m[t,s], TIMES[t], STATE_NAMES[s]) for t in range(7) for s in range(9)]
    flat.sort(reverse=True)
    print("\n[top mismatch cells] (time, state, mean-shift):", flush=True)
    for v, t, s in flat[:6]:
        print(f"   t={t:4.1f}ps  {s:11s}  shift {v:.2f}", flush=True)

    print("\n" + "=" * 76, flush=True)
    print("해석:", flush=True)
    if free_sh > cond_sh * 1.3:
        print("  → 모델은 *조건이 명시한 시점*은 비교적 맞추나 *자유/후반 시점*에서 크게 빗나감.", flush=True)
        print("    = 명시 조건은 만족하나 나머지 동역학은 미보정 → PINN이 *전체 trajectory*를 강제해야.", flush=True)
    else:
        print("  → mismatch가 조건 시점에도 고르게 분포 → 모델이 조건 자체를 tight히 못 맞춤.", flush=True)
    print("=" * 76, flush=True)

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    Path("outputs/metrics/mismatch_characterization.json").write_text(json.dumps(
        {"times": TIMES.tolist(), "states": STATE_NAMES,
         "mean_shift": shift_m.tolist(), "var_ratio": vrat_m.tolist(),
         "cond_shift": float(cond_sh), "free_shift": float(free_sh)}, indent=2), encoding="utf-8")
    print("saved: outputs/metrics/mismatch_characterization.json", flush=True)


if __name__ == "__main__":
    main()
