#!/usr/bin/env python3
"""Deeper ENAQT ??2D noise landscape (lambda x omega_c) + coherence mechanism.

The 1D lambda sweep showed an interior optimum. To push toward a real ENAQT claim
(docs/22_recheck): (1) sweep BOTH noise knobs ??reorganization energy lambda AND bath cutoff
omega_c (dephasing character) ??does a 2D interior optimum exist? (2) coherence mechanism ??does
the efficiency peak coincide with INTERMEDIATE coherence (noise breaks destructive interference
without freezing transport = Goldilocks), not max coherence (lambda=0) or min (large lambda)?

omega_c is overridden via the module global `simulator.OMEGA_C` (power_spectrum reads it at call).

Usage:  python scripts/enaqt_deep.py [--n 25] [--validate-only]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from fmo_hamiltonian import simulator
from fmo_hamiltonian.constants import H_FMO_CM
from fmo_hamiltonian.sampling import h27_to_matrix
from fmo_hamiltonian.trajectory_features import compute_eta_t

PREP = ROOT / "outputs/flow_pilot_pH_given_c_20260603_62k_prepare/prepared_flow_pilot_data.npz"
LAMBDAS = [0.0, 5.0, 15.0, 35.0, 70.0, 140.0]
OMEGAS = [25.0, 53.0, 106.0, 212.0, 424.0]            # bath cutoff grid (106 = current)
TLIST = np.arange(0.0, 50.0 + 0.25, 0.5, dtype=float)
OMEGA_DEFAULT = 106.0


def sim_point(H7, lam, omega_c):
    """Return (eta50, eta20, c_l1) for one (lambda, omega_c)."""
    simulator.OMEGA_C = float(omega_c)                # override module global
    out = simulator.simulate(H7, lambda_reorg=lam, tlist=TLIST, return_traj=True)
    _, rho = out["_traj"]
    pop = np.real(np.diagonal(rho, axis1=1, axis2=2))
    eta = compute_eta_t(pop)
    return float(eta[-1]), float(np.interp(20.0, TLIST, eta)), float(out.get("c_l1", np.nan))


def landscape_2d(H7):
    eta50 = np.zeros((len(OMEGAS), len(LAMBDAS)))
    for a, oc in enumerate(OMEGAS):
        for b, lam in enumerate(LAMBDAS):
            eta50[a, b], _, _ = sim_point(H7, lam, oc)
    return eta50


def lambda_curve_with_coherence(H7, omega_c=OMEGA_DEFAULT):
    e50 = []; cl1 = []
    for lam in LAMBDAS:
        e, _, c = sim_point(H7, lam, omega_c)
        e50.append(e); cl1.append(c)
    return np.array(e50), np.array(cl1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    fmo = np.asarray(H_FMO_CM, float)

    # validate omega_c override actually changes dynamics
    e_lo, _, _ = sim_point(fmo, 35.0, 25.0)
    e_hi, _, _ = sim_point(fmo, 35.0, 424.0)
    print(f"=== omega_c override check (FMO, lambda=35) ===", flush=True)
    print(f"  eta50 at omega_c=25 -> {e_lo:.3f} ; omega_c=424 -> {e_hi:.3f}  "
          f"(differ => override works: {abs(e_lo-e_hi)>1e-3})", flush=True)
    if args.validate_only:
        simulator.OMEGA_C = OMEGA_DEFAULT
        return

    # ---- Part 1: 2D landscape for standard FMO ----
    print("\n=== Part 1: standard FMO 2D noise landscape (lambda x omega_c) ===", flush=True)
    t0 = time.perf_counter()
    fmo_2d = landscape_2d(fmo)
    ai, bi = np.unravel_index(np.argmax(fmo_2d), fmo_2d.shape)
    interior2d = (0 < ai < len(OMEGAS) - 1) and (0 < bi < len(LAMBDAS) - 1)
    print(f"  eta50 grid (rows omega_c={OMEGAS}, cols lambda={LAMBDAS}):", flush=True)
    for a, oc in enumerate(OMEGAS):
        print(f"    omega_c={oc:5.0f}: {[f'{x:.3f}' for x in fmo_2d[a]]}", flush=True)
    print(f"  -> 2D optimum at omega_c={OMEGAS[ai]}, lambda={LAMBDAS[bi]} "
          f"(eta50={fmo_2d[ai,bi]:.3f}); interior-in-2D={interior2d}  ({time.perf_counter()-t0:.0f}s)",
          flush=True)

    # ---- Part 2: coherence mechanism (1D lambda, with c_l1) ----
    print("\n=== Part 2: coherence mechanism (eta50 peak vs coherence) ===", flush=True)
    fmo_e50, fmo_cl1 = lambda_curve_with_coherence(fmo)
    star = int(np.argmax(fmo_e50))
    print(f"  FMO eta50(lambda): {[f'{x:.3f}' for x in fmo_e50]}", flush=True)
    print(f"  FMO c_l1 (coherence): {[f'{x:.3f}' for x in fmo_cl1]}", flush=True)
    print(f"  FMO eta50 peaks at lambda={LAMBDAS[star]}; coherence there={fmo_cl1[star]:.3f} "
          f"vs max(lambda=0)={fmo_cl1[0]:.3f} -> coherence fraction at opt = {fmo_cl1[star]/max(fmo_cl1[0],1e-9):.2f}",
          flush=True)

    d = np.load(PREP, allow_pickle=True)
    H27 = d["H_gauge_27"]; test = d["split_test"]; pg = d["priority_group"].astype(str)
    rng = np.random.default_rng(0)
    pop_results = {}
    for g in ("fast_high", "non_high"):
        idx = np.array([i for i in test if pg[i] == g], dtype=int)
        idx = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        print(f"\n  [{g}] {len(idx)} H x {len(LAMBDAS)} lambda (omega_c=106) ...", flush=True)
        coh_frac = []; star_cl1 = []; t0 = time.perf_counter()
        e50_all = []; cl1_all = []
        for j, i in enumerate(idx):
            e50, cl1 = lambda_curve_with_coherence(h27_to_matrix(H27[i]))
            s = int(np.argmax(e50))
            coh_frac.append(cl1[s] / max(cl1[0], 1e-9))
            star_cl1.append(cl1[s]); e50_all.append(e50); cl1_all.append(cl1)
            if (j + 1) % 10 == 0:
                print(f"    {j+1}/{len(idx)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        coh_frac = np.array(coh_frac)
        pop_results[g] = {
            "n": len(idx),
            "median_coherence_fraction_at_optimum": float(np.median(coh_frac)),
            "frac_optimum_below_full_coherence": float(np.mean(coh_frac < 0.9)),
            "mean_eta50_by_lambda": np.array(e50_all).mean(0).round(4).tolist(),
            "mean_cl1_by_lambda": np.array(cl1_all).mean(0).round(4).tolist(),
        }
        print(f"  [{g}] median coherence-at-optimum / coherence-at-lambda0 = "
              f"{pop_results[g]['median_coherence_fraction_at_optimum']:.2f}  "
              f"(frac with optimum below full coherence: {pop_results[g]['frac_optimum_below_full_coherence']:.2f})",
              flush=True)

    # figures
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
        im = ax[0].imshow(fmo_2d, origin="lower", aspect="auto", cmap="viridis")
        ax[0].set_xticks(range(len(LAMBDAS)), [str(int(x)) for x in LAMBDAS])
        ax[0].set_yticks(range(len(OMEGAS)), [str(int(x)) for x in OMEGAS])
        ax[0].set_xlabel("lambda (cm^-1)"); ax[0].set_ylabel("omega_c (cm^-1)")
        ax[0].set_title("Standard FMO eta50 ??2D noise landscape")
        ax[0].plot(bi, ai, "r*", ms=16); fig.colorbar(im, ax=ax[0])
        ax2 = ax[1]; ax3 = ax2.twinx()
        ax2.plot(LAMBDAS, fmo_e50, "o-", color="C0", label="eta50")
        ax3.plot(LAMBDAS, fmo_cl1, "s--", color="C3", label="coherence c_l1")
        ax2.axvline(LAMBDAS[star], ls=":", color="gray")
        ax2.set_xscale("symlog", linthresh=5); ax2.set_xlabel("lambda (cm^-1)")
        ax2.set_ylabel("eta50", color="C0"); ax3.set_ylabel("coherence c_l1", color="C3")
        ax2.set_title("FMO: efficiency peaks at INTERMEDIATE coherence (ENAQT)")
        fig.tight_layout(); fig.savefig(ROOT / "outputs/enaqt_deep.png", dpi=120); plt.close(fig)
        print("\nsaved figure: outputs/enaqt_deep.png", flush=True)
    except Exception as e:
        print(f"(figure skip: {e})", flush=True)

    simulator.OMEGA_C = OMEGA_DEFAULT
    results = {
        "lambda_grid": LAMBDAS, "omega_c_grid": OMEGAS,
        "fmo_eta50_2d": fmo_2d.round(4).tolist(),
        "fmo_2d_optimum": {"omega_c": OMEGAS[ai], "lambda": LAMBDAS[bi],
                           "eta50": float(fmo_2d[ai, bi]), "interior_in_2d": bool(interior2d)},
        "fmo_eta50_by_lambda": fmo_e50.round(4).tolist(),
        "fmo_cl1_by_lambda": fmo_cl1.round(4).tolist(),
        "fmo_coherence_fraction_at_optimum": float(fmo_cl1[star] / max(fmo_cl1[0], 1e-9)),
        "population": pop_results,
    }
    (ROOT / "outputs/enaqt_deep.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved: outputs/enaqt_deep.json", flush=True)
    print("\nInterpretation: an interior optimum in 2D together with a peak at intermediate coherence", flush=True)
    print("      supports an ENAQT-style Goldilocks regime rather than monotonic noise benefit.", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()

