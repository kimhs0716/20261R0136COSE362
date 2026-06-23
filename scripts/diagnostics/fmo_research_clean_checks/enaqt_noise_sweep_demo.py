#!/usr/bin/env python3
"""Recommendation #4 demo ??population-level ENAQT noise sweep (not n=1 representative).

ENAQT signature = transport efficiency is NON-MONOTONIC in environment coupling
(reorganization energy lambda): an interior optimum lambda* > 0 beats both the coherent
limit (lambda->0) and the over-damped limit (large lambda).

The #46 dashboard sweeps lambda for ONE representative per family. This extends it to MANY
real held-out members per group + the standard FMO as a positive control, so the
non-monotonicity becomes a population-level claim.

For each H: simulate at a lambda grid, record eta5/eta10/eta20/eta50 and t80; find lambda*
(argmax eta50); classify interior-peak (ENAQT) vs monotonic; measure the gain available over
the current lambda=35 operating point.

Usage:  python scripts/enaqt_noise_sweep_demo.py [--n 40] [--validate-only]
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
from fmo_hamiltonian.trajectory_features import compute_eta_t, compute_arrival_times

PREP = ROOT / "outputs/flow_pilot_pH_given_c_20260603_62k_prepare/prepared_flow_pilot_data.npz"
LAMBDAS = [0.0, 5.0, 15.0, 35.0, 70.0, 140.0, 280.0]   # cm^-1 ; 35 = current operating point
TLIST = np.arange(0.0, 50.0 + 0.25, 0.5, dtype=float)
CUR = LAMBDAS.index(35.0)


def eta_curve(H7):
    """eta5/10/20/50 + t80 across the lambda grid for one H."""
    e5 = []; e10 = []; e20 = []; e50 = []; t80 = []
    for lam in LAMBDAS:
        out = simulator.simulate(H7, lambda_reorg=lam, tlist=TLIST, return_traj=True)
        _, rho = out["_traj"]
        pop = np.real(np.diagonal(rho, axis1=1, axis2=2))
        eta = compute_eta_t(pop)
        e5.append(float(np.interp(5.0, TLIST, eta)))
        e10.append(float(np.interp(10.0, TLIST, eta)))
        e20.append(float(np.interp(20.0, TLIST, eta)))
        e50.append(float(eta[-1]))
        arr = compute_arrival_times(eta, TLIST)
        t80.append(float(arr.get("t80", np.nan)) if np.isfinite(arr.get("t80", np.nan)) else np.nan)
    return dict(eta5=e5, eta10=e10, eta20=e20, eta50=e50, t80=t80)


def classify(e50, margin=0.02):
    """interior-peak (ENAQT) vs monotonic, on eta50(lambda)."""
    e = np.asarray(e50, float)
    star = int(np.argmax(e))
    interior = (0 < star < len(e) - 1) and (e[star] - e[0] > margin) and (e[star] - e[-1] > margin)
    if interior:
        kind = "interior_peak_ENAQT"
    elif star == len(e) - 1:
        kind = "monotonic_increasing"
    elif star == 0:
        kind = "monotonic_decreasing"
    else:
        kind = "weak_interior"
    return kind, star


def summarize(group, curves):
    e50 = np.array([c["eta50"] for c in curves])           # (n, L)
    e20 = np.array([c["eta20"] for c in curves])
    kinds = []; stars = []; gains = []
    for c in curves:
        k, s = classify(c["eta50"]); kinds.append(k); stars.append(s)
        gains.append(max(c["eta50"]) - c["eta50"][CUR])    # available gain over lambda=35
    kinds = np.array(kinds); stars = np.array(stars); gains = np.array(gains)
    n = len(curves)
    frac_interior = float(np.mean([k in ("interior_peak_ENAQT", "weak_interior") for k in kinds]))
    frac_strict = float(np.mean(kinds == "interior_peak_ENAQT"))
    return {
        "group": group, "n": n,
        "mean_eta50_by_lambda": e50.mean(0).round(4).tolist(),
        "mean_eta20_by_lambda": e20.mean(0).round(4).tolist(),
        "lambda_grid": LAMBDAS,
        "frac_interior_optimum": frac_interior,
        "frac_strict_ENAQT": frac_strict,
        "lambda_star_hist": {str(LAMBDAS[i]): int(np.sum(stars == i)) for i in range(len(LAMBDAS))},
        "median_gain_over_lambda35": float(np.median(gains)),
        "mean_pop_eta50_argmax_lambda": LAMBDAS[int(np.argmax(e50.mean(0)))],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    print("=== positive control: standard FMO eta(lambda) ===", flush=True)
    fmo = eta_curve(np.asarray(H_FMO_CM, float))
    for name in ("eta5", "eta20", "eta50"):
        print(f"  {name:5s} vs lambda{LAMBDAS}: {[round(x,3) for x in fmo[name]]}", flush=True)
    k, s = classify(fmo["eta50"])
    print(f"  -> standard FMO eta50 argmax lambda = {LAMBDAS[s]}  ({k})", flush=True)
    print(f"  (ENAQT expectation: interior peak, i.e. best at intermediate lambda > 0)", flush=True)
    if args.validate_only:
        return

    d = np.load(PREP, allow_pickle=True)
    H27 = d["H_gauge_27"]; test = d["split_test"]; pg = d["priority_group"].astype(str)
    rng = np.random.default_rng(0)

    results = {"standard_fmo": {"eta50_by_lambda": [round(x, 4) for x in fmo["eta50"]],
                                "eta20_by_lambda": [round(x, 4) for x in fmo["eta20"]],
                                "lambda_star": LAMBDAS[s], "kind": k, "lambda_grid": LAMBDAS}}
    curves_by_group = {}
    for g in ("fast_high", "non_high"):
        idx = np.array([i for i in test if pg[i] == g], dtype=int)
        idx = np.sort(rng.choice(idx, min(args.n, len(idx)), replace=False))
        print(f"\n=== {g}: sweeping {len(idx)} real held-out H x {len(LAMBDAS)} lambdas ===", flush=True)
        curves = []; t0 = time.perf_counter()
        for j, i in enumerate(idx):
            curves.append(eta_curve(h27_to_matrix(H27[i])))
            if (j + 1) % 10 == 0:
                print(f"  {g}: {j+1}/{len(idx)} ({time.perf_counter()-t0:.0f}s)", flush=True)
        curves_by_group[g] = curves
        summ = summarize(g, curves)
        results[g] = summ
        print(f"  mean eta50 vs lambda: {summ['mean_eta50_by_lambda']}", flush=True)
        print(f"  population mean-eta50 peaks at lambda = {summ['mean_pop_eta50_argmax_lambda']}", flush=True)
        print(f"  frac with interior optimum (ENAQT-like): {summ['frac_interior_optimum']:.2f} "
              f"(strict {summ['frac_strict_ENAQT']:.2f})", flush=True)
        print(f"  lambda* histogram: {summ['lambda_star_hist']}", flush=True)
        print(f"  median available gain over lambda=35: {summ['median_gain_over_lambda35']:+.3f}", flush=True)

    # figure
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
        ax[0].plot(LAMBDAS, fmo["eta50"], "o-", label="standard FMO", color="k", lw=2)
        for g in curves_by_group:
            e50 = np.array([c["eta50"] for c in curves_by_group[g]])
            ax[0].plot(LAMBDAS, e50.mean(0), "s-", label=f"{g} mean (n={len(e50)})")
        ax[0].axvline(35, ls="--", color="gray", alpha=0.6); ax[0].set_xscale("symlog", linthresh=5)
        ax[0].set_xlabel("reorganization energy lambda (cm^-1)"); ax[0].set_ylabel("eta50")
        ax[0].set_title("ENAQT noise sweep: eta50 vs lambda"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
        for g in curves_by_group:
            stars = [LAMBDAS[classify(c["eta50"])[1]] for c in curves_by_group[g]]
            ax[1].hist(stars, bins=[-1,2.5,10,25,50,105,210,400], alpha=0.6, label=g)
        ax[1].axvline(35, ls="--", color="gray"); ax[1].set_xscale("symlog", linthresh=5)
        ax[1].set_xlabel("per-H optimal lambda*"); ax[1].set_ylabel("count")
        ax[1].set_title("Distribution of per-H noise optimum"); ax[1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(ROOT / "outputs/enaqt_noise_sweep_demo.png", dpi=120); plt.close(fig)
        print("\nsaved figure: outputs/enaqt_noise_sweep_demo.png", flush=True)
    except Exception as e:
        print(f"(figure skip: {e})", flush=True)

    (ROOT / "outputs/enaqt_noise_sweep_demo.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print("saved: outputs/enaqt_noise_sweep_demo.json", flush=True)
    print("\n?댁꽍: ?쒖? FMO媛 interior peak?닿퀬 fast_high population mean??以묎컙 lambda?먯꽌 ?뺤젏?대㈃", flush=True)
    print("      ??ENAQT-like noise-assisted optimum??population ?섏??먯꽌 ?뺤씤 (n=1 ??쒓? ?꾨떂).", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()

