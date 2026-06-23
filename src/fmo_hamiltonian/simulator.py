"""FMO forward simulator wrapper.

QuTiP is imported lazily inside `simulate`, so utilities can be imported without QuTiP installed.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    CM2RADPS,
    DEFAULT_LAMBDA_REORG,
    DEFAULT_N_TIME,
    DEFAULT_T_MAX,
    DIM,
    GAMMA_RECOMB,
    IDX_IN,
    IDX_LOSS,
    IDX_TRAP,
    IDX_TRAP_SITE,
    KAPPA,
    KB,
    N_SITE,
    OMEGA_C,
    SEC_CUTOFF,
    TEMP_K,
)

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def power_spectrum(lambda_reorg: float):
    """Return Drude-Lorentz bath noise power spectrum S(w), with w in rad/ps."""
    lam = lambda_reorg * CM2RADPS
    gam = OMEGA_C * CM2RADPS
    kt = KB * TEMP_K * CM2RADPS

    def spectrum(w):
        w = float(w)
        if abs(w) < 1e-9:
            return 4 * lam * kt / gam
        j = 2 * lam * gam * w / (w**2 + gam**2)
        return float(j * (1.0 / np.tanh(w / (2 * kt)) + 1.0))

    return spectrum


def simulate(
    h_cm: np.ndarray,
    lambda_reorg: float = DEFAULT_LAMBDA_REORG,
    *,
    t_max: float = DEFAULT_T_MAX,
    n_time: int = DEFAULT_N_TIME,
    tlist: np.ndarray | None = None,
    return_traj: bool = False,
):
    """Simulate H -> scalar labels, optionally returning dense rho(t)."""
    import qutip as qt

    h9 = np.zeros((DIM, DIM), dtype=float)
    h9[:N_SITE, :N_SITE] = np.asarray(h_cm, dtype=float) * CM2RADPS
    h = qt.Qobj(h9)

    rho0 = qt.basis(DIM, IDX_IN).proj()
    if tlist is None:
        tlist = np.linspace(0.0, t_max, n_time)
    else:
        tlist = np.asarray(tlist, dtype=float)
        if tlist.ndim != 1 or len(tlist) < 2:
            raise ValueError("tlist must be a 1-d array with at least two time points")
        t_max = float(tlist[-1])

    s = power_spectrum(lambda_reorg)
    a_ops = [[qt.projection(DIM, n, n), s] for n in range(N_SITE)]
    c_trap = np.sqrt(KAPPA) * qt.projection(DIM, IDX_TRAP, IDX_TRAP_SITE)
    c_loss = [
        np.sqrt(GAMMA_RECOMB) * qt.projection(DIM, IDX_LOSS, n)
        for n in range(N_SITE)
    ]

    res = qt.brmesolve(h, rho0, tlist, a_ops=a_ops, c_ops=[c_trap] + c_loss,
                       sec_cutoff=SEC_CUTOFF)
    rho_t = np.array([state.full() for state in res.states])
    labels = labels_from_rho(rho_t, tlist)
    if return_traj:
        labels["_traj"] = (tlist, rho_t)
    return labels


def labels_from_rho(rho_t: np.ndarray, tlist: np.ndarray) -> dict[str, float]:
    """Convert rho(t) to scalar labels."""
    sys = rho_t[:, :N_SITE, :N_SITE]
    eta = float(rho_t[-1, IDX_TRAP, IDX_TRAP].real)
    p_trap = np.real(rho_t[:, IDX_TRAP, IDX_TRAP])
    tau_transfer = float(tlist[-1] - trapz(p_trap, tlist) / max(eta, 1e-9))

    tr_sys = np.real(np.trace(sys, axis1=1, axis2=2))
    mask = tr_sys > 0.05
    if mask.sum() < 2:
        return {
            "eta": eta,
            "tau_transfer": tau_transfer,
            "ipr": float("nan"),
            "purity": float("nan"),
            "c_l1": float("nan"),
        }

    s = sys[mask]
    t = tlist[mask]
    w = tr_sys[mask]
    rho_n = s / w[:, None, None]
    pops = np.real(np.diagonal(rho_n, axis1=1, axis2=2))
    ipr_t = np.sum(pops**2, axis=1)
    purity_t = np.real(np.trace(rho_n @ rho_n, axis1=1, axis2=2))
    cl1_t = np.sum(np.abs(rho_n) * (1.0 - np.eye(N_SITE)[None]), axis=(1, 2))

    denom = trapz(w, t) + 1e-12
    return {
        "eta": eta,
        "tau_transfer": tau_transfer,
        "ipr": float(trapz(ipr_t * w, t) / denom),
        "purity": float(trapz(purity_t * w, t) / denom),
        "c_l1": float(trapz(cl1_t * w, t) / denom),
    }
