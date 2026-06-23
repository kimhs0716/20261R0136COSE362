from __future__ import annotations

import numpy as np
import qutip as qt

trapz = np.trapezoid

CM2RADPS = 2 * np.pi * 0.0299792458
KB = 0.6950348

N_SITE = 7
IDX_IN = 0
IDX_TRAP_SITE = 2
IDX_TRAP = 7
IDX_LOSS = 8
DIM = 9

TEMP_K = 300.0
OMEGA_C = 106.0
KAPPA = 1.0
GAMMA_RECOMB = 1.0 / 1000.0
T_MAX = 50.0
N_TIME = 1500
SEC_CUTOFF = np.inf

SITE_ENERGY_RANGE = (0.0, 450.0)
GEOM_BOX = 4.0
GEOM_RMIN = 1.0
GEOM_DIP = 150.0

H_FMO_CM = np.array(
    [
        [-21.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
        [-87.7, 99.0, 30.8, 8.2, 0.7, 11.8, 4.3],
        [5.5, 30.8, -221.0, -53.5, -2.2, -9.6, 6.0],
        [-5.9, 8.2, -53.5, -111.0, -70.7, -17.0, -63.3],
        [6.7, 0.7, -2.2, -70.7, 49.0, 81.1, -1.3],
        [-13.7, 11.8, -9.6, -17.0, 81.1, 199.0, 39.7],
        [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 9.0],
    ],
    dtype=np.float64,
)


def power_spectrum(lambda_reorg: float):
    """Drude-Lorentz bath noise power spectrum S(w)."""
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


def simulate(H_cm, lambda_reorg: float = 35.0, return_traj: bool = False) -> dict:
    """7x7 Hamiltonian에서 eta, tau_transfer, ipr, purity, c_l1을 계산한다."""
    h9 = np.zeros((DIM, DIM), dtype=np.float64)
    h9[:N_SITE, :N_SITE] = np.asarray(H_cm, dtype=np.float64) * CM2RADPS
    h = qt.Qobj(h9)

    rho0 = qt.basis(DIM, IDX_IN).proj()
    tlist = np.linspace(0.0, T_MAX, N_TIME)

    spectrum = power_spectrum(lambda_reorg)
    a_ops = [[qt.projection(DIM, n, n), spectrum] for n in range(N_SITE)]
    c_trap = np.sqrt(KAPPA) * qt.projection(DIM, IDX_TRAP, IDX_TRAP_SITE)
    c_loss = [np.sqrt(GAMMA_RECOMB) * qt.projection(DIM, IDX_LOSS, n) for n in range(N_SITE)]

    res = qt.brmesolve(h, rho0, tlist, a_ops=a_ops, c_ops=[c_trap] + c_loss, sec_cutoff=SEC_CUTOFF)
    rho_t = np.array([state.full() for state in res.states])

    labels = labels_from_trajectory(rho_t, tlist)
    if return_traj:
        labels["_traj"] = (tlist, rho_t)
    return labels


def labels_from_trajectory(rho_t: np.ndarray, tlist: np.ndarray) -> dict:
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


def sample_H_geom(rng: np.random.Generator) -> np.ndarray:
    """데이터셋 생성과 같은 geometric prior에서 random Hamiltonian을 샘플링한다."""
    pos = np.zeros((N_SITE, 3), dtype=np.float64)
    n = 0
    while n < N_SITE:
        p = rng.uniform(0.0, GEOM_BOX, size=3)
        if n == 0 or np.all(np.linalg.norm(pos[:n] - p, axis=1) > GEOM_RMIN):
            pos[n] = p
            n += 1

    mu = rng.normal(size=(N_SITE, 3))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)

    h = np.zeros((N_SITE, N_SITE), dtype=np.float64)
    for i in range(N_SITE):
        for j in range(i + 1, N_SITE):
            d = pos[i] - pos[j]
            r = float(np.linalg.norm(d))
            rhat = d / r
            kappa = mu[i] @ mu[j] - 3.0 * (mu[i] @ rhat) * (mu[j] @ rhat)
            h[i, j] = h[j, i] = GEOM_DIP * kappa / r**3

    diag = rng.uniform(*SITE_ENERGY_RANGE, size=N_SITE)
    h[np.diag_indices(N_SITE)] = diag - diag.mean()
    return h.astype(np.float32)
