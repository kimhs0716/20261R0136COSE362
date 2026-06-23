"""Hamiltonian geometry sampling and representation helpers."""

from __future__ import annotations

import numpy as np

from .constants import GEOM_BOX, GEOM_DIP, GEOM_RMIN, N_SITE

_IU = np.triu_indices(N_SITE)
_DIAG_IDX = np.where(_IU[0] == _IU[1])[0]
_OFF_IDX = np.where(_IU[0] != _IU[1])[0]


def sample_h_geom(
    rng: np.random.Generator,
    *,
    geom_box: float = GEOM_BOX,
    r_min: float = GEOM_RMIN,
    dipole_scale: float = GEOM_DIP,
    return_meta: bool = False,
):
    """Sample a 7-site Hamiltonian from a simple geometric point-dipole prior."""
    pos = np.zeros((N_SITE, 3), dtype=float)
    n = 0
    while n < N_SITE:
        p = rng.uniform(0.0, geom_box, size=3)
        if n == 0 or np.all(np.linalg.norm(pos[:n] - p, axis=1) > r_min):
            pos[n] = p
            n += 1

    mu = rng.normal(size=(N_SITE, 3))
    mu /= np.linalg.norm(mu, axis=1, keepdims=True)

    h = np.zeros((N_SITE, N_SITE), dtype=float)
    dist_mat = np.zeros_like(h)
    kappa_mat = np.zeros_like(h)
    for i in range(N_SITE):
        for j in range(i + 1, N_SITE):
            d = pos[i] - pos[j]
            r = float(np.linalg.norm(d))
            rhat = d / r
            kappa = float(mu[i] @ mu[j] - 3.0 * (mu[i] @ rhat) * (mu[j] @ rhat))
            h[i, j] = h[j, i] = dipole_scale * kappa / r**3
            dist_mat[i, j] = dist_mat[j, i] = r
            kappa_mat[i, j] = kappa_mat[j, i] = kappa

    diag = rng.uniform(0.0, 450.0, size=N_SITE)
    h[np.diag_indices(N_SITE)] = diag - diag.mean()

    if not return_meta:
        return h

    return h, {
        "geom_pos": pos.astype(np.float32),
        "dipole_mu": mu.astype(np.float32),
        "dist_mat": dist_mat.astype(np.float32),
        "kappa_mat": kappa_mat.astype(np.float32),
    }


def h_to_params(h: np.ndarray) -> np.ndarray:
    """7x7 symmetric matrix -> 28-d upper-triangular parameter vector."""
    h = np.asarray(h)
    return h[_IU].astype(np.float32)


def params_to_h(params: np.ndarray) -> np.ndarray:
    """28-d upper-triangular parameter vector -> 7x7 symmetric matrix."""
    params = np.asarray(params)
    h = np.zeros(params.shape[:-1] + (N_SITE, N_SITE), dtype=params.dtype)
    h[..., _IU[0], _IU[1]] = params
    h = h + np.swapaxes(h, -1, -2)
    diag = np.diagonal(h, axis1=-2, axis2=-1)
    h[..., np.arange(N_SITE), np.arange(N_SITE)] = diag / 2.0
    return h


def gauge_fix_encode(h28: np.ndarray) -> np.ndarray:
    """28-d H params -> 27-d gauge-fixed vector."""
    h28 = np.asarray(h28)
    diag = h28[..., _DIAG_IDX]
    off = h28[..., _OFF_IDX]
    return np.concatenate([diag[..., :6], off], axis=-1).astype(np.float32)


def gauge_fix_decode(h27: np.ndarray) -> np.ndarray:
    """27-d gauge-fixed vector -> 28-d H params."""
    h27 = np.asarray(h27)
    diag_6 = h27[..., :6]
    off = h27[..., 6:]
    diag_7 = -diag_6.sum(axis=-1, keepdims=True)
    diag = np.concatenate([diag_6, diag_7], axis=-1)
    out = np.zeros(h27.shape[:-1] + (28,), dtype=h27.dtype)
    out[..., _DIAG_IDX] = diag
    out[..., _OFF_IDX] = off
    return out


def h27_to_matrix(h27: np.ndarray) -> np.ndarray:
    """27-d gauge-fixed vector -> 7x7 symmetric matrix."""
    return params_to_h(gauge_fix_decode(h27))

