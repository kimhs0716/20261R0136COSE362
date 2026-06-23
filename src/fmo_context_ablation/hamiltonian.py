from __future__ import annotations

import numpy as np


N_SITE = 7
N_H28 = 28
N_H27 = 27

_IU = np.triu_indices(N_SITE)
_DIAG_IDX = np.where(_IU[0] == _IU[1])[0]
_OFF_IDX = np.where(_IU[0] != _IU[1])[0]


def gauge_fix_encode(h28: np.ndarray) -> np.ndarray:
    """28D upper-triangular H vector를 trace-zero gauge의 27D 벡터로 바꾼다."""
    h28 = np.asarray(h28)
    diag = h28[..., _DIAG_IDX]
    off = h28[..., _OFF_IDX]
    return np.concatenate([diag[..., :6], off], axis=-1)


def gauge_fix_decode(h27: np.ndarray) -> np.ndarray:
    """27D gauge vector를 28D upper-triangular H vector로 복원한다."""
    h27 = np.asarray(h27)
    diag_6 = h27[..., :6]
    off = h27[..., 6:]
    if off.shape[-1] != 21:
        raise ValueError(f"Expected 21 off-diagonal values, got {off.shape[-1]}")
    diag_7 = -diag_6.sum(axis=-1, keepdims=True)
    diag_full = np.concatenate([diag_6, diag_7], axis=-1)
    out = np.zeros(h27.shape[:-1] + (N_H28,), dtype=h27.dtype)
    out[..., _DIAG_IDX] = diag_full
    out[..., _OFF_IDX] = off
    return out


def h28_to_matrix(h28: np.ndarray) -> np.ndarray:
    """28D upper-triangular H vector를 7x7 symmetric matrix로 바꾼다."""
    h28 = np.asarray(h28)
    h = np.zeros(h28.shape[:-1] + (N_SITE, N_SITE), dtype=h28.dtype)
    h[..., _IU[0], _IU[1]] = h28
    h = h + np.swapaxes(h, -1, -2) - np.eye(N_SITE, dtype=h28.dtype) * np.diagonal(h, axis1=-2, axis2=-1)[..., None, :]
    return h


def h27_to_matrix(h27: np.ndarray) -> np.ndarray:
    """27D gauge vector를 7x7 symmetric matrix로 바꾼다."""
    return h28_to_matrix(gauge_fix_decode(h27))


def matrix_to_h28(h: np.ndarray) -> np.ndarray:
    """7x7 symmetric matrix를 28D upper-triangular vector로 바꾼다."""
    h = np.asarray(h)
    return h[..., _IU[0], _IU[1]]


def matrix_to_h27(h: np.ndarray) -> np.ndarray:
    """7x7 symmetric matrix를 27D gauge vector로 바꾼다."""
    return gauge_fix_encode(matrix_to_h28(h))
