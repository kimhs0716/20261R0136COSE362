"""Pure NumPy trajectory and Hamiltonian feature utilities.

These functions are intentionally model-free. They are for deciding what labels and trajectory
summaries are informative before any flow/diffusion training.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .constants import IDX_LOSS, IDX_TRAP, N_SITE
from .sampling import gauge_fix_decode, params_to_h

DEFAULT_PATH_GROUPS = ("site1", "site2", "sink34", "detour567", "trap", "loss", "residual")


def interp_trajectory_at_times(values_t, times, query_times, *, allow_extrapolate=False):
    """Interpolate time series at requested times.

    Supports shapes `(T,)`, `(N, T)`, and `(N, T, D)`. Query values outside the simulated
    range become NaN unless `allow_extrapolate=True`.
    """
    values = np.asarray(values_t, dtype=float)
    times = np.asarray(times, dtype=float)
    query = np.asarray(query_times, dtype=float)
    if times.ndim != 1:
        raise ValueError("times must be 1-d")
    if values.shape[-1] == len(times) and values.ndim == 1:
        return _interp_1d(values, times, query, allow_extrapolate)
    if values.ndim == 2:
        return np.stack([_interp_1d(row, times, query, allow_extrapolate) for row in values], axis=0)
    if values.ndim == 3:
        rows = []
        for sample in values:
            cols = [_interp_1d(sample[:, j], times, query, allow_extrapolate) for j in range(sample.shape[1])]
            rows.append(np.stack(cols, axis=-1))
        return np.stack(rows, axis=0)
    raise ValueError(f"unsupported values_t shape: {values.shape}")


def _interp_1d(values, times, query, allow_extrapolate):
    left = None if allow_extrapolate else np.nan
    right = None if allow_extrapolate else np.nan
    return np.interp(query, times, values, left=left, right=right)


def compute_eta_t(pop_t, trap_index: int = IDX_TRAP):
    """Extract eta(t), defined here as trap population over time."""
    pop = np.asarray(pop_t)
    if pop.shape[-1] <= trap_index:
        raise ValueError(f"pop_t last dimension {pop.shape[-1]} does not include trap_index={trap_index}")
    return pop[..., trap_index].astype(np.float32)


def compute_arrival_times(trap_t, times, thresholds=(0.25, 0.50, 0.80, 0.90)):
    """Compute first trap-arrival times by linear interpolation.

    Works for one trajectory `(T,)` or a batch `(N, T)`. Returns a dict of arrays keyed by
    `t25`, `t50`, `t80`, `t90`.
    """
    trap = np.asarray(trap_t, dtype=float)
    times = np.asarray(times, dtype=float)
    if trap.ndim == 1:
        return {f"t{int(th * 100)}": _first_crossing(times, trap, th) for th in thresholds}
    if trap.ndim == 2:
        out = {}
        for th in thresholds:
            out[f"t{int(th * 100)}"] = np.array([_first_crossing(times, row, th) for row in trap], dtype=np.float32)
        return out
    raise ValueError(f"trap_t must have shape (T,) or (N,T), got {trap.shape}")


def _first_crossing(times, values, threshold):
    hit = np.flatnonzero(values >= threshold)
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    if i == 0:
        return float(times[0])
    t0, t1 = times[i - 1], times[i]
    y0, y1 = values[i - 1], values[i]
    if abs(y1 - y0) < 1e-12:
        return float(t1)
    return float(t0 + (threshold - y0) * (t1 - t0) / (y1 - y0))


def compute_path_groups(pop_t, state_layout=None):
    """Compute pathway groups from population trajectories.

    Default state convention:
    columns 0..6 = site1..site7, 7 = trap, 8 = loss.
    Returns `(path_t, group_names)` where group order is:
    `site1, site2, sink34, detour567, trap, loss, residual`.
    """
    if state_layout is not None:
        raise NotImplementedError("custom state_layout is not implemented yet")
    pop = np.asarray(pop_t, dtype=float)
    if pop.shape[-1] < 9:
        raise ValueError("pop_t last dimension must contain 7 sites + trap + loss")
    site1 = pop[..., 0]
    site2 = pop[..., 1]
    sink34 = pop[..., 2] + pop[..., 3]
    detour567 = pop[..., 4] + pop[..., 5] + pop[..., 6]
    trap = pop[..., 7]
    loss = pop[..., 8]
    residual = pop[..., :7].sum(axis=-1)
    path = np.stack([site1, site2, sink34, detour567, trap, loss, residual], axis=-1)
    return path.astype(np.float32), list(DEFAULT_PATH_GROUPS)


def compute_path_at_times(pop_t, times, summary_times):
    """Interpolate path groups at summary times."""
    path_t, names = compute_path_groups(pop_t)
    return interp_trajectory_at_times(path_t, times, summary_times), names


def compute_windowed_residence(path_t, times, windows=((0, 5), (5, 10), (10, 20)), group_names=None):
    """Window-averaged pathway residence using trapezoidal integration / window length."""
    path = np.asarray(path_t, dtype=float)
    times = np.asarray(times, dtype=float)
    if group_names is None:
        group_names = list(DEFAULT_PATH_GROUPS[:path.shape[-1]])
    if path.ndim == 2:
        path = path[None, ...]
    rows = []
    names = []
    for start, end in windows:
        mask = (times >= start) & (times <= end)
        width = float(end - start)
        if mask.sum() < 2 or width <= 0:
            vals = np.full((path.shape[0], path.shape[-1]), np.nan, dtype=np.float32)
        else:
            vals = np.trapezoid(path[:, mask, :], times[mask], axis=1) / width
        rows.append(vals)
        names.extend([f"residence_{name}_{_fmt_time(start)}_{_fmt_time(end)}" for name in group_names])
    return np.concatenate(rows, axis=1).astype(np.float32), names


def summarize_timeseries_windows(values_t, times, prefix, windows=((0, 5), (5, 10), (10, 20))):
    """Summarize optional scalar time series such as cl1/purity/ipr over windows."""
    values = np.asarray(values_t, dtype=float)
    times = np.asarray(times, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    cols = []
    names = []
    for start, end in windows:
        mask = (times >= start) & (times <= end)
        if mask.sum() == 0:
            mean = np.full(values.shape[0], np.nan, dtype=np.float32)
        else:
            mean = np.nanmean(values[:, mask], axis=1).astype(np.float32)
        cols.append(mean[:, None])
        names.append(f"{prefix}_mean_{_fmt_time(start)}_{_fmt_time(end)}")
    max_val = np.nanmax(values, axis=1)
    max_idx = np.nanargmax(values, axis=1)
    cols.append(max_val[:, None].astype(np.float32))
    cols.append(times[max_idx][:, None].astype(np.float32))
    names.extend([f"{prefix}_max", f"{prefix}_tmax"])
    return np.concatenate(cols, axis=1), names


def compute_hamiltonian_features(h_matrix_or_params):
    """Compute simple physical features from H.

    Accepts a 7x7 matrix, a 28-d upper-triangular vector, or a 27-d gauge-fixed vector.
    """
    h = _as_matrix(h_matrix_or_params)
    if h.ndim == 2:
        h = h[None, ...]
    rows = []
    names = [
        "diag_mean",
        "diag_std",
        "diag_range",
        "offdiag_mean_abs",
        "offdiag_max_abs",
        "offdiag_fro_norm",
        "spectral_width",
        "min_eigen_gap",
        "max_eigen_gap",
        "site1_to_others_abs_sum",
        "site2_to_sink34_abs_sum",
        "site1_to_sink34_abs_sum",
        "detour_coupling_abs_sum",
        "direct_shortcut_to_sink_abs_sum",
    ]
    iu = np.triu_indices(N_SITE, 1)
    for mat in h:
        diag = np.diag(mat)
        off = mat[iu]
        eigs = np.linalg.eigvalsh(mat)
        gaps = np.diff(np.sort(eigs))
        detour_edges = [abs(mat[4, 5]), abs(mat[5, 6]), abs(mat[4, 6])]
        rows.append([
            float(diag.mean()),
            float(diag.std()),
            float(diag.max() - diag.min()),
            float(np.mean(np.abs(off))),
            float(np.max(np.abs(off))),
            float(np.sqrt(np.sum(off**2))),
            float(eigs.max() - eigs.min()),
            float(np.min(gaps)) if len(gaps) else math.nan,
            float(np.max(gaps)) if len(gaps) else math.nan,
            float(np.sum(np.abs(mat[0, 1:]))),
            float(abs(mat[1, 2]) + abs(mat[1, 3])),
            float(abs(mat[0, 2]) + abs(mat[0, 3])),
            float(np.sum(detour_edges)),
            float(abs(mat[0, 2]) + abs(mat[0, 3])),
        ])
    return np.asarray(rows, dtype=np.float32), names


def _as_matrix(h):
    arr = np.asarray(h)
    if arr.shape[-2:] == (N_SITE, N_SITE):
        return arr.astype(float)
    if arr.shape[-1] == 28:
        return params_to_h(arr).astype(float)
    if arr.shape[-1] == 27:
        return params_to_h(gauge_fix_decode(arr)).astype(float)
    raise ValueError(f"cannot interpret Hamiltonian shape {arr.shape}")


def _fmt_time(x):
    return str(float(x)).replace(".", "p").replace("-", "m")


def parse_time_list(text: str | Iterable[float]):
    if isinstance(text, str):
        return np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=np.float32)
    return np.array(list(text), dtype=np.float32)

