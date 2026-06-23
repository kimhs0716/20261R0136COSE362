"""Shared constants for the clean FMO workspace."""

from __future__ import annotations

import numpy as np

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
DEFAULT_LAMBDA_REORG = 35.0
DEFAULT_T_MAX = 15.0
DEFAULT_N_TIME = 600
SEC_CUTOFF = np.inf

GEOM_BOX = 4.0
GEOM_RMIN = 1.0
GEOM_DIP = 150.0

H_FMO_CM = np.array([
    [-21.0, -87.7, 5.5, -5.9, 6.7, -13.7, -9.9],
    [-87.7, 99.0, 30.8, 8.2, 0.7, 11.8, 4.3],
    [5.5, 30.8, -221.0, -53.5, -2.2, -9.6, 6.0],
    [-5.9, 8.2, -53.5, -111.0, -70.7, -17.0, -63.3],
    [6.7, 0.7, -2.2, -70.7, 49.0, 81.1, -1.3],
    [-13.7, 11.8, -9.6, -17.0, 81.1, 199.0, 39.7],
    [-9.9, 4.3, 6.0, -63.3, -1.3, 39.7, 9.0],
], dtype=float)

