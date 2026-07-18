"""Deterministic 200-point parameter grid shared by all gates (spec: 200 points)."""
from itertools import product

_K = [70, 85, 100, 115, 130]
_SIGMA = [0.15, 0.35]
_T = [0.1, 0.5, 1.0, 2.0]
_RQ = [(0.0, 0.0), (0.02, 0.0), (0.05, 0.02), (0.08, 0.0), (0.05, 0.05)]

PARAM_GRID = [
    {"S": 100.0, "K": float(K), "T": T, "r": r, "sigma": sig, "q": q}
    for K, sig, T, (r, q) in product(_K, _SIGMA, _T, _RQ)
]
assert len(PARAM_GRID) == 200
