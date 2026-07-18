"""Monte Carlo pricing of EUROPEAN options by GBM terminal-value sampling.

American exercise requires Longstaff-Schwartz regression and is out of scope
(see spec). Use engine.binomial for American options.
"""
from __future__ import annotations

import math

import numpy as np

from engine.bs import validate_inputs


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call",
             n_paths: int = 100_000, seed: int = 0) -> tuple[float, float]:
    validate_inputs(S, K, T, sigma, kind)
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2 (got {n_paths})")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = math.exp(-r * T)
    price = disc * float(payoff.mean())
    std_error = disc * float(payoff.std(ddof=1)) / math.sqrt(n_paths)
    return price, std_error
