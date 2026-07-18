"""Cox-Ross-Rubinstein binomial tree, European and American exercise."""
from __future__ import annotations

import math

import numpy as np

from engine.bs import validate_inputs


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, kind: str = "call", american: bool = False,
              steps: int = 1000) -> float:
    validate_inputs(S, K, T, sigma, kind)
    if steps < 1:
        raise ValueError(f"steps must be >= 1 (got {steps})")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"CRR risk-neutral probability {p:.4f} outside (0,1); invalid parameter regime")
    disc = math.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(steps + 1)
    ST = S * u ** j * d ** (steps - j)
    values = np.maximum(sign * (ST - K), 0.0)
    for i in range(steps, 0, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            jj = np.arange(i)
            S_i = S * u ** jj * d ** ((i - 1) - jj)
            values = np.maximum(values, np.maximum(sign * (S_i - K), 0.0))
    return float(values[0])


def crr_greeks(S: float, K: float, T: float, r: float, sigma: float,
               q: float = 0.0, kind: str = "call", american: bool = False,
               steps: int = 500) -> dict:
    """Central finite-difference Greeks on the tree. Same units as bs_greeks."""
    def price(**over):
        kw = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        kw.update(over)
        return crr_price(kind=kind, american=american, steps=steps, **kw)

    h_S = 0.005 * S
    h_sig = 1e-3
    h_r = 1e-4
    h_T = min(1e-3, T / 10.0)
    base = price()
    up_S, dn_S = price(S=S + h_S), price(S=S - h_S)
    return {
        "delta": (up_S - dn_S) / (2 * h_S),
        "gamma": (up_S - 2 * base + dn_S) / (h_S * h_S),
        "vega": (price(sigma=sigma + h_sig) - price(sigma=sigma - h_sig)) / (2 * h_sig),
        "theta": -(price(T=T + h_T) - price(T=T - h_T)) / (2 * h_T),
        "rho": (price(r=r + h_r) - price(r=r - h_r)) / (2 * h_r),
    }
