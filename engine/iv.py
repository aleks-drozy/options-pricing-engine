"""Implied volatility by Brent root-finding on Black-Scholes."""
from __future__ import annotations

import math

from scipy.optimize import brentq

from engine.bs import bs_price

_SIG_LO, _SIG_HI = 1e-4, 5.0


class IVError(ValueError):
    """Quote cannot be inverted to a Black-Scholes volatility."""


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call") -> float:
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    lower = max(S * df_q - K * df_r, 0.0) if kind == "call" else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if kind == "call" else K * df_r
    if not lower < price < upper:
        raise IVError(f"price {price} outside no-arbitrage bounds ({lower:.6f}, {upper:.6f})")

    def objective(sig: float) -> float:
        return bs_price(S, K, T, r, sig, q, kind) - price

    try:
        sig = brentq(objective, _SIG_LO, _SIG_HI, xtol=1e-10)
    except ValueError as exc:
        raise IVError(f"no root in [{_SIG_LO}, {_SIG_HI}]: {exc}") from exc
    if abs(bs_price(S, K, T, r, sig, q, kind) - price) > 1e-6 * S:
        raise IVError("re-price check failed")
    return float(sig)
