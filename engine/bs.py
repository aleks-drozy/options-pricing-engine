"""Black-Scholes closed-form pricing (continuous dividend yield q).

Conventions: T in years; sigma per sqrt(year); r, q continuously compounded.
"""
from __future__ import annotations

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_KINDS = ("call", "put")


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def validate_inputs(S: float, K: float, T: float, sigma: float, kind: str) -> None:
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive (got S={S}, K={K})")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T})")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma})")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS} (got {kind!r})")


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
    return d1, d1 - srt


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    validate_inputs(S, K, T, sigma, kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    if kind == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)
