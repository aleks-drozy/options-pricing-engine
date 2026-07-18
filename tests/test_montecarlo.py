import math
import pytest
from engine.montecarlo import mc_price
from engine.bs import bs_price

CASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def test_mc_within_3se_of_bs():
    price, se = mc_price(n_paths=200_000, seed=42, **CASE)
    assert abs(price - bs_price(**CASE)) < 3 * se


def test_mc_deterministic_given_seed():
    a = mc_price(n_paths=50_000, seed=7, **CASE)
    b = mc_price(n_paths=50_000, seed=7, **CASE)
    assert a == b


def test_se_shrinks_like_sqrt_n():
    _, se_small = mc_price(n_paths=10_000, seed=1, **CASE)
    _, se_big = mc_price(n_paths=1_000_000, seed=1, **CASE)
    ratio = se_small / se_big
    assert ratio == pytest.approx(math.sqrt(100), rel=0.15)


def test_put_parity_within_se():
    c, se_c = mc_price(kind="call", n_paths=400_000, seed=3, **CASE)
    p, se_p = mc_price(kind="put", n_paths=400_000, seed=3, **CASE)
    lhs = c - p
    rhs = CASE["S"] - CASE["K"] * math.exp(-CASE["r"] * CASE["T"])
    assert abs(lhs - rhs) < 3 * (se_c + se_p)
