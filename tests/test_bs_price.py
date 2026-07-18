import math
import pytest
from engine.bs import bs_price, d1_d2

# Classic textbook case: S=100,K=100,T=1,r=5%,sigma=20%,q=0
CALL_REF = 10.450583572185565
PUT_REF = 5.573526022256971


def test_bs_call_reference():
    assert bs_price(100, 100, 1.0, 0.05, 0.20) == pytest.approx(CALL_REF, rel=1e-12)


def test_bs_put_reference():
    assert bs_price(100, 100, 1.0, 0.05, 0.20, kind="put") == pytest.approx(PUT_REF, rel=1e-12)


def test_put_call_parity_with_yield():
    S, K, T, r, sigma, q = 105, 95, 0.75, 0.04, 0.3, 0.02
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-12)


def test_call_increases_in_sigma_and_S():
    base = bs_price(100, 100, 1, 0.05, 0.2)
    assert bs_price(100, 100, 1, 0.05, 0.3) > base
    assert bs_price(110, 100, 1, 0.05, 0.2) > base


def test_tiny_T_approaches_intrinsic():
    assert bs_price(120, 100, 1e-10, 0.05, 0.2) == pytest.approx(20.0, abs=1e-6)
    assert bs_price(80, 100, 1e-10, 0.05, 0.2, kind="put") == pytest.approx(20.0, abs=1e-6)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(K=0), dict(T=0), dict(sigma=-0.1), dict(kind="straddle"),
])
def test_invalid_inputs_raise(bad):
    kw = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0, kind="call")
    kw.update(bad)
    with pytest.raises(ValueError):
        bs_price(kw["S"], kw["K"], kw["T"], kw["r"], kw["sigma"], kw["q"], kw["kind"])


def test_d1_d2_relation():
    d1, d2 = d1_d2(100, 100, 1.0, 0.05, 0.2, 0.0)
    assert d1 - d2 == pytest.approx(0.2)
