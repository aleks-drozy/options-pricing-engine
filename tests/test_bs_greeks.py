import pytest
from engine.bs import bs_greeks, bs_price

CASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def fd(param, h, kind="call"):
    up = dict(CASE); dn = dict(CASE)
    up[param] += h; dn[param] -= h
    return (bs_price(kind=kind, **up) - bs_price(kind=kind, **dn)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["delta"] == pytest.approx(fd("S", 1e-4, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vega_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["vega"] == pytest.approx(fd("sigma", 1e-5, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_rho_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["rho"] == pytest.approx(fd("r", 1e-6, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_matches_fd(kind):
    # theta = -dV/dT
    g = bs_greeks(kind=kind, **CASE)
    assert g["theta"] == pytest.approx(-fd("T", 1e-6, kind), rel=1e-5)


def test_gamma_matches_second_fd():
    h = 1e-2
    up = dict(CASE); dn = dict(CASE)
    up["S"] += h; dn["S"] -= h
    second = (bs_price(**up) - 2 * bs_price(**CASE) + bs_price(**dn)) / (h * h)
    assert bs_greeks(**CASE)["gamma"] == pytest.approx(second, rel=1e-5)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_greeks_match_fd_with_dividend_yield(kind):
    # q > 0 exercises the e^{-qT} terms that a q=0 case leaves untested.
    case = dict(CASE, q=0.02)

    def fdq(param, h):
        up = dict(case); dn = dict(case)
        up[param] += h; dn[param] -= h
        return (bs_price(kind=kind, **up) - bs_price(kind=kind, **dn)) / (2 * h)

    g = bs_greeks(kind=kind, **case)
    assert g["delta"] == pytest.approx(fdq("S", 1e-4), rel=1e-6)
    assert g["vega"] == pytest.approx(fdq("sigma", 1e-5), rel=1e-6)
    assert g["theta"] == pytest.approx(-fdq("T", 1e-6), rel=1e-5)


def test_call_delta_bounds_and_put_call_delta_relation():
    import math
    g_c = bs_greeks(kind="call", **CASE)["delta"]
    g_p = bs_greeks(kind="put", **CASE)["delta"]
    assert 0 < g_c < 1 and -1 < g_p < 0
    assert g_c - g_p == pytest.approx(math.exp(-CASE["q"] * CASE["T"]), rel=1e-12)
