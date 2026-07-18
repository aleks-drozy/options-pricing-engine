import pytest
from engine.binomial import crr_price, crr_greeks
from engine.bs import bs_price, bs_greeks

CASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def test_european_converges_to_bs():
    bs = bs_price(**CASE)
    err_200 = abs(crr_price(steps=200, **CASE) - bs)
    err_2000 = abs(crr_price(steps=2000, **CASE) - bs)
    assert err_2000 < err_200
    assert err_2000 < 0.01


def test_american_geq_european():
    for kind in ("call", "put"):
        eu = crr_price(kind=kind, american=False, steps=800, **{k: v for k, v in CASE.items() if k != "kind"})
        am = crr_price(kind=kind, american=True, steps=800, **{k: v for k, v in CASE.items() if k != "kind"})
        assert am >= eu - 1e-9


def test_american_call_no_dividend_equals_european():
    eu = crr_price(kind="call", american=False, steps=1000, **CASE)
    am = crr_price(kind="call", american=True, steps=1000, **CASE)
    assert am == pytest.approx(eu, abs=1e-9)


def test_deep_itm_american_put_carries_premium():
    kw = dict(S=60, K=100, T=1.0, r=0.08, sigma=0.2, q=0.0, kind="put")
    eu = crr_price(american=False, steps=800, **kw)
    am = crr_price(american=True, steps=800, **kw)
    assert am > eu + 0.05
    assert am >= 40.0 - 1e-9  # never below intrinsic


def test_invalid_regime_raises():
    # sigma tiny + huge drift -> p outside (0,1)
    with pytest.raises(ValueError):
        crr_price(100, 100, 1.0, 0.9, 0.01, steps=10)


def test_tree_greeks_close_to_bs_for_european():
    g_tree = crr_greeks(steps=500, **CASE)
    g_bs = bs_greeks(**CASE)
    assert g_tree["delta"] == pytest.approx(g_bs["delta"], abs=2e-3)
    assert g_tree["vega"] == pytest.approx(g_bs["vega"], rel=2e-2)
    assert g_tree["rho"] == pytest.approx(g_bs["rho"], rel=2e-2)
