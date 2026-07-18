import pytest
from engine.iv import implied_vol, IVError
from engine.bs import bs_price


@pytest.mark.parametrize("sigma", [0.05, 0.2, 0.8, 2.5])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_round_trip(sigma, kind):
    S, K, T, r, q = 100, 110, 0.5, 0.03, 0.01
    price = bs_price(S, K, T, r, sigma, q, kind)
    assert implied_vol(price, S, K, T, r, q, kind) == pytest.approx(sigma, abs=1e-7)


def test_below_intrinsic_raises():
    # call lower bound: S e^-qT - K e^-rT ~= 100 - 95*e^-0.03 > 7
    with pytest.raises(IVError):
        implied_vol(0.01, 100, 95, 1.0, 0.03, 0.0, "call")


def test_above_upper_bound_raises():
    with pytest.raises(IVError):
        implied_vol(101.0, 100, 95, 1.0, 0.03, 0.0, "call")  # call worth more than S


def test_negative_price_raises():
    with pytest.raises(IVError):
        implied_vol(-1.0, 100, 100, 1.0, 0.03, 0.0, "call")
