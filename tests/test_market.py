from pathlib import Path
import pytest
from market.snapshot import load_snapshot, filter_quotes
from market.smile import compute_smile

FIX = Path(__file__).parent / "fixtures" / "mini_chain.json"


def test_filter_counts_add_up():
    snap = load_snapshot(FIX)
    kept, counts = filter_quotes(snap)
    assert counts["total"] == 8
    assert counts["kept"] == len(kept) == 4
    assert counts["no_bid_or_ask"] == 1
    assert counts["wide_spread"] == 1
    assert counts["moneyness"] == 1
    assert counts["dte"] == 1


def test_kept_quotes_have_mid_and_moneyness():
    kept, _ = filter_quotes(load_snapshot(FIX))
    q = next(x for x in kept if x["strike"] == 90)
    assert q["mid"] == pytest.approx(11.70)
    assert q["moneyness"] == pytest.approx(0.9)
    assert q["dte"] == 90


def test_smile_recovers_ivs():
    smile = compute_smile(load_snapshot(FIX))
    assert len(smile["per_expiry"]) == 1
    points = smile["per_expiry"][0]["points"]
    ivs = [p["iv"] for p in points if p["iv"] is not None]
    assert len(ivs) >= 3
    assert all(0.01 < iv < 1.0 for iv in ivs)
    assert smile["term_structure"][0]["atm_iv"] == pytest.approx(
        next(p["iv"] for p in points if p["strike"] == 100 and p["kind"] == "call"), rel=1e-9)
