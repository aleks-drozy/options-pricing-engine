"""Implied-vol smile and term structure from a filtered snapshot."""
from __future__ import annotations

from engine.iv import IVError, implied_vol
from market.snapshot import filter_quotes


def compute_smile(snapshot: dict) -> dict:
    spot, r, q = snapshot["spot"], snapshot["r"], snapshot["q"]
    kept, _counts = filter_quotes(snapshot)
    failures = 0
    by_expiry: dict[str, dict] = {}
    for quote in kept:
        T = quote["dte"] / 365.0
        try:
            iv = implied_vol(quote["mid"], spot, quote["strike"], T, r, q, quote["kind"])
        except IVError:
            iv = None
            failures += 1
        entry = by_expiry.setdefault(
            quote["expiry"], {"expiry": quote["expiry"], "dte": quote["dte"], "points": []})
        entry["points"].append({"strike": quote["strike"], "moneyness": quote["moneyness"],
                                "kind": quote["kind"], "mid": quote["mid"], "iv": iv})
    per_expiry = sorted(by_expiry.values(), key=lambda e: e["dte"])
    term = []
    for entry in per_expiry:
        candidates = [p for p in entry["points"] if p["iv"] is not None]
        if candidates:
            atm = min(candidates, key=lambda p: (abs(p["moneyness"] - 1.0), p["kind"] != "call"))
            term.append({"expiry": entry["expiry"], "dte": entry["dte"], "atm_iv": atm["iv"]})
    return {"per_expiry": per_expiry, "term_structure": term, "iv_failures": failures}
