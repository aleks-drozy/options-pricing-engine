"""Load and quality-filter a committed option-chain snapshot.

Filter rules (spec): bid > 0 and ask > 0; spread <= max($0.10, 25% of mid);
moneyness K/S in [0.5, 1.5]; expiry 7..400 DTE. Dropped quotes are COUNTED.
"""
from __future__ import annotations

import json
from pathlib import Path

DTE_MIN, DTE_MAX = 7, 400
MONEYNESS_LO, MONEYNESS_HI = 0.5, 1.5


def load_snapshot(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def filter_quotes(snapshot: dict) -> tuple[list[dict], dict]:
    spot = snapshot["spot"]
    counts = {"total": 0, "kept": 0, "no_bid_or_ask": 0,
              "wide_spread": 0, "moneyness": 0, "dte": 0}
    kept: list[dict] = []
    for exp in snapshot["expiries"]:
        dte_ok = DTE_MIN <= exp["dte"] <= DTE_MAX
        for quote in exp["quotes"]:
            counts["total"] += 1
            if not dte_ok:
                counts["dte"] += 1
                continue
            bid, ask = quote["bid"], quote["ask"]
            if bid <= 0 or ask <= 0:
                counts["no_bid_or_ask"] += 1
                continue
            mid = 0.5 * (bid + ask)
            if ask - bid > max(0.10, 0.25 * mid):
                counts["wide_spread"] += 1
                continue
            moneyness = quote["strike"] / spot
            if not MONEYNESS_LO <= moneyness <= MONEYNESS_HI:
                counts["moneyness"] += 1
                continue
            counts["kept"] += 1
            kept.append({**quote, "mid": mid, "moneyness": moneyness,
                         "dte": exp["dte"], "expiry": exp["expiry"]})
    return kept, counts
