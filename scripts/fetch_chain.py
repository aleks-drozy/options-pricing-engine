"""MANUAL script: fetch SPY option chain -> data/spy_chain.json.

Never run by tests or CI (network). Re-run to refresh the committed snapshot.
Usage: python scripts/fetch_chain.py
"""
import datetime as dt
import json
import math
from pathlib import Path

import yfinance as yf

MAX_EXPIRIES = 12


def main() -> None:
    spy = yf.Ticker("SPY")
    spot = float(spy.fast_info["lastPrice"])
    irx = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
    r = float(irx.iloc[-1]) / 100.0
    divs = spy.dividends
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=365)
    trailing = float(divs[divs.index >= cutoff].sum())
    q = trailing / spot

    today = dt.date.today()
    expiries = []
    for expiry_str in spy.options:
        expiry = dt.date.fromisoformat(expiry_str)
        dte = (expiry - today).days
        if not 7 <= dte <= 400:
            continue
        chain = spy.option_chain(expiry_str)
        quotes = []
        for kind, frame in (("call", chain.calls), ("put", chain.puts)):
            for row in frame.itertuples():
                bid = float(row.bid) if not math.isnan(row.bid) else 0.0
                ask = float(row.ask) if not math.isnan(row.ask) else 0.0
                quotes.append({"strike": float(row.strike), "kind": kind,
                               "bid": bid, "ask": ask})
        expiries.append({"expiry": expiry_str, "dte": dte, "quotes": quotes})
        if len(expiries) >= MAX_EXPIRIES:
            break

    out = {"fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
           "spot": spot, "r": r, "q": q,
           "r_source": "13-week T-bill (^IRX) latest close / 100 via yfinance",
           "q_source": "SPY trailing 12-month dividends / spot via yfinance",
           "expiries": expiries}
    Path("data").mkdir(exist_ok=True)
    Path("data/spy_chain.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    n_quotes = sum(len(e["quotes"]) for e in expiries)
    print(f"spot={spot:.2f} r={r:.4f} q={q:.4f} expiries={len(expiries)} quotes={n_quotes}")


if __name__ == "__main__":
    main()
