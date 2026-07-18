"""Generate charts/*.png. Run: python scripts/make_charts.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from engine.binomial import crr_price
from engine.bs import bs_greeks, bs_price
from engine.montecarlo import mc_price
from market.smile import compute_smile
from market.snapshot import load_snapshot

OUT = Path("charts"); OUT.mkdir(exist_ok=True)
BASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
S_GRID = np.linspace(50, 150, 201)


def payoff():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, kind in zip(axes, ("call", "put")):
        pay = np.maximum(S_GRID - 100, 0) if kind == "call" else np.maximum(100 - S_GRID, 0)
        prices = [bs_price(float(s), 100, 1.0, 0.05, 0.2, 0.0, kind) for s in S_GRID]
        ax.plot(S_GRID, pay, label="payoff at expiry")
        ax.plot(S_GRID, prices, label="BS value, T=1y")
        ax.set_title(f"{kind} K=100"); ax.set_xlabel("S"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "payoff.png", dpi=150); plt.close(fig)


def convergence_tree():
    bs = bs_price(**BASE)
    steps = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000]
    errs = [abs(crr_price(steps=n, **BASE) - bs) for n in steps]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(steps, errs, "o-", label="|tree - BS|")
    ax.loglog(steps, [errs[0] * steps[0] / n for n in steps], "--", label="O(1/N) reference")
    ax.set_xlabel("tree steps N"); ax.set_ylabel("abs error"); ax.legend()
    ax.set_title("CRR converges to Black-Scholes")
    fig.tight_layout(); fig.savefig(OUT / "convergence_tree.png", dpi=150); plt.close(fig)


def convergence_mc():
    bs = bs_price(**BASE)
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    errs, ses = [], []
    for n in ns:
        price, se = mc_price(n_paths=n, seed=5, **BASE)
        errs.append(abs(price - bs)); ses.append(se)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.loglog(ns, errs, "o-", label="|MC - BS| (seed 5)")
    ax.loglog(ns, ses, "--", label="standard error")
    ax.loglog(ns, [ses[0] * (ns[0] / n) ** 0.5 for n in ns], ":", label="O(1/sqrt N) reference")
    ax.set_xlabel("paths N"); ax.legend(); ax.set_title("Monte Carlo error shrinks like 1/sqrt(N)")
    fig.tight_layout(); fig.savefig(OUT / "convergence_mc.png", dpi=150); plt.close(fig)


def greeks():
    names = ["delta", "gamma", "vega", "theta"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for ax, name in zip(axes.flat, names):
        for kind in ("call", "put"):
            vals = [bs_greeks(float(s), 100, 1.0, 0.05, 0.2, 0.0, kind)[name] for s in S_GRID]
            ax.plot(S_GRID, vals, label=kind)
        ax.set_title(name); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "greeks.png", dpi=150); plt.close(fig)


def smile():
    snap = load_snapshot("data/spy_chain.json")
    sm = compute_smile(snap)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    shown = 0
    for entry in sm["per_expiry"]:
        pts = [(p["moneyness"], p["iv"]) for p in entry["points"]
               if p["iv"] is not None and p["kind"] == "call"]
        if len(pts) >= 5 and shown < 4:
            xs, ys = zip(*sorted(pts))
            axes[0].plot(xs, [y * 100 for y in ys], "o-", ms=3, label=f'{entry["dte"]}d')
            shown += 1
    axes[0].set_xlabel("moneyness K/S"); axes[0].set_ylabel("implied vol %")
    axes[0].set_title(f'SPY smile ({snap["fetched_utc"][:10]}) - flat under BS, not in reality')
    axes[0].legend()
    ts = sm["term_structure"]
    axes[1].plot([t["dte"] for t in ts], [t["atm_iv"] * 100 for t in ts], "o-")
    axes[1].set_xlabel("days to expiry"); axes[1].set_ylabel("ATM implied vol %")
    axes[1].set_title("IV term structure")
    fig.tight_layout(); fig.savefig(OUT / "smile.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    payoff(); convergence_tree(); convergence_mc(); greeks(); smile()
    print("charts written to charts/")
