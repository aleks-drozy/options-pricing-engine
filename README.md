# Options Pricing Engine

[![tests](https://github.com/aleks-drozy/options-pricing-engine/actions/workflows/tests.yml/badge.svg)](https://github.com/aleks-drozy/options-pricing-engine/actions/workflows/tests.yml)

**▶ Live explorer: [aleks-drozy.github.io/options-pricing-engine](https://aleks-drozy.github.io/options-pricing-engine/)** — sliders for S, K, sigma, T, r, q; live BS/tree/MC prices, payoff + Greeks, and a self-check badge that re-verifies its own numbers on every load.

**Three pricers, one truth — and the market disagrees with all of them.**

Black-Scholes closed form, a CRR binomial tree (European and American), and a
seeded Monte Carlo simulation all price the same vanilla option. Seven
machine-checked gates prove the three agree with each other to within their
own numerical tolerances. Then the engine turns on a real SPY option chain and
inverts the market's own prices back to volatility — and the market quotes a
different `sigma` for every strike. Flat-vol Black-Scholes can't explain that;
the smile is the market pricing in the fat tails and skew that a lognormal
model assumes away. See [WRITEUP.md](WRITEUP.md) for the full finding.

## Verdict: 7/7 gates PASS

| # | Gate | Result |
|---|---|---|
| 1 | Put-call parity (BS closed form) | ✓ worst gap `2.84e-14` (< 1e-10) |
| 2 | Put-call parity (Monte Carlo) | ✓ 16/16 grid points, 0 failures |
| 3 | Tree → BS convergence | ✓ 0 failures; averaged error shrinks N=200 → N=2000 |
| 4 | MC → BS convergence | ✓ error `0.00858` vs SE `0.01472` at N=1e6; 95.0% CI coverage over 200 seeds |
| 5 | Greeks (closed form vs FD) | ✓ worst rel. error `5.57e-6`; American tree Greeks sane |
| 6 | No-arbitrage (American ≥ European) | ✓ 0 violations, worst `q=0` gap `4.55e-13` |
| 7 | IV round-trip (synthetic + real SPY snapshot) | ✓ synthetic worst error `8.42e-13`; 3587 kept quotes, 3485 resolved (97.2%) + 102 counted failures |

Real numbers, regenerated from [`results/validation.json`](results/validation.json)
by `run_validate.py` — not hand-typed. Three gate criteria have been amended
after gate runs themselves surfaced numerical edge cases; see
[WRITEUP.md](WRITEUP.md#4-limitations) for all three, with rationale.

This is a **research project, not trading advice.**

## Quick start

```bash
pip install -r requirements-dev.txt

python -m pytest              # 56 tests, no network
python run_validate.py        # -> results/validation.json (the 7-gate verdict above)

python -m scripts.make_charts # -> charts/*.png
python -m scripts.build_viz   # -> docs/index.html (the live explorer)
```

Run these as modules (`python -m scripts.x`), not as scripts
(`python scripts/x.py`) — the scripts import from the repo root
(`engine`, `market`, `validation`), which only resolves correctly when
Python's working directory is on the path as a package, i.e. via `-m`.

`data/spy_chain.json` is a committed snapshot; `scripts/fetch_chain.py` (which
refreshes it via `yfinance`) is run manually and never touched by CI or tests.

## Project structure

```
options-pricing-engine/
├── engine/
│   ├── bs.py          # bs_price(S,K,T,r,sigma,q,kind) + closed-form greeks
│   ├── binomial.py    # crr_price(..., steps, american: bool) + FD greeks
│   ├── montecarlo.py  # mc_price(..., n_paths, seed) -> (price, std_error)
│   └── iv.py          # implied_vol(price, S,K,T,r,q,kind) via brentq
├── market/
│   ├── snapshot.py    # load data/spy_chain.json; quote-quality filter
│   └── smile.py       # per-expiry IV smile + term structure tables
├── validation/
│   ├── parity.py      # put-call parity gate
│   ├── convergence.py # tree->BS and MC->BS gates
│   ├── greeks_check.py# closed-form vs FD gate
│   └── noarb.py       # American>=European; Am call == Eu call when q=0
├── scripts/
│   ├── fetch_chain.py # yfinance pull -> data/spy_chain.json (manual, not CI)
│   ├── make_charts.py # charts/*.png
│   ├── make_golden.py # viz golden table -> results/golden.json
│   └── build_viz.py   # inject data+golden into viz/template.html -> docs/index.html
├── viz/template.html
├── tests/             # pytest, no network, 56 tests
├── run_validate.py    # runs all gates -> results/validation.json, non-zero exit on fail
├── data/spy_chain.json    # committed SPY snapshot
├── results/ · charts/ · docs/   (docs/ = GitHub Pages, self-check PASS 24/24)
└── README.md · WRITEUP.md · requirements.txt · .github/workflows/tests.yml
```

## Further reading

- [WRITEUP.md](WRITEUP.md) — method, the full 7-gate results table, the smile
  finding, and limitations
- [docs/superpowers/specs/2026-07-18-options-pricing-engine-design.md](docs/superpowers/specs/2026-07-18-options-pricing-engine-design.md) — the design spec, including the amendments log
- [docs/superpowers/plans/2026-07-18-options-pricing-engine.md](docs/superpowers/plans/2026-07-18-options-pricing-engine.md) — the 11-task implementation plan
