# Options Pricing Engine — Design

**Date:** 2026-07-18 · **Status:** approved pending user spec review
**Repo:** `options-pricing-engine` (public GitHub, `aleks-drozy`)

## Goal

A quant-developer portfolio piece that prices vanilla options three independent ways,
proves the three methods agree (machine-checked gates), computes Greeks, and then
shows the one thing the models can't explain: the real SPY volatility smile.
Narrative: *three pricers, one truth — and the market disagrees with all of them.*

## Scope

**In:**
- European calls/puts: Black-Scholes closed form, Monte Carlo (GBM terminal value,
  seeded NumPy Generator), CRR binomial tree.
- American calls/puts: CRR binomial tree with early exercise.
- Continuous dividend yield `q` supported in all three pricers (SPY has one; a
  pricer that ignores it would misprice the real chain we show).
- Greeks — Delta, Gamma, Vega, Theta, Rho: closed form for BS; central finite
  differences on the tree for American.
- Implied volatility: Brent inversion of BS against real SPY mid prices; smile per
  expiry + IV term structure.
- Interactive explorer (GitHub Pages, self-contained HTML) with a golden-table
  self-check badge.
- Validation gate runner → `results/validation.json`; charts; WRITEUP.md; CI.

**Out (explicitly):** exotics, Longstaff-Schwartz American MC, discrete dividends,
stochastic-vol models (Heston etc.), live trading anything, portfolio/risk metrics
(that's idea #4, a separate project).

## Structure

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
├── tests/             # pytest, NO network
├── run_validate.py    # runs all gates -> results/validation.json, non-zero exit on fail
├── data/spy_chain.json    # committed snapshot (see Market data)
├── results/ · charts/ · docs/   (docs/ = GitHub Pages)
└── README.md · WRITEUP.md · requirements.txt · .github/workflows/tests.yml
```

## Numerical specifications

- **BS closed form** with continuous yield: `d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)`;
  price and all five Greeks analytic. Inputs validated: `S,K,σ,T > 0`, else `ValueError`.
- **CRR tree:** `u = exp(σ√Δt)`, `d = 1/u`, `p = (exp((r-q)Δt) - d)/(u - d)`;
  backward induction; American takes `max(continuation, intrinsic)` per node.
  Raise if `p` falls outside (0,1) (parameter regime where CRR is invalid).
- **Monte Carlo:** terminal-value sampling `S_T = S·exp((r - q - σ²/2)T + σ√T·Z)`,
  discounted payoff mean; returns `(price, std_error)`; `numpy.random.Generator(PCG64(seed))`.
  European only (stated in docstring: American MC needs LSMC — out of scope).
- **Greeks (tree):** central differences — bump `h_S = 0.5% · S`, `h_σ = 1e-3`,
  `h_r = 1e-4`, `h_T = min(1e-3, T/10)`; Gamma from the S-bump triple.
- **Implied vol:** `brentq` on σ ∈ [1e-4, 5.0], tolerance `xtol=1e-10`; verify by
  re-pricing: `|BS(σ_iv) - mid| < 1e-6·S`; unresolvable quotes → NaN, counted, reported.

## Market data (reproducibility contract)

- `scripts/fetch_chain.py` is run **manually once** (and on any future refresh);
  CI and tests NEVER touch the network.
- Snapshot = `data/spy_chain.json`: `{fetched_utc, spot, r, q, expiries: [{expiry,
  dte, quotes: [{strike, kind, bid, ask, mid}]}]}`.
  - `r`: 13-week T-bill yield (^IRX) captured at fetch time; `q`: SPY trailing
    12-month dividend yield, both recorded in the file with their source noted.
- **Filter rules (stated, counted, applied in `market/snapshot.py`):** keep quotes
  with `bid > 0` and `ask > 0`; spread ≤ max($0.10, 25% of mid); moneyness
  `K/S ∈ [0.5, 1.5]`; expiries 7–400 DTE. Filtered counts land in WRITEUP + explorer
  footnote — dropped garbage is reported, never silent.

## Validation gates (`run_validate.py`)

| # | Gate | Pass criterion |
|---|---|---|
| 1 | Put-call parity (BS, closed form) | `|C − P − (S·e^{−qT} − K·e^{−rT})| < 1e−10` across a 200-point grid |
| 2 | Parity (MC) | parity gap < 3× combined std error, same grid sample |
| 3 | Tree → BS convergence | error at N=2000 < error at N=200 for every grid point, and `|tree(2000) − BS| < max(0.01, 0.1% · price)` |
| 4 | MC → BS convergence | with N=1e6: `|MC − BS| < 3·SE`; across 200 seeds at N=1e5, the 95% CI covers BS ≥ 90% of runs |
| 5 | Greeks | BS closed form vs FD on BS: rel. err < 1e−4; tree FD Greeks finite and Delta ∈ [−1,0]∪[0,1] by kind |
| 6 | No-arbitrage | American ≥ European − 1e−9 everywhere; `q=0` ⇒ `|Am call − Eu call| < 1e−9`; both tree N=1000 |
| 7 | IV round-trip | for synthetic BS prices at known σ: recovered σ within 1e−7; on the real snapshot: 100% of kept quotes either resolve or are counted as filtered |

Output `results/validation.json` with per-gate numbers; non-zero exit if any gate
fails; README badge states the gate result honestly.

## Explorer (`docs/index.html`, GitHub Pages)

- Self-contained (inline CSS/JS, data embedded at build time by `build_viz.py`).
- Controls: sliders S ∈ [50,200], K ∈ [50,200], σ ∈ [1%,100%], T ∈ [0.02,3]y,
  r ∈ [0,10%], q ∈ [0,5%]; toggles call/put, European/American.
- Live outputs (recomputed in JS per input): price by method (BS + tree N=500;
  MC shown from embedded precomputed curves with its std-error band), payoff
  diagram at expiry + P&L, Greeks readouts and Delta/Gamma curves vs S.
- **Golden self-check:** `make_golden.py` emits 24 cases (call/put × Eu/Am ×
  ITM/ATM/OTM × short/long T, incl. q>0). On load, JS recomputes all 24;
  badge shows PASS (green) only if BS rel. err < 1e−9 and tree(500) rel. err
  < 1e−6 on every case; otherwise FAIL (red) with the worst case shown.
  The page proves its own numbers or says so.
- Market tab: SPY smile per expiry (IV vs moneyness), term structure, snapshot
  date + filter counts footnote.

## Testing & CI

- pytest, no network, target ~60+ tests: golden values (BS prices/Greeks vs
  hand-checked references), parity properties, monotonicity (price ↑ in σ; call ↑
  in S), degenerate limits (T→0 → intrinsic; σ→0 → discounted forward payoff),
  American-premium cases (deep ITM put), CRR validity guard, IV round-trip,
  snapshot filter unit tests on a fixture chain, golden.json freshness (regenerating
  golden values in-test matches the committed file → explorer can't drift).
- GitHub Actions: `pip install -r requirements-dev.txt && pytest && python run_validate.py`
  on push/PR + `workflow_dispatch` (checkout@v5, setup-python@v6 — Node 24 majors).
- Pages deployed from `docs/` on main (same pattern as football-trajectory).

## Delivery

1. Working engine + gates green locally → initial public commit.
2. Charts + WRITEUP (method, gate results, the smile finding, limitations —
   incl. European-only MC, continuous-yield simplification, snapshot-in-time r/q).
3. Explorer live on GitHub Pages, self-check PASS.
4. Vault: new numbered project folder (18) seeded per conventions; README links
   from portfolio later (separate task, not this build).

## Amendments

- 2026-07-18 (build, Task 7): Gate 5 criterion amended to rel < 1e-4 OR abs < 1e-8 (far-OTM greeks ~1e-6 make the FD reference cancellation-noise-dominated). Gate 3 monotonicity waived below a 1e-9 absolute noise floor (deep-ITM/short-T tree already exact to machine epsilon at 200 steps). Both discovered by the gate run itself; intent of both gates unchanged.
- 2026-07-18 (final review, F1): gate sampling de-aliased from the r/q grid axis (strides 5/10 -> 7/13); this exposed CRR sawtooth oscillation at K=115 with r>0 that strict two-point monotonicity mis-scores — gate 3's shrinkage comparison amended to adjacent-step averages (N, N+1); raw absolute tolerance unchanged. Both the sampling hole and the oscillation were caught by the gates' own final review.
