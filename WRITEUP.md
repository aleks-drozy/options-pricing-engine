# Options Pricing Engine — method, validation, and the smile

## 1. Method

Three independent ways to price the same vanilla option, built so that none of
them can lean on the others:

| Pricer | File | What it does |
|---|---|---|
| Black-Scholes closed form | `engine/bs.py` | Analytic price + all five Greeks from `d1`/`d2` |
| CRR binomial tree | `engine/binomial.py` | Backward induction, European or American exercise |
| Monte Carlo | `engine/montecarlo.py` | GBM terminal-value sampling, European only |

**Black-Scholes.** Standard closed form with a continuous dividend yield `q`
folded into both drift and discounting:

```
d1 = [ln(S/K) + (r - q + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
call = S e^(-qT) N(d1) - K e^(-rT) N(d2)
put  = K e^(-rT) N(-d2) - S e^(-qT) N(-d1)
```

Greeks (delta, gamma, vega, theta, rho) are the analytic derivatives of this
formula, not finite differences — `engine/bs.py:bs_greeks`.

**CRR binomial tree.** `u = e^(sigma*sqrt(dt))`, `d = 1/u`,
`p = (e^((r-q)dt) - d) / (u - d)`, backward induction from terminal payoffs.
American exercise takes `max(continuation, intrinsic)` at every node. The
risk-neutral probability `p` is checked to land in `(0,1)`; outside that range
the parameter regime is invalid for CRR and the call raises rather than
returning a silently wrong number.

**Monte Carlo.** Terminal-value sampling only —
`S_T = S * exp((r - q - sigma^2/2) T + sigma*sqrt(T)*Z)` — discounted payoff
mean, with a `numpy.random.Generator(PCG64(seed))` for reproducibility. Returns
`(price, std_error)` so every MC number in this repo carries its own
uncertainty. European only; American MC needs Longstaff-Schwartz regression,
which is explicitly out of scope (see Limitations).

**Dividend yield `q`.** All three pricers accept it, because the instrument
this repo actually prices — SPY — pays one. A pricer that silently assumed
`q=0` would misprice every real quote in `data/spy_chain.json`.

**Implied volatility.** `engine/iv.py` inverts Black-Scholes against a market
mid price with `scipy.optimize.brentq` over `sigma in [1e-4, 5.0]`
(`xtol=1e-10`), then re-prices at the recovered `sigma` and requires the
residual to be below `1e-6*S` before returning. Prices outside the
no-arbitrage bounds, or quotes brentq can't bracket a root for, raise
`IVError` — counted, never silently dropped (see §3).

## 2. Validation — all 7 gates pass

`run_validate.py` runs seven machine-checked gates and writes
`results/validation.json`. These are the actual numbers from the last run:

| # | Gate | Pass criterion | Result |
|---|---|---|---|
| 1 | Put-call parity (BS) | `\|C - P - (S e^{-qT} - K e^{-rT})\| < 1e-10` across a 200-point grid | **PASS** — worst gap `2.84e-14` |
| 2 | Put-call parity (MC) | parity gap < 3x combined std error, same grid | **PASS** — 16/16 sample points, 0 failures |
| 3 | Tree -> BS convergence | adjacent-step-averaged error (mean of N, N+1 prices) at N=2000 < that at N=200 (or below a 1e-9 noise floor), and raw `\|tree(2000) - BS\| < max(0.01, 0.1%*price)` | **PASS** — 0 failures across the sampled grid |
| 4 | MC -> BS convergence | at N=1e6, `\|MC - BS\| < 3*SE`; across 200 seeds at N=1e5, 95% CI covers BS >= 90% of runs | **PASS** — error `0.00858` vs SE `0.01472` at N=1e6; CI coverage **95.0%** over 200 seeds |
| 5 | Greeks | BS closed form vs central FD: rel. err < 1e-4 OR abs. err < 1e-8; tree FD Greeks finite, delta in range | **PASS** — worst rel. error `5.57e-6`, 0 comparisons needed the absolute floor on the current sampling; American put tree Greeks sane (delta `-0.412`, gamma `0.051`, vega `37.48`, theta `-2.237`, rho `-30.21`) |
| 6 | No-arbitrage | American >= European - 1e-9 everywhere; `q=0` => `\|Am call - Eu call\| < 1e-9` | **PASS** — 0 violations, worst `q=0` call gap `4.55e-13` |
| 7 | IV round-trip | synthetic BS prices recover `sigma` within 1e-7; on the real SPY snapshot, every kept quote either resolves or is counted as a filtered failure, and >= 80% of kept quotes resolve | **PASS** — synthetic worst error `8.42e-13`; snapshot: 3587 kept, 3485 resolved (97.2%), 102 counted failures, `3485 + 102 = 3587` |

A final-review audit of the gates themselves caught that every sampling stride
was a multiple of the parameter grid's r/q period — so the sampled gates had
only ever seen `r=0, q=0` — and de-aliasing the strides immediately exposed a
real CRR sawtooth oscillation at nonzero `r` that pointwise monotonicity
mis-scores: the gates audit the engine, and auditing the gates found real
things in both (see §4, third amendment).

Full machine output: [`results/validation.json`](results/validation.json).

## 3. The smile finding

Black-Scholes assumes one constant `sigma` prices every strike and expiry off
the same underlying. If that were true, inverting the market's own quoted
prices back to volatility (§1, `implied_vol`) would return the same number
everywhere. It doesn't.

**Snapshot.** SPY, spot **743.29**, `r` **3.71%** (13-week T-bill, `^IRX`
close), `q` **1.01%** (trailing 12-month dividend yield), fetched
2026-07-18T13:46 UTC across **12 expiries**, dte **9-90**.

**Filter counts** (`market/snapshot.py`, applied before anything below is
computed):

| | count |
|---|---|
| total quotes | 3755 |
| dropped — no valid bid/ask | 116 |
| dropped — spread too wide | 0 |
| dropped — moneyness outside [0.5, 1.5] | 52 |
| dropped — DTE outside [7, 400] | 0 |
| **kept** | **3587** |

Of the 3587 kept quotes, 3485 resolved to an implied vol and **102 did not** —
counted as `iv_failures`, not silently discarded. That's quotes whose mid
price falls outside brentq's bracketable range or fails the re-price check
(typically wide-but-passing-filter quotes at the moneyness edge, or a stale
cross where mid violates the model's own no-arbitrage bounds).

**Term structure.** The at-the-money IV is close to flat across the whole
9-90 DTE window — roughly 13-16%, with no obvious trend by expiry (see
`charts/smile.png`, right panel). Flat-vol BS would predict exactly this *for
a single strike held at the money* — that part of the assumption survives.

**Skew is where it breaks.** Every expiry with enough kept quotes to plot
shows a clear smirk, not a flat line: out-of-the-money puts (`K/S` below 1)
carry meaningfully higher implied vol than at-the-money, out-of-the-money
calls carry somewhat higher IV than at-the-money too, and the effect is
asymmetric — puts are steeper. That asymmetry is the market pricing in
exactly what geometric Brownian motion assumes away: a lognormal terminal
distribution has thin tails and no skew, and equity index markets have
neither — crashes are more common and more violent than a lognormal admits,
so downside protection (OTM puts) trades at a persistent premium over what a
single flat `sigma` would say it's worth. That premium *is* the smile.

Nominal IVs at the extreme low-moneyness edge of a few longer-dated expiries
(e.g. 62d, 90d) print very high (60-240%) — worth flagging as a
data-quality caveat rather than a pricing finding: those are thin, wide-quote
strikes right at the moneyness filter's boundary, where the passing-but-wide
spread makes the implied vol noisy rather than a clean read on tail pricing.
The skew itself is visible well inside that noisy edge, on strikes with tight
markets, so the finding doesn't depend on it.

Charts: `charts/smile.png` (per-expiry smile + term structure),
`charts/payoff.png`, `charts/greeks.png`, `charts/convergence_tree.png`,
`charts/convergence_mc.png`.

## 4. Limitations

- **European-only Monte Carlo.** `engine/montecarlo.py` prices European
  options exclusively; American exercise under MC needs Longstaff-Schwartz
  regression, which is out of scope (stated in the module docstring). American
  prices and Greeks in this repo come only from the CRR tree.
- **Continuous-yield simplification.** All three pricers model SPY's dividend
  as a continuous yield `q`, not the discrete quarterly cash dividends SPY
  actually pays. This is standard for index options but is an approximation,
  not the real payment schedule.
- **Snapshot `r`/`q` are point-in-time, not term structures.** A single `r`
  (13-week T-bill close) and a single `q` (trailing 12-month yield) are used
  for every expiry in the chain, from 9 to 90 DTE. Real term structures for
  both the risk-free rate and forward dividend expectations vary by maturity;
  this snapshot doesn't capture that.
- **Quotes are Friday-close, fetched 12h+ stale.** The snapshot was pulled
  Saturday 2026-07-18 at 13:46 UTC, when markets are closed — so every quote
  reflects Friday 2026-07-17's close, already 12+ hours old at fetch time and
  older by the time anyone reads this.
- **Term structure coverage is short-dated only.** `scripts/fetch_chain.py`
  caps at `MAX_EXPIRIES = 12`, taken nearest-dated first, so the snapshot
  spans only ~9-90 DTE. There's no LEAPS-length expiry in this dataset; any
  claim about the smile or term structure is scoped to that window and
  shouldn't be extrapolated further out.
- **No discrete dividends.** Follows directly from the continuous-yield
  point above — no pricer here models an actual ex-dividend date or cash
  amount.
- **American Greeks carry FD-on-tree noise.** `crr_greeks` (§1) uses central
  finite differences on the binomial tree rather than a closed form. Gate 5
  keeps an absolute-error floor (`rel < 1e-4 OR abs < 1e-8`) because FD
  cancellation noise dominates a pure relative-error test when the true Greek
  is tiny (far-OTM Greeks are ~1e-6 in magnitude); rescues are counted in
  `results/validation.json` (`abs_rescued` — 0 on the current sampling). Tree
  Greeks are therefore reliable in magnitude and sign but noisier at the
  margins than the BS closed form.
- **Three gate criteria have been amended after gate runs**, each discovered
  by running (or auditing) the gates themselves rather than decided in
  advance (full text in the spec's Amendments section,
  [`docs/specs/2026-07-18-options-pricing-engine.md`](docs/specs/2026-07-18-options-pricing-engine.md)):
  - **Gate 5 (Greeks):** amended from a pure relative-error criterion to
    `rel < 1e-4 OR abs < 1e-8`. Far-OTM Greeks are ~1e-6 in magnitude, which
    makes the finite-difference reference itself cancellation-noise-dominated
    — a pure relative test at that scale compares rounding noise, not the
    model. The absolute floor rescues exactly those cases without weakening
    the criterion anywhere it has signal to check.
  - **Gate 3 (tree convergence):** the monotonic-improvement requirement
    (error at N=2000 < error at N=200) was waived below a 1e-9 absolute noise
    floor. Deep-ITM, short-`T` grid points are already exact to machine
    epsilon at 200 steps, so "must strictly improve" has nothing left to
    measure — it was failing on rounding, not on the tree actually getting
    worse.
  - **Gate 3 again (final review):** the gate samplers' strides were
    multiples of the parameter grid's r/q period, so gates 2/3/5/6 had only
    ever sampled `r=0, q=0` points. De-aliasing the strides (5/10 → 7/13,
    coprime with the period) surfaced CRR's well-known sawtooth error
    oscillation at `K=115` with `r>0`: the tree's error alternates phase as
    the strike crosses tree layers, so pointwise monotonicity between two
    arbitrary step counts mis-scores it. The shrinkage comparison now uses
    adjacent-step averages (mean of the N and N+1 prices, standard
    Broadie-Detemple-style smoothing); the raw absolute tolerance is
    unchanged.

  None of the amendments changes what any gate is actually checking for.

## Links

- [`README.md`](README.md) — quick start, verdict table, live explorer
- [`docs/specs/2026-07-18-options-pricing-engine.md`](docs/specs/2026-07-18-options-pricing-engine.md) — the design spec, committed before the engine was written, including the Amendments log
- [`results/validation.json`](results/validation.json) — machine-generated gate output
