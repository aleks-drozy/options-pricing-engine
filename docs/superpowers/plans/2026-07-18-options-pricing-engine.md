# Options Pricing Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Vanilla option pricer (BS / Monte Carlo / CRR binomial, European + American) with Greeks, seven machine-checked validation gates, a committed SPY chain snapshot driving an implied-vol smile, and a GitHub Pages explorer with a golden-table self-check.

**Architecture:** Flat Python package dirs (`engine/`, `market/`, `validation/`, `scripts/`) mirroring the football-trajectory repo layout; pure functions over classes; every numerical claim enforced by `run_validate.py` gates; explorer is a build-time-injected self-contained HTML page.

**Tech Stack:** Python 3.12, NumPy, SciPy (brentq only), matplotlib, pytest, GitHub Actions (checkout@v5 / setup-python@v6), GitHub Pages from `docs/`.

## Global Constraints

- All work happens at the repo root (git repo exists; spec + this plan committed).
- Tests NEVER touch the network. `scripts/fetch_chain.py` is manual-only, never imported by tests or CI.
- All pricers take `(S, K, T, r, sigma, q=0.0, kind="call")` in that order; `kind` ∈ {"call","put"}; extra args after. Validate `S,K,sigma,T > 0`, `kind` valid → `ValueError`.
- Continuous dividend yield `q` in every pricer.
- Vega per 1.0 of vol; Theta per year; Rho per 1.0 of rate. State in docstrings.
- MC is European-only (LSMC out of scope) — say so in `mc_price` docstring.
- Seeds: `np.random.default_rng(seed)`; no `Date.now`-style nondeterminism anywhere in tests.
- Windows: write all files UTF-8; never rely on cp1252 (known machine trap).
- Commits: conventional messages, each ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.

---

### Task 1: Scaffold + Black-Scholes price

**Files:**
- Create: `requirements.txt`, `requirements-dev.txt`, `pytest.ini`, `.gitignore`, `.github/workflows/tests.yml`, `engine/__init__.py`, `engine/bs.py`, `tests/test_bs_price.py`

**Interfaces:**
- Produces: `engine.bs.bs_price(S, K, T, r, sigma, q=0.0, kind="call") -> float`; `engine.bs.d1_d2(S, K, T, r, sigma, q) -> tuple[float, float]`; `engine.bs._norm_cdf(x) -> float`, `engine.bs._norm_pdf(x) -> float`; `engine.bs.validate_inputs(S, K, T, sigma, kind)` (raises ValueError).

- [ ] **Step 1: Scaffold files**

`requirements.txt`:
```
numpy>=2.0
scipy>=1.13
matplotlib>=3.9
```
`requirements-dev.txt`:
```
-r requirements.txt
pytest>=8.0
```
`pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -q
```
`.gitignore`:
```
__pycache__/
*.pyc
.venv/
.pytest_cache/
```
`.github/workflows/tests.yml`:
```yaml
name: tests
on:
  push: {branches: [main]}
  pull_request:
  workflow_dispatch:
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - uses: actions/setup-python@v6
        with: {python-version: '3.12', cache: pip}
      - run: pip install -r requirements-dev.txt
      - run: python -m pytest
      - run: python run_validate.py
```
(`run_validate.py` arrives in Task 7; until then CI would fail on that line — acceptable, repo isn't pushed until Task 11.)

`engine/__init__.py`: empty file.

- [ ] **Step 2: Write failing tests** — `tests/test_bs_price.py`:

```python
import math
import pytest
from engine.bs import bs_price, d1_d2

# Classic textbook case: S=100,K=100,T=1,r=5%,sigma=20%,q=0
CALL_REF = 10.450583572185565
PUT_REF = 5.573526022256971


def test_bs_call_reference():
    assert bs_price(100, 100, 1.0, 0.05, 0.20) == pytest.approx(CALL_REF, rel=1e-12)


def test_bs_put_reference():
    assert bs_price(100, 100, 1.0, 0.05, 0.20, kind="put") == pytest.approx(PUT_REF, rel=1e-12)


def test_put_call_parity_with_yield():
    S, K, T, r, sigma, q = 105, 95, 0.75, 0.04, 0.3, 0.02
    c = bs_price(S, K, T, r, sigma, q, "call")
    p = bs_price(S, K, T, r, sigma, q, "put")
    assert c - p == pytest.approx(S * math.exp(-q * T) - K * math.exp(-r * T), abs=1e-12)


def test_call_increases_in_sigma_and_S():
    base = bs_price(100, 100, 1, 0.05, 0.2)
    assert bs_price(100, 100, 1, 0.05, 0.3) > base
    assert bs_price(110, 100, 1, 0.05, 0.2) > base


def test_tiny_T_approaches_intrinsic():
    assert bs_price(120, 100, 1e-10, 0.05, 0.2) == pytest.approx(20.0, abs=1e-6)
    assert bs_price(80, 100, 1e-10, 0.05, 0.2, kind="put") == pytest.approx(20.0, abs=1e-6)


@pytest.mark.parametrize("bad", [
    dict(S=-1), dict(K=0), dict(T=0), dict(sigma=-0.1), dict(kind="straddle"),
])
def test_invalid_inputs_raise(bad):
    kw = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0, kind="call")
    kw.update(bad)
    with pytest.raises(ValueError):
        bs_price(kw["S"], kw["K"], kw["T"], kw["r"], kw["sigma"], kw["q"], kw["kind"])


def test_d1_d2_relation():
    d1, d2 = d1_d2(100, 100, 1.0, 0.05, 0.2, 0.0)
    assert d1 - d2 == pytest.approx(0.2)
```

- [ ] **Step 3: Run to verify failure** — `python -m pytest tests/test_bs_price.py -v` → FAIL (`ModuleNotFoundError: engine.bs`).

- [ ] **Step 4: Implement** — `engine/bs.py`:

```python
"""Black-Scholes closed-form pricing (continuous dividend yield q).

Conventions: T in years; sigma per sqrt(year); r, q continuously compounded.
"""
from __future__ import annotations

import math

_SQRT_2PI = math.sqrt(2.0 * math.pi)
_KINDS = ("call", "put")


def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def validate_inputs(S: float, K: float, T: float, sigma: float, kind: str) -> None:
    if S <= 0 or K <= 0:
        raise ValueError(f"S and K must be positive (got S={S}, K={K})")
    if T <= 0:
        raise ValueError(f"T must be positive (got {T})")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma})")
    if kind not in _KINDS:
        raise ValueError(f"kind must be one of {_KINDS} (got {kind!r})")


def d1_d2(S: float, K: float, T: float, r: float, sigma: float, q: float) -> tuple[float, float]:
    srt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / srt
    return d1, d1 - srt


def bs_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call") -> float:
    validate_inputs(S, K, T, sigma, kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    if kind == "call":
        return S * df_q * _norm_cdf(d1) - K * df_r * _norm_cdf(d2)
    return K * df_r * _norm_cdf(-d2) - S * df_q * _norm_cdf(-d1)
```

- [ ] **Step 5: Run to verify pass** — `python -m pytest tests/test_bs_price.py -v` → all PASS.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: scaffold + Black-Scholes price with dividend yield"`

---

### Task 2: Black-Scholes Greeks

**Files:**
- Create: `tests/test_bs_greeks.py`
- Modify: `engine/bs.py` (append)

**Interfaces:**
- Consumes: `bs_price`, `d1_d2`, `_norm_cdf`, `_norm_pdf`, `validate_inputs` from Task 1.
- Produces: `engine.bs.bs_greeks(S, K, T, r, sigma, q=0.0, kind="call") -> dict` with keys `delta, gamma, vega, theta, rho` (floats; vega per 1.0 vol, theta per year, rho per 1.0 rate).

- [ ] **Step 1: Write failing tests** — `tests/test_bs_greeks.py`:

```python
import pytest
from engine.bs import bs_greeks, bs_price

CASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def fd(param, h, kind="call"):
    up = dict(CASE); dn = dict(CASE)
    up[param] += h; dn[param] -= h
    return (bs_price(kind=kind, **up) - bs_price(kind=kind, **dn)) / (2 * h)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_delta_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["delta"] == pytest.approx(fd("S", 1e-4, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_vega_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["vega"] == pytest.approx(fd("sigma", 1e-5, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_rho_matches_fd(kind):
    g = bs_greeks(kind=kind, **CASE)
    assert g["rho"] == pytest.approx(fd("r", 1e-6, kind), rel=1e-6)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_theta_matches_fd(kind):
    # theta = -dV/dT
    g = bs_greeks(kind=kind, **CASE)
    assert g["theta"] == pytest.approx(-fd("T", 1e-6, kind), rel=1e-5)


def test_gamma_matches_second_fd():
    h = 1e-2
    up = dict(CASE); dn = dict(CASE)
    up["S"] += h; dn["S"] -= h
    second = (bs_price(**up) - 2 * bs_price(**CASE) + bs_price(**dn)) / (h * h)
    assert bs_greeks(**CASE)["gamma"] == pytest.approx(second, rel=1e-5)


def test_call_delta_bounds_and_put_call_delta_relation():
    import math
    g_c = bs_greeks(kind="call", **CASE)["delta"]
    g_p = bs_greeks(kind="put", **CASE)["delta"]
    assert 0 < g_c < 1 and -1 < g_p < 0
    assert g_c - g_p == pytest.approx(math.exp(-CASE["q"] * CASE["T"]), rel=1e-12)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_bs_greeks.py -v` → FAIL (`ImportError: bs_greeks`).

- [ ] **Step 3: Implement** — append to `engine/bs.py`:

```python
def bs_greeks(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, kind: str = "call") -> dict:
    """Closed-form Greeks. vega per 1.0 vol, theta per YEAR, rho per 1.0 rate."""
    validate_inputs(S, K, T, sigma, kind)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    df_q = math.exp(-q * T)
    df_r = math.exp(-r * T)
    sqrt_T = math.sqrt(T)
    pdf1 = _norm_pdf(d1)
    gamma = df_q * pdf1 / (S * sigma * sqrt_T)
    vega = S * df_q * pdf1 * sqrt_T
    common_theta = -S * df_q * pdf1 * sigma / (2.0 * sqrt_T)
    if kind == "call":
        delta = df_q * _norm_cdf(d1)
        theta = common_theta + q * S * df_q * _norm_cdf(d1) - r * K * df_r * _norm_cdf(d2)
        rho = K * T * df_r * _norm_cdf(d2)
    else:
        delta = df_q * (_norm_cdf(d1) - 1.0)
        theta = common_theta - q * S * df_q * _norm_cdf(-d1) + r * K * df_r * _norm_cdf(-d2)
        rho = -K * T * df_r * _norm_cdf(-d2)
    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta, "rho": rho}
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_bs_greeks.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: closed-form Black-Scholes Greeks"`

---

### Task 3: CRR binomial tree (European + American) + tree Greeks

**Files:**
- Create: `engine/binomial.py`, `tests/test_binomial.py`

**Interfaces:**
- Consumes: `engine.bs.bs_price`, `engine.bs.validate_inputs`.
- Produces: `engine.binomial.crr_price(S, K, T, r, sigma, q=0.0, kind="call", american=False, steps=1000) -> float`; `engine.binomial.crr_greeks(S, K, T, r, sigma, q=0.0, kind="call", american=False, steps=500) -> dict` (same keys/units as `bs_greeks`).

- [ ] **Step 1: Write failing tests** — `tests/test_binomial.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_binomial.py -v` → FAIL (no module).

- [ ] **Step 3: Implement** — `engine/binomial.py`:

```python
"""Cox-Ross-Rubinstein binomial tree, European and American exercise."""
from __future__ import annotations

import math

import numpy as np

from engine.bs import validate_inputs


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              q: float = 0.0, kind: str = "call", american: bool = False,
              steps: int = 1000) -> float:
    validate_inputs(S, K, T, sigma, kind)
    if steps < 1:
        raise ValueError(f"steps must be >= 1 (got {steps})")
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"CRR risk-neutral probability {p:.4f} outside (0,1); invalid parameter regime")
    disc = math.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(steps + 1)
    ST = S * u ** j * d ** (steps - j)
    values = np.maximum(sign * (ST - K), 0.0)
    for i in range(steps, 0, -1):
        values = disc * (p * values[1:] + (1.0 - p) * values[:-1])
        if american:
            jj = np.arange(i)
            S_i = S * u ** jj * d ** ((i - 1) - jj)
            values = np.maximum(values, np.maximum(sign * (S_i - K), 0.0))
    return float(values[0])


def crr_greeks(S: float, K: float, T: float, r: float, sigma: float,
               q: float = 0.0, kind: str = "call", american: bool = False,
               steps: int = 500) -> dict:
    """Central finite-difference Greeks on the tree. Same units as bs_greeks."""
    def price(**over):
        kw = dict(S=S, K=K, T=T, r=r, sigma=sigma, q=q)
        kw.update(over)
        return crr_price(kind=kind, american=american, steps=steps, **kw)

    h_S = 0.005 * S
    h_sig = 1e-3
    h_r = 1e-4
    h_T = min(1e-3, T / 10.0)
    base = price()
    up_S, dn_S = price(S=S + h_S), price(S=S - h_S)
    return {
        "delta": (up_S - dn_S) / (2 * h_S),
        "gamma": (up_S - 2 * base + dn_S) / (h_S * h_S),
        "vega": (price(sigma=sigma + h_sig) - price(sigma=sigma - h_sig)) / (2 * h_sig),
        "theta": -(price(T=T + h_T) - price(T=T - h_T)) / (2 * h_T),
        "rho": (price(r=r + h_r) - price(r=r - h_r)) / (2 * h_r),
    }
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_binomial.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: CRR binomial tree with American exercise and FD Greeks"`

---

### Task 4: Monte Carlo pricer

**Files:**
- Create: `engine/montecarlo.py`, `tests/test_montecarlo.py`

**Interfaces:**
- Consumes: `engine.bs.validate_inputs`, `engine.bs.bs_price`.
- Produces: `engine.montecarlo.mc_price(S, K, T, r, sigma, q=0.0, kind="call", n_paths=100_000, seed=0) -> tuple[float, float]` returning `(price, std_error)`.

- [ ] **Step 1: Write failing tests** — `tests/test_montecarlo.py`:

```python
import math
import pytest
from engine.montecarlo import mc_price
from engine.bs import bs_price

CASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def test_mc_within_3se_of_bs():
    price, se = mc_price(n_paths=200_000, seed=42, **CASE)
    assert abs(price - bs_price(**CASE)) < 3 * se


def test_mc_deterministic_given_seed():
    a = mc_price(n_paths=50_000, seed=7, **CASE)
    b = mc_price(n_paths=50_000, seed=7, **CASE)
    assert a == b


def test_se_shrinks_like_sqrt_n():
    _, se_small = mc_price(n_paths=10_000, seed=1, **CASE)
    _, se_big = mc_price(n_paths=1_000_000, seed=1, **CASE)
    ratio = se_small / se_big
    assert ratio == pytest.approx(math.sqrt(100), rel=0.15)


def test_put_parity_within_se():
    c, se_c = mc_price(kind="call", n_paths=400_000, seed=3, **CASE)
    p, se_p = mc_price(kind="put", n_paths=400_000, seed=3, **CASE)
    lhs = c - p
    rhs = CASE["S"] - CASE["K"] * math.exp(-CASE["r"] * CASE["T"])
    assert abs(lhs - rhs) < 3 * (se_c + se_p)
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_montecarlo.py -v` → FAIL.

- [ ] **Step 3: Implement** — `engine/montecarlo.py`:

```python
"""Monte Carlo pricing of EUROPEAN options by GBM terminal-value sampling.

American exercise requires Longstaff-Schwartz regression and is out of scope
(see spec). Use engine.binomial for American options.
"""
from __future__ import annotations

import math

import numpy as np

from engine.bs import validate_inputs


def mc_price(S: float, K: float, T: float, r: float, sigma: float,
             q: float = 0.0, kind: str = "call",
             n_paths: int = 100_000, seed: int = 0) -> tuple[float, float]:
    validate_inputs(S, K, T, sigma, kind)
    if n_paths < 2:
        raise ValueError(f"n_paths must be >= 2 (got {n_paths})")
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal(n_paths)
    ST = S * np.exp((r - q - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * Z)
    payoff = np.maximum(ST - K, 0.0) if kind == "call" else np.maximum(K - ST, 0.0)
    disc = math.exp(-r * T)
    price = disc * float(payoff.mean())
    std_error = disc * float(payoff.std(ddof=1)) / math.sqrt(n_paths)
    return price, std_error
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_montecarlo.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: seeded Monte Carlo European pricer with standard error"`

---

### Task 5: Implied volatility

**Files:**
- Create: `engine/iv.py`, `tests/test_iv.py`

**Interfaces:**
- Consumes: `engine.bs.bs_price`.
- Produces: `engine.iv.IVError(ValueError)`; `engine.iv.implied_vol(price, S, K, T, r, q=0.0, kind="call") -> float`.

- [ ] **Step 1: Write failing tests** — `tests/test_iv.py`:

```python
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
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest tests/test_iv.py -v` → FAIL.

- [ ] **Step 3: Implement** — `engine/iv.py`:

```python
"""Implied volatility by Brent root-finding on Black-Scholes."""
from __future__ import annotations

import math

from scipy.optimize import brentq

from engine.bs import bs_price

_SIG_LO, _SIG_HI = 1e-4, 5.0


class IVError(ValueError):
    """Quote cannot be inverted to a Black-Scholes volatility."""


def implied_vol(price: float, S: float, K: float, T: float, r: float,
                q: float = 0.0, kind: str = "call") -> float:
    df_q, df_r = math.exp(-q * T), math.exp(-r * T)
    lower = max(S * df_q - K * df_r, 0.0) if kind == "call" else max(K * df_r - S * df_q, 0.0)
    upper = S * df_q if kind == "call" else K * df_r
    if not lower < price < upper:
        raise IVError(f"price {price} outside no-arbitrage bounds ({lower:.6f}, {upper:.6f})")

    def objective(sig: float) -> float:
        return bs_price(S, K, T, r, sig, q, kind) - price

    try:
        sig = brentq(objective, _SIG_LO, _SIG_HI, xtol=1e-10)
    except ValueError as exc:
        raise IVError(f"no root in [{_SIG_LO}, {_SIG_HI}]: {exc}") from exc
    if abs(bs_price(S, K, T, r, sig, q, kind) - price) > 1e-6 * S:
        raise IVError("re-price check failed")
    return float(sig)
```

- [ ] **Step 4: Run to verify pass** — `python -m pytest tests/test_iv.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: Brent implied-vol solver with no-arbitrage bounds"`

---

### Task 6: Market snapshot loader + smile

**Files:**
- Create: `market/__init__.py`, `market/snapshot.py`, `market/smile.py`, `tests/test_market.py`, `tests/fixtures/mini_chain.json`

**Interfaces:**
- Consumes: `engine.iv.implied_vol`, `engine.iv.IVError`.
- Produces: `market.snapshot.load_snapshot(path) -> dict` (raw JSON); `market.snapshot.filter_quotes(snapshot) -> tuple[list[dict], dict]` where each kept quote dict gains `"mid"`, `"dte"`, `"expiry"`, `"moneyness"` keys and the second element is `counts = {"total", "kept", "no_bid_or_ask", "wide_spread", "moneyness", "dte"}`; `market.smile.compute_smile(snapshot) -> dict` with `{"per_expiry": [{expiry, dte, points: [{strike, moneyness, kind, mid, iv|null}]}], "term_structure": [{expiry, dte, atm_iv}], "iv_failures": int}`.

- [ ] **Step 1: Fixture** — `tests/fixtures/mini_chain.json` (hand-built; BS prices at sigma=0.2/0.25 so IVs recover exactly; includes one zero-bid quote, one wide-spread quote, one far-OTM moneyness reject, one 3-DTE expiry to be dropped):

```json
{
  "fetched_utc": "2026-07-18T10:00:00Z",
  "spot": 100.0,
  "r": 0.04,
  "q": 0.012,
  "r_source": "^IRX close / 100 (fixture)",
  "q_source": "trailing 12m dividends / spot (fixture)",
  "expiries": [
    {"expiry": "2026-07-21", "dte": 3, "quotes": [
      {"strike": 100, "kind": "call", "bid": 1.0, "ask": 1.1}
    ]},
    {"expiry": "2026-10-16", "dte": 90, "quotes": [
      {"strike": 90, "kind": "call", "bid": 11.60, "ask": 11.80},
      {"strike": 100, "kind": "call", "bid": 4.55, "ask": 4.75},
      {"strike": 100, "kind": "put", "bid": 4.05, "ask": 4.25},
      {"strike": 110, "kind": "call", "bid": 1.05, "ask": 1.15},
      {"strike": 100, "kind": "call", "bid": 0.0, "ask": 4.8},
      {"strike": 105, "kind": "call", "bid": 1.0, "ask": 2.0},
      {"strike": 220, "kind": "call", "bid": 0.05, "ask": 0.07}
    ]}
  ]
}
```

- [ ] **Step 2: Write failing tests** — `tests/test_market.py`:

```python
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
```

- [ ] **Step 3: Run to verify failure** — `python -m pytest tests/test_market.py -v` → FAIL.

- [ ] **Step 4: Implement** — `market/__init__.py` empty. `market/snapshot.py`:

```python
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
```

`market/smile.py`:

```python
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
```

- [ ] **Step 5: Run to verify pass** — `python -m pytest tests/test_market.py -v` → PASS (fixture mids were generated from BS at ~0.2-0.25 vol; if an IV lands outside (0.01, 1.0) adjust the fixture bid/ask — not the filter).
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: snapshot quote filter and implied-vol smile"`

---

### Task 7: Validation gates + runner

**Files:**
- Create: `validation/__init__.py`, `validation/grid.py`, `validation/parity.py`, `validation/convergence.py`, `validation/greeks_check.py`, `validation/noarb.py`, `run_validate.py`, `tests/test_gates.py`

**Interfaces:**
- Consumes: everything from Tasks 1-5.
- Produces: `validation.grid.PARAM_GRID: list[dict]` (exactly 200 dicts with keys S,K,T,r,sigma,q); each gate module exposes `run() -> dict` with at least `{"gate": str, "passed": bool, ...details}`; `run_validate.py` writes `results/validation.json` `{"generated_by", "gates": [...], "all_passed": bool}` and exits 1 on failure.

- [ ] **Step 1: Grid** — `validation/grid.py`:

```python
"""Deterministic 200-point parameter grid shared by all gates (spec: 200 points)."""
from itertools import product

_K = [70, 85, 100, 115, 130]
_SIGMA = [0.15, 0.35]
_T = [0.1, 0.5, 1.0, 2.0]
_RQ = [(0.0, 0.0), (0.02, 0.0), (0.05, 0.02), (0.08, 0.0), (0.05, 0.05)]

PARAM_GRID = [
    {"S": 100.0, "K": float(K), "T": T, "r": r, "sigma": sig, "q": q}
    for K, sig, T, (r, q) in product(_K, _SIGMA, _T, _RQ)
]
assert len(PARAM_GRID) == 200
```

- [ ] **Step 2: Write failing tests** — `tests/test_gates.py`:

```python
import subprocess
import sys
import json
from pathlib import Path

from validation.grid import PARAM_GRID
from validation.parity import run as parity_run
from validation.convergence import run as convergence_run
from validation.greeks_check import run as greeks_run
from validation.noarb import run as noarb_run


def test_grid_has_200_points():
    assert len(PARAM_GRID) == 200


def test_parity_gate_passes():
    res = parity_run()
    assert res["passed"], res


def test_greeks_gate_passes():
    res = greeks_run()
    assert res["passed"], res


def test_noarb_gate_passes():
    res = noarb_run()
    assert res["passed"], res


def test_convergence_gate_passes():
    res = convergence_run()
    assert res["passed"], res


def test_runner_writes_json_and_exits_zero():
    proc = subprocess.run([sys.executable, "run_validate.py"], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    data = json.loads(Path("results/validation.json").read_text(encoding="utf-8"))
    assert data["all_passed"] is True
    assert len(data["gates"]) == 7
```

- [ ] **Step 3: Run to verify failure** — `python -m pytest tests/test_gates.py -v` → FAIL.

- [ ] **Step 4: Implement gates.** `validation/__init__.py` empty. `validation/parity.py`:

```python
"""Gates 1-2: put-call parity, closed-form (1e-10) and Monte Carlo (3 SE)."""
import math

from engine.bs import bs_price
from engine.montecarlo import mc_price
from validation.grid import PARAM_GRID

MC_SAMPLE_STRIDE = 10  # every 10th grid point for the MC leg (20 points)


def run() -> dict:
    worst_bs = 0.0
    for g in PARAM_GRID:
        c = bs_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "call")
        p = bs_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "put")
        rhs = g["S"] * math.exp(-g["q"] * g["T"]) - g["K"] * math.exp(-g["r"] * g["T"])
        worst_bs = max(worst_bs, abs(c - p - rhs))
    mc_fails = 0
    for g in PARAM_GRID[::MC_SAMPLE_STRIDE]:
        c, se_c = mc_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "call",
                           n_paths=200_000, seed=11)
        p, se_p = mc_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "put",
                           n_paths=200_000, seed=11)
        rhs = g["S"] * math.exp(-g["q"] * g["T"]) - g["K"] * math.exp(-g["r"] * g["T"])
        if abs(c - p - rhs) > 3 * (se_c + se_p):
            mc_fails += 1
    passed = worst_bs < 1e-10 and mc_fails == 0
    return {"gate": "put_call_parity", "passed": passed,
            "worst_bs_gap": worst_bs, "mc_points": len(PARAM_GRID[::MC_SAMPLE_STRIDE]),
            "mc_failures": mc_fails}
```

`validation/convergence.py`:

```python
"""Gates 3-4: tree->BS error shrinks and lands inside tolerance; MC->BS within
3 SE at 1e6 paths; MC 95% CI covers BS >= 90% of 200 seeded runs at 1e5 paths."""
from engine.bs import bs_price
from engine.binomial import crr_price
from engine.montecarlo import mc_price
from validation.grid import PARAM_GRID

BASE = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0}
TREE_SAMPLE_STRIDE = 5  # 40 grid points for the tree leg


def run() -> dict:
    tree_fails = []
    for g in PARAM_GRID[::TREE_SAMPLE_STRIDE]:
        bs = bs_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "call")
        e200 = abs(crr_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "call", steps=200) - bs)
        e2000 = abs(crr_price(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], "call", steps=2000) - bs)
        if not (e2000 < e200 and e2000 < max(0.01, 0.001 * bs)):
            tree_fails.append({**g, "e200": e200, "e2000": e2000})

    bs_base = bs_price(**BASE, kind="call")
    mc_1m, se_1m = mc_price(**BASE, kind="call", n_paths=1_000_000, seed=99)
    mc_big_ok = abs(mc_1m - bs_base) < 3 * se_1m

    covered = 0
    n_runs = 200
    for seed in range(n_runs):
        price, se = mc_price(**BASE, kind="call", n_paths=100_000, seed=seed)
        if abs(price - bs_base) <= 1.96 * se:
            covered += 1
    coverage = covered / n_runs

    passed = not tree_fails and mc_big_ok and coverage >= 0.90
    return {"gate": "convergence", "passed": passed, "tree_failures": tree_fails,
            "mc_1m_error": abs(mc_1m - bs_base), "mc_1m_se": se_1m,
            "mc_ci_coverage": coverage}
```

`validation/greeks_check.py`:

```python
"""Gate 5: closed-form Greeks match finite differences of bs_price; tree Greeks sane."""
from engine.bs import bs_greeks, bs_price
from engine.binomial import crr_greeks
from validation.grid import PARAM_GRID

GREEKS_SAMPLE_STRIDE = 10
_BUMPS = {"delta": ("S", 1e-4), "vega": ("sigma", 1e-5), "rho": ("r", 1e-6)}


def _fd(g: dict, kind: str, param: str, h: float) -> float:
    up, dn = dict(g), dict(g)
    up[param] += h
    dn[param] -= h
    return (bs_price(up["S"], up["K"], up["T"], up["r"], up["sigma"], up["q"], kind)
            - bs_price(dn["S"], dn["K"], dn["T"], dn["r"], dn["sigma"], dn["q"], kind)) / (2 * h)


def run() -> dict:
    worst_rel = 0.0
    for g in PARAM_GRID[::GREEKS_SAMPLE_STRIDE]:
        for kind in ("call", "put"):
            greeks = bs_greeks(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], kind)
            for name, (param, h) in _BUMPS.items():
                fd_val = _fd(g, kind, param, h)
                denom = max(abs(fd_val), 1e-6)
                worst_rel = max(worst_rel, abs(greeks[name] - fd_val) / denom)
    tree = crr_greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "put", american=True, steps=500)
    tree_ok = all(v == v and abs(v) < 1e6 for v in tree.values()) and -1.0 <= tree["delta"] <= 0.0
    passed = worst_rel < 1e-4 and tree_ok
    return {"gate": "greeks", "passed": passed, "worst_rel_error": worst_rel,
            "american_put_tree_greeks": tree}
```

`validation/noarb.py`:

```python
"""Gate 6: American >= European; American call == European call when q == 0."""
from engine.binomial import crr_price
from validation.grid import PARAM_GRID

NOARB_SAMPLE_STRIDE = 5


def run() -> dict:
    violations = []
    worst_call_gap = 0.0
    for g in PARAM_GRID[::NOARB_SAMPLE_STRIDE]:
        args = (g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"])
        for kind in ("call", "put"):
            eu = crr_price(*args, kind, american=False, steps=1000)
            am = crr_price(*args, kind, american=True, steps=1000)
            if am < eu - 1e-9:
                violations.append({**g, "kind": kind, "eu": eu, "am": am})
            if kind == "call" and g["q"] == 0.0:
                worst_call_gap = max(worst_call_gap, abs(am - eu))
    passed = not violations and worst_call_gap < 1e-9
    return {"gate": "no_arbitrage", "passed": passed,
            "violations": violations, "worst_q0_call_gap": worst_call_gap}
```

- [ ] **Step 5: Runner** — `run_validate.py` (also implements gate 7 inline):

```python
"""Run all validation gates -> results/validation.json. Non-zero exit on failure."""
import json
import sys
from pathlib import Path

from engine.bs import bs_price
from engine.iv import IVError, implied_vol
from market.smile import compute_smile
from market.snapshot import filter_quotes, load_snapshot
from validation.convergence import run as convergence_run
from validation.greeks_check import run as greeks_run
from validation.noarb import run as noarb_run
from validation.parity import run as parity_run

SNAPSHOT = Path("data/spy_chain.json")


def iv_roundtrip_gate() -> dict:
    worst = 0.0
    for sigma in (0.08, 0.2, 0.6, 1.5):
        for kind in ("call", "put"):
            price = bs_price(100, 90, 0.4, 0.03, sigma, 0.01, kind)
            worst = max(worst, abs(implied_vol(price, 100, 90, 0.4, 0.03, 0.01, kind) - sigma))
    return {"gate": "iv_roundtrip_synthetic", "passed": worst < 1e-7, "worst_error": worst}


def iv_snapshot_gate() -> dict:
    if not SNAPSHOT.exists():
        return {"gate": "iv_snapshot", "passed": False,
                "error": "data/spy_chain.json missing - run scripts/fetch_chain.py"}
    snap = load_snapshot(SNAPSHOT)
    kept, counts = filter_quotes(snap)
    smile = compute_smile(snap)
    resolved = sum(1 for e in smile["per_expiry"] for p in e["points"] if p["iv"] is not None)
    accounted = resolved + smile["iv_failures"] == len(kept)
    return {"gate": "iv_snapshot", "passed": accounted, "kept_quotes": len(kept),
            "resolved": resolved, "iv_failures": smile["iv_failures"], "filter_counts": counts}


def main() -> int:
    gates = [parity_run(), convergence_run(), greeks_run(), noarb_run(),
             iv_roundtrip_gate(), iv_snapshot_gate()]
    # split parity result into its two spec gates for reporting
    parity = gates[0]
    gates[0] = {"gate": "parity_closed_form", "passed": parity["worst_bs_gap"] < 1e-10,
                "worst_bs_gap": parity["worst_bs_gap"]}
    gates.insert(1, {"gate": "parity_mc", "passed": parity["mc_failures"] == 0,
                     "mc_points": parity["mc_points"], "mc_failures": parity["mc_failures"]})
    all_passed = all(g["passed"] for g in gates)
    out = {"generated_by": "run_validate.py", "gates": gates, "all_passed": all_passed}
    Path("results").mkdir(exist_ok=True)
    Path("results/validation.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for g in gates:
        print(("PASS" if g["passed"] else "FAIL"), g["gate"])
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
```

Note: `iv_snapshot_gate` needs `data/spy_chain.json`. Until Task 9 commits the real snapshot, create a placeholder by copying the fixture: `cp tests/fixtures/mini_chain.json data/spy_chain.json` (replaced in Task 9 by the real fetch; the gate logic is identical either way).

- [ ] **Step 6: Run to verify pass** — `python -m pytest tests/test_gates.py -v` → PASS (allow ~1-2 min: the MC coverage loop is 200×1e5 draws). Then `python run_validate.py` → 7 PASS lines, exit 0.
- [ ] **Step 7: Commit** — `git add -A && git commit -m "feat: seven machine-checked validation gates + runner"`

---

### Task 8: Golden table + charts

**Files:**
- Create: `scripts/__init__.py` (empty), `scripts/make_golden.py`, `scripts/make_charts.py`, `tests/test_golden.py`

**Interfaces:**
- Consumes: `bs_price`, `crr_price`, `mc_price`, `bs_greeks`, `compute_smile`, `load_snapshot`.
- Produces: `results/golden.json` — `{"tree_steps": 500, "cases": [{id, S, K, T, r, q, sigma, kind, american, bs, tree500}]}` (24 cases; `bs` is `null` when `american` is true); `scripts.make_golden.build_cases() -> list[dict]`; `charts/payoff.png`, `charts/convergence_tree.png`, `charts/convergence_mc.png`, `charts/greeks.png`, `charts/smile.png`.

- [ ] **Step 1: Write failing test** — `tests/test_golden.py`:

```python
import json
from pathlib import Path
from scripts.make_golden import build_cases, write_golden


def test_24_cases_cover_grid():
    cases = build_cases()
    assert len(cases) == 24
    assert {c["kind"] for c in cases} == {"call", "put"}
    assert {c["american"] for c in cases} == {True, False}
    assert all(("bs" in c and "tree500" in c) for c in cases)
    for c in cases:
        assert (c["bs"] is None) == c["american"]


def test_written_file_matches_freshly_built(tmp_path):
    out = tmp_path / "golden.json"
    write_golden(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tree_steps"] == 500
    assert data["cases"] == build_cases()


def test_committed_golden_is_fresh():
    committed = Path("results/golden.json")
    assert committed.exists(), "run scripts/make_golden.py"
    data = json.loads(committed.read_text(encoding="utf-8"))
    assert data["cases"] == build_cases(), "committed golden.json is stale - regenerate"
```

- [ ] **Step 2: Run to verify failure**, then implement — `scripts/make_golden.py`:

```python
"""Emit results/golden.json - the explorer's self-check table (24 cases)."""
import json
from itertools import product
from pathlib import Path

from engine.binomial import crr_price
from engine.bs import bs_price

TREE_STEPS = 500
_SIGMA, _R, _Q = 0.25, 0.04, 0.015
_MONEY = {"itm": 80.0, "atm": 100.0, "otm": 120.0}


def build_cases() -> list[dict]:
    cases = []
    for kind, american, (mlabel, K), T in product(
            ("call", "put"), (False, True), _MONEY.items(), (0.25, 2.0)):
        S = 100.0
        case = {"id": f"{kind}-{'am' if american else 'eu'}-{mlabel}-T{T}",
                "S": S, "K": K, "T": T, "r": _R, "q": _Q, "sigma": _SIGMA,
                "kind": kind, "american": american,
                "bs": None if american else bs_price(S, K, T, _R, _SIGMA, _Q, kind),
                "tree500": crr_price(S, K, T, _R, _SIGMA, _Q, kind,
                                     american=american, steps=TREE_STEPS)}
        cases.append(case)
    return cases


def write_golden(path: str | Path = "results/golden.json") -> None:
    Path(path).parent.mkdir(exist_ok=True)
    Path(path).write_text(
        json.dumps({"tree_steps": TREE_STEPS, "cases": build_cases()}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    write_golden()
    print("wrote results/golden.json")
```

Note: 2 kinds × 2 styles × 3 moneyness × 2 maturities = 24 ✓.

- [ ] **Step 3:** `python scripts/make_golden.py` then `python -m pytest tests/test_golden.py -v` → PASS.

- [ ] **Step 4: Charts** — `scripts/make_charts.py` (no test — visual artifact; must run cleanly):

```python
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
```

- [ ] **Step 5:** `python scripts/make_charts.py` → 5 PNGs (smile panel will look sparse until Task 9's real snapshot; rerun after Task 9). `python -m pytest` → all green.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: golden self-check table + charts"`

---

### Task 9: Real SPY snapshot (manual fetch)

**Files:**
- Create: `scripts/fetch_chain.py`, `requirements-fetch.txt`
- Replace: `data/spy_chain.json` (placeholder from Task 7 → real snapshot)

**Interfaces:**
- Consumes: snapshot JSON schema from Task 6 (spec: `{fetched_utc, spot, r, q, r_source, q_source, expiries:[{expiry, dte, quotes:[{strike, kind, bid, ask}]}]}`).
- Produces: committed real `data/spy_chain.json`.

- [ ] **Step 1:** `requirements-fetch.txt`:

```
-r requirements.txt
yfinance>=0.2.50
```

- [ ] **Step 2: Implement** — `scripts/fetch_chain.py`:

```python
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
```

- [ ] **Step 3: Run the fetch (manual, network):** `pip install -r requirements-fetch.txt && python scripts/fetch_chain.py`. Expected: one line with plausible values (spot 500-800, r 0.02-0.06, q 0.008-0.02, expiries ≈ 12, quotes in the thousands). If yfinance is rate-limited, wait and retry; do NOT fake data.
- [ ] **Step 4: Re-run downstream:** `python run_validate.py` (gate 7 now runs on real data — expect PASS with a nonzero `iv_failures` count, that's the honest bit) and `python scripts/make_charts.py` (smile now real). `python -m pytest` still green (tests use the fixture, not the snapshot).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "data: committed SPY chain snapshot + manual fetch script"`

---

### Task 10: Explorer (template + build)

**Files:**
- Create: `viz/template.html`, `scripts/build_viz.py`, `tests/test_build_viz.py`
- Output (generated, committed): `docs/index.html`

**Interfaces:**
- Consumes: `results/golden.json`, `data/spy_chain.json`, `market.smile.compute_smile`, `engine.montecarlo.mc_price`, `market.snapshot.filter_quotes`.
- Produces: `scripts.build_viz.build(template="viz/template.html", out="docs/index.html") -> None`. Template contains literal tokens `__GOLDEN__`, `__SMILE__`, `__MC__`, `__META__` replaced by JSON. JS must implement: `normCdf` (Cody/West double-precision), `bsPrice`, `bsGreeks`, `crrPrice(S,K,T,r,sigma,q,kind,american,steps)` — same formulas as Python (Tasks 1-3).

- [ ] **Step 1: Write failing test** — `tests/test_build_viz.py`:

```python
import json
import re
from pathlib import Path
from scripts.build_viz import build


def test_build_injects_all_tokens(tmp_path):
    out = tmp_path / "index.html"
    build(out=out)
    html = out.read_text(encoding="utf-8")
    assert "__GOLDEN__" not in html and "__SMILE__" not in html
    assert "__MC__" not in html and "__META__" not in html
    m = re.search(r"const GOLDEN = (\{.*?\});\n", html, re.S)
    assert m, "GOLDEN payload missing"
    golden = json.loads(m.group(1))
    assert len(golden["cases"]) == 24


def test_committed_docs_index_exists_and_is_fresh():
    html = Path("docs/index.html").read_text(encoding="utf-8")
    assert "__GOLDEN__" not in html
    assert "self-check" in html.lower()
```

- [ ] **Step 2: build script** — `scripts/build_viz.py`:

```python
"""Inject data payloads into viz/template.html -> docs/index.html."""
import json
from pathlib import Path

from engine.montecarlo import mc_price
from market.smile import compute_smile
from market.snapshot import filter_quotes, load_snapshot

BASE = dict(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)


def mc_payload() -> dict:
    ns = [1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    rows = []
    for n in ns:
        price, se = mc_price(n_paths=n, seed=5, **BASE)
        rows.append({"n": n, "price": price, "se": se})
    return {"base": BASE, "kind": "call", "seed": 5, "rows": rows}


def build(template: str | Path = "viz/template.html",
          out: str | Path = "docs/index.html") -> None:
    snap = load_snapshot("data/spy_chain.json")
    _, counts = filter_quotes(snap)
    smile = compute_smile(snap)
    meta = {"fetched_utc": snap["fetched_utc"], "spot": snap["spot"],
            "r": snap["r"], "q": snap["q"], "filter_counts": counts,
            "iv_failures": smile["iv_failures"]}
    golden = json.loads(Path("results/golden.json").read_text(encoding="utf-8"))
    html = Path(template).read_text(encoding="utf-8")
    for token, payload in (("__GOLDEN__", golden), ("__SMILE__", smile),
                           ("__MC__", mc_payload()), ("__META__", meta)):
        html = html.replace(token, json.dumps(payload))
    Path(out).parent.mkdir(exist_ok=True)
    Path(out).write_text(html, encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
```

- [ ] **Step 3: template** — `viz/template.html`. Full requirements (implementer writes the page to these binding specs; ~450 lines):
  - Single self-contained file: inline `<style>` + `<script>`, no external requests. Dark quant-terminal aesthetic consistent with Alex's portfolio (near-black background `#0a0e14`, cyan accent `#22d3ee`, gold `#eab308` for highlights, `system-ui`/`ui-monospace` stack — no webfonts).
  - Header: title "Options Pricing Engine", subtitle one-liner, **self-check badge** `<span id="badge">` and snapshot date from `META`.
  - Tab bar: `Pricer` | `Convergence` | `Market smile` (plain JS tab switching, no framework).
  - **Pricer tab:** six range sliders with live value labels — S [50,200] step 0.5 default 100; K [50,200] step 0.5 default 100; sigma [0.01,1.0] step 0.005 default 0.2; T [0.02,3] step 0.01 default 1.0; r [0,0.10] step 0.001 default 0.05; q [0,0.05] step 0.001 default 0. Toggles: call/put, European/American. Readout table: BS price (blank + note "n/a - use tree" when American), tree N=500 price, and the five Greeks (BS closed form for European, tree FD for American). Canvas payoff chart (payoff at expiry + current model value curve vs S, redrawn on input; plain canvas 2D, no library).
  - **Convergence tab:** two canvas log-log charts from the embedded `MC` payload and a client-computed tree-error series (steps [10,20,50,100,200,500,1000] vs BS at the tab's fixed BASE params), each with its reference slope line, axis labels, and a one-paragraph honest explanation.
  - **Market smile tab:** canvas chart of IV (%) vs moneyness for up to 4 expiries (legend = DTE) from `SMILE`, term-structure chart, and a footnote rendering `META.filter_counts` + `META.iv_failures` verbatim ("dropped quotes are counted, not hidden").
  - **JS math (mirrors Python exactly):** `normCdf` via West's double-precision algorithm (Graeme West, "Better approximations to cumulative normal functions", accurate to ~1e-15 — NOT Abramowitz-Stegun, which is too coarse for the 1e-9 gate); `bsPrice`, `bsGreeks` (same formulas as Task 2), `crrPrice` iterative arrays (same recursion as Task 3 incl. the American intrinsic max and the `p∉(0,1)` guard).
  - **Self-check on load:** for each of the 24 `GOLDEN.cases`: if `bs` non-null assert `|jsBS - bs| / |bs| < 1e-9`; always assert `|jsTree(steps=GOLDEN.tree_steps) - tree500| / |tree500| < 1e-6`. Badge: green `SELF-CHECK PASS (24/24)` or red `SELF-CHECK FAIL` + worst case id and relative error. No silent pass.
  - Footer: link to GitHub repo, snapshot date, "prices are model values, not advice".
- [ ] **Step 4:** `python scripts/build_viz.py` then `python -m pytest tests/test_build_viz.py -v` → PASS.
- [ ] **Step 5: Browser verify (required):** open `docs/index.html` via the preview browser; confirm badge shows PASS 24/24, sliders move all readouts, all three tabs render, no console errors.
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: interactive explorer with golden self-check badge"`

---

### Task 11: README + WRITEUP + publish

**Files:**
- Create: `README.md`, `WRITEUP.md`, `.gitattributes` (`* text=auto eol=lf`)

**Interfaces:** none new — this task publishes.

- [ ] **Step 1: WRITEUP.md** — sections (write from the actual `results/validation.json` numbers, never invented): Method (three pricers, formulas summarized, q handling); Validation (the 7 gates table with the real measured numbers); The smile finding (what flat-vol BS predicts vs the SPY snapshot: smile chart, term structure, filter counts, iv_failures — interpret honestly: the smile is the market pricing in what GBM assumes away, fat tails and skew); Limitations (European-only MC; continuous-yield simplification; snapshot-in-time r/q; 12h-stale quotes; no discrete dividends; American Greeks by FD carry tree noise).
- [ ] **Step 2: README.md** — structure copied from football-trajectory's register: hero line ("Three pricers, one truth — and the market disagrees with all of them"), tests badge, **live explorer link** (https://aleks-drozy.github.io/options-pricing-engine/), verdict-style gate table (all 7 with one-line results), quick start (`pip install -r requirements-dev.txt; pytest; python run_validate.py`), project structure block, links to WRITEUP/spec/plan, "research project, not trading advice" line.
- [ ] **Step 3:** Full local green sweep: `python -m pytest && python run_validate.py` → exit 0.
- [ ] **Step 4: Publish:** `gh repo create aleks-drozy/options-pricing-engine --public --source . --push`. Then enable Pages from `docs/` on main: `gh api -X POST repos/aleks-drozy/options-pricing-engine/pages -f "source[branch]=main" -f "source[path]=/docs"`. Watch CI: `gh run watch` → tests job green.
- [ ] **Step 5: Live verify:** fetch the Pages URL until it serves; open in preview browser; badge PASS. Confirm README renders (badge shows passing).
- [ ] **Step 6:** Private project-notes update (external to this repo).

---

## Self-Review (done at write time)

- **Spec coverage:** BS/MC/CRR ✓(T1,3,4) American ✓(T3) q ✓(all) Greeks ✓(T2,3) IV ✓(T5) snapshot+filters ✓(T6,9) 7 gates ✓(T7: parity split into closed-form+MC = gates 1-2; convergence = 3-4; greeks 5; noarb 6; IV round-trip + snapshot accounting = 7) golden ✓(T8) charts ✓(T8) explorer+self-check ✓(T10) CI/no-network ✓(T1, fetch isolated in T9) README/WRITEUP/Pages ✓(T11) vault ✓(T11).
- **Type consistency:** all pricers share `(S,K,T,r,sigma,q,kind)` ordering; `crr_price` extras `(american, steps)`; `mc_price` returns tuple — checked against every call site in gates/scripts above.
- **Placeholders:** template task expresses the page as binding requirements rather than 450 literal HTML lines — every behavior, constant, range, formula source, and threshold is specified; no TBDs anywhere.
