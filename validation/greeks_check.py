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
    abs_rescued = 0
    all_pass = True
    for g in PARAM_GRID[::GREEKS_SAMPLE_STRIDE]:
        for kind in ("call", "put"):
            greeks = bs_greeks(g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"], kind)
            for name, (param, h) in _BUMPS.items():
                fd_val = _fd(g, kind, param, h)
                abs_diff = abs(greeks[name] - fd_val)
                denom = max(abs(fd_val), 1e-6)
                rel_error = abs_diff / denom
                worst_rel = max(worst_rel, rel_error)
                # Far-OTM greeks (~1e-6) make the FD reference noise-dominated: the
                # signal is buried under ~1e-10 cancellation error, so a pure relative
                # criterion ends up comparing rounding noise instead of the model. A
                # comparison also passes when the absolute gap is below machine-noise
                # scale, and we count how many needed that rescue.
                if rel_error < 1e-4 or abs_diff < 1e-8:
                    if rel_error >= 1e-4:
                        abs_rescued += 1
                else:
                    all_pass = False
    tree = crr_greeks(100, 100, 1.0, 0.05, 0.2, 0.0, "put", american=True, steps=500)
    tree_ok = all(v == v and abs(v) < 1e6 for v in tree.values()) and -1.0 <= tree["delta"] <= 0.0
    passed = all_pass and tree_ok
    return {"gate": "greeks", "passed": passed, "worst_rel_error": worst_rel,
            "american_put_tree_greeks": tree, "abs_rescued": abs_rescued}
