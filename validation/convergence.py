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
