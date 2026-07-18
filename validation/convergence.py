"""Gates 3-4: tree->BS error shrinks and lands inside tolerance; MC->BS within
3 SE at 1e6 paths; MC 95% CI covers BS >= 90% of 200 seeded runs at 1e5 paths.

Amended criteria (see spec Amendments): the shrinkage comparison uses
adjacent-step averages (N, N+1) to damp CRR's sawtooth oscillation, and
monotonicity is waived below a 1e-9 absolute noise floor where the tree is
already exact to machine epsilon. The raw-error absolute tolerance is
unamended."""
from engine.bs import bs_price
from engine.binomial import crr_price
from engine.montecarlo import mc_price
from validation.grid import PARAM_GRID

BASE = {"S": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "q": 0.0}
TREE_SAMPLE_STRIDE = 7  # coprime with the grid's r/q period (5) so r,q>0 points get sampled


def run() -> dict:
    tree_fails = []
    for g in PARAM_GRID[::TREE_SAMPLE_STRIDE]:
        args = (g["S"], g["K"], g["T"], g["r"], g["sigma"], g["q"])
        bs = bs_price(*args, "call")
        # CRR error oscillates ("sawtooth") as the strike moves across tree
        # nodes, so pointwise monotonicity between two arbitrary step counts
        # mis-scores the phase. Averaging adjacent step counts (N, N+1) damps
        # the odd-even oscillation (standard practice, cf. Broadie-Detemple
        # smoothing) - the absolute tolerance still binds the RAW e2000 error.
        p200 = crr_price(*args, "call", steps=200)
        p2000 = crr_price(*args, "call", steps=2000)
        e200 = abs(p200 - bs)
        e2000 = abs(p2000 - bs)
        e_avg200 = abs((p200 + crr_price(*args, "call", steps=201)) / 2 - bs)
        e_avg2000 = abs((p2000 + crr_price(*args, "call", steps=2001)) / 2 - bs)
        # For deep-ITM short-T points the tree is already exact to machine epsilon
        # at both step counts, so requiring monotonic improvement below a 1e-9
        # absolute noise floor is meaningless - accept either monotonic progress
        # or "already below the noise floor".
        if not ((e_avg2000 < e_avg200 or e_avg2000 < 1e-9) and e2000 < max(0.01, 0.001 * bs)):
            tree_fails.append({**g, "e200": e200, "e2000": e2000,
                               "e_avg200": e_avg200, "e_avg2000": e_avg2000})

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
