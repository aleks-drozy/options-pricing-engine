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
