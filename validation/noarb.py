"""Gate 6: American >= European; American call == European call when q == 0."""
from engine.binomial import crr_price
from validation.grid import PARAM_GRID

NOARB_SAMPLE_STRIDE = 7  # coprime with the grid's r/q period (5) so r,q>0 points get sampled


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
