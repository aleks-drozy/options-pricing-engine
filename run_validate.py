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
