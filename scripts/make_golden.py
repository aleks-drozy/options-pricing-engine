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
