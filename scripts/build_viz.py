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
    # newline="\n": byte-identical output on Windows and Linux (repo is eol=lf)
    Path(out).write_text(html, encoding="utf-8", newline="\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    build()
