import json
import math
import re
from pathlib import Path
from scripts.build_viz import build

_PAYLOAD_RE = re.compile(r"^const (GOLDEN|SMILE|MC|META) = (.*);$", re.M)


def _shell_and_payloads(html: str) -> tuple[str, dict]:
    """Split built HTML into the template shell and its four parsed payloads."""
    payloads = {}

    def _stash(m):
        payloads[m.group(1)] = json.loads(m.group(2))
        return f"const {m.group(1)} = __{m.group(1)}__;"

    return _PAYLOAD_RE.sub(_stash, html), payloads


def _close(a, b) -> bool:
    """Deep equality with float tolerance (committed payloads are Windows-built;
    Linux libm differs by ~1 ULP in erfc-derived values - cf. tests/test_golden.py)."""
    if isinstance(a, float) or isinstance(b, float):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


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


def test_committed_docs_index_exists_and_is_fresh(tmp_path):
    # Build fresh to a tmp path and require the committed docs/index.html to
    # match it: the template shell byte-for-byte, and the four data payloads
    # structurally (floats at 1e-9 - the committed page was built on Windows
    # and CI's Linux libm drifts by ~1 ULP in erfc-derived numbers, exactly
    # like results/golden.json; payloads are otherwise deterministic:
    # golden.json committed, snapshot committed, MC seeded).
    fresh_path = tmp_path / "index.html"
    build(out=fresh_path)
    committed = Path("docs/index.html").read_text(encoding="utf-8")
    fresh = fresh_path.read_text(encoding="utf-8")
    c_shell, c_payloads = _shell_and_payloads(committed)
    f_shell, f_payloads = _shell_and_payloads(fresh)
    assert c_shell == f_shell, "committed docs/index.html template is stale - rerun scripts.build_viz"
    assert set(c_payloads) == {"GOLDEN", "SMILE", "MC", "META"} == set(f_payloads)
    for name in sorted(c_payloads):
        assert _close(c_payloads[name], f_payloads[name]), f"{name} payload is stale - rerun scripts.build_viz"
