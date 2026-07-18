import math
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


def assert_cases_match(actual, expected):
    # Committed golden values are Windows-generated; Linux libm's erfc differs
    # by ~1 ULP, so computed fields (bs, tree500) are compared with
    # math.isclose at rel_tol=1e-12 / abs_tol=1e-15 - far tighter than the
    # explorer's own self-check thresholds (1e-9 / 1e-6), so genuine drift is
    # still caught. Inputs (S, K, T, r, q, sigma) and non-numeric fields
    # (id, kind, american) are not computed, so they still compare exactly.
    assert len(actual) == len(expected)
    assert [c["id"] for c in actual] == [c["id"] for c in expected]
    for a, e in zip(actual, expected):
        assert a["id"] == e["id"]
        for field in ("kind", "american"):
            assert a[field] == e[field]
        for field in ("S", "K", "T", "r", "q", "sigma"):
            assert a[field] == e[field], f"{a['id']}: {field} differs"
        for field in ("bs", "tree500"):
            av, ev = a[field], e[field]
            if av is None or ev is None:
                assert av is None and ev is None, f"{a['id']}: {field} differs"
            else:
                assert math.isclose(av, ev, rel_tol=1e-12, abs_tol=1e-15), \
                    f"{a['id']}: {field} differs ({av!r} vs {ev!r})"


def test_written_file_matches_freshly_built(tmp_path):
    out = tmp_path / "golden.json"
    write_golden(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tree_steps"] == 500
    assert_cases_match(data["cases"], build_cases())


def test_committed_golden_is_fresh():
    committed = Path("results/golden.json")
    assert committed.exists(), "run scripts/make_golden.py"
    data = json.loads(committed.read_text(encoding="utf-8"))
    assert_cases_match(data["cases"], build_cases())
