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


def test_written_file_matches_freshly_built(tmp_path):
    out = tmp_path / "golden.json"
    write_golden(out)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["tree_steps"] == 500
    assert data["cases"] == build_cases()


def test_committed_golden_is_fresh():
    committed = Path("results/golden.json")
    assert committed.exists(), "run scripts/make_golden.py"
    data = json.loads(committed.read_text(encoding="utf-8"))
    assert data["cases"] == build_cases(), "committed golden.json is stale - regenerate"
