import json
import re
from pathlib import Path
from scripts.build_viz import build


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


def test_committed_docs_index_exists_and_is_fresh():
    html = Path("docs/index.html").read_text(encoding="utf-8")
    assert "__GOLDEN__" not in html
    assert "self-check" in html.lower()
