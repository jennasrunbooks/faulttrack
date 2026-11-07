"""Checks on web/index.html: it reads only keys that exist, and loads nothing external."""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INDEX = (ROOT / "web" / "index.html").read_text(encoding="utf-8")


def _page_keys() -> list[str]:
    match = re.search(r"const KEYS = \[(.*?)\];", INDEX, re.DOTALL)
    assert match, "index.html must declare `const KEYS = [...]`"
    return re.findall(r'"([^"]+)"', match.group(1))


def test_every_key_the_page_reads_is_in_results_json() -> None:
    results = json.loads((ROOT / "web" / "results.json").read_text(encoding="utf-8"))
    keys = _page_keys()
    assert keys
    missing = [k for k in keys if k not in results]
    assert not missing, f"results.json lacks keys the page reads: {missing}"


def test_page_has_no_external_requests() -> None:
    assert not re.search(r"<script[^>]*\bsrc=", INDEX, re.IGNORECASE)
    assert not re.search(
        r"<link[^>]*rel=[\"']stylesheet[\"'][^>]*href=[\"']https?:",
        INDEX,
        re.IGNORECASE,
    )
    assert "@import" not in INDEX
