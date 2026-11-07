"""README and REVIEWING.md: headline numbers match web/results.json; links resolve.

Offline: reads only committed files, never the network.
"""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
DOCS = [README, ROOT / "REVIEWING.md"]

# [text](target) or ![alt](target); the target ends at whitespace or ")".
_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_FENCE = re.compile(r"^```.*?^```", re.MULTILINE | re.DOTALL)
_INLINE_CODE = re.compile(r"`[^`\n]*`")


def _results() -> dict:
    with open(ROOT / "web" / "results.json", encoding="utf-8") as f:
        return json.load(f)


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _table_row(text: str, starts_with: str) -> str:
    """Return the single Markdown table line that starts with `starts_with`."""
    rows = [line for line in text.splitlines() if line.startswith(starts_with)]
    assert len(rows) == 1, f"expected one README row starting {starts_with!r}: {rows}"
    return rows[0]


def _section(text: str, heading: str) -> str:
    """Return the text under a `## ` or `### ` heading, up to the next heading."""
    match = re.search(
        rf"^#{{2,3}} {re.escape(heading)}\n(.*?)(?=^#{{2,3}} |\Z)",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert match, f"README has no section {heading!r}"
    return match.group(1)


def test_final_model_row_matches_results_json() -> None:
    r = _results()
    row = _table_row(_readme(), "| **Final model**")
    mean, std = r["fault_cv_macro_f1_mean"], r["fault_cv_macro_f1_std"]
    lo, hi = r["fault_cv_macro_f1_min"], r["fault_cv_macro_f1_max"]
    assert f"{mean:.3f} ± {std:.3f}" in row
    assert f"{lo:.3f} / {hi:.3f}" in row


def test_first_results_sentence_gives_headline_and_loso() -> None:
    r = _results()
    first = _section(_readme(), "Results").strip().split("\n", 1)[0]
    assert f"macro F1 {r['fault_cv_macro_f1_mean']:.2f}" in first
    assert f"about {r['loso']['mean']:.2f}" in first
    assert "(#generalizing-to-an-unseen-system)" in first


def test_loso_row_matches_results_json() -> None:
    loso = _results()["loso"]
    row = _table_row(_readme(), "| **Random forest (final model)**")
    assert f"{loso['mean']:.3f} ± {loso['std']:.3f}" in row
    for system in ("re2ob", "re2ss", "re2tt"):
        assert f"{loso['macro_f1'][system]:.3f}" in row, system


def test_suspect_hit_rate_matches_results_json() -> None:
    r = _results()
    text = _readme()
    row = _table_row(text, "| Suspect service")
    assert f"{r['suspect_hit_rate']:.3f}" in row
    assert "Rule" in row and "not the model" in row
    by_system = r["suspect_hit_rate_by_system"]
    names = {"re2ob": "OB", "re2ss": "SS", "re2tt": "TT"}
    for system, short in names.items():
        assert f"{short} {by_system[system]:.3f}" in text, system


def test_readme_has_no_tbd() -> None:
    assert "TBD" not in _readme()


def _relative_links(path: Path) -> list[str]:
    text = _FENCE.sub("", path.read_text(encoding="utf-8"))
    text = _INLINE_CODE.sub("", text)
    targets = []
    for target in _LINK.findall(text):
        if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
            continue  # http(s):, mailto: and other schemes
        target = target.split("#", 1)[0]
        if target:  # a bare #anchor points inside this file
            targets.append(target)
    return targets


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: p.name)
def test_relative_links_resolve(doc: Path) -> None:
    links = _relative_links(doc)
    assert links, f"no relative links found in {doc.name}; is the pattern broken?"
    missing = [t for t in links if not (doc.parent / t).exists()]
    assert not missing, f"{doc.name} links to paths that don't exist: {missing}"


def test_link_check_catches_a_missing_path(tmp_path: Path) -> None:
    doc = tmp_path / "doc.md"
    doc.write_text(
        "[ok](doc.md#top) [gone](nope/) [web](https://example.com) [here](#a)\n"
        "`[code](skip.md)`\n```\n[fenced](skip2.md)\n```\n",
        encoding="utf-8",
    )
    assert _relative_links(doc) == ["doc.md", "nope/"]
