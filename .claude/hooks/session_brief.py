#!/usr/bin/env python3
"""SessionStart: add the date, deadline and plan table (spec §10) to the agent's context."""

import os
import re
import sys
from datetime import date
from pathlib import Path

DEADLINE = date(2026, 11, 16)


def plan_table(spec: str) -> str:
    section = re.search(r"^## 10\..*?(?=^## \d+\.|\Z)", spec, re.S | re.M)
    if not section:
        return ""
    rows: list[str] = []
    for line in section.group(0).splitlines():
        if line.startswith("|"):
            rows.append(line)
        elif rows:
            break  # first table only: the plan, not the CI or Dependabot tables
    return "\n".join(rows)


def main() -> int:
    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", "."))
    today = date.today()
    left = (DEADLINE - today).days
    print(
        f"FaultTrack session brief. Today is {today:%a %b %d, %Y}. "
        f"Deadline: Mon Nov 16, 2026, 17:00 ({left} days left)."
    )
    print(
        "Before your first edit, follow 'Before your first edit on any issue' in AGENTS.md."
    )
    try:
        table = plan_table((root / "product-spec.md").read_text(encoding="utf-8"))
    except OSError:
        table = ""
    if table:
        print("\nPlan (product-spec.md §10):\n" + table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
