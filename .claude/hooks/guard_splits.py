#!/usr/bin/env python3
"""PreToolUse guard: block edits that add row-level CV splits (AGENTS.md hard rule 7).

FaultTrack has 5 rows per case, so splits must be grouped by case_id. A line can opt
out with the marker `# faulttrack: allow-split` (for example a test asserting that
ungrouped splits leak). Files under spike/ are exempt.
"""

import json
import re
import sys

BLOCKED = re.compile(
    r"\b(train_test_split|(?<!Group)(?<!Stratified)KFold|StratifiedKFold|"
    r"(?<!Group)ShuffleSplit|StratifiedShuffleSplit|LeaveOneOut|RepeatedKFold|"
    r"RepeatedStratifiedKFold)\s*\("
)
MARKER = "faulttrack: allow-split"


def new_text(tool_input: dict) -> str:
    parts = [tool_input.get(k, "") for k in ("content", "new_string", "new_source")]
    parts += [e.get("new_string", "") for e in tool_input.get("edits", [])]
    return "\n".join(p for p in parts if isinstance(p, str))


def main() -> int:
    event = json.load(sys.stdin)
    tool_input = event.get("tool_input", {})
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if "/spike/" in path or path.startswith("spike/"):
        return 0
    for line in new_text(tool_input).splitlines():
        if BLOCKED.search(line) and MARKER not in line:
            print(
                "Blocked by .claude/hooks/guard_splits.py: row-level split found:\n"
                f"  {line.strip()}\n"
                "Each case has 5 rows, so use GroupKFold(n_splits=5) with "
                "groups=df['case_id'], or faulttrack.evaluation.grouped_cv() "
                "(AGENTS.md hard rule 7).",
                file=sys.stderr,
            )
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
