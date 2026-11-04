#!/usr/bin/env python3
"""PreToolUse guard for Bash: block commands AGENTS.md forbids (exit 2 = block).

Rules match the command itself, not text it carries: heredoc bodies and quoted
strings (commit messages, PR bodies, echo text) are removed before matching, so
"no pip" in a PR description doesn't trigger the pip rule.
"""

import json
import re
import sys

RULES = [
    (
        r"\bgit\s+push\b.*(\s--force\b|\s-f\b|\s--force-with-lease\b|\s\+\S)",
        "Force pushes are blocked. Merge origin/main into your branch and push normally "
        "(AGENTS.md hard rule 4).",
    ),
    (
        r"\bgit\s+rebase\b",
        "Rebasing is blocked. Use `git fetch origin && git merge origin/main` "
        "(AGENTS.md hard rule 4).",
    ),
    (
        r"\bgit\s+tag\b|\bgit\s+push\b.*\s--tags\b|\bgh\s+release\b",
        "Tags and releases are maintainer-only (AGENTS.md hard rule 1).",
    ),
    (
        r"\bgh\s+pr\s+merge\b",
        "Agents don't merge PRs. A human squash-merges after review.",
    ),
    (
        r"(^|[\s;&|(])(fly|flyctl)\s",
        "Deploys are maintainer-only (AGENTS.md hard rule 1).",
    ),
    (
        r"(^|[\s;&|(])(pip3?|uv\s+pip)\s",
        "This project uses uv only. Use `uv add` / `uv add --dev` (AGENTS.md hard rule 8).",
    ),
    (
        r"(^|[\s;&|(<])(\./)?\.env(?!\.example\b)(\.[\w-]+)?\b",
        "Reading .env files is blocked (AGENTS.md hard rule 2).",
    ),
    (
        r"\bgit\s+(add|commit)\b.*\bdata/raw\b",
        "data/raw/ is never committed (AGENTS.md hard rule 2).",
    ),
]


_HEREDOC_START = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def strip_text(command: str) -> str:
    """Return the command with heredoc bodies and quoted strings blanked out."""
    lines = command.split("\n")
    out: list[str] = []
    end_marker: str | None = None
    for line in lines:
        if end_marker is not None:
            if line.strip() == end_marker:
                end_marker = None
            continue
        out.append(line)
        start = _HEREDOC_START.search(line)
        if start:
            end_marker = start.group(2)
    text = "\n".join(out)
    text = re.sub(r"'[^']*'", "''", text)
    return re.sub(r'"(?:\\.|[^"\\])*"', '""', text)


def main() -> int:
    event = json.load(sys.stdin)
    command = strip_text(event.get("tool_input", {}).get("command", ""))
    for pattern, message in RULES:
        if re.search(pattern, command):
            print(f"Blocked by .claude/hooks/guard_bash.py: {message}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
