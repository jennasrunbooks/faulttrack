#!/usr/bin/env python3
"""PostToolUse: ruff-fix and format the edited file; report what's left (exit 2)."""

import json
import subprocess
import sys
from pathlib import Path


def run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--quiet", "ruff", *args],
        capture_output=True,
        text=True,
        check=False,
    )


def main() -> int:
    event = json.load(sys.stdin)
    tool_input = event.get("tool_input", {})
    path = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    if Path(path).suffix not in {".py", ".ipynb"} or not Path(path).exists():
        return 0
    run("check", "--fix", "--quiet", path)
    run("format", "--quiet", path)
    left = run("check", "--quiet", path)
    if left.returncode != 0:
        print(
            f"ruff found issues it couldn't fix in {path}:\n{left.stdout}",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
