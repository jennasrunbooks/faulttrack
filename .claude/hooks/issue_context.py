#!/usr/bin/env python3
"""UserPromptSubmit: when the prompt names an issue, add its body and comments to the context."""

import json
import re
import subprocess
import sys

REPO = "jennasrunbooks/faulttrack"
MAX_CHARS = 12000
ISSUE_REF = re.compile(r"(?:#|\bissue\s+#?)(\d{1,5})\b", re.I)


def main() -> int:
    prompt = json.load(sys.stdin).get("prompt", "")
    numbers = list(dict.fromkeys(ISSUE_REF.findall(prompt)))[:3]
    for n in numbers:
        try:
            out = subprocess.run(
                ["gh", "issue", "view", n, "--repo", REPO, "--comments"],
                capture_output=True, text=True, timeout=15, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            continue
        if out.returncode == 0 and out.stdout.strip():
            body = out.stdout[:MAX_CHARS]
            print(f"<issue number=\"{n}\" source=\"gh issue view {n} --comments\">\n"
                  f"{body}\n</issue>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
