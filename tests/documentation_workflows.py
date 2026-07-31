"""Execute the maintained documentation workflows from their Markdown sources."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = [
    ROOT / "docs/index.md",
    ROOT / "docs/getting-started/quickstart.md",
    *sorted((ROOT / "docs/user-guide").glob("*.md")),
]
WORKFLOW_PATTERN = re.compile(
    r"<!-- ml4t-exec -->\s*```python\n(?P<source>.*?)\n```",
    flags=re.DOTALL,
)


def documented_workflows() -> list[tuple[Path, int, str]]:
    """Return every explicitly executable documentation block."""
    workflows: list[tuple[Path, int, str]] = []
    for document in DOCUMENTS:
        text = document.read_text(encoding="utf-8")
        matches = list(WORKFLOW_PATTERN.finditer(text))
        if not matches:
            raise AssertionError(f"{document.relative_to(ROOT)} has no executable workflow")
        workflows.extend(
            (document, text.count("\n", 0, match.start()) + 1, match.group("source"))
            for match in matches
        )
    return workflows


def run_documented_workflows() -> None:
    """Run each workflow in an isolated interpreter."""
    failures: list[str] = []
    for document, line_number, source in documented_workflows():
        result = subprocess.run(
            [sys.executable, "-c", source],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            relative_path = document.relative_to(ROOT)
            failures.append(
                f"{relative_path}:{line_number} exited {result.returncode}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
    if failures:
        raise AssertionError("\n\n".join(failures))


if __name__ == "__main__":
    run_documented_workflows()
