"""Regression tests for code advertised in the maintained documentation."""

import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("documentation_workflows.py")


def test_documented_workflows_run() -> None:
    """Every maintained page must expose a working adoption path."""
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, (
        f"documentation workflows failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
