"""Regression tests for code advertised in the maintained documentation."""

import subprocess
import sys
from pathlib import Path

RUNNER = Path(__file__).with_name("documentation_workflows.py")


def test_documented_workflows_run() -> None:
    """Every maintained page must expose a working adoption path."""
    # No outer timeout: the runner executes each workflow serially under its own 60 s limit, so
    # a fixed total budget fails on a loaded machine when no workflow is slow.
    result = subprocess.run(
        [sys.executable, str(RUNNER)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"documentation workflows failed\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
