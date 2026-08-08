"""Execute every advertised example as an end user would."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples"
EXAMPLES = sorted(path for path in EXAMPLE_DIR.glob("*.py"))


@pytest.mark.parametrize("script", EXAMPLES, ids=lambda path: path.name)
def test_advertised_example_runs(script: Path) -> None:
    """Advertised examples must complete in a clean subprocess."""
    if script.name == "fdiff_example.py" and find_spec("statsmodels") is None:
        pytest.skip("Fractional-differencing diagnostics require statsmodels")

    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=EXAMPLE_DIR.parent,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    assert result.returncode == 0, (
        f"{script.name} failed with exit {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
