"""Execute the configured installed-suite command, including abnormal exits."""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def _run_suite(tmp_path: Path, test: str, *, hook: str = "", options: str = ""):
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    command = next(
        step["run"]
        for step in workflow["jobs"]["test"]["steps"]
        if "python -m pytest tests/" in step.get("run", "")
    )
    # Bind the runner's interpreter input, without replacing its shell/exit logic.
    command = command.replace("--python .artifact-venv", f"--python {shlex.quote(sys.executable)}")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_case.py").write_text(test)
    (tmp_path / "pytest.ini").write_text(f"[pytest]\naddopts = {options}\n")
    (tmp_path / "conftest.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def pytest_sessionstart(session):\n"
        "    with Path('sessions').open('a') as stream:\n"
        "        stream.write('started\\n')\n"
        "    if os.name == 'posix':\n"
        "        import resource\n"
        "        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n" + hook
    )
    return subprocess.run(
        ["bash", "--noprofile", "--norc", "-eo", "pipefail", "-c", command],
        cwd=tmp_path,
        env={
            **os.environ,
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
            "PYTEST_PLUGINS": "pytest_cov.plugin,pytest_timeout",
            "PYTEST_ADDOPTS": "",
        },
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )


def test_routine_wheel_cell_runs_the_full_suite_once(tmp_path: Path) -> None:
    result = _run_suite(tmp_path, "def test_pass():\n    assert 2 + 2 == 4\n")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "sessions").read_text().splitlines() == ["started"]


@pytest.mark.parametrize(
    ("test", "hook", "options", "message"),
    [
        ("def test_fails():\n    assert False, 'real-test-failure'\n", "", "", "real-test-failure"),
        ("def test_broken(:\n", "", "", "SyntaxError"),
        ("VALUE = 1\n", "", "", "no tests ran"),
        ("def test_pass(): pass\n", "", "--not-a-real-pytest-option", "unrecognized arguments"),
        (
            "def test_pass(): pass\n",
            "def pytest_collection(session):\n    raise RuntimeError('real-internal-error')\n",
            "",
            "real-internal-error",
        ),
        ("def test_crashes():\n    import os\n    os.abort()\n", "", "", ""),
        (
            "import time\nimport pytest\n@pytest.mark.timeout(0.2)\n"
            "def test_times_out():\n    time.sleep(30)\n",
            "",
            "",
            "Timeout",
        ),
        (
            "def test_pass(): pass\n",
            "def pytest_sessionfinish(session, exitstatus):\n"
            "    assert exitstatus == 0\n"
            "    print('1 passed', flush=True)\n"
            "    os.abort()\n",
            "",
            "1 passed",
        ),
    ],
    ids=[
        "failure",
        "collection",
        "empty",
        "usage",
        "internal",
        "crash",
        "timeout",
        "pass-then-crash",
    ],
)
def test_configured_wheel_command_propagates_real_failure(
    tmp_path: Path, test: str, hook: str, options: str, message: str
) -> None:
    result = _run_suite(tmp_path, test, hook=hook, options=options)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
