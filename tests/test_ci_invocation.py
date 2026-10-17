"""Execute the configured installed-suite command, including abnormal exits."""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).parents[1]


def _bash() -> str:
    if os.name == "nt":
        git = shutil.which("git")
        if git is not None:
            for parent in Path(git).resolve().parents:
                for relative in ("bin/bash.exe", "usr/bin/bash.exe"):
                    candidate = parent / relative
                    if candidate.is_file():
                        return str(candidate)
        raise AssertionError("The configured Bash workflow requires Git Bash on Windows, not WSL")
    bash = shutil.which("bash")
    assert bash is not None, "The configured Bash workflow requires Bash"
    return bash


def _run_suite(
    tmp_path: Path, test: str, *, hook: str = "", options: str = "", native: bool = False
):
    workflow = yaml.load((ROOT / ".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
    command = next(
        step["run"]
        for step in workflow["jobs"]["test"]["steps"]
        if "python -m pytest tests/" in step.get("run", "")
    )
    # Bind the runner's interpreter input, without replacing its shell/exit logic.
    python = shlex.quote(Path(sys.executable).as_posix())
    command = command.replace("--python .artifact-venv", f"--python {python}")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/test_case.py").write_text(test)
    (tmp_path / "pytest.ini").write_text(f"[pytest]\naddopts = -s {options}\n")
    (tmp_path / "conftest.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "def pytest_sessionstart(session):\n"
        "    with Path('sessions').open('a') as stream:\n"
        "        stream.write('started\\n')\n"
        "    if os.name == 'posix':\n"
        "        import resource\n"
        "        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "    else:\n"
        "        import ctypes\n"
        "        ctypes.windll.kernel32.SetErrorMode(3)\n"
        "        ctypes.CDLL('ucrtbase')._set_abort_behavior(0, 3)\n" + hook
    )
    if native:
        from scripts.ci.native_stress import FEATURE_ORDER, MACD

        direct = tmp_path / MACD[0].split("::")[0]
        direct.parent.mkdir(parents=True)
        body = test.removeprefix("import time\nimport pytest\n")
        if body.startswith("VALUE ="):
            body = "pass\n"
        direct.write_text(
            "import time\nimport pytest\nclass TestMACDNumbaDirectCoverage:\n"
            + "".join("    " + line + "\n" for line in body.replace("():", "(self):").splitlines())
        )
        (tmp_path / MACD[1]).write_text(test)
        (tmp_path / FEATURE_ORDER[0].split("::")[0]).write_text(
            "\n".join(f"def {node.split('::')[1]}(): pass" for node in FEATURE_ORDER)
        )
        (tmp_path / "tests/features/test_rolling_map_first_call.py").write_text(test)
        command = (
            f"{python} "
            f"{shlex.quote((ROOT / 'scripts/ci/native_stress.py').as_posix())} --repetitions 2"
        )
    return subprocess.run(
        [_bash(), "--noprofile", "--norc", "-eo", "pipefail", "-c", command],
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


def test_native_stress_runs_fresh_ordered_processes_and_cold_call_control(tmp_path: Path) -> None:
    result = _run_suite(tmp_path, "def test_pass():\n    assert 2 + 2 == 4\n", native=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (tmp_path / "sessions").read_text().splitlines() == ["started"] * 5


@pytest.mark.parametrize("native", [False, True], ids=["full-suite", "native-stress"])
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
        (
            "def test_crashes():\n    import os\n"
            "    print('native-abort', flush=True)\n    os.abort()\n",
            "",
            "",
            "native-abort",
        ),
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
    tmp_path: Path, test: str, hook: str, options: str, message: str, native: bool
) -> None:
    result = _run_suite(tmp_path, test, hook=hook, options=options, native=native)
    assert result.returncode != 0, result.stdout + result.stderr
    assert message in result.stdout + result.stderr
