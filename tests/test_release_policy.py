"""Regression tests for the release qualification policy."""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"
SHA_PIN = re.compile(r"^[^@]+@[0-9a-f]{40}$")


def _load_workflow(name: str) -> dict[str, Any]:
    with (WORKFLOWS / name).open(encoding="utf-8") as workflow_file:
        return yaml.load(workflow_file, Loader=yaml.BaseLoader)


def _external_actions(value: Any) -> list[str]:
    if isinstance(value, dict):
        actions = []
        for key, child in value.items():
            if key == "uses" and isinstance(child, str) and not child.startswith("./"):
                actions.append(child)
            actions.extend(_external_actions(child))
        return actions
    if isinstance(value, list):
        return [action for child in value for action in _external_actions(child)]
    return []


def test_ci_qualifies_required_python_platform_matrix() -> None:
    workflow = _load_workflow("ci.yml")
    assert "workflow_call" in workflow["on"]

    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    actual = set(itertools.product(matrix["os"], matrix["python-version"]))
    actual.update((entry["os"], entry["python-version"]) for entry in matrix["include"])

    stable = {"3.12", "3.13", "3.14"}
    prerelease = {"3.15"}
    platforms = {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert actual == set(itertools.product(platforms, stable | prerelease))


def test_each_matrix_cell_runs_all_release_checks_without_masking_failures() -> None:
    steps = _load_workflow("ci.yml")["jobs"]["test"]["steps"]
    commands = {step["name"]: step["run"] for step in steps if "run" in step}
    setup = next(
        step for step in steps if step.get("name") == "Set up Python ${{ matrix.python-version }}"
    )

    assert {
        "Install dependencies",
        "Import package",
        "Run tests",
        "Run ty check",
        "Build package",
    } <= commands.keys()
    assert "--extra ta --extra store" in commands["Install dependencies"]
    assert setup["with"] == {
        "python-version": "${{ matrix.python-version }}",
        "allow-prereleases": "true",
    }
    assert "--python-version ${{ matrix.python-version }}" in commands["Run ty check"]
    assert "--python python" in commands["Build package"]

    test_command = commands["Run tests"]
    assert "pytest" in test_command
    assert "set +e" not in test_command
    assert "PYTEST_EXIT" not in test_command
    assert "exit 0" not in test_command


def test_release_publishes_only_the_qualified_artifact() -> None:
    ci_jobs = _load_workflow("ci.yml")["jobs"]
    build_steps = ci_jobs["build"]["steps"]
    build_commands = {step["name"]: step["run"] for step in build_steps if "run" in step}
    assert ci_jobs["build"]["needs"] == ["lint", "typecheck", "security", "test", "coverage"]
    assert "twine check dist/*" in build_commands["Validate package metadata"]
    assert "uv pip install" in build_commands["Validate wheel installation"]
    assert 'python -c "import ml4t.engineer"' in build_commands["Validate wheel installation"]

    release_jobs = _load_workflow("release.yml")["jobs"]

    assert release_jobs["qualification"]["uses"] == "./.github/workflows/ci.yml"
    assert release_jobs["publish"]["needs"] == "qualification"

    publish_steps = release_jobs["publish"]["steps"]
    download = next(step for step in publish_steps if step["name"] == "Download build artifacts")
    assert download["with"] == {"name": "dist", "path": "dist/"}


def test_ci_enforces_independent_line_and_branch_coverage_thresholds() -> None:
    coverage_steps = _load_workflow("ci.yml")["jobs"]["coverage"]["steps"]
    commands = {step["name"]: step["run"] for step in coverage_steps if "run" in step}
    measurement = next(
        step for step in coverage_steps if step.get("name") == "Measure line and branch coverage"
    )

    assert measurement["env"] == {"NUMBA_DISABLE_JIT": "1"}
    assert "--cov-report=json:coverage.json" in commands["Measure line and branch coverage"]
    assert commands["Enforce release thresholds"] == (
        "uv run python scripts/check_coverage.py coverage.json"
    )


def test_ci_audits_core_and_complete_locked_environments() -> None:
    steps = _load_workflow("ci.yml")["jobs"]["security"]["steps"]
    commands = {step["name"]: step["run"] for step in steps if "run" in step}

    export = commands["Export locked environments"]
    assert "--no-dev --no-emit-project" in export
    assert "--all-extras --all-groups --no-emit-project" in export

    audit = commands["Audit locked environments"]
    assert audit.count("pip-audit --requirement") == 2
    assert "core.txt" in audit
    assert "complete.txt" in audit


def test_all_external_actions_are_pinned_to_full_commit_shas() -> None:
    actions = []
    for path in WORKFLOWS.glob("*.yml"):
        actions.extend(_external_actions(_load_workflow(path.name)))

    assert actions
    assert all(SHA_PIN.fullmatch(action) for action in actions), actions
