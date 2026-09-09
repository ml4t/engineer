"""Regression tests for the release qualification policy."""

from __future__ import annotations

import itertools
import re
import tomllib
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import Requirement

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
    assert "workflow_dispatch" in workflow["on"]
    assert "release/**" in workflow["on"]["push"]["branches"]

    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]
    actual = set(itertools.product(matrix["os"], matrix["python-version"]))

    stable = {"3.12", "3.13", "3.14"}
    platforms = {"ubuntu-latest", "macos-latest", "windows-latest"}
    assert actual == set(itertools.product(platforms, stable))
    assert "include" not in matrix


def test_each_matrix_cell_runs_all_release_checks_without_masking_failures() -> None:
    steps = _load_workflow("ci.yml")["jobs"]["test"]["steps"]
    commands = {step["name"]: step["run"] for step in steps if "run" in step and "name" in step}
    setup = next(
        step for step in steps if step.get("name") == "Set up Python ${{ matrix.python-version }}"
    )

    assert {
        "Verify candidate identity",
        "Run ty check",
        "Export installed-wheel test environment",
        "Install the candidate wheel in a clean environment",
        "Import the installed candidate",
        "Run the installed-wheel suite repeatedly",
        "Run documented workflows from the installed candidate",
    } <= commands.keys()
    assert setup["with"] == {"python-version": "${{ matrix.python-version }}"}
    identity = commands["Verify candidate identity"]
    assert "candidate.py verify candidate" in identity
    assert '--expected-commit "${{ github.sha }}"' in identity
    assert '--expected-tree "$(git rev-parse HEAD^{tree})"' in identity
    assert (
        next(step for step in steps if step.get("name") == "Verify candidate identity")["shell"]
        == "bash"
    )

    assert "--python-version ${{ matrix.python-version }}" in commands["Run ty check"]
    assert "--python .artifact-venv --no-project" in commands["Run ty check"]
    assert "--exclude" not in commands["Run ty check"]
    assert "--extra-search-path" not in commands["Run ty check"]

    test_command = commands["Run the installed-wheel suite repeatedly"]
    assert "for iteration in {1..10}" in test_command
    assert "--python .artifact-venv --no-project" in test_command
    assert "python -m pytest tests/" in test_command
    assert "set +e" not in test_command
    assert "PYTEST_EXIT" not in test_command
    assert "exit 0" not in test_command

    export_command = commands["Export installed-wheel test environment"]
    assert "--group dev --extra ta --extra store --extra viz" in export_command
    assert "--no-emit-project" in export_command
    wheel_install = commands["Install the candidate wheel in a clean environment"]
    assert 'UV_CACHE_DIR="${{ runner.temp }}/fresh-wheel-cache"' in wheel_install
    assert 'uv pip install --python .artifact-venv "$wheel"' in wheel_install
    assert "--requirements" in wheel_install
    assert wheel_install.index('"$wheel"') < wheel_install.index("--requirements")

    step_names = [step.get("name") for step in steps]
    assert step_names.index(
        "Install the candidate wheel in a clean environment"
    ) < step_names.index("Run the installed-wheel suite repeatedly")


def test_release_publishes_only_the_qualified_artifact() -> None:
    ci_jobs = _load_workflow("ci.yml")["jobs"]
    candidate_steps = ci_jobs["build-candidate"]["steps"]
    candidate_commands = {
        step["name"]: step["run"] for step in candidate_steps if "run" in step and "name" in step
    }
    assert ci_jobs["build-candidate"]["needs"] == [
        "lint",
        "typecheck",
        "security",
        "coverage",
        "documentation",
    ]
    assert candidate_commands["Build source and wheel artifacts once"] == (
        "uv build --out-dir candidate/dist"
    )
    version_step = next(
        step for step in candidate_steps if step.get("name") == "Set release candidate version"
    )
    assert version_step["if"] == "${{ inputs.release_version != '' }}"
    assert version_step["env"] == {"RELEASE_VERSION": "${{ inputs.release_version }}"}
    assert "SETUPTOOLS_SCM_PRETEND_VERSION" in version_step["run"]
    manifest = candidate_commands["Record candidate commit, tree, version, and SHA256 digests"]
    assert "candidate.py create candidate" in manifest
    assert "github.sha" in manifest
    assert "git rev-parse HEAD^{tree}" in manifest
    assert (
        "twine check candidate/dist/*"
        in candidate_commands["Validate artifact metadata and manifest"]
    )

    upload = next(
        step for step in candidate_steps if "actions/upload-artifact@" in step.get("uses", "")
    )
    assert upload["with"]["name"] == "release-candidate"
    assert ci_jobs["build"]["needs"] == ["build-candidate", "test"]
    assert not any("uv build" in step.get("run", "") for step in ci_jobs["build"]["steps"])

    release = _load_workflow("release.yml")
    release_jobs = release["jobs"]
    assert set(release["on"]["workflow_dispatch"]["inputs"]) == {"version", "candidate_commit"}
    assert release_jobs["select-candidate"]["needs"] == [
        "validate",
        "ecosystem-qualification",
        "qualification",
    ]
    assert release_jobs["qualification"]["uses"] == "./.github/workflows/ci.yml"
    assert release_jobs["qualification"]["with"]["release_version"] == (
        "${{ needs.validate.outputs.version }}"
    )
    assert release_jobs["docs"]["needs"] == ["validate", "select-candidate"]
    assert release_jobs["publish"]["needs"] == [
        "validate",
        "ecosystem-qualification",
        "select-candidate",
        "docs",
    ]
    assert release_jobs["github-release"]["needs"] == ["validate", "publish"]
    assert release_jobs["verify-release"]["needs"] == ["validate", "github-release"]

    publish_steps = release_jobs["publish"]["steps"]
    publisher = next(
        step for step in publish_steps if "pypa/gh-action-pypi-publish@" in step.get("uses", "")
    )
    assert publisher["with"]["packages-dir"] == "candidate/dist/"
    assert not any(
        "uv build" in step.get("run", "")
        for job in release_jobs.values()
        for step in job.get("steps", [])
    )

    release_step = next(
        step
        for step in release_jobs["github-release"]["steps"]
        if step.get("name") == "Create tag and GitHub release from the candidate commit"
    )
    assert '--target "$CANDIDATE_COMMIT"' in release_step["run"]
    assert "candidate/candidate.json" in release_step["run"]

    verify_commands = {
        step["name"]: step["run"]
        for step in release_jobs["verify-release"]["steps"]
        if "run" in step and "name" in step
    }
    assert (
        "release.py verify candidate"
        in verify_commands["Verify PyPI metadata and artifact SHA256 digests"]
    )
    assert (
        "release.py smoke-test"
        in verify_commands["Install the published wheel and run the README quick start"]
    )
    assert (
        "candidate.py verify released"
        in verify_commands["Verify GitHub release artifacts against the manifest"]
    )
    assert (
        "verify_docs_deployment.py"
        in verify_commands["Repeat deployed documentation identity check"]
    )


def test_ci_enforces_independent_line_and_branch_coverage_thresholds() -> None:
    coverage_steps = _load_workflow("ci.yml")["jobs"]["coverage"]["steps"]
    commands = {
        step["name"]: step["run"] for step in coverage_steps if "run" in step and "name" in step
    }
    measurement = next(
        step for step in coverage_steps if step.get("name") == "Measure line and branch coverage"
    )

    assert measurement["env"] == {"NUMBA_DISABLE_JIT": "1"}
    assert "--cov-report=json:coverage.json" in commands["Measure line and branch coverage"]
    assert commands["Enforce release thresholds"] == (
        "uv run python scripts/check_coverage.py coverage.json"
    )


def test_standalone_typecheck_installs_optional_type_dependencies() -> None:
    steps = _load_workflow("ci.yml")["jobs"]["typecheck"]["steps"]
    install = next(step["run"] for step in steps if step.get("name") == "Install dependencies")

    assert "--extra store --extra viz" in install


def test_ci_audits_core_and_complete_locked_environments() -> None:
    steps = _load_workflow("ci.yml")["jobs"]["security"]["steps"]
    commands = {step["name"]: step["run"] for step in steps if "run" in step and "name" in step}

    export = commands["Export locked environments"]
    assert "--no-dev --no-emit-project" in export
    assert "--all-extras --all-groups --no-emit-project" in export

    audit = commands["Audit runtime and contributor environments"]
    assert audit.count("python -m pip_audit") == 2
    assert "core.txt" in audit
    assert "complete.txt" in audit


def test_core_dependency_uses_stable_bounded_specs_contract() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    requirement = next(
        Requirement(value)
        for value in project["project"]["dependencies"]
        if Requirement(value).name == "ml4t-specs"
    )
    assert str(requirement.specifier) == "<0.2.0,>=0.1.0"
    specs = next(package for package in lock["package"] if package["name"] == "ml4t-specs")
    assert specs["version"] in requirement.specifier


def test_package_supports_python_312_through_314() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]

    assert project["project"]["requires-python"] == ">=3.12,<3.15"
    assert "numba>=0.57.0" in dependencies
    assert "statsmodels>=0.14.0" in project["project"]["optional-dependencies"]["stats"]
    assert "pyarrow>=14.0.0" in project["project"]["optional-dependencies"]["store"]
    assert "pyarrow>=14.0.0" in project["project"]["optional-dependencies"]["viz"]
    assert not any(
        dependency.startswith(("pyarrow", "scipy", "scikit-learn", "statsmodels"))
        for dependency in dependencies
    )
    assert "pydantic>=2.0.0,<3" in dependencies
    assert not any(dependency.startswith("matplotlib") for dependency in dependencies)
    assert "matplotlib>=3.7.0" in project["project"]["optional-dependencies"]["viz"]


def test_package_metadata_declares_stable_status() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

    assert "Development Status :: 5 - Production/Stable" in project["classifiers"]
    assert "Development Status :: 4 - Beta" not in project["classifiers"]


def test_all_external_actions_are_pinned_to_full_commit_shas() -> None:
    actions = []
    for path in WORKFLOWS.glob("*.yml"):
        actions.extend(_external_actions(_load_workflow(path.name)))

    assert actions
    assert all(SHA_PIN.fullmatch(action) for action in actions), actions
