"""Tests for the release coverage threshold checker."""

import json
from pathlib import Path

import pytest

from scripts.check_coverage import RELEASE_CRITICAL_MODULES, check_coverage


def _write_report(path: Path, *, lines: tuple[int, int], branches: tuple[int, int]) -> None:
    critical_summary = {
        "covered_lines": 95,
        "num_statements": 100,
        "covered_branches": 90,
        "num_branches": 100,
    }
    path.write_text(
        json.dumps(
            {
                "totals": {
                    "covered_lines": lines[0],
                    "num_statements": lines[1],
                    "covered_branches": branches[0],
                    "num_branches": branches[1],
                },
                "files": {
                    module: {"summary": critical_summary.copy()}
                    for module in RELEASE_CRITICAL_MODULES
                },
            }
        )
    )


def test_release_coverage_thresholds_are_independent(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, lines=(900, 1000), branches=(850, 1000))

    assert check_coverage(report) == (90.0, 85.0)

    _write_report(report, lines=(899, 1000), branches=(850, 1000))
    with pytest.raises(ValueError, match="line coverage"):
        check_coverage(report)

    _write_report(report, lines=(900, 1000), branches=(849, 1000))
    with pytest.raises(ValueError, match="branch coverage"):
        check_coverage(report)


@pytest.mark.parametrize(
    "document",
    [
        {},
        {"totals": []},
        {
            "totals": {
                "covered_lines": True,
                "num_statements": 1,
                "covered_branches": 1,
                "num_branches": 1,
            }
        },
        {
            "totals": {
                "covered_lines": 0,
                "num_statements": 0,
                "covered_branches": 0,
                "num_branches": 0,
            }
        },
    ],
)
def test_invalid_coverage_reports_fail_closed(tmp_path: Path, document: object) -> None:
    report = tmp_path / "coverage.json"
    report.write_text(json.dumps(document))

    with pytest.raises(ValueError):
        check_coverage(report)


def test_missing_release_critical_module_fails_closed(tmp_path: Path) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, lines=(900, 1000), branches=(850, 1000))
    document = json.loads(report.read_text())
    document["files"].pop(RELEASE_CRITICAL_MODULES[0])
    report.write_text(json.dumps(document))

    with pytest.raises(ValueError, match="is missing from the report"):
        check_coverage(report)


@pytest.mark.parametrize(
    ("summary_update", "message"),
    [
        ({"covered_lines": 949, "num_statements": 1000}, "line coverage"),
        ({"covered_branches": 899, "num_branches": 1000}, "branch coverage"),
    ],
)
def test_release_critical_thresholds_are_independent(
    tmp_path: Path,
    summary_update: dict[str, int],
    message: str,
) -> None:
    report = tmp_path / "coverage.json"
    _write_report(report, lines=(900, 1000), branches=(850, 1000))
    document = json.loads(report.read_text())
    first_module = RELEASE_CRITICAL_MODULES[0]
    document["files"][first_module]["summary"].update(summary_update)
    report.write_text(json.dumps(document))

    with pytest.raises(ValueError, match=rf"{first_module} {message}"):
        check_coverage(report)
