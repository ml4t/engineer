"""Enforce the release line and branch coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LINE_THRESHOLD = 90.0
BRANCH_THRESHOLD = 85.0


def _percentage(covered: int, total: int, metric: str) -> float:
    if total <= 0:
        raise ValueError(f"Coverage report contains no {metric}")
    return 100.0 * covered / total


def _integer(totals: dict[str, Any], field: str) -> int:
    value = totals.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Coverage report field {field!r} must be an integer")
    return value


def check_coverage(report_path: Path) -> tuple[float, float]:
    """Read a coverage.py JSON report and enforce both release thresholds."""
    try:
        report = json.loads(report_path.read_text())
        totals = report["totals"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid coverage report {report_path}: {exc}") from exc

    if not isinstance(totals, dict):
        raise ValueError("Coverage report totals must be a mapping")

    line_rate = _percentage(
        _integer(totals, "covered_lines"),
        _integer(totals, "num_statements"),
        "statements",
    )
    branch_rate = _percentage(
        _integer(totals, "covered_branches"),
        _integer(totals, "num_branches"),
        "branches",
    )

    failures = []
    if line_rate < LINE_THRESHOLD:
        failures.append(f"line coverage {line_rate:.2f}% is below {LINE_THRESHOLD:.2f}%")
    if branch_rate < BRANCH_THRESHOLD:
        failures.append(f"branch coverage {branch_rate:.2f}% is below {BRANCH_THRESHOLD:.2f}%")
    if failures:
        raise ValueError("; ".join(failures))

    return line_rate, branch_rate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()

    try:
        line_rate, branch_rate = check_coverage(args.report)
    except ValueError as exc:
        parser.exit(1, f"Coverage check failed: {exc}\n")

    print(f"Coverage thresholds passed: lines {line_rate:.2f}%, branches {branch_rate:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
