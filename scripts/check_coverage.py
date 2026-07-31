"""Enforce the release line and branch coverage thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

LINE_THRESHOLD = 90.0
BRANCH_THRESHOLD = 85.0
CRITICAL_LINE_THRESHOLD = 95.0
CRITICAL_BRANCH_THRESHOLD = 90.0

# These modules implement or govern the documented primary workflows. Feature-family
# implementations remain subject to the project thresholds and independent numerical
# reference tests; this higher set covers shared orchestration, leakage-sensitive data
# preparation, stateful bar sampling, and label generation.
RELEASE_CRITICAL_MODULES = (
    "src/ml4t/engineer/api.py",
    "src/ml4t/engineer/bars/base.py",
    "src/ml4t/engineer/bars/imbalance.py",
    "src/ml4t/engineer/bars/run.py",
    "src/ml4t/engineer/bars/tick.py",
    "src/ml4t/engineer/bars/vectorized.py",
    "src/ml4t/engineer/bars/volume.py",
    "src/ml4t/engineer/config/base.py",
    "src/ml4t/engineer/config/data_contract.py",
    "src/ml4t/engineer/config/experiment.py",
    "src/ml4t/engineer/config/labeling.py",
    "src/ml4t/engineer/config/preprocessing_config.py",
    "src/ml4t/engineer/config/spec_bridge.py",
    "src/ml4t/engineer/core/decorators.py",
    "src/ml4t/engineer/core/dispatch.py",
    "src/ml4t/engineer/core/lookbacks.py",
    "src/ml4t/engineer/core/registry.py",
    "src/ml4t/engineer/core/schemas.py",
    "src/ml4t/engineer/core/validation.py",
    "src/ml4t/engineer/dataset.py",
    "src/ml4t/engineer/discovery/catalog.py",
    "src/ml4t/engineer/labeling/atr_barriers.py",
    "src/ml4t/engineer/labeling/calendar.py",
    "src/ml4t/engineer/labeling/horizon_labels.py",
    "src/ml4t/engineer/labeling/meta_labels.py",
    "src/ml4t/engineer/labeling/numba_ops.py",
    "src/ml4t/engineer/labeling/percentile_labels.py",
    "src/ml4t/engineer/labeling/triple_barrier.py",
    "src/ml4t/engineer/labeling/uniqueness.py",
    "src/ml4t/engineer/labeling/utils.py",
    "src/ml4t/engineer/preprocessing.py",
)


def _percentage(covered: int, total: int, metric: str) -> float:
    if total <= 0:
        raise ValueError(f"Coverage report contains no {metric}")
    return 100.0 * covered / total


def _integer(totals: dict[str, Any], field: str) -> int:
    value = totals.get(field)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"Coverage report field {field!r} must be an integer")
    return value


def _rates(
    summary: object,
    context: str,
    *,
    allow_no_branches: bool = False,
) -> tuple[float, float]:
    if not isinstance(summary, dict):
        raise ValueError(f"Coverage report {context} summary must be a mapping")
    line_rate = _percentage(
        _integer(summary, "covered_lines"),
        _integer(summary, "num_statements"),
        f"statements for {context}",
    )
    covered_branches = _integer(summary, "covered_branches")
    num_branches = _integer(summary, "num_branches")
    branch_rate = (
        100.0
        if allow_no_branches and num_branches == 0
        else _percentage(covered_branches, num_branches, f"branches for {context}")
    )
    return line_rate, branch_rate


def check_coverage(report_path: Path) -> tuple[float, float]:
    """Read a coverage.py JSON report and enforce both release thresholds."""
    try:
        report = json.loads(report_path.read_text())
        totals = report["totals"]
        files = report["files"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Invalid coverage report {report_path}: {exc}") from exc

    if not isinstance(totals, dict):
        raise ValueError("Coverage report totals must be a mapping")
    if not isinstance(files, dict):
        raise ValueError("Coverage report files must be a mapping")

    line_rate, branch_rate = _rates(totals, "totals")

    failures = []
    if line_rate < LINE_THRESHOLD:
        failures.append(f"line coverage {line_rate:.2f}% is below {LINE_THRESHOLD:.2f}%")
    if branch_rate < BRANCH_THRESHOLD:
        failures.append(f"branch coverage {branch_rate:.2f}% is below {BRANCH_THRESHOLD:.2f}%")

    for module in RELEASE_CRITICAL_MODULES:
        entry = files.get(module)
        if not isinstance(entry, dict) or "summary" not in entry:
            failures.append(f"release-critical module {module} is missing from the report")
            continue
        module_line_rate, module_branch_rate = _rates(
            entry["summary"], module, allow_no_branches=True
        )
        if module_line_rate < CRITICAL_LINE_THRESHOLD:
            failures.append(
                f"{module} line coverage {module_line_rate:.2f}% is below "
                f"{CRITICAL_LINE_THRESHOLD:.2f}%"
            )
        if module_branch_rate < CRITICAL_BRANCH_THRESHOLD:
            failures.append(
                f"{module} branch coverage {module_branch_rate:.2f}% is below "
                f"{CRITICAL_BRANCH_THRESHOLD:.2f}%"
            )
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
