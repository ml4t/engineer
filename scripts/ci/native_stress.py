"""Fresh-process regressions for native bounds, call ordering and cold compilation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

MACD = [
    "tests/features/momentum/test_momentum_numba_coverage.py::TestMACDNumbaDirectCoverage",
    "tests/features/momentum/test_macd.py",
]
FEATURE_ORDER = [
    "tests/test_all_features_integration.py::test_all_features_compute_individually",
    "tests/test_all_features_integration.py::test_all_features_compute_together",
    "tests/test_all_features_integration.py::test_all_features_isolate_panel_assets",
]


def _run(nodes: list[str]) -> None:
    subprocess.run(
        [sys.executable, "-m", "pytest", *nodes, "-q", "--tb=short", "--no-cov"],
        check=True,
        timeout=600,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", action="store_true")
    parser.add_argument("--repetitions", type=int, default=10)
    args = parser.parse_args()
    if args.matrix:
        workflow = yaml.load(Path(".github/workflows/ci.yml").read_text(), Loader=yaml.BaseLoader)
        print("matrix=" + json.dumps(workflow["jobs"]["test"]["strategy"]["matrix"]))
        return
    if args.repetitions < 1:
        parser.error("repetitions must be positive")
    for iteration in range(args.repetitions):
        print(f"Fresh MACD process {iteration + 1}/{args.repetitions}", flush=True)
        _run(MACD if iteration % 2 == 0 else list(reversed(MACD)))
    for nodes in (FEATURE_ORDER, list(reversed(FEATURE_ORDER))):
        print("Fresh public feature-order process", flush=True)
        _run(nodes)
    print("Cold-call regressions, each with a fresh interpreter and empty Numba cache", flush=True)
    _run(["tests/features/test_rolling_map_first_call.py"])


if __name__ == "__main__":
    main()
