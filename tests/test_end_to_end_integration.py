# mypy: disable-error-code="call-arg,no-untyped-def,operator"
"""End-to-end integration tests for complete ml4t-engineer workflow.

.. deprecated:: 0.1.0a8
    Feature outcome analysis has moved to ml4t-diagnostic.
    These tests require ml4t-diagnostic to be installed.

Tests the full feature engineering pipeline:
1. Relationships - Compute correlation matrices
2. Outcome - Analyze feature-outcome relationships (IC, importance, drift)
3. Selection - Systematic feature filtering
4. Visualization - Generate comprehensive plots

Note: Outcome/diagnostics functionality is available in the separate ml4t-diagnostic library.
Install with: pip install ml4t-diagnostic
"""

from __future__ import annotations

import pytest

# These tests require ml4t-diagnostic
try:
    from ml4t.diagnostic.evaluation import FeatureOutcome  # noqa: F401

    HAS_DIAGNOSTIC = True
except ImportError:
    HAS_DIAGNOSTIC = False

pytestmark = pytest.mark.skipif(
    not HAS_DIAGNOSTIC,
    reason=(
        "These integration tests require ml4t-diagnostic. Install with: pip install ml4t-diagnostic"
    ),
)


# Note: All tests in this file test the integration between ml4t-engineer
# and ml4t-diagnostic. Since outcome functionality has moved to ml4t-diagnostic,
# these tests are now optional and only run when ml4t-diagnostic is installed.
#
# For feature computation tests that don't require outcome analysis,
# see tests/test_*.py files.
#
# To run these tests, install ml4t-diagnostic:
#   pip install ml4t-diagnostic
#
# Then run:
#   pytest tests/test_end_to_end_integration.py -v


def test_placeholder():
    """Placeholder to indicate integration tests require ml4t-diagnostic."""
    # This test exists to show that the integration tests are available
    # but require ml4t-diagnostic to be installed.
    pass
