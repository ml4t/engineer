import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("execution", ["eager", "lazy"])
def test_registered_hurst_runs_without_optional_scipy_and_preserves_rs_ols_values(execution):
    workflow = """
import sys
sys.modules['scipy'] = None
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from ml4t.engineer import compute_features

close = 100 + np.cumsum(np.random.default_rng(42).normal(0, .1, 300))
prices = pl.DataFrame({
    'timestamp': [datetime(2025, 1, 1) + timedelta(hours=i) for i in range(300)],
    'open': close, 'high': close + 1, 'low': close - 1, 'close': close,
    'volume': np.ones(300),
})
data = prices.lazy() if sys.argv[1] == 'lazy' else prices
result = compute_features(data, ['hurst_exponent'], calendar=None)
if isinstance(result, pl.LazyFrame):
    result = result.collect()
values = result['hurst_exponent'].to_numpy()
assert np.isnan(values[:99]).all()
assert np.isfinite(values[99:]).all()
np.testing.assert_allclose(
    values[[99, 100, 150, 299]],
    [0.6039476488469495, 0.6072413764071118, 0.6468637069496758, 0.6657816121301358],
    rtol=1e-12, atol=1e-12,
)
assert result.select(prices.columns).equals(prices)
"""
    completed = subprocess.run(
        [sys.executable, "-c", workflow, execution],
        env={**os.environ, "NUMBA_DISABLE_JIT": "0"},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
